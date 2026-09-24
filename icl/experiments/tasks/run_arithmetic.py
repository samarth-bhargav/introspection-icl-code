"""Full math (emotion-gated arithmetic) introspection run in a single model load.

Two phases share one model load:

  Phase 1 (type-1 c_max calibration): fix K = --k_type1, sweep --cmax_grid and
      record accuracy at each c_max_fraction.  The c_max maximizing overall judge
      accuracy is selected as the operating point (ties broken toward the
      smaller c_max, i.e. the least activation perturbation).

  Phase 2 (type-2 K sweep): using the selected c_max, sweep K over --k_values.

Each phase uses --prompt_variations prompt texts x --n_tests samples per config,
with per-sample logs carrying prompt_variation_id and the target/distractor/none
condition so distractor-specific and per-prompt stats are plottable.

Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m icl.experiments.tasks.run_arithmetic \
        --model qwen3-32b --gpu 0 --k_values 0-20 --k_type1 10 \
        --prompt_variations 10 --n_tests 30 --max_operand 9 \
        --target_prob 0.5 --distractor_prob 0.25 --max_new_tokens 8 --temperature 0 \
        --judge_base_url http://127.0.0.1:8002/v1 --judge_model synonym-judge \
        --out_dir evals/regen/generation_qwen3-32b/math
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .gated_arithmetic import (
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    run_emotion_math_multiplier,
)
from .task_utils import load_artifacts


def _parse_k_values(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(part))
    deduped = list(dict.fromkeys(values))
    if not deduped:
        raise ValueError("no K values parsed")
    if any(k < 0 for k in deduped):
        raise ValueError(f"K values must be non-negative: {deduped}")
    return deduped


def _parse_cmax_grid(text: str | None) -> list[float]:
    if not text:
        return [round(i / 10, 1) for i in range(11)]  # 0.0 .. 1.0 step 0.1
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_concepts(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def _accuracy(bucket: dict[str, Any] | None) -> float | None:
    if not bucket:
        return None
    return bucket.get("accuracy")


def _injection_accuracies(payload: dict[str, Any]) -> dict[str, float | None]:
    by_injection = payload.get("by_injection") or {}
    return {kind: _accuracy(by_injection.get(kind)) for kind in ("target", "distractor", "none")}


def _prompt_variation_accuracies(payload: dict[str, Any]) -> dict[str, float | None]:
    agg = (payload.get("row_aggregates") or {}).get("by_prompt_variation") or {}
    return {vid: _accuracy(bucket) for vid, bucket in agg.items()}


def _config_summary(payload: dict[str, Any], *, k: int, cmax: float, path: Path) -> dict[str, Any]:
    return {
        "k": k,
        "cmax_fraction": cmax,
        "path": str(path),
        "n_samples": payload.get("n_samples"),
        "n_prompt_variations": payload.get("n_prompt_variations"),
        "overall_accuracy": _accuracy(payload.get("overall")),
        "control_accuracy": _accuracy(payload.get("control_overall")),
        "by_injection_accuracy": _injection_accuracies(payload),
        "by_prompt_variation_accuracy": _prompt_variation_accuracies(payload),
        "wall_seconds": payload.get("wall_seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3-32b")
    parser.add_argument("--gpu", default=None, help="optional CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--emotion", default="anger",
                        help="fixed target emotion; used only when --emotion_pool is empty")
    parser.add_argument(
        "--emotion_pool",
        default="anger,disgust,fear,joy,love,sadness",
        help="comma list of emotions; target is drawn at random per test sample "
        "(named in the system prompt) and distractors are the other pool emotions. "
        "Pass an empty string to fall back to the single fixed --emotion.",
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--k_values", default="0-20", help="type-2 K sweep, e.g. 0-20")
    parser.add_argument("--k_type1", type=int, default=10, help="fixed K for the type-1 c_max sweep")
    parser.add_argument("--cmax_grid", default=None, help="comma list; default 0.0..1.0 step 0.1")
    parser.add_argument("--prompt_variations", type=int, default=10)
    parser.add_argument("--n_tests", type=int, default=30, help="samples per prompt variation per config")
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--max_operand", type=int, default=9)
    parser.add_argument("--target_prob", type=float, default=0.5)
    parser.add_argument("--distractor_prob", type=float, default=0.25)
    parser.add_argument("--distractor_concepts", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--judge_base_url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge_timeout", type=float, default=30.0)
    parser.add_argument("--judge_debug", action="store_true")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--no_control", action="store_true")
    parser.add_argument(
        "--fixed_cmax",
        type=float,
        default=None,
        help="skip type-1 selection and use this c_max for the type-2 sweep",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env")
    except ImportError:
        pass

    import torch
    from icl import get_model_and_tokenizer
    from icl.common.prompt_variations import VARIATIONS

    torch.set_grad_enabled(False)

    k_values = _parse_k_values(args.k_values)
    cmax_grid = _parse_cmax_grid(args.cmax_grid)
    distractor_concepts = _parse_concepts(args.distractor_concepts)
    emotion_pool = _parse_concepts(args.emotion_pool)  # None when empty string
    if args.prompt_variations > 0:
        available = VARIATIONS["math_introspection"]
        if args.prompt_variations > len(available):
            raise ValueError(
                f"requested {args.prompt_variations} prompt variations, "
                f"only {len(available)} defined for math_introspection"
            )
        prompt_variations = available[: args.prompt_variations]
    else:
        prompt_variations = None

    out_dir = Path(args.out_dir)
    type1_dir = out_dir / "type1_cmax_sweep"
    type2_dir = out_dir / "type2_k_sweep"
    type1_dir.mkdir(parents=True, exist_ok=True)
    type2_dir.mkdir(parents=True, exist_ok=True)

    n_pv = args.prompt_variations if args.prompt_variations > 0 else 1
    per_config = n_pv * args.n_rollouts * args.n_tests
    emo_desc = f"emotion_pool={emotion_pool}" if emotion_pool else f"emotion={args.emotion}"
    print(
        f"[run_arithmetic] model={args.model} {emo_desc} "
        f"prompt_variations={n_pv} n_tests={args.n_tests} -> {per_config} samples/config | "
        f"type1 K={args.k_type1} cmax_grid={cmax_grid} | type2 K={k_values[0]}..{k_values[-1]} | "
        f"judge={args.judge_base_url} out={out_dir}",
        flush=True,
    )

    model, tok = get_model_and_tokenizer(args.model)
    library, cmax = load_artifacts(args.model)

    def _run(*, k: int, cmax_fraction: float, out_path: Path) -> dict[str, Any]:
        return run_emotion_math_multiplier(
            model,
            tok,
            library,
            cmax,
            model_name=args.model,
            emotion=args.emotion,
            layer=args.layer,
            cmax_fraction=cmax_fraction,
            n_demos=k,
            n_tests=args.n_tests,
            n_rollouts=args.n_rollouts,
            max_operand=args.max_operand,
            emotion_prob=args.target_prob,
            distractor_prob=args.distractor_prob,
            distractor_concepts=distractor_concepts,
            emotion_pool=emotion_pool,
            prompt_variations=prompt_variations,
            balanced_tests=True,
            seed=args.seed + 100000 * k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            judge_base_url=args.judge_base_url,
            judge_model=args.judge_model,
            judge_timeout=args.judge_timeout,
            judge_debug=args.judge_debug,
            with_control=not args.no_control,
            out_path=out_path,
            verbose=True,
        )

    t0 = time.time()

    # ── Phase 1: type-1 c_max calibration at fixed K ──────────────────
    type1_summaries: list[dict[str, Any]] = []
    if args.fixed_cmax is not None:
        chosen_cmax = args.fixed_cmax
        print(f"[run_arithmetic] skipping type-1 sweep; using fixed cmax={chosen_cmax}", flush=True)
    else:
        print(f"[run_arithmetic] === Phase 1: type-1 c_max sweep @ K={args.k_type1} ===", flush=True)
        for cf in cmax_grid:
            out_path = type1_dir / f"math_{args.model}_k{args.k_type1}_cmax{cf:g}.json"
            payload = _run(k=args.k_type1, cmax_fraction=cf, out_path=out_path)
            summary = _config_summary(payload, k=args.k_type1, cmax=cf, path=out_path)
            type1_summaries.append(summary)
            print(
                f"[run_arithmetic] type1 cmax={cf:g} overall_acc={summary['overall_accuracy']:.3f} "
                f"target={summary['by_injection_accuracy'].get('target')} "
                f"none={summary['by_injection_accuracy'].get('none')}",
                flush=True,
            )
        # Select operating c_max: max overall accuracy, tie-break smaller cmax.
        chosen = max(
            type1_summaries,
            key=lambda s: ((s["overall_accuracy"] or 0.0), -s["cmax_fraction"]),
        )
        chosen_cmax = chosen["cmax_fraction"]
        print(
            f"[run_arithmetic] selected cmax_fraction={chosen_cmax:g} "
            f"(overall_acc={chosen['overall_accuracy']:.3f})",
            flush=True,
        )

    # ── Phase 2: type-2 K sweep at the selected c_max ─────────────────
    print(f"[run_arithmetic] === Phase 2: type-2 K sweep @ cmax={chosen_cmax:g} ===", flush=True)
    type2_summaries: list[dict[str, Any]] = []
    for k in k_values:
        out_path = type2_dir / f"math_{args.model}_k{k}.json"
        payload = _run(k=k, cmax_fraction=chosen_cmax, out_path=out_path)
        summary = _config_summary(payload, k=k, cmax=chosen_cmax, path=out_path)
        type2_summaries.append(summary)
        print(
            f"[run_arithmetic] type2 K={k} overall_acc={summary['overall_accuracy']:.3f} "
            f"target={summary['by_injection_accuracy'].get('target')} "
            f"distractor={summary['by_injection_accuracy'].get('distractor')} "
            f"none={summary['by_injection_accuracy'].get('none')}",
            flush=True,
        )

    summary_payload = {
        "experiment": "math_full_introspection",
        "task": "math",
        "model": args.model,
        "emotion": args.emotion,
        "layer": args.layer,
        "n_prompt_variations": n_pv,
        "n_tests": args.n_tests,
        "n_rollouts": args.n_rollouts,
        "samples_per_config": per_config,
        "max_operand": args.max_operand,
        "target_prob": args.target_prob,
        "distractor_prob": args.distractor_prob,
        "none_prob": 1.0 - args.target_prob - args.distractor_prob,
        "type1": {
            "k": args.k_type1,
            "cmax_grid": cmax_grid,
            "by_cmax": type1_summaries,
        },
        "selected_cmax_fraction": chosen_cmax,
        "type2": {
            "k_values": k_values,
            "cmax_fraction": chosen_cmax,
            "by_k": type2_summaries,
        },
        "generation": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature},
        "judge": {
            "base_url": args.judge_base_url,
            "model": args.judge_model,
            "timeout": args.judge_timeout,
        },
        "out_dir": str(out_dir),
        "timestamp": datetime.now().isoformat(),
        "wall_seconds": time.time() - t0,
    }
    summary_path = out_dir / f"summary_math_full_{args.model}.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"[run_arithmetic] wrote {summary_path} (wall={summary_payload['wall_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
