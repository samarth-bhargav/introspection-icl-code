"""Calibration (type1) + K-sweep (type2) for the amendment-successor task.

Loads one model once and runs two phases, mirroring the magnitude/layer
introspection convention (``icl.experiments.introspection_sweep``):

  type1 / calibration : prompt variations, fixed n_demos and n_tests,
      sweeping ``cmax_fraction`` over a grid (default 0.0..1.0 step 0.1).  The
      operating fraction f* maximizes overall accuracy, breaking ties toward
      the smaller fraction.

  type2 / K-sweep : at f*, sweep K = n_demos over a range (default 0..10) using
      the 10 successor prompt variations, ``samples_per_var`` randomized tests
      each, with a no-steer control per row.  Per-K metrics pool all variations.

Outputs (under ``--out_root`` = evals/regen by default):
  successor_cmax_sweep/<model>/fraction_<f>.json     per-fraction calibration run
  successor_cmax_sweep/<model>/calibration_<model>.json   curve + chosen f*
  successor_k_sweep/<model>/k<K>_var<V>.json         per (K, variation) run
  successor_k_sweep/<model>/summary_<model>.json     per-K pooled metrics

Example:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m icl.experiments.tasks.run_successor \
      --model qwen3-32b --out_root evals/regen
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

from .amendment_successor import run_amendment_successor
from .task_utils import load_artifacts
from .gated_arithmetic import DEFAULT_JUDGE_BASE_URL, DEFAULT_JUDGE_MODEL
from .successor_prompts import SUCCESSOR_PROMPT_VARIATIONS

DEFAULT_FRACTIONS = [round(0.1 * i, 2) for i in range(0, 11)]  # 0.0 .. 1.0


# ── pooling helpers ────────────────────────────────────────────────────────
def _get_bucket(payload: dict[str, Any], *path: str) -> dict[str, Any] | None:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


def _pool(payloads: list[dict[str, Any]], *path: str) -> dict[str, Any]:
    """Pool a (possibly nested) bucket across many run payloads."""
    n = 0
    n_correct = 0
    for p in payloads:
        bucket = _get_bucket(p, *path)
        if bucket is None:
            continue
        n += int(bucket.get("n", 0) or 0)
        n_correct += int(bucket.get("n_correct", 0) or 0)
    return {"n": n, "n_correct": n_correct, "accuracy": (n_correct / n) if n else None}


def _acc(bucket: dict[str, Any] | None) -> float | None:
    return None if bucket is None else bucket.get("accuracy")


# ── per-run caching ────────────────────────────────────────────────────────
def _load_cached(
    path: Path,
    *,
    model: str,
    n_demos: int,
    n_tests: int,
    cmax_fraction: float | None,
    prompt_variation: int | None,
    randomize_emotion: bool | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("experiment") != "amendment_successor":
        return None
    if payload.get("model") != model:
        return None
    if payload.get("n_demos") != n_demos or payload.get("n_tests") != n_tests:
        return None
    if prompt_variation is not None and payload.get("prompt_variation") != prompt_variation:
        return None
    if randomize_emotion is not None and bool(payload.get("randomize_emotion", False)) != randomize_emotion:
        return None
    if cmax_fraction is not None and abs(float(payload.get("cmax_fraction", -1)) - cmax_fraction) > 1e-6:
        return None
    if not payload.get("rows"):
        return None
    return payload


def _run_one(
    *,
    model,
    tok,
    library,
    cmax,
    model_name: str,
    emotion: str,
    randomize_emotion: bool,
    emotion_pool: list[str] | None,
    cmax_fraction: float,
    n_demos: int,
    n_tests: int,
    target_prob: float,
    distractor_prob: float,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    judge_base_url: str,
    judge_model: str,
    judge_timeout: float,
    with_control: bool,
    system_prompt_template: str | None,
    user_prompt_template: str | None,
    prompt_variation: int,
    out_path: Path,
    resume: bool,
    demo_briefs: list[str] | None = None,
) -> dict[str, Any]:
    if resume:
        cached = _load_cached(
            out_path,
            model=model_name,
            n_demos=n_demos,
            n_tests=n_tests,
            cmax_fraction=cmax_fraction,
            prompt_variation=prompt_variation,
            randomize_emotion=randomize_emotion,
        )
        if cached is not None:
            print(f"[successor_sweep] reuse {out_path.name}", flush=True)
            return cached
    return run_amendment_successor(
        model,
        tok,
        library,
        cmax,
        model_name=model_name,
        emotion=emotion,
        randomize_emotion=randomize_emotion,
        emotion_pool=emotion_pool,
        cmax_fraction=cmax_fraction,
        n_demos=n_demos,
        n_tests=n_tests,
        n_rollouts=1,
        emotion_prob=target_prob,
        distractor_prob=distractor_prob,
        balanced_tests=True,
        seed=seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        judge_timeout=judge_timeout,
        with_control=with_control,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        prompt_variation=prompt_variation,
        demo_briefs=demo_briefs,
        out_path=out_path,
        verbose=False,
    )


# ── phase 1: calibration (type1) ───────────────────────────────────────────
def run_calibration(
    *,
    model,
    tok,
    library,
    cmax,
    model_name: str,
    fractions: list[float],
    n_demos: int,
    n_tests: int,
    variations: list[dict[str, str]],
    common: dict[str, Any],
    out_dir: Path,
    seed_base: int,
    resume: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "by_variation"
    raw_dir.mkdir(parents=True, exist_ok=True)
    curve: list[dict[str, Any]] = []
    for frac in fractions:
        t0 = time.time()
        # 10 prompt variations x n_tests samples each, pooled to ~10*n_tests per fraction.
        var_payloads: list[dict[str, Any]] = []
        for vi, var in enumerate(variations):
            p = _run_one(
                model=model,
                tok=tok,
                library=library,
                cmax=cmax,
                model_name=model_name,
                cmax_fraction=frac,
                n_demos=n_demos,
                n_tests=n_tests,
                seed=seed_base + int(round(frac * 100)) * 100 + vi,
                system_prompt_template=var["system_prompt"],
                user_prompt_template=var["user_prompt"],
                prompt_variation=vi,
                out_path=raw_dir / f"fraction_{frac:.2f}_var{vi}.json",
                resume=resume,
                **common,
            )
            var_payloads.append(p)
        overall = _pool(var_payloads, "overall")
        target = _pool(var_payloads, "by_injection", "target")
        same = _pool(var_payloads, "by_condition", "same")
        control = _pool(var_payloads, "control_overall")
        # Pooled fraction file with ALL rows across the 10 variations, so the
        # type1 figure (plot_strength_sweep.successor_type1) computes its 95%
        # CI over the full ~300 per-sample points. (Per-variation raw files live
        # in the by_variation/ subdir, which the figure's non-recursive glob skips.)
        pooled_rows = [r for p in var_payloads for r in (p.get("rows") or [])]
        pooled = {
            "experiment": "amendment_successor",
            "model": model_name,
            "cmax_fraction": frac,
            "n_demos": n_demos,
            "n_tests": sum(int(p.get("n_tests") or 0) for p in var_payloads),
            "n_variations": len(variations),
            "samples_per_variation": n_tests,
            "overall": overall,
            "by_injection_target": target,
            "rows": pooled_rows,
            "timestamp": datetime.now().isoformat(),
        }
        (out_dir / f"fraction_{frac:.2f}.json").write_text(json.dumps(pooled, indent=2))
        curve.append(
            {
                "cmax_fraction": frac,
                "overall_accuracy": overall.get("accuracy"),
                "overall_n": overall.get("n"),
                "target_accuracy": target.get("accuracy"),
                "same_accuracy": same.get("accuracy"),
                "control_accuracy": control.get("accuracy"),
            }
        )
        print(
            f"[calib] {model_name} f={frac:.2f} overall={overall.get('accuracy')} "
            f"(n={overall.get('n')}) target={target.get('accuracy')} "
            f"wall={time.time() - t0:.1f}s",
            flush=True,
        )

    accs = [(c["cmax_fraction"], c["overall_accuracy"]) for c in curve if c["overall_accuracy"] is not None]
    if not accs:
        raise ValueError("successor calibration produced no accuracy measurements")
    max_acc = max(a for _, a in accs)
    best_frac = max(accs, key=lambda t: (t[1], -t[0]))[0]

    calib = {
        "experiment": "successor_calibration",
        "model": model_name,
        "n_demos": n_demos,
        "n_variations": len(variations),
        "samples_per_variation": n_tests,
        "n_tests": n_tests * len(variations),
        "fractions": fractions,
        "max_accuracy": max_acc,
        "selection_metric": "accuracy",
        "best_cmax_fraction": best_frac,
        "curve": curve,
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / f"calibration_{model_name}.json").write_text(json.dumps(calib, indent=2))
    print(
        f"[calib] {model_name} CHOSEN f*={best_frac:.2f} "
        f"(max_acc={max_acc:.3f})",
        flush=True,
    )
    return calib


# ── phase 2: K-sweep (type2) ────────────────────────────────────────────────
def run_k_sweep(
    *,
    model,
    tok,
    library,
    cmax,
    model_name: str,
    best_frac: float,
    k_values: list[int],
    variations: list[dict[str, str]],
    samples_per_var: int,
    common: dict[str, Any],
    out_dir: Path,
    seed_base: int,
    resume: bool,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_k: list[dict[str, Any]] = []
    for k in k_values:
        t0 = time.time()
        var_payloads: list[dict[str, Any]] = []
        var_overall: list[dict[str, Any]] = []
        for vi, var in enumerate(variations):
            out_path = out_dir / f"k{k}_var{vi}.json"
            payload = _run_one(
                model=model,
                tok=tok,
                library=library,
                cmax=cmax,
                model_name=model_name,
                cmax_fraction=best_frac,
                n_demos=k,
                n_tests=samples_per_var,
                seed=seed_base + 1000 * k + vi,
                system_prompt_template=var["system_prompt"],
                user_prompt_template=var["user_prompt"],
                prompt_variation=vi,
                out_path=out_path,
                resume=resume,
                **common,
            )
            var_payloads.append(payload)
            ob = _get_bucket(payload, "overall") or {}
            cb = _get_bucket(payload, "control_overall") or {}
            var_overall.append(
                {
                    "variation": vi,
                    "overall_accuracy": ob.get("accuracy"),
                    "control_accuracy": cb.get("accuracy"),
                }
            )

        overall = _pool(var_payloads, "overall")
        target = _pool(var_payloads, "by_injection", "target")
        distractor = _pool(var_payloads, "by_injection", "distractor")
        none = _pool(var_payloads, "by_injection", "none")
        same = _pool(var_payloads, "by_condition", "same")
        nxt = _pool(var_payloads, "by_condition", "next")
        control = _pool(var_payloads, "control_overall")
        control_target = _pool(var_payloads, "control_by_injection", "target")
        improvement = (
            overall["accuracy"] - control["accuracy"]
            if overall["accuracy"] is not None and control["accuracy"] is not None
            else None
        )
        target_improvement = (
            target["accuracy"] - control_target["accuracy"]
            if target["accuracy"] is not None and control_target["accuracy"] is not None
            else None
        )
        rec = {
            "k": k,
            "n_variations": len(variations),
            "samples_per_var": samples_per_var,
            "n_rows": overall["n"],
            "overall_accuracy": overall["accuracy"],
            "target_next_accuracy": target["accuracy"],
            "same_accuracy": same["accuracy"],
            "next_accuracy": nxt["accuracy"],
            "distractor_accuracy": distractor["accuracy"],
            "none_accuracy": none["accuracy"],
            "control_accuracy": control["accuracy"],
            "control_target_accuracy": control_target["accuracy"],
            "improvement": improvement,
            "target_improvement": target_improvement,
            "per_variation": var_overall,
            "wall_seconds": time.time() - t0,
        }
        by_k.append(rec)
        print(
            f"[ksweep] {model_name} K={k} overall={overall['accuracy']} "
            f"target/next={target['accuracy']} control={control['accuracy']} "
            f"impr={improvement} wall={time.time() - t0:.1f}s",
            flush=True,
        )
    return by_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3-32b")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--emotion", default="anger", help="fixed gating emotion (ignored if --randomize_emotion)")
    parser.add_argument(
        "--randomize_emotion",
        action="store_true",
        help="draw the gating emotion per conversation from --emotion_pool; distractors are the other pool emotions",
    )
    parser.add_argument(
        "--emotion_pool",
        default=None,
        help="comma list of emotion concepts to draw from; default = config.EMOTION_CONCEPTS",
    )
    parser.add_argument("--out_root", default="evals/regen")
    parser.add_argument(
        "--run_name",
        default=None,
        help="output dir prefix; default 'successor' (fixed) or 'successor_emotions' (randomized) "
        "-> <run_name>_cmax_sweep/ and <run_name>_k_sweep/",
    )
    # calibration
    parser.add_argument("--fractions", default=None, help="comma list; default 0.0..1.0 step 0.1")
    parser.add_argument("--calib_demos", type=int, default=8)
    parser.add_argument("--calib_tests", type=int, default=30)
    parser.add_argument("--skip_calibration", action="store_true")
    parser.add_argument("--fraction", type=float, default=None, help="fixed f* (skips calibration)")
    # K-sweep
    parser.add_argument("--k_values", default="0-10")
    parser.add_argument("--n_variations", type=int, default=10)
    parser.add_argument("--samples_per_var", type=int, default=30)
    parser.add_argument("--skip_ksweep", action="store_true")
    # shared
    parser.add_argument("--target_prob", type=float, default=0.5)
    parser.add_argument("--distractor_prob", type=float, default=0.25)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge_base_url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge_timeout", type=float, default=30.0)
    parser.add_argument("--no_control", action="store_true",
                        help="skip the per-row no-steer control generation (halves cost; "
                        "type1 strength-sweep figure does not use control)")
    parser.add_argument("--no_self_demos", action="store_true",
                        help="use canned AMENDMENT_BRIEFS for demo labels instead of the "
                        "model's own cached summaries (default: use self-generated demos)")
    parser.add_argument("--self_briefs_path", default=None,
                        help="path to amendment_self_briefs.json (default: "
                        "icl/artifacts/<model>/amendment_self_briefs.json)")
    parser.add_argument("--no_resume", action="store_true")
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

    torch.set_grad_enabled(False)

    fractions = (
        [round(float(x), 4) for x in args.fractions.split(",") if x.strip()]
        if args.fractions
        else DEFAULT_FRACTIONS
    )
    k_values: list[int] = []
    for part in args.k_values.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            k_values.extend(range(int(lo), int(hi) + 1))
        elif part:
            k_values.append(int(part))
    variations = SUCCESSOR_PROMPT_VARIATIONS[: args.n_variations]
    resume = not args.no_resume

    from icl.experiments import config as C

    emotion_pool = (
        [e.strip() for e in args.emotion_pool.split(",") if e.strip()]
        if args.emotion_pool
        else (list(C.EMOTION_CONCEPTS) if args.randomize_emotion else None)
    )
    run_name = args.run_name or ("successor_emotions" if args.randomize_emotion else "successor")

    out_root = Path(args.out_root)
    calib_dir = out_root / f"{run_name}_cmax_sweep" / args.model
    ksweep_dir = out_root / f"{run_name}_k_sweep" / args.model

    # No-reference design: use the model's own cached amendment summaries as demo
    # answer labels so the in-context demos are in-distribution (default ON).
    demo_briefs = None
    if not args.no_self_demos:
        sb_path = (Path(args.self_briefs_path) if args.self_briefs_path
                   else C.ARTIFACTS_ROOT / args.model / "amendment_self_briefs.json")
        if not sb_path.exists():
            raise FileNotFoundError(
                f"self-demo briefs not found: {sb_path}. Run python -m icl.experiments.tasks.generate_briefs --model "
                f"{args.model}, or pass --no_self_demos to use canned briefs."
            )
        demo_briefs = json.loads(sb_path.read_text())["briefs"]
        print(f"[successor_sweep] using {len(demo_briefs)} model-generated demo briefs "
              f"from {sb_path}", flush=True)

    common = {
        "emotion": args.emotion,
        "randomize_emotion": args.randomize_emotion,
        "emotion_pool": emotion_pool,
        "target_prob": args.target_prob,
        "distractor_prob": args.distractor_prob,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "judge_base_url": args.judge_base_url,
        "judge_model": args.judge_model,
        "judge_timeout": args.judge_timeout,
        "with_control": not args.no_control,
        "demo_briefs": demo_briefs,
    }

    emo_desc = f"random{emotion_pool}" if args.randomize_emotion else args.emotion
    print(
        f"[successor_sweep] model={args.model} run_name={run_name} emotion={emo_desc} "
        f"fractions={fractions} K={k_values} n_var={len(variations)} "
        f"samples/var={args.samples_per_var} calib(demos={args.calib_demos},tests={args.calib_tests})",
        flush=True,
    )
    t_load = time.time()
    model, tok = get_model_and_tokenizer(args.model)
    library, cmax = load_artifacts(args.model)
    print(f"[successor_sweep] model loaded in {time.time() - t_load:.1f}s", flush=True)

    t_all = time.time()
    calib: dict[str, Any] | None = None
    if args.fraction is not None or args.skip_calibration:
        best_frac = args.fraction if args.fraction is not None else 1.0
        print(f"[successor_sweep] skipping calibration, f*={best_frac}", flush=True)
    else:
        calib = run_calibration(
            model=model,
            tok=tok,
            library=library,
            cmax=cmax,
            model_name=args.model,
            fractions=fractions,
            n_demos=args.calib_demos,
            n_tests=args.calib_tests,
            variations=variations,
            common=common,
            out_dir=calib_dir,
            seed_base=args.seed + 7,
            resume=resume,
        )
        best_frac = calib["best_cmax_fraction"]

    by_k: list[dict[str, Any]] = []
    if not args.skip_ksweep:
        by_k = run_k_sweep(
            model=model,
            tok=tok,
            library=library,
            cmax=cmax,
            model_name=args.model,
            best_frac=best_frac,
            k_values=k_values,
            variations=variations,
            samples_per_var=args.samples_per_var,
            common=common,
            out_dir=ksweep_dir,
            seed_base=args.seed + 13,
            resume=resume,
        )

    summary = {
        "experiment": "successor_sweep",
        "model": args.model,
        "run_name": run_name,
        "emotion": None if args.randomize_emotion else args.emotion,
        "randomize_emotion": args.randomize_emotion,
        "emotion_pool": emotion_pool if args.randomize_emotion else None,
        "best_cmax_fraction": best_frac,
        "calibration": calib,
        "k_values": k_values,
        "n_variations": len(variations),
        "samples_per_var": args.samples_per_var,
        "target_prob": args.target_prob,
        "distractor_prob": args.distractor_prob,
        "generation": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature},
        "judge": {"base_url": args.judge_base_url, "model": args.judge_model},
        "by_k": by_k,
        "timestamp": datetime.now().isoformat(),
        "wall_seconds": time.time() - t_all,
    }
    if args.skip_ksweep:
        # Calibration-only run: do NOT touch the existing type2 ksweep summary.
        print(
            f"[successor_sweep] skip_ksweep set -> preserving existing "
            f"{ksweep_dir / f'summary_{args.model}.json'} (calibration written to {calib_dir})",
            flush=True,
        )
    else:
        ksweep_dir.mkdir(parents=True, exist_ok=True)
        summary_path = ksweep_dir / f"summary_{args.model}.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[successor_sweep] wrote {summary_path} (total wall={time.time() - t_all:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
