"""Single-pass layer/magnitude introspection sweeps (type1, type2, prompt-var).

type1  : fixed K=30, sweep strength over the 21-point grid -> used to pick the
         per-model star (m* for magnitude, f* for layer) by argmax accuracy@K=30.
type2  : full K=0..61 readout at one chosen strength (the star).
         With --prompt_variation N (0..9) the trigger/system come from
         icl.common.prompt_variations.VARIATIONS (prompt-variation panels);
         otherwise the canonical main-text prompt is used.

Usage:
    python -m icl.experiments.introspection_sweep --model qwen3-8b --gpu 0 --task magnitude --mode type1
    python -m icl.experiments.introspection_sweep --model qwen3-8b --gpu 0 --task magnitude --mode type2 --strength 1.75
    python -m icl.experiments.introspection_sweep --pick_star evals/regen/magnitude/type1_qwen3-8b.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def pick_star(type1_json, k: int) -> dict:
    """Pick the operating strength α* from a type1 sweep at K=k.

    Parsimony "knee" rule: the SMALLEST strength (>0) whose accuracy is within
    one binomial standard error of the maximum accuracy on the grid. The paper
    selects the argmax ("maximizes mean p(correct)"), but its contrastive-vector
    curves PEAK; our one-vs-rest layer curves are strength-robust PLATEAUS, so a
    raw argmax lands on a noisy high point. The knee gives the minimal injection
    achieving statistically-best performance — matching the paper's operating
    points (α* ~1-1.75) and yielding milder, cleaner generalisation. For peaked
    (magnitude) curves the knee coincides with the argmax peak.
    """
    import math
    data = json.loads(Path(type1_json).read_text())
    n = data.get("n_samples", 100)
    curve = []
    for entry in data["per_strength"]:
        rec = next((r for r in entry["by_k"] if r["k"] == k), None)
        if rec is None:
            continue
        curve.append((entry["strength"], rec["accuracy"], rec["mean_p"]))
    max_acc = max(c[1] for c in curve)
    se = math.sqrt(max_acc * (1 - max_acc) / n) if 0 < max_acc < 1 else 0.0
    thresh = max_acc - se
    knee = next((c for c in curve if c[0] > 0 and c[1] >= thresh),
                max(curve, key=lambda t: t[1]))
    return {"star": knee[0], "accuracy": knee[1], "mean_p": knee[2],
            "max_acc": max_acc, "curve": curve}


def sweep_and_save(model, tok, library, cmax, *, model_name, task, mode,
                   strength=None, prompt_variation=-1, n_samples, seed=13,
                   kmax=None, concepts_pool=None):
    """Run one sweep (model already loaded) and write JSON. Returns (path, payload)."""
    from icl.common.prompt_variations import VARIATIONS
    from icl.experiments import config as C
    from icl.experiments import singlepass as SP

    concepts_pool = concepts_pool or C.CONCEPTS
    if prompt_variation >= 0:
        var = VARIATIONS[f"{task}_introspection"][prompt_variation]
        trigger, system = var["prompt_text"], var["system_prompt"]
    else:
        trigger = C.MAGNITUDE_TRIGGER if task == "magnitude" else C.LAYER_TRIGGER
        system = C.MAGNITUDE_SYSTEM if task == "magnitude" else C.LAYER_SYSTEM

    if mode == "type1":
        strengths = C.STRENGTH_GRID
        T = (kmax if kmax is not None else C.TYPE1_K) + 1
    else:
        if strength is None:
            raise ValueError("strength required for type2")
        strengths = [strength]
        T = (kmax if kmax is not None else C.TYPE2_KMAX) + 1
    T = min(T, len(concepts_pool))

    mag_layer = C.MAGNITUDE_LAYER[model_name]
    layer_anchors = dict(zip(C.LAYER_LABELS, C.LAYER_ANCHORS[model_name]))

    t0 = time.time()
    print(f"[sweep] {model_name} task={task} mode={mode} var={prompt_variation} "
          f"strengths={len(strengths)} T={T} n_samples={n_samples}", flush=True)
    per_strength = SP.run_sweep(
        model, tok, library, task=task, strengths=strengths, T=T,
        n_samples=n_samples, seed=seed, system_prompt=system, trigger=trigger,
        cmax=cmax, magnitude_layer=mag_layer, layer_anchors=layer_anchors,
        concepts_pool=concepts_pool, cmax_floor=C.CMAX_FLOOR)

    payload = {
        "task": task, "mode": mode, "model": model_name,
        "prompt_variation": prompt_variation, "trigger": trigger,
        "system_prompt": system, "n_samples": n_samples, "seed": seed, "T": T,
        "magnitude_layer": mag_layer if task == "magnitude" else None,
        "layer_anchors": layer_anchors if task == "layer" else None,
        "strengths": strengths, "timestamp": datetime.now().isoformat(),
        "per_strength": per_strength,
    }
    out_dir = C.EVALS_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_var{prompt_variation}" if prompt_variation >= 0 else ""
    out_path = out_dir / f"{mode}_{model_name}{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[sweep] wrote {out_path} (wall={time.time()-t0:.1f}s)", flush=True)
    if mode == "type1":
        sk = min(C.TYPE1_K, T - 1)
        star = pick_star(out_path, sk)
        ref = C.PAPER_MSTAR[model_name] if task == "magnitude" else C.PAPER_FSTAR[model_name]
        print(f"[sweep] STAR ({task}) = {star['star']} acc@K{sk}={star['accuracy']:.3f} "
              f"(paper ref {ref})", flush=True)
    return out_path, payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pick_star", default=None)
    ap.add_argument("--model")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--task", choices=["magnitude", "layer"])
    ap.add_argument("--mode", choices=["type1", "type2"])
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--prompt_variation", type=int, default=-1)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--kmax", type=int, default=None)
    ap.add_argument("--smoke_concepts", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if args.pick_star:
        from icl.experiments import config as C
        print(json.dumps({k: v for k, v in pick_star(args.pick_star, C.TYPE1_K).items()
                          if k != "curve"}))
        return

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    try:
        from dotenv import load_dotenv
        load_dotenv(repo_root / "notebooks" / ".env")
    except ImportError:
        pass

    import torch
    from icl import get_model_and_tokenizer
    from icl.steering.concepts import ConceptLibrary
    from icl.experiments import config as C
    torch.set_grad_enabled(False)

    n_samples = args.n_samples or C.DEFAULT_N_SAMPLES
    pool = C.CONCEPTS if args.smoke_concepts <= 0 else C.CONCEPTS[:args.smoke_concepts]
    model, tok = get_model_and_tokenizer(args.model)
    library = ConceptLibrary.load(C.library_path(args.model))
    cmax = C.load_cmax(args.model)
    sweep_and_save(model, tok, library, cmax, model_name=args.model, task=args.task,
                   mode=args.mode, strength=args.strength,
                   prompt_variation=args.prompt_variation, n_samples=n_samples,
                   seed=args.seed, kmax=args.kmax, concepts_pool=pool)


if __name__ == "__main__":
    main()
