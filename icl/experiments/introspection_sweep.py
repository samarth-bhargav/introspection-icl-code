"""Single-pass layer/magnitude introspection sweeps (type1, type2, prompt-var).

type1  : fixed K=30, sweep strength over the 21-point grid -> used to pick the
         per-model star (m* for magnitude, f* for layer) by maximum mean p(correct) at K=30.
type2  : full K=0..61 readout at one chosen strength (the star).
         With --prompt_variation N (0..9) the trigger/system come from
         icl.common.prompt_variations.VARIATIONS (prompt-variation panels);
         otherwise all ten prompt variations are pooled.

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
    """Choose the strength maximizing mean correct-label probability."""
    from icl.experiments.selection import pick_mean_probability
    return pick_mean_probability(type1_json, k)


def sweep_and_save(model, tok, library, cmax, *, model_name, task, mode,
                   strength=None, prompt_variation=-1, n_samples, seed=13,
                   kmax=None, concepts_pool=None, n_prompt_variations=10):
    """Run one sweep (model already loaded) and write JSON. Returns (path, payload)."""
    from icl.common.prompt_variations import VARIATIONS
    from icl.experiments import config as C
    from icl.experiments import singlepass as SP

    if prompt_variation < 0:
        from icl.experiments.magnitude import _pool_prompt_sweeps
        variants = VARIATIONS[f"{task}_introspection"][:n_prompt_variations]
        if not variants or len(variants) != n_prompt_variations:
            raise ValueError("n_prompt_variations must be between 1 and 10")
        runs = []
        for vi, variant in enumerate(variants):
            _, payload = sweep_and_save(model, tok, library, cmax, model_name=model_name,
                task=task, mode=mode, strength=strength, prompt_variation=vi,
                n_samples=n_samples, seed=seed, kmax=kmax, concepts_pool=concepts_pool)
            runs.append((vi, variant, payload["per_strength"]))
        payload.pop("trigger", None)
        payload.pop("system_prompt", None)
        payload.update(prompt_variation=None, n_samples=n_samples * len(variants),
                       n_samples_per_prompt=n_samples, n_prompt_variations=len(variants),
                       prompt_variations=variants, per_strength=_pool_prompt_sweeps(runs))
        path = C.EVALS_ROOT / task / f"{mode}_{model_name}.json"
        path.write_text(json.dumps(payload, indent=2))
        return path, payload

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
              f"(historical ref {ref})", flush=True)
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
    try:
        from dotenv import load_dotenv
        load_dotenv(repo_root / ".env")
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
