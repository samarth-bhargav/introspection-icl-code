"""Layer-introspection generalisation to held-out test layers.

Demos inject at the 3 anchor layers (early/middle/late) with per-concept
c_max-scaling at f*; the test query is injected at every decoder layer in turn.
Mirrors icl.plotting.plot_layer_generalization on the regenerated mean-diff lib.

Usage:
    python -m icl.experiments.run_layer_gen --model gemma-31b --gpu 0 --cmax_fraction 1.25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from random import Random


def _run_layer_gen_prompt(model, tok, library, cmax, *, model_name, cmax_fraction,
                  n_examples=20, n_samples=30, seed=42, concepts=None,
                  system_prompt=None, trigger=None, prompt_variation=0):
    from icl import ICLQuery, run_queries
    from icl.experiments import config as C

    concepts = concepts or C.CONCEPTS
    anchors = C.LAYER_ANCHORS[model_name]
    layer_groups = list(zip(C.LAYER_LABELS, anchors))
    n_layers = C.N_LAYERS[model_name]
    test_layers = list(range(n_layers))
    out_tok = C.LAYER_OUTPUT_TOKENS

    def cmax_for(concept, layer):
        # The paper uses the nearest of the three anchors even when another
        # task has independently calibrated this layer (e.g. the magnitude layer).
        return C.layer_generalization_cmax(cmax, concept, layer, anchors)

    t0 = time.time()
    print(f"[layer_gen] {model_name} anchors={anchors} f={cmax_fraction} "
          f"layers={n_layers} n_samples={n_samples}", flush=True)
    records = []
    for tl in test_layers:
        rng = Random(seed + 31 * tl)
        per_class = {lab: 0 for lab in C.LAYER_LABELS}
        p_sums = {lab: 0.0 for lab in C.LAYER_LABELS}
        samples, queries = [], []
        for _ in range(n_samples):
            test_concept = rng.choice(concepts)
            pairs = [(c, g) for c in concepts if c != test_concept for g in layer_groups]
            chosen = rng.sample(pairs, min(n_examples, len(pairs)))
            ex_c = [c for c, _ in chosen]
            ex_g = [g for _, g in chosen]
            all_c = ex_c + [test_concept]
            all_l = [g[1] for g in ex_g] + [tl]
            scales = [cmax_fraction * cmax_for(c, l) for c, l in zip(all_c, all_l)]
            queries.append(ICLQuery(
                concepts=all_c, words=[g[0] for g in ex_g], output_tokens=out_tok,
                system_prompt=system_prompt or C.LAYER_SYSTEM, injection_layer=all_l,
                injection_scale=scales, prompts=[trigger or C.LAYER_TRIGGER] * len(all_c)))
        for r in run_queries(queries, model=model, tokenizer=tok, concept_library=library):
            pred = max(r.probabilities, key=r.probabilities.get)
            per_class[pred] += 1
            for lab in C.LAYER_LABELS:
                p_sums[lab] += r.probabilities.get(lab, 0.0)
            samples.append({"predicted": pred, "prompt_variation": prompt_variation,
                            "sample_id": len(samples), "query": r.query.__dict__,
                            "probabilities": {k: float(r.probabilities.get(k, 0.0))
                                              for k in C.LAYER_LABELS}})
        mean_p = {k: p_sums[k] / n_samples for k in C.LAYER_LABELS}
        records.append({"test_layer": tl, "counts": per_class, "mean_p": mean_p,
                        "n_total": n_samples, "samples": samples})
        print(f"  L{tl:>2} p(e)={mean_p['early']:.2f} p(m)={mean_p['middle']:.2f} "
              f"p(l)={mean_p['late']:.2f}", flush=True)

    payload = {"experiment": "layer_generalization", "model": model_name,
               "anchors": list(anchors), "labels": C.LAYER_LABELS,
               "cmax_fraction": cmax_fraction, "n_examples": n_examples,
               "n_samples": n_samples, "seed": seed,
               "timestamp": datetime.now().isoformat(), "records": records}
    out_dir = C.EVALS_ROOT / "layer_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer_generalization_{model_name}_var{prompt_variation}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[layer_gen] wrote {out_path} (wall={time.time()-t0:.1f}s)", flush=True)
    return out_path


def run_layer_gen(model, tok, library, cmax, *, model_name, cmax_fraction,
                  n_examples=20, n_samples=30, seed=42, concepts=None, prompt_variations=10):
    """Pool all prompt variations, retaining each sample and prompt identity."""
    from icl.common.prompt_variations import VARIATIONS
    from icl.experiments import config as C
    variants = VARIATIONS["layer_introspection"][:prompt_variations]
    if not variants or len(variants) != prompt_variations:
        raise ValueError("prompt_variations must be between 1 and 10")
    payloads = []
    for vi, variant in enumerate(variants):
        path = _run_layer_gen_prompt(model, tok, library, cmax, model_name=model_name,
            cmax_fraction=cmax_fraction, n_examples=n_examples, n_samples=n_samples,
            seed=seed, concepts=concepts, prompt_variation=vi,
            system_prompt=variant["system_prompt"], trigger=variant["prompt_text"])
        payloads.append(json.loads(path.read_text()))
    pooled = dict(payloads[0])
    pooled.update(n_samples=n_samples * prompt_variations, n_samples_per_prompt=n_samples,
                  n_prompt_variations=prompt_variations, prompt_variations=variants)
    records = []
    for i, record in enumerate(payloads[0]["records"]):
        samples = [s for p in payloads for s in p["records"][i]["samples"]]
        n = len(samples)
        records.append({"test_layer": record["test_layer"], "samples": samples,
                        "n_total": n,
                        "counts": {lab: sum(s["predicted"] == lab for s in samples) for lab in C.LAYER_LABELS},
                        "mean_p": {lab: sum(s["probabilities"][lab] for s in samples)/n for lab in C.LAYER_LABELS}})
    pooled["records"] = records
    path = C.EVALS_ROOT / "layer_generalization" / f"layer_generalization_{model_name}.json"
    path.write_text(json.dumps(pooled, indent=2))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gemma-31b")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--cmax_fraction", type=float, required=True)
    ap.add_argument("--n_examples", type=int, default=20)
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--prompt_variations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
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
    model, tok = get_model_and_tokenizer(args.model)
    library = ConceptLibrary.load(C.library_path(args.model))
    cmax = C.load_cmax(args.model)
    run_layer_gen(model, tok, library, cmax, model_name=args.model,
                  cmax_fraction=args.cmax_fraction, n_examples=args.n_examples,
                  n_samples=args.n_samples, seed=args.seed, prompt_variations=args.prompt_variations)


if __name__ == "__main__":
    main()
