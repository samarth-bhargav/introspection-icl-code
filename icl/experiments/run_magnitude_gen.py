"""Magnitude generalisation to unseen test magnitudes (PDF Fig 3 / App H).

Demos: k in-context examples at anchor magnitudes {αL,αM,αH} = m*·{0.25,1,2.5}
(c_max-scaled at the magnitude layer). Test: inject at the magnitude layer with
a swept absolute multiplier α (also c_max-scaled), and record mean p(low/medium/
high). Shows whether the model maps unseen α onto the right label.

Usage:
    python -m icl.experiments.run_magnitude_gen --model gemma-31b --gpu 0 --m_star 1.25
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


def run_magnitude_gen(model, tok, library, cmax, *, model_name, m_star,
                      n_examples=20, n_samples=30, seed=42,
                      sweep=(0.0, 5.0, 0.1), concepts=None):
    from icl import ICLQuery, run_queries
    from icl.experiments import config as C

    concepts = concepts or C.CONCEPTS
    layer = C.MAGNITUDE_LAYER[model_name]
    out_tok = C.MAGNITUDE_OUTPUT_TOKENS
    base = C.MAGNITUDE_BASE
    anchor_levels = [(lab, m_star * base[lab]) for lab in C.MAGNITUDE_LABELS]
    lo, hi, step = sweep
    test_alphas = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]

    def cm(concept):
        return C.cmax_or_floor(cmax, concept, layer)

    t0 = time.time()
    print(f"[mag_gen] {model_name} layer={layer} m*={m_star} anchors={anchor_levels} "
          f"alphas={len(test_alphas)} n_samples={n_samples}", flush=True)
    records = []
    for ta in test_alphas:
        rng = Random(seed + int(round(ta * 1000)))
        per_class = {lab: 0 for lab in C.MAGNITUDE_LABELS}
        p_sums = {lab: 0.0 for lab in C.MAGNITUDE_LABELS}
        samples, queries = [], []
        for _ in range(n_samples):
            chosen = rng.sample(concepts, min(n_examples + 1, len(concepts)))
            demo_concepts = chosen[:-1]
            test_concept = chosen[-1]
            demo_levels = [rng.choice(anchor_levels) for _ in demo_concepts]
            words = [lab for lab, _ in demo_levels]
            all_c = demo_concepts + [test_concept]
            scales = [frac * cm(c) for c, (lab, frac) in zip(demo_concepts, demo_levels)] \
                + [ta * cm(test_concept)]
            queries.append(ICLQuery(
                concepts=all_c, words=words, output_tokens=out_tok,
                system_prompt=C.MAGNITUDE_SYSTEM, injection_layer=[layer] * len(all_c),
                injection_scale=scales, prompts=[C.MAGNITUDE_TRIGGER] * len(all_c)))
        for r in run_queries(queries, model=model, tokenizer=tok, concept_library=library):
            pred = max(r.probabilities, key=r.probabilities.get)
            per_class[pred] += 1
            for lab in C.MAGNITUDE_LABELS:
                p_sums[lab] += r.probabilities.get(lab, 0.0)
            samples.append({"predicted": pred,
                            "probabilities": {k: float(r.probabilities.get(k, 0.0))
                                              for k in C.MAGNITUDE_LABELS}})
        mean_p = {k: p_sums[k] / n_samples for k in C.MAGNITUDE_LABELS}
        records.append({"test_alpha": ta, "counts": per_class, "mean_p": mean_p,
                        "n_total": n_samples, "samples": samples})
        if abs(ta * 10 - round(ta * 10)) < 1e-9 and int(round(ta * 10)) % 5 == 0:
            print(f"  α={ta:.2f} p(low)={mean_p['low']:.2f} p(med)={mean_p['medium']:.2f} "
                  f"p(high)={mean_p['high']:.2f}", flush=True)

    payload = {"experiment": "magnitude_generalization", "model": model_name,
               "layer": layer, "labels": C.MAGNITUDE_LABELS, "m_star": m_star,
               "anchor_levels": anchor_levels, "n_examples": n_examples,
               "n_samples": n_samples, "seed": seed,
               "timestamp": datetime.now().isoformat(), "records": records}
    out_dir = C.EVALS_ROOT / "magnitude_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"magnitude_generalization_{model_name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[mag_gen] wrote {out_path} (wall={time.time()-t0:.1f}s)", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--m_star", type=float, required=True)
    ap.add_argument("--n_examples", type=int, default=20)
    ap.add_argument("--n_samples", type=int, default=30)
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
    run_magnitude_gen(model, tok, library, cmax, model_name=args.model,
                      m_star=args.m_star, n_examples=args.n_examples,
                      n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
