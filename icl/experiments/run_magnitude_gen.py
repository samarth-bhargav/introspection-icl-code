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
import os
import sys
from pathlib import Path


def run_magnitude_gen(model, tok, library, cmax, *, model_name, m_star,
                      n_examples=20, n_samples=30, seed=42,
                      sweep=(0.0, 5.0, 0.1), concepts=None, prompt_variations=10):
    """Use the same generator, prompts, and output path as the magnitude CLI."""
    from icl.experiments import magnitude as M
    from icl.experiments import config as C
    M.C.ARTIFACTS_ROOT, M.C.EVALS_ROOT = C.ARTIFACTS_ROOT, C.EVALS_ROOT
    out_root = C.EVALS_ROOT / "constitution_source_magnitude"
    M.run_magnitude_generalization(model, tok, library, cmax, model_name=model_name,
        out_root=out_root, m_star=m_star, n_examples=n_examples, n_samples=n_samples,
        prompt_variations=prompt_variations, seed=seed, sweep=sweep,
        concepts=concepts or C.CONCEPTS, force=True)
    return M._maggen_path(out_root, model_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--m_star", type=float, required=True)
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
    run_magnitude_gen(model, tok, library, cmax, model_name=args.model,
                      m_star=args.m_star, n_examples=args.n_examples,
                      n_samples=args.n_samples, seed=args.seed, prompt_variations=args.prompt_variations)


if __name__ == "__main__":
    main()
