"""Full per-model pipeline (model loaded ONCE): build -> sweeps -> generalisation.

Stages (select with --stages, default all):
  build      : build_and_calibrate (mean-diff lib + c_max). --rebuild to redo vectors.
  magnitude  : type1 strength sweep -> m*; type2 ICL sweep; 10 prompt-var sweeps.
  layer      : type1 strength sweep -> f*; type2 ICL sweep; 10 prompt-var sweeps.
  maggen     : magnitude generalisation (PDF Fig 3 / App H) at m*.
  layergen   : layer generalisation (PDF Fig 5 / App I) at f*.

m*/f* for maggen/layergen are taken from this session's type1 sweep if run, else
read from the existing type1 JSON on disk.

Usage:
    python -m icl.experiments.run_model --model gemma-31b --gpu 0
    python -m icl.experiments.run_model --model qwen3-8b --gpu 0 --stages build,maggen,layergen --rebuild
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ALL_STAGES = ["build", "magnitude", "layer", "maggen", "layergen"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--prompt_variations", type=int, default=10)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--force_cmax", action="store_true")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    stages = set(s.strip() for s in args.stages.split(",") if s.strip())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
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
    from icl.experiments.build_library import build_and_calibrate
    from icl.experiments.introspection_sweep import sweep_and_save, pick_star
    from icl.experiments.run_magnitude_gen import run_magnitude_gen
    from icl.experiments.run_layer_gen import run_layer_gen
    torch.set_grad_enabled(False)

    m = args.model
    concepts = C.CONCEPTS[:12] if args.smoke else C.CONCEPTS
    n_samples = args.n_samples or (6 if args.smoke else C.DEFAULT_N_SAMPLES)
    n_var = 2 if args.smoke else args.prompt_variations
    type1_kmax = 8 if args.smoke else C.TYPE1_K
    type2_kmax = 10 if args.smoke else C.TYPE2_KMAX
    star_k = min(type1_kmax, len(concepts) - 1)
    gen_ns = 6 if args.smoke else 30

    t0 = time.time()
    print(f"==== run_model {m} gpu={args.gpu} stages={sorted(stages)} "
          f"n_samples={n_samples} ====", flush=True)
    model, tok = get_model_and_tokenizer(m)

    if "build" in stages:
        library = build_and_calibrate(model, tok, m, concepts=concepts,
                                      rebuild=args.rebuild, force_cmax=args.force_cmax)
    else:
        library = ConceptLibrary.load(C.library_path(m))
    cmax = C.load_cmax(m)

    def star_for(task):
        """m*/f*: from this session's type1 if run, else from existing type1 JSON."""
        p = C.EVALS_ROOT / task / f"type1_{m}.json"
        return pick_star(p, star_k)["star"]

    stars = {}
    for task in ["magnitude", "layer"]:
        if task in stages:
            p1, _ = sweep_and_save(model, tok, library, cmax, model_name=m, task=task,
                                   mode="type1", n_samples=n_samples, kmax=type1_kmax,
                                   concepts_pool=concepts)
            star = pick_star(p1, star_k)["star"]
            stars[task] = star
            print(f"==== {m} {task} STAR = {star} (k={star_k}) ====", flush=True)
            sweep_and_save(model, tok, library, cmax, model_name=m, task=task,
                           mode="type2", strength=star, n_samples=n_samples,
                           kmax=type2_kmax, concepts_pool=concepts)
            for v in range(n_var):
                sweep_and_save(model, tok, library, cmax, model_name=m, task=task,
                               mode="type2", strength=star, prompt_variation=v,
                               n_samples=n_samples, kmax=type2_kmax, concepts_pool=concepts)

    if "maggen" in stages:
        ms = stars.get("magnitude") or star_for("magnitude")
        run_magnitude_gen(model, tok, library, cmax, model_name=m, m_star=ms,
                          n_samples=gen_ns, concepts=concepts)
    if "layergen" in stages:
        fs = stars.get("layer") or star_for("layer")
        run_layer_gen(model, tok, library, cmax, model_name=m, cmax_fraction=fs,
                      n_samples=gen_ns, concepts=concepts)

    print(f"==== run_model {m} DONE wall={time.time()-t0:.1f}s stars={stars} ====", flush=True)


if __name__ == "__main__":
    main()
