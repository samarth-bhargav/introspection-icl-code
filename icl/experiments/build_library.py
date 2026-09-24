"""Build a one-vs-rest mean-difference concept library + c_max for one model.

Implements the notebook (concept_name_probe.ipynb) method literally:
  per concept, 20 description prompts; greedy-generate 20 tokens; mean the layer
  residual over generated continuation positions only -> per-prompt vector; concept
  vector = mean(its prompts) - mean(ALL OTHER concepts' prompts), L2-normalised
  per layer. No negative templates.

Vectors are extracted at every layer in config.build_layers(model) (all decoder
layers for Gemma, since layer-generalisation needs them); c_max is computed via
the comprehension binary search at config.cmax_layers(model).

Usage:
    python -m icl.experiments.build_library --model qwen3-8b --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone


def _log(msg: str) -> None:
    print(f"[build_library] {msg}", flush=True)


def build_and_calibrate(model, tok, model_name, *, concepts=None,
                        n_generated_tokens=20, rebuild=False, force_cmax=False):
    """Build the mean-diff library + c_max for `model_name` (model already loaded)."""
    import torch
    from icl.experiments.telemetry import emit
    from steering_vectors import SteeringVector
    from icl import compute_max_strength
    from icl.steering.concepts import ConceptLibrary, format_as_chat
    from icl.steering.injection import _get_target_module
    from icl.steering.ranges import _QA_PAIRS
    from icl.experiments import config as C

    concepts = concepts or C.CONCEPTS
    layers = C.build_layers(model_name)
    cmax_ls = [l for l in C.cmax_layers(model_name) if l in layers]
    lib_path = C.library_path(model_name)
    cmax_p = C.cmax_path(model_name)
    meta_p = C.metadata_path(model_name)
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    t0 = time.time()
    _log(f"build layers ({len(layers)}): "
         f"{layers if len(layers) <= 12 else f'{layers[0]}..{layers[-1]}'}; "
         f"c_max layers {cmax_ls}; concepts {len(concepts)}")

    def generate_continuation(prompt_text: str):
        chat = format_as_chat(tok, prompt_text)
        input_ids = tok.encode(chat, return_tensors="pt",
                               add_special_tokens=False).to(device)
        with torch.no_grad():
            full_ids = model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                      max_new_tokens=n_generated_tokens, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
        return full_ids, input_ids.shape[1]

    def multilayer_means(full_ids, prompt_len: int):
        captured: dict[int, "torch.Tensor"] = {}
        handles = []

        def mk(li):
            def hook(_m, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                gen_hs = hs[0, prompt_len:, :]
                if gen_hs.shape[0] == 0:
                    raise RuntimeError("no generated-token activations captured")
                captured[li] = gen_hs.float().mean(dim=0).cpu()
            return hook

        for li in layers:
            handles.append(_get_target_module(model, li).register_forward_hook(mk(li)))
        try:
            with torch.no_grad():
                model(full_ids)
        finally:
            for h in handles:
                h.remove()
        return captured

    # ── Build (or load) the mean-diff library ──────────────────────
    if lib_path.exists() and meta_p.exists() and not rebuild:
        previous = json.loads(meta_p.read_text())
        if (previous.get("concepts") != concepts or
                previous.get("n_generated_tokens") != n_generated_tokens or
                previous.get("description_prompt_templates") != C.DESCRIPTION_PROMPT_TEMPLATES):
            raise ValueError("Cached vector construction differs from this run; use --rebuild")
        _log(f"loading existing library: {lib_path}")
        library = ConceptLibrary.load(lib_path)
    else:
        _log("building library ...")
        per_concept_sum: dict[str, dict[int, "torch.Tensor"]] = {}
        n_prompts = len(C.DESCRIPTION_PROMPT_TEMPLATES)
        for ci, concept in enumerate(concepts, 1):
            cs = time.time()
            acc: dict[int, "torch.Tensor"] = {}
            for prompt_id, tmpl in enumerate(C.DESCRIPTION_PROMPT_TEMPLATES):
                full, prompt_len = generate_continuation(tmpl.format(concept=concept))
                emit("vector_extraction", model=model_name, concept=concept,
                     prompt_id=prompt_id, prompt=tmpl.format(concept=concept),
                     token_ids=full[0].tolist(), prompt_length=prompt_len,
                     generated_text=tok.decode(full[0, prompt_len:], skip_special_tokens=True))
                for li, v in multilayer_means(full, prompt_len).items():
                    acc[li] = v if li not in acc else acc[li] + v
            per_concept_sum[concept] = acc
            _log(f"  [{ci:>3}/{len(concepts)}] {concept:<14} ({time.time()-cs:.1f}s)")

        total_count = len(concepts) * n_prompts
        total_sum = {li: sum(per_concept_sum[c][li] for c in concepts) for li in layers}
        vectors: dict[str, "SteeringVector"] = {}
        for concept in concepts:
            la = {}
            for li in layers:
                concept_mean = per_concept_sum[concept][li] / n_prompts
                others_mean = (total_sum[li] - per_concept_sum[concept][li]) / (total_count - n_prompts)
                diff = concept_mean - others_mean
                norm = diff.norm()
                if norm > 0:
                    diff = diff / norm
                la[li] = diff
            vectors[concept] = SteeringVector(layer_activations=la, layer_type="decoder_block")
        library = ConceptLibrary(vectors=vectors, concept_names=list(concepts))
        library.save(lib_path)
        meta_p.write_text(json.dumps({
            "method": "one_vs_rest_mean_difference_generation_activations",
            "pooling": "mean_generation_tokens",
            "model": model_name, "layers": layers,
            "n_generated_tokens": n_generated_tokens, "generation_temperature": 0.0,
            "n_prompts_per_concept": n_prompts,
            "description_prompt_templates": C.DESCRIPTION_PROMPT_TEMPLATES,
            "concepts": concepts, "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        _log(f"library saved: {lib_path} ({time.time()-t0:.1f}s elapsed)")

    # ── c_max calibration ──────────────────────────────────────────
    calibration_questions = [list(pair) for pair in _QA_PAIRS]
    if cmax_p.exists() and not force_cmax and not rebuild:
        cache = json.loads(cmax_p.read_text())
        if (cache.get("metadata", {}).get("questions") != calibration_questions or
                cache.get("metadata", {}).get("calibration_version") != 2):
            _log("calibration suite changed; recomputing c_max")
            force_cmax = True
    else:
        force_cmax = True
    if force_cmax:
        cache = {"metadata": {
            "method": "compute_max_strength", "model": model_name,
            "library_path": str(lib_path), "threshold": C.CMAX_THRESHOLD,
            "precision": C.CMAX_PRECISION, "search_lo": C.CMAX_SEARCH_LO,
            "search_hi": C.CMAX_SEARCH_HI,
            "questions": calibration_questions, "injection_span": "user_content_only",
            "calibration_version": 2,
            "scaling": "coefficient_times_live_L2_norm_times_unit_vector",
            "created_at": datetime.now(timezone.utc).isoformat()}, "ranges": {}}

    _log(f"computing c_max for {len(concepts)} concepts x {len(cmax_ls)} layers {cmax_ls}")
    for layer in cmax_ls:
        ls = str(layer)
        cl = time.time()
        n_done = 0
        for concept in concepts:
            if ls in cache["ranges"].get(concept, {}) and not force_cmax:
                continue
            cm = compute_max_strength(
                model, tok, library, concept, layer,
                threshold=C.CMAX_THRESHOLD, precision=C.CMAX_PRECISION,
                search_lo=C.CMAX_SEARCH_LO, search_hi=C.CMAX_SEARCH_HI)
            cache["ranges"].setdefault(concept, {})[ls] = {"c_max": cm}
            emit("calibration_result", model=model_name, concept=concept, layer=layer,
                 raw_cmax=cm, effective_cmax=C.CMAX_FLOOR if cm is None else cm,
                 lower_bound_failed=cm is None)
            n_done += 1
            if n_done % 10 == 0:
                cmax_p.write_text(json.dumps(cache, indent=2))
        cmax_p.write_text(json.dumps(cache, indent=2))
        vals = [cache["ranges"][c][ls]["c_max"] for c in concepts
                if cache["ranges"].get(c, {}).get(ls, {}).get("c_max") is not None]
        broken = sum(1 for c in concepts
                     if cache["ranges"].get(c, {}).get(ls, {}).get("c_max") is None)
        med = sorted(vals)[len(vals) // 2] if vals else float("nan")
        _log(f"  L{layer}: median c_max={med:.3f} n={len(vals)}/{len(concepts)} "
             f"broken={broken} ({time.time()-cl:.1f}s)")
    _log(f"build+calibrate DONE {model_name} wall={time.time()-t0:.1f}s")
    return library


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n_generated_tokens", type=int, default=20)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--force_cmax", action="store_true")
    ap.add_argument("--smoke_concepts", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dotenv import load_dotenv
        load_dotenv(repo_root / ".env")
    except ImportError:
        pass

    import torch
    from icl import get_model_and_tokenizer
    from icl.experiments import config as C
    torch.set_grad_enabled(False)

    concepts = C.CONCEPTS if args.smoke_concepts <= 0 else C.CONCEPTS[:args.smoke_concepts]
    _log(f"model={args.model} gpu={args.gpu}")
    model, tok = get_model_and_tokenizer(args.model)
    build_and_calibrate(model, tok, args.model, concepts=concepts,
                        n_generated_tokens=args.n_generated_tokens,
                        rebuild=args.rebuild, force_cmax=args.force_cmax)


if __name__ == "__main__":
    main()
