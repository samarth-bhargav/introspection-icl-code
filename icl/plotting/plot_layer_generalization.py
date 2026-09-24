"""Test-layer generalisation sweep for 3-class layer introspection.

Layer analog of the magnitude generalization panels (``icl.plotting.plot_magnitude``).  Demos inject at
the per-model anchor layers (early/middle/late, see Appendix
``app:anchor_layers``) with per-concept c_max-scaling
(scale = cmax_fraction × n_{c,l}^M(concept, layer)).  The *test* query
injection layer is then swept over every decoder layer in turn.

Records per test-layer mean p(early), p(middle), p(late) plus per-sample
raw probabilities, writes a JSON log + a PNG + an HTML.

Default usage (Gemma-31B on CUDA device 0)::

    CUDA_VISIBLE_DEVICES=0 python -m icl.plotting.plot_layer_generalization \
        --model gemma-31b --cmax_fraction 1.25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from random import Random


# 62-concept generic library (matches icl.common.layer_introspection).
CONCEPTS = [
    "love", "anger", "fear", "joy", "sadness", "disgust", "science", "politics",
    "religion", "sports", "music", "cooking", "honesty", "creativity",
    "intelligence", "kindness", "freedom", "justice", "power", "beauty", "truth",
    "ocean", "mountain", "city", "forest", "fire", "space", "car", "airplane",
    "boat", "bicycle", "truck", "ship", "motorcycle", "helicopter", "dog", "cat",
    "fish", "spider", "horse", "bird", "snake", "rabbit", "pain", "skill", "queen",
    "rage", "law", "charm", "stone", "light", "peace", "film", "lake", "bread",
    "pizza", "rice", "soup", "cake", "pie", "coffee", "tea",
]
LABELS = ["early", "middle", "late"]
OUTPUT_TOKENS = [[lab, f" {lab}"] for lab in LABELS]
TRIGGER = "The injection depth is"
SYSTEM_PROMPT = (
    "You are participating in a word association experiment. "
    "Each input triggers a specific response: early, middle, or late. "
    "Learn the pattern from examples. "
    "Respond with only the target word — no preamble, no explanation."
)

_CMAX_FLOOR = 0.1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gemma-31b",
                   help="model short name (default: gemma-31b)")
    p.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES; if unset, do not modify the env")
    p.add_argument("--anchors", default=None,
                   help="comma-separated anchor layers (early,middle,late). "
                        "Default: per-model overrides from layer_introspection.")
    p.add_argument("--cmax_fraction", type=float, default=1.25,
                   help="fraction of c_max applied at each turn (default: 1.25 "
                        "matches Gemma's TYPE2_LAYER_FIXED_CF optimum)")
    p.add_argument("--sweep_start", type=int, default=0)
    p.add_argument("--sweep_end", type=int, default=None,
                   help="last test layer (default: model's last decoder layer)")
    p.add_argument("--sweep_stride", type=int, default=1)
    p.add_argument("--test_layers", default=None,
                   help="comma-separated explicit test layers")
    p.add_argument("--n_examples", type=int, default=20)
    p.add_argument("--n_samples", type=int, default=30,
                   help="queries per test layer per prompt variation (default: 30; "
                        "with 10 variations → 300 samples pooled per layer)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="evals/layer_generalization",
                   help="dir for JSON logs (and per-run PNG/HTML)")
    p.add_argument("--writeup_plot_dir", default="writeup/plots",
                   help="dir for the canonical PNG/HTML used by the paper")
    p.add_argument("--tag", default=None)
    p.add_argument("--no_save_writeup", action="store_true")
    p.add_argument("--rerender_from", default=None,
                   help="path to an existing sweep JSON; if set, skip running")
    p.add_argument("--prompt_variations", type=int, default=10,
                   help="how many prompt variations to pool over (0 to disable)")
    p.add_argument("--variation_shard", default=None,
                   help="comma-separated indices into VARIATIONS['layer_introspection']")
    return p


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    args = _build_parser().parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    if args.rerender_from:
        _rerender(Path(args.rerender_from), args)
        return

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from tqdm.auto import tqdm
    from icl import (
        ICLQuery,
        build_prompt,
        ensure_concepts,
        extract_token_logits,
        find_trigger_positions,
        forward_with_multi_layer_steering,
        get_concept_library,
        get_library_path,
        get_model_and_tokenizer,
        MODEL_REGISTRY,
    )
    from icl.common.layer_introspection import get_layer_groups
    from icl.common.helpers import _load_cmax_cache

    info = MODEL_REGISTRY.get(args.model)
    if info is None:
        sys.exit(f"ERROR: unknown model {args.model!r}. Known: {list(MODEL_REGISTRY)}")
    n_layers = info["layers"]

    if args.anchors:
        anchors = _parse_int_list(args.anchors)
        if len(anchors) != 3:
            sys.exit(f"ERROR: --anchors needs exactly 3 layers, got {anchors}")
        layer_groups = list(zip(LABELS, anchors))
    else:
        layer_groups = get_layer_groups(args.model)
        anchors = [g[1] for g in layer_groups]

    test_layers = (_parse_int_list(args.test_layers) if args.test_layers
                   else list(range(args.sweep_start,
                                   (n_layers - 1 if args.sweep_end is None else args.sweep_end) + 1,
                                   args.sweep_stride)))

    if args.prompt_variations and args.prompt_variations > 0:
        from icl.common.prompt_variations import VARIATIONS
        all_vars = VARIATIONS["layer_introspection"][:args.prompt_variations]
        if args.variation_shard:
            shard_idx = [int(i) for i in args.variation_shard.split(",")]
            variations = [(i, all_vars[i]) for i in shard_idx]
        else:
            variations = list(enumerate(all_vars))
    else:
        variations = [(0, {"prompt_text": TRIGGER, "system_prompt": SYSTEM_PROMPT})]

    # Pre-load c_max for every (concept, anchor) needed for demos.
    cmax_cache = _load_cmax_cache(args.model)
    cmax_table = {(c, l): cmax_cache.get((c, l), _CMAX_FLOOR)
                  for c in CONCEPTS for l in anchors}
    n_floor_demo = sum(1 for c in CONCEPTS for l in anchors
                       if (c, l) not in cmax_cache)

    # For each test layer, fall back to the NEAREST anchor's c_max if not
    # calibrated. This keeps the test injection comparable to demo injections
    # — both scale by the same per-concept comprehension threshold.
    n_calibrated_test = 0
    for tl in test_layers:
        if tl in anchors:
            n_calibrated_test += len(CONCEPTS)
            continue  # already in cmax_table from the demo loop
        cached = all((c, tl) in cmax_cache for c in CONCEPTS)
        if cached:
            for c in CONCEPTS:
                cmax_table[(c, tl)] = cmax_cache[(c, tl)]
            n_calibrated_test += len(CONCEPTS)
        else:
            nearest = min(anchors, key=lambda a: abs(a - tl))
            for c in CONCEPTS:
                cmax_table[(c, tl)] = cmax_table[(c, nearest)]
    n_floor_test = (len(CONCEPTS) * len(test_layers)) - n_calibrated_test
    total_pairs = len(CONCEPTS) * len(test_layers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writeup_dir = Path(args.writeup_plot_dir)

    print(f"\n{'=' * 72}")
    print(f"  Layer generalisation sweep (c_max-scaled)")
    print(f"  Model            : {args.model} ({info['hf_id']}, {n_layers} layers)")
    print(f"  Anchors          : {layer_groups}")
    print(f"  cmax_fraction    : {args.cmax_fraction}")
    print(f"  c_max anchors    : {len(CONCEPTS)*len(anchors) - n_floor_demo}/{len(CONCEPTS)*len(anchors)} (concept × anchor) calibrated")
    print(f"  c_max test       : {total_pairs - n_floor_test}/{total_pairs} (concept × test_layer) calibrated; {n_floor_test} use floor=0.1")
    print(f"  Test layers      : {len(test_layers)} ({test_layers[0]}..{test_layers[-1]} stride {args.sweep_stride})")
    print(f"  n_examples       : {args.n_examples}")
    print(f"  n_samples        : {args.n_samples} per layer per variation")
    print(f"  prompt_variations: {len(variations)} (indices: {[i for i, _ in variations]})")
    print(f"  seed             : {args.seed}")
    print(f"{'=' * 72}\n")

    model, tok = get_model_and_tokenizer(args.model)
    lib_path = get_library_path(args.model)
    clib = ensure_concepts(
        get_concept_library(model=model, tokenizer=tok, model_name=args.model),
        CONCEPTS, model=model, tokenizer=tok, library_path=lib_path,
    )
    needed = set(anchors) | set(test_layers)
    bad = [c for c in CONCEPTS
           if needed - set(clib.get_vector(c).layer_activations.keys())]
    if bad:
        sys.exit(f"ERROR: concept library missing layer activations for: {bad}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    shard_tag = (f"_shard{args.variation_shard.replace(',', '-')}"
                 if args.variation_shard else "")
    run_name = (f"layer_generalization_{args.model}_anchors-"
                f"{'-'.join(map(str, anchors))}{tag}{shard_tag}_{ts}")
    json_path = out_dir / f"{run_name}.json"

    sweep_records = _run_sweep(
        model=model, tok=tok, clib=clib,
        test_layers=test_layers, layer_groups=layer_groups,
        cmax_fraction=args.cmax_fraction,
        cmax_table=cmax_table, variations=variations,
        n_examples=args.n_examples, n_samples=args.n_samples,
        seed=args.seed, json_path=json_path,
        run_meta={
            "model": args.model, "hf_id": info["hf_id"], "n_layers": n_layers,
            "anchors": anchors, "labels": LABELS,
            "layer_groups": [{"class": c, "layer": l} for c, l in layer_groups],
            "test_layers": test_layers,
            "cmax_fraction": args.cmax_fraction,
            "n_examples": args.n_examples, "n_samples": args.n_samples,
            "n_prompt_variations": len(variations),
            "variation_indices": [i for i, _ in variations],
            "seed": args.seed,
            "cmax_floor": _CMAX_FLOOR,
        },
        ICLQuery=ICLQuery, build_prompt=build_prompt,
        find_trigger_positions=find_trigger_positions,
        forward_with_multi_layer_steering=forward_with_multi_layer_steering,
        extract_token_logits=extract_token_logits,
        tqdm=tqdm,
    )

    png_path = out_dir / f"{run_name}.png"
    _render_png(sweep_records, layer_groups, args.model,
                args.cmax_fraction, args.n_examples, args.n_samples,
                len(variations), png_path)

    if not args.no_save_writeup:
        writeup_dir.mkdir(parents=True, exist_ok=True)
        canon_png = writeup_dir / f"layer_generalization_{args.model.replace('-', '')}.png"
        _render_png(sweep_records, layer_groups, args.model,
                    args.cmax_fraction, args.n_examples, args.n_samples,
                    len(variations), canon_png)


def _run_sweep(*, model, tok, clib, test_layers, layer_groups, cmax_fraction,
               cmax_table, variations, n_examples, n_samples, seed, json_path,
               run_meta, ICLQuery, build_prompt, find_trigger_positions,
               forward_with_multi_layer_steering, extract_token_logits, tqdm):
    sweep_records: list[dict] = []
    anchors = [l for _, l in layer_groups]

    for test_layer in tqdm(test_layers, desc="test layers"):
        per_class_counts = {lab: 0 for lab in LABELS}
        p_sums = {lab: 0.0 for lab in LABELS}
        samples: list[dict] = []
        nearest_class = min(layer_groups, key=lambda g: abs(g[1] - test_layer))[0]

        n_total = 0
        for var_idx, var in variations:
            trigger = var["prompt_text"]
            system_prompt = var["system_prompt"]
            rng = Random(seed + 31 * test_layer + 17 * var_idx)

            for i in range(n_samples):
                test_concept = rng.choice(CONCEPTS)
                available_pairs = [(c, g) for c in CONCEPTS if c != test_concept
                                   for g in layer_groups]
                n_choose = min(n_examples, len(available_pairs))
                chosen = rng.sample(available_pairs, n_choose)
                example_concepts = [c for c, _ in chosen]
                example_groups = [g for _, g in chosen]

                words = [g[0] for g in example_groups]
                all_concepts = example_concepts + [test_concept]
                prompts_list = [trigger] * len(all_concepts)
                all_layers = [g[1] for g in example_groups] + [test_layer]
                # Per-turn absolute scale = cmax_fraction × c_max(concept, that_turn's_layer).
                all_scales = [cmax_fraction * cmax_table[(c, l)]
                              for c, l in zip(all_concepts, all_layers)]

                dummy_query = ICLQuery(
                    concepts=all_concepts, words=words,
                    output_tokens=OUTPUT_TOKENS, system_prompt=system_prompt,
                    injection_layer=all_layers, injection_scale=all_scales,
                    prompts=prompts_list,
                )
                prompt_str = build_prompt(dummy_query, tok)
                input_ids = tok.encode(prompt_str, return_tensors="pt",
                                       add_special_tokens=False).to("cuda")
                triggers = find_trigger_positions(input_ids[0], tok, prompts_list)

                triples = []
                for positions, concept_name, layer_idx in zip(triggers, all_concepts, all_layers):
                    sv = clib.get_vector(concept_name)
                    triples.append((positions, layer_idx, sv))

                logits = forward_with_multi_layer_steering(
                    model, input_ids, triples, scales=all_scales,
                )
                _, probs = extract_token_logits(logits, OUTPUT_TOKENS, tok)
                predicted = max(probs, key=probs.get)

                per_class_counts[predicted] += 1
                for k in p_sums:
                    p_sums[k] += probs.get(k, 0.0)

                samples.append({
                    "idx": i,
                    "var_idx": var_idx,
                    "test_concept": test_concept,
                    "predicted": predicted,
                    "probabilities": {k: float(probs.get(k, 0.0)) for k in LABELS},
                })
                n_total += 1

        mean_p = {k: (p_sums[k] / n_total if n_total else 0.0) for k in p_sums}
        majority = max(per_class_counts, key=per_class_counts.get)
        record = {
            "test_layer": test_layer,
            "nearest_anchor_class": nearest_class,
            "counts": per_class_counts,
            "mean_p": mean_p,
            "majority_class": majority,
            "n_total": n_total,
            "samples": samples,
        }
        sweep_records.append(record)

        with open(json_path, "w") as f:
            json.dump({**run_meta, "records": sweep_records}, f, indent=2)

        print(f"  layer {test_layer:3d}  majority={majority:<6s}  "
              f"p(early)={mean_p['early']:.2f} p(mid)={mean_p['middle']:.2f} "
              f"p(late)={mean_p['late']:.2f}  nearest={nearest_class}  "
              f"n={n_total}")

    print(f"\nWrote {json_path}")
    return sweep_records


def _series_from_records(sweep_records):
    import numpy as np
    z = 1.96
    xs = [r["test_layer"] for r in sweep_records]
    means: dict[str, list[float]] = {lab: [] for lab in LABELS}
    lo: dict[str, list[float]] = {lab: [] for lab in LABELS}
    hi: dict[str, list[float]] = {lab: [] for lab in LABELS}
    for r in sweep_records:
        samples = r.get("samples", [])
        for lab in LABELS:
            vals = np.array([s["probabilities"][lab] for s in samples],
                            dtype=float)
            n = len(vals)
            m = float(vals.mean()) if n else 0.0
            sd = float(vals.std(ddof=1)) if n > 1 else 0.0
            half = z * sd / (n ** 0.5) if n else 0.0
            means[lab].append(m)
            lo[lab].append(max(0.0, m - half))
            hi[lab].append(min(1.0, m + half))
    return xs, means, lo, hi


def _model_display(short: str) -> str:
    return {
        "gemma-31b": "Gemma-31B",
        "qwen3-32b": "Qwen3-32B",
        "qwen3-8b": "Qwen3-8B",
        "olmo-32b": "OLMo-32B",
        "olmo-7b": "OLMo-7B",
    }.get(short, short)


def _render_png(sweep_records, layer_groups, model_name, cmax_fraction,
                n_examples, n_samples, n_variations, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, means, lo, hi = _series_from_records(sweep_records)
    series = [
        ("early",  "#2563eb", f"early (anchor L{layer_groups[0][1]})"),
        ("middle", "#16a34a", f"middle (anchor L{layer_groups[1][1]})"),
        ("late",   "#dc2626", f"late (anchor L{layer_groups[2][1]})"),
    ]
    anchors = [l for _, l in layer_groups]

    fig, ax = plt.subplots(figsize=(11, 5))
    for lab, color, legend in series:
        ax.fill_between(xs, lo[lab], hi[lab], color=color, alpha=0.18, linewidth=0)
        ax.plot(xs, means[lab], "-", color=color, lw=2, label=legend)

    for a, col in zip(anchors, ("#2563eb", "#16a34a", "#dc2626")):
        ax.axvline(a, ls="--", lw=1.2, color=col, alpha=0.7)
    ax.axhline(1/3, color="gray", ls=":", lw=1.5, alpha=0.7,
               label="Chance (0.333)")

    ax.set_xlabel(r"Test Query Injection Layer $\ell$", fontsize=11)
    ax.set_ylabel(r"Mean $P(\mathrm{label})$", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{_model_display(model_name)} Layer Introspection Generalization",
        fontsize=13,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
              framealpha=0.95)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")

    # Plotly HTML twin in matching style.
    import plotly.graph_objects as go
    html_path = out_path.with_suffix(".html")
    pfig = go.Figure()
    plotly_legends = [
        ("early",  "#2563eb", f"early (anchor L{layer_groups[0][1]})"),
        ("middle", "#16a34a", f"middle (anchor L{layer_groups[1][1]})"),
        ("late",   "#dc2626", f"late (anchor L{layer_groups[2][1]})"),
    ]
    for lab, color, legend in plotly_legends:
        pfig.add_trace(go.Scatter(
            x=list(xs) + list(xs)[::-1],
            y=list(hi[lab]) + list(lo[lab])[::-1],
            fill="toself", fillcolor=color, line=dict(width=0),
            opacity=0.18, hoverinfo="skip", showlegend=False, name=f"{lab} CI",
        ))
        pfig.add_trace(go.Scatter(
            x=xs, y=means[lab], mode="lines",
            line=dict(color=color, width=2), name=legend,
        ))
    for a, col in zip(anchors, ("#2563eb", "#16a34a", "#dc2626")):
        pfig.add_vline(x=a, line=dict(color=col, dash="dash", width=1.2), opacity=0.7)
    pfig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="gray", dash="dot", width=1.5),
        name="Chance (0.333)",
    ))
    pfig.add_hline(y=1/3, line=dict(color="gray", dash="dot", width=1.5))
    pfig.update_layout(
        title=f"{_model_display(model_name)} Layer Introspection Generalization",
        xaxis_title="Test Query Injection Layer ℓ",
        yaxis_title="Mean P(label)",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        legend=dict(x=1.02, y=1.0),
        width=1100, height=500,
    )
    pfig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  → {html_path}")


def _rerender(json_path: Path, args) -> None:
    payload = json.loads(json_path.read_text())
    records = payload["records"]
    anchors = payload["anchors"]
    layer_groups = list(zip(LABELS, anchors))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = json_path.stem
    png_path = out_dir / f"{base}.png"
    _render_png(records, layer_groups, payload["model"],
                payload["cmax_fraction"], payload["n_examples"],
                payload["n_samples"],
                payload.get("n_prompt_variations", 1),
                png_path)
    if not args.no_save_writeup:
        writeup_dir = Path(args.writeup_plot_dir)
        writeup_dir.mkdir(parents=True, exist_ok=True)
        canon_png = writeup_dir / f"layer_generalization_{payload['model'].replace('-', '')}.png"
        _render_png(records, layer_groups, payload["model"],
                    payload["cmax_fraction"], payload["n_examples"],
                    payload["n_samples"],
                    payload.get("n_prompt_variations", 1),
                    canon_png)


if __name__ == "__main__":
    main()
