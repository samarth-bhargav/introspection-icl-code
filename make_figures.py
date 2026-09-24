#!/usr/bin/env python3
"""Render paper figures from evaluation logs; check required inputs first."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent
MODELS = ("gemma-31b", "qwen3-32b", "qwen3-8b", "olmo-32b", "olmo-7b")
MAG = "evals/regen/constitution_source_magnitude"
GATED = "evals/full_6emo"
LAYER_PANELS = (
    "type1/type1_layer_introspection.html",
    "type2/type2_layer_introspection_merged.html",
    "type2/type2_layer_introspection_prompt_sigma.html",
)


@dataclass(frozen=True)
class Step:
    name: str
    commands: tuple[tuple[str, ...], ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]  # PNGs; renderers also write HTML companions.
    hint: str


def module(name, *args):
    return ("-m", f"icl.plotting.{name}", *args)


def gated_inputs(task):
    if task == "math":
        return tuple(f"{GATED}/generation_{m}/math/type2_k_sweep/math_{m}_k*.json" for m in MODELS)
    return tuple(f"{GATED}/successor_emotions_k_sweep/{m}/k*_var*.json" for m in MODELS)


def steps():
    layer_html = tuple(f"layer_generalization_{m.replace('-', '')}.html" for m in MODELS)
    return (
        Step("magnitude", (module("plot_magnitude", "--plots-dir", "plots_new"),),
             tuple(f"{MAG}/magnitude/{kind}_{m}.json" for m in MODELS for kind in ("type1", "type2"))
             + tuple(f"{MAG}/magnitude_generalization/magnitude_generalization_{m}.json" for m in MODELS),
             ("plots_new/type1/type1_magnitude_introspection.png",
              "plots_new/type2/type2_magnitude_introspection_merged.png",
              "plots_new/magnitude_generalization_combined.png")
             + tuple(f"plots_new/magnitude_generalization_{m.replace('-', '')}.png" for m in MODELS),
             "Run icl.experiments.magnitude for each model."),
        Step("magnitude-sensitivity", (module("plot_prompt_sigma", "--plots-dir", "plots"),),
             tuple(f"{MAG}/magnitude/type2_{m}.json" for m in MODELS),
             ("plots/type2/type2_magnitude_introspection_prompt_sigma.png",),
             "Run magnitude with prompt variations; the JSON must contain by_prompt summaries."),
        Step("gated", (module("plot_gated"),), gated_inputs("math") + gated_inputs("successor"),
             tuple(f"plots/full_6emo/type2/type2_{task}{suffix}.png"
                   for task in ("math_introspection", "amendment_successor")
                   for suffix in ("", "_prompt_sigma")),
             "Generate the six-emotion arithmetic and successor k sweeps."),
        Step("anger", (module("plot_anger"),), gated_inputs("math") + gated_inputs("successor"),
             tuple(f"plots/anger/type2/type2_{task}.png" for task in ("math_introspection", "amendment_successor")),
             "Use six-emotion logs with per-row target_emotion/gating_emotion fields."),
        Step("math-modes", (module("plot_math_modes", "--gen_root", GATED,
                                   "--out_dir", "plots/full_6emo/math_modes"),), gated_inputs("math"),
             tuple(f"plots/full_6emo/math_modes/math_mode_{mode}.png" for mode in ("target", "distractor", "none")),
             "Generate the six-emotion arithmetic k sweep."),
        Step("successor-modes", (module("plot_successor_modes", "--root", f"{GATED}/successor_emotions_k_sweep",
                                        "--plots-dir", "plots/full_6emo", "--subdir", "successor"),),
             gated_inputs("successor"),
             tuple(f"plots/full_6emo/successor/successor_{mode}.png" for mode in ("target", "distractor", "none")),
             "Generate the six-emotion successor k sweep."),
        Step("gated-strength", (module("plot_strength_sweep", "--repo", ".", "--plots-dir", "plots", "--kind", "type1"),),
             tuple(f"evals/regen/generation_{m}/math/type1_cmax_sweep/*cmax*.json" for m in MODELS)
             + tuple(f"evals/regen/successor_cmax_sweep/{m}/fraction_*.json" for m in MODELS),
             tuple(f"plots/type1/type1_{task}.png" for task in ("math_introspection", "amendment_successor")),
             "Requires the separate behavioral strength sweeps; see docs/reproduction-status.md."),
        Step("layer-generalization",
             tuple(module("plot_layer_generalization", "--rerender_from",
                          f"evals/regen/layer_generalization/layer_generalization_{m}.json",
                          "--out_dir", "plots/_raw", "--writeup_plot_dir", "plots") for m in MODELS)
             + (module("apply_paper_styling", "--plots-dir", "plots", "--only", *layer_html),
                module("build_combined_layer", "--plots-dir", "plots")),
             tuple(f"evals/regen/layer_generalization/layer_generalization_{m}.json" for m in MODELS),
             tuple(f"plots/{p.removesuffix('.html')}.png" for p in layer_html)
             + ("plots/layer_generalization_combined.png",),
             "Run icl.experiments.run_model with the layergen stage for each model."),
        Step("layer-sweeps", (module("plot_layer_sweeps"),),
             tuple(f"evals/regen/layer/type1_{m}.json" for m in MODELS)
             + tuple(f"evals/regen/layer/type2_{m}_var*.json" for m in MODELS),
             tuple(f"plots/{p.removesuffix('.html')}.png" for p in LAYER_PANELS),
             "Run the layer strength and prompt-variation sweeps for each model."),
    )


def missing_inputs(step, root=REPO):
    """Check file presence per model, not numerical reproduction or run completeness."""
    return [pattern for pattern in step.inputs
            if not any(p.is_file() and p.stat().st_size for p in root.glob(pattern))]


def run_step(step, root=REPO):
    # Remove no files: compare output metadata to detect a renderer silently skipping.
    before = {p: (root / p).stat().st_mtime_ns if (root / p).exists() else None
              for p in step.outputs}
    for argv in step.commands:
        print(f"[{step.name}] python {' '.join(argv)}", flush=True)
        subprocess.run([sys.executable, *argv], cwd=root, check=True)
    missing = [p for p in step.outputs if not (root / p).is_file()
               or not (root / p).stat().st_size
               or (root / p).stat().st_mtime_ns == before[p]]
    if missing:
        raise RuntimeError(f"{step.name} did not write expected outputs: {', '.join(missing)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list figure groups and commands")
    ap.add_argument("--check", action="store_true", help="check input file presence without rendering")
    ap.add_argument("--only", help="comma-separated figure groups from --list (default: all)")
    args = ap.parse_args()
    selected = list(steps())
    if args.only:
        names = set(args.only.split(","))
        unknown = names - {step.name for step in selected}
        if unknown:
            ap.error(f"unknown figure groups: {', '.join(sorted(unknown))}")
        selected = [step for step in selected if step.name in names]
    if args.list:
        for step in selected:
            print(step.name)
            for argv in step.commands:
                print(f"  python {' '.join(argv)}")
        return
    blocked = False
    for step in selected:
        missing = missing_inputs(step)
        print(f"{step.name}: {'MISSING INPUTS' if missing else 'input files present'}")
        if missing:
            blocked = True
            for path in missing:
                print(f"  {path}")
            print(f"  {step.hint}")
    if blocked:
        raise SystemExit(1)
    if args.check:
        print("File presence only: this does not verify schemas, complete sweeps, or numerical agreement.")
        return
    for sub in ("plots", "plots_new"):
        (REPO / sub).mkdir(parents=True, exist_ok=True)
    for step in selected:
        run_step(step)
    print(f"Rendered {len(selected)} figure groups under plots/ and plots_new/.")


if __name__ == "__main__":
    main()
