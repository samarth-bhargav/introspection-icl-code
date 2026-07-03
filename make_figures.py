#!/usr/bin/env python3
"""Regenerate every figure used in the paper PDF from the shipped eval data.

This is a thin orchestrator: it just runs the individual plotting entry points
(documented one-by-one in README.md) in the right order, writing into
``plots/`` and ``plots_new/`` so the output paths match the paper's
``\\includegraphics`` paths exactly.

None of these steps need a GPU or the judge daemon — they read the JSON eval
logs under ``evals/`` (and, for the three layer-introspection panels whose raw
sweep logs were not preserved, the shipped Plotly ``.html`` companions).

    python make_figures.py            # all figures
    python make_figures.py --list     # print the figure -> command map and exit

Run it from the repo root with this repo on PYTHONPATH, e.g.

    PYTHONPATH=. /path/to/venv/bin/python make_figures.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable

# Each step: (human label, argv). argv[0] is either "-m <module>" or a script.
STEPS: list[tuple[str, list[str]]] = [
    # ── Magnitude introspection (paper "plots_new/") ─────────────────────────
    ("magnitude type1/type2 + generalization (+combined) -> plots_new/",
     ["-m", "icl.plotting.plot_magnitude", "--plots-dir", "plots_new"]),
    ("magnitude prompt-sensitivity -> plots/type2/",
     ["-m", "icl.plotting.plot_prompt_sigma", "--plots-dir", "plots"]),

    # ── Emotion-gated arithmetic + amendment successor (full 6-emotion run) ──
    ("gated type2 math+successor (+prompt-sigma) -> plots/full_6emo/type2/",
     ["gated6/plot_full_6emo_figs.py"]),
    ("gated anger-only type2 math+successor -> plots/anger/type2/",
     ["gated6/plot_anger_figs.py"]),
    ("gated per-mode math panels -> plots/full_6emo/math_modes/",
     ["-m", "icl.plotting.plot_math_modes",
      "--gen_root", "evals/full_6emo", "--out_dir", "plots/full_6emo/math_modes"]),
    ("gated per-mode successor panels -> plots/full_6emo/successor/",
     ["-m", "icl.plotting.plot_successor_modes",
      "--root", "evals/full_6emo/successor_emotions_k_sweep",
      "--plots-dir", "plots/full_6emo", "--subdir", "successor"]),
    # type1 (strength-sweep) gated panels — generated from the pre-6emo strength
    # sweeps that the paper still uses for these appendix figures.
    ("gated type1 math+successor strength sweeps -> plots/type1/",
     ["-m", "icl.plotting.plot_strength_sweep", "--repo", ".", "--plots-dir", "plots"]),

    # ── Layer-introspection generalization (rendered from eval JSON) ─────────
    ("layer generalization per-model (raw) -> plots/",
     ["-m", "icl.plotting.plot_layer_generalization",
      "--rerender_from", "evals/regen/layer_generalization/layer_generalization_gemma-31b.json",
      "--out_dir", "plots/_raw", "--writeup_plot_dir", "plots"]),
]

# The five per-model layer-generalization rerenders (one entry per model).
_LAYERGEN_MODELS = ["gemma-31b", "qwen3-32b", "qwen3-8b", "olmo-32b", "olmo-7b"]
for _m in _LAYERGEN_MODELS[1:]:
    STEPS.append((f"layer generalization {_m} (raw) -> plots/",
                  ["-m", "icl.plotting.plot_layer_generalization",
                   "--rerender_from", f"evals/regen/layer_generalization/layer_generalization_{_m}.json",
                   "--out_dir", "plots/_raw", "--writeup_plot_dir", "plots"]))

# Uniform paper styling for the figures that were emitted "raw" (the layer
# generalization rerenders) and for the three shipped layer-introspection
# panels whose raw sweep logs were not preserved.
_STYLE_ONLY = [
    "type1/type1_layer_introspection.html",
    "type2/type2_layer_introspection_merged.html",
    "type2/type2_layer_introspection_prompt_sigma.html",
    "layer_generalization_gemma31b.html",
    "layer_generalization_qwen332b.html",
    "layer_generalization_qwen38b.html",
    "layer_generalization_olmo32b.html",
    "layer_generalization_olmo7b.html",
]
STEPS.append(("apply paper styling to layer panels -> plots/",
              ["-m", "icl.plotting.apply_paper_styling", "--plots-dir", "plots", "--only", *_STYLE_ONLY]))
STEPS.append(("combined side-by-side layer generalization -> plots/",
              ["-m", "icl.plotting.build_combined_layer", "--plots-dir", "plots"]))


def _run(label: str, argv: list[str]) -> None:
    cmd = [PY, *argv]
    print(f"\n=== {label}\n    $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print steps and exit")
    args = ap.parse_args()
    if args.list:
        for label, argv in STEPS:
            print(f"{label}\n    python {' '.join(argv)}\n")
        return

    # Create the output-dir skeleton up front (some renderers require their
    # --plots-dir to already exist and don't create it themselves).
    for sub in ("plots/type1", "plots/type2", "plots/full_6emo/type2",
                "plots/full_6emo/math_modes", "plots/full_6emo/successor",
                "plots/anger/type2", "plots_new/type1", "plots_new/type2"):
        (REPO / sub).mkdir(parents=True, exist_ok=True)

    # Seed the three layer-introspection panels whose raw sweep logs were not
    # preserved: their Plotly .html companions live in figure_sources/ and are
    # re-styled in place by the apply_paper_styling step.
    seeded = 0
    for src in (REPO / "figure_sources").rglob("*.html"):
        dst = REPO / "plots" / src.relative_to(REPO / "figure_sources")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            seeded += 1
    if seeded:
        print(f"seeded {seeded} layer-introspection .html from figure_sources/")

    for label, argv in STEPS:
        _run(label, argv)

    # Drop the throwaway raw-render scratch dir.
    shutil.rmtree(REPO / "plots" / "_raw", ignore_errors=True)
    print("\nAll figures regenerated under plots/ and plots_new/.")


if __name__ == "__main__":
    main()
