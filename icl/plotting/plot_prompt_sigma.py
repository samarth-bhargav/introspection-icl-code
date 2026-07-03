"""Port the magnitude type2 prompt-sigma appendix figure from the
constitution-source 10x30 rerun.

The paper's appendix (app:prompt_sigma) shows, for the magnitude ICL sweep, the
mean P(correct) over the 10 prompt variants with a band of +/-1 sigma across
those 10 prompt-level means (prompt sensitivity, not per-sample noise).

The 20%-depth rerun stores per-prompt summaries inline under ``by_prompt`` in
each pooled type2 JSON, so this script builds the prompt-sigma series from there
and renders it with the **same Plotly paper styling** as the other magnitude
figures (via ``plot_magnitude._render_model_overlay``), so
the appendix figure is visually consistent with the rest of the paper. It writes
to the existing paper filename so the TeX ``\\includegraphics`` does not change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reuse the exact Plotly overlay + paper styling used by the main magnitude
# figures so this appendix plot matches them.
from icl.plotting.plot_magnitude import (
    MODELS,
    _render_model_overlay,
)


_REPO = Path(__file__).resolve().parents[2]


def _prompt_sigma_rows_from_type2(path: Path) -> list[dict]:
    """Build mean +/-1 sigma-over-prompts rows from a pooled type2 JSON."""
    data = json.loads(path.read_text())
    rows: list[dict] = []
    for rec in data["per_strength"][0]["by_k"]:
        by_prompt = rec.get("by_prompt", [])
        prompt_means = [float(bp["mean_p"]) for bp in by_prompt if "mean_p" in bp]
        if not prompt_means:
            continue
        arr = np.asarray(prompt_means, dtype=float)
        mean = float(arr.mean())
        sigma = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        rows.append({
            "x": int(rec["k"]),
            "mp": mean,
            "mp_lo": max(0.0, mean - sigma),
            "mp_hi": min(1.0, mean + sigma),
        })
    rows.sort(key=lambda r: r["x"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--source-root",
        default=str(_REPO / "evals" / "regen" / "constitution_source_magnitude"),
    )
    ap.add_argument(
        "--plots-dir",
        default="/workspace/Introspection ICL - ARXIV/plots",
    )
    args = ap.parse_args()

    source_root = Path(args.source_root)
    plots_dir = Path(args.plots_dir)

    by_model: dict[str, list[dict]] = {}
    for model in MODELS:
        path = source_root / "magnitude" / f"type2_{model}.json"
        if not path.exists():
            print(f"  missing {path}, skipping {model}")
            continue
        rows = _prompt_sigma_rows_from_type2(path)
        if rows:
            by_model[model] = rows
            print(f"  {model}: {len(rows)} K-points")

    if not by_model:
        raise SystemExit("no type2 by_prompt data found; rerun with prompt variations")

    _render_model_overlay(
        by_model,
        plots_dir / "type2" / "type2_magnitude_introspection_prompt_sigma.html",
        title="Injection Magnitude Classification (Prompt Sensitivity)",
        xlabel="In-Context Examples (<i>k</i>)",
    )


if __name__ == "__main__":
    main()
