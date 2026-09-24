"""Per-injection-mode accuracy-vs-K appendix figures for the successor task.

For each of the three test-injection modes the model can see on the held-out
turn --- ``target`` (correct emotion injected -> answer the next amendment),
``distractor`` (an unrelated concept injected -> answer the same amendment),
and ``none`` (nothing injected -> answer the same amendment) --- we plot the
mean P(correct) as a function of the number of in-context examples ``k``, one
line per model, with a 95% confidence interval.

As with the paper's main introspection figures, the point estimate at each
``k`` is the mean over all per-sample correctness values for that mode (pooled
across the 10 prompt variations) and the band is the 95% CI over those data
points, computed with the same ``_ci`` helper the paper uses
(``mean +/- 1.96 * SEM``). Styling reuses the paper's Plotly primitives so the
figures match the rest of the paper; the 3-class chance line is intentionally
omitted (this is a 27-way free-generation task, not 3-way classification).

Usage:
    .venv/bin/python -m icl.plotting.plot_successor_modes \
        --root evals/regen/successor_k_sweep \
        --plots-dir plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go

from icl.plotting.plot_magnitude import (
    MODEL_COLORS,
    MODEL_DISPLAY,
    MODELS,
    _apply_standard_layout,
    _ci,
    _write,
)

MODES = {
    "target": "Correct Injection",
    "distractor": "Distractor Injection",
    "none": "No Injection",
}
# Dotted-gray baseline accuracy per mode. 0.5 for the correct-injection (target)
# graph; 0.75 for the distractor and no-injection (control) graphs (best mixed
# trivial strategy on those conditions).
MODE_BASELINES = {"target": 0.5, "distractor": 0.75, "none": 0.75}
XLABEL = "In-Context Examples (<i>k</i>)"


def _rows_for_mode(model_dir: Path, model: str, mode: str) -> list[dict]:
    """Build [{x, mp, mp_lo, mp_hi}] across K for one model + injection mode.

    At each K, pool the per-sample correctness (0/1) for ``mode`` across all 10
    prompt variations, then take mean + 95% CI via the paper's ``_ci`` helper.
    """
    ks = sorted(
        {int(p.stem.split("_")[0][1:]) for p in model_dir.glob("k*_var*.json")}
    )
    rows: list[dict] = []
    for k in ks:
        values: list[float] = []
        for var_path in sorted(model_dir.glob(f"k{k}_var*.json")):
            payload = json.loads(var_path.read_text())
            for r in payload.get("rows", []):
                if (r.get("test_injection") or {}).get("kind") == mode and r.get("p_correct") is not None:
                    values.append(float(r["p_correct"]))
        if not values:
            continue
        mean, lo, hi = _ci(values)
        rows.append({"x": k, "mp": mean, "mp_lo": lo, "mp_hi": hi})
    rows.sort(key=lambda r: r["x"])
    return rows


def _render_mode(by_model: dict[str, list[dict]], out_html: Path, *, title: str,
                 baseline: float = 0.5) -> None:
    fig = go.Figure()
    for model in MODELS:
        rows = by_model.get(model, [])
        if not rows:
            continue
        xs = [r["x"] for r in rows]
        means = [r["mp"] for r in rows]
        lo = [r["mp_lo"] for r in rows]
        hi = [r["mp_hi"] for r in rows]
        color = MODEL_COLORS[model]
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=hi + lo[::-1],
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            opacity=0.18,
            hoverinfo="skip",
            showlegend=False,
            name=f"{MODEL_DISPLAY[model]} (95% CI)",
        ))
        fig.add_trace(go.Scatter(
            x=xs,
            y=means,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=MODEL_DISPLAY[model],
        ))
    _apply_standard_layout(fig, title=title, xlabel=XLABEL, ylabel="Accuracy")
    fig.add_hline(y=baseline, line=dict(color="gray", dash="dot", width=1.5),
                  annotation_text=f"Baseline ({baseline:.2f})", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 1.05])
    _write(fig, out_html)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="evals/regen/successor_k_sweep")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--subdir", default="successor", help="subdir under plots-dir for outputs")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.plots_dir) / args.subdir

    for mode, mode_label in MODES.items():
        by_model: dict[str, list[dict]] = {}
        for model in MODELS:
            model_dir = root / model
            if not model_dir.is_dir():
                continue
            rows = _rows_for_mode(model_dir, model, mode)
            if rows:
                by_model[model] = rows
        if not by_model:
            print(f"  no data for mode={mode}, skipping")
            continue
        out_html = out_dir / f"successor_{mode}.html"
        _render_mode(
            by_model,
            out_html,
            title=f"Amendment Successor — {mode_label}",
            baseline=MODE_BASELINES.get(mode, 0.5),
        )


if __name__ == "__main__":
    main()
