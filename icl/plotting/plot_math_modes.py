"""Math (emotion-gated arithmetic) introspection: per-mode accuracy vs K.

Three appendix plots, one per test-turn injection mode:

  * target      — the gated emotion IS injected (model should double)
  * distractor  — a different concept is injected (model should NOT double)
  * none        — nothing injected (model should NOT double)

Each plot shows mean P(correct) for that mode versus the number of in-context
examples K, one line per model, with 95% Wilson confidence-interval bands.
Rendering reuses the paper's shared Plotly primitives (``_apply_standard_layout``
+ ``_write`` from ``plot_magnitude``) so these figures match
the rest of the paper --- in particular the per-mode amendment-successor
appendix figures, which share the same lines+markers Plotly style. ``_write``
emits both the ``.html`` twin and the ``.png`` (via kaleido) used in the paper.

Usage:
    python -m icl.plotting.plot_math_modes \
        --gen_root evals/regen \
        --out_dir plots/math_modes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go

from icl.plotting._intro_common import wilson_ci
from icl.plotting.plot_magnitude import (
    MODEL_COLORS,
    MODEL_DISPLAY,
    MODELS,
    _apply_standard_layout,
    _write,
)

# (test_injection.kind, short title, file slug)
MODES = [
    ("target", "Correct Injection", "target"),
    ("distractor", "Distractor Injection", "distractor"),
    ("none", "No Injection", "none"),
]
# Dotted-gray baseline accuracy per mode: 0.5 for the correct-injection (target)
# graph; 0.75 for the distractor and no-injection (control) graphs.
MODE_BASELINES = {"target": 0.5, "distractor": 0.75, "none": 0.75}
# Match the successor per-mode appendix figures (Amendment Successor — <Mode>).
XLABEL = "In-Context Examples (<i>k</i>)"


def load_mode_series(model: str, mode: str, gen_root: Path) -> list[dict]:
    """Per-K {x, mp, mp_lo, mp_hi, n} for one model/mode from the type-2 sweep.

    Accuracy is the fraction of correct test-turn answers among rows whose test
    injection kind == *mode*; the band is the Wilson 95% CI of that proportion.
    """
    d = gen_root / f"generation_{model}" / "math" / "type2_k_sweep"
    rows: list[dict] = []
    for fp in sorted(d.glob(f"math_{model}_k*.json")):
        try:
            e = json.loads(fp.read_text())
        except Exception:
            continue
        k = e.get("n_demos")
        if k is None:
            continue
        n = 0
        n_correct = 0
        for r in e.get("rows", []):
            if (r.get("test_injection") or {}).get("kind") == mode:
                n += 1
                n_correct += int(bool(r.get("correct")))
        if n == 0:
            continue
        mp = n_correct / n
        lo, hi = wilson_ci(n_correct, n)
        rows.append({"x": k, "mp": mp, "mp_lo": lo, "mp_hi": hi, "n": n})
    rows.sort(key=lambda r: r["x"])
    return rows


def render_modes(by_model, out_path: Path, title: str, xlabel: str, baseline: float = 0.5) -> None:
    """Multi-model overlay with 95% CI shading, in the paper's shared Plotly
    style (matches the per-mode amendment-successor appendix figures): mean line
    with markers + shaded CI band per model, dotted-gray baseline line, y-axis
    Accuracy starting at 0. ``_write`` emits both the ``.html`` twin and the
    ``.png`` used in the paper."""
    has_data = any(by_model.get(m) for m in MODELS)
    if not has_data:
        print(f"  no data for {out_path.name}, skipping")
        return

    fig = go.Figure()
    for m in MODELS:
        rows = by_model.get(m, [])
        if not rows:
            continue
        xs = [r["x"] for r in rows]
        mps = [r["mp"] for r in rows]
        lo = [r["mp_lo"] for r in rows]
        hi = [r["mp_hi"] for r in rows]
        c = MODEL_COLORS[m]
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], y=hi + lo[::-1], fill="toself", fillcolor=c,
            line=dict(width=0), opacity=0.18, hoverinfo="skip",
            showlegend=False, name=f"{MODEL_DISPLAY[m]} (95% CI)",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=mps, mode="lines+markers", line=dict(color=c, width=2),
            marker=dict(size=5), name=MODEL_DISPLAY[m],
        ))
    _apply_standard_layout(fig, title=title, xlabel=xlabel, ylabel="Accuracy")
    fig.add_hline(y=baseline, line=dict(color="gray", dash="dot", width=1.5),
                  annotation_text=f"Baseline ({baseline:.2f})", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 1.05])
    _write(fig, out_path.with_suffix(".html"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen_root", default="evals/regen")
    ap.add_argument("--out_dir", default="evals/regen/plots/math_modes")
    args = ap.parse_args()

    gen_root = Path(args.gen_root)
    out_dir = Path(args.out_dir)

    for mode, short, slug in MODES:
        by_model = {}
        for m in MODELS:
            rows = load_mode_series(m, mode, gen_root)
            if rows:
                by_model[m] = rows
        title = f"Emotion-Gated Arithmetic — {short}"
        render_modes(by_model, out_dir / f"math_mode_{slug}.png", title, XLABEL,
                     baseline=MODE_BASELINES.get(mode, 0.5))


if __name__ == "__main__":
    main()
