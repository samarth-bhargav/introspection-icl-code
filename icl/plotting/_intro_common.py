"""Shared helpers for plot_type1 / plot_type2: data loading, CIs, styling.

All metrics are stored and rendered as proportions in ``[0, 1]`` (not %).
``render`` writes one matplotlib PNG and one plotly HTML for every plot so
titles can be tweaked downstream without re-running the script.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# Gemma listed first so it appears at the top of every legend.
MODELS = ["gemma-31b", "qwen3-32b", "qwen3-8b", "olmo-32b", "olmo-7b"]
MODEL_COLORS = {
    "gemma-31b": "#FFA15A",
    "qwen3-32b": "#636EFA",
    "qwen3-8b":  "#EF553B",
    "olmo-32b":  "#AB63FA",
    "olmo-7b":   "#00CC96",
}
MODEL_DISPLAY = {
    "gemma-31b": "Gemma-31B",
    "qwen3-32b": "Qwen3-32B",
    "qwen3-8b":  "Qwen3-8B",
    "olmo-32b":  "OLMo-32B",
    "olmo-7b":   "OLMo-7B",
}
CHANCE_3 = 1.0 / 3.0  # 3-class chance, as a proportion


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion → (lo, hi) in [0, 1]."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def normal_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Normal-approx 95% CI for a mean of [0,1] values → (lo, hi) in [0, 1]."""
    if not values:
        return 0.0, 0.0
    a = np.array(values, dtype=float)
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return max(0.0, m - z * se), min(1.0, m + z * se)


def load_dir(path: Path, x_param: str) -> dict[str, list[dict]]:
    """Load all JSONs in *path*, group by params.model, extract series.

    All numeric fields (``acc``, ``mp``, CIs) are returned as proportions in
    ``[0, 1]``.
    """
    by_model: dict[str, list[dict]] = defaultdict(list)
    for fp in sorted(path.glob("*.json")):
        e = json.loads(fp.read_text())
        params = e.get("params", {})
        model = params.get("model")
        x = params.get(x_param)
        if model is None or x is None:
            continue
        n_total = e["n_total"]
        n_correct = e["n_correct"]
        p_corrects = [s["p_correct"] for s in e["samples"]]
        acc_lo, acc_hi = wilson_ci(n_correct, n_total)
        mp = float(np.mean(p_corrects)) if p_corrects else 0.0
        mp_lo, mp_hi = normal_ci(p_corrects)
        by_model[model].append({
            "x": x,
            "acc": e["accuracy"],
            "acc_lo": acc_lo, "acc_hi": acc_hi,
            "mp": mp, "mp_lo": mp_lo, "mp_hi": mp_hi,
        })
    for m in by_model:
        by_model[m].sort(key=lambda r: r["x"])
    return by_model


def _render_plotly(by_model: dict[str, list[dict]], out_path: Path,
                   title: str, xlabel: str, chance: float,
                   xlog: bool) -> None:
    """Write the same data as ``render`` but as an interactive HTML."""
    import plotly.graph_objects as go

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
        # Translucent CI band (no legend entry).
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=hi + lo[::-1],
            fill="toself",
            fillcolor=c,
            line=dict(width=0),
            opacity=0.18,
            hoverinfo="skip",
            showlegend=False,
            name=f"{MODEL_DISPLAY[m]} (CI)",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=mps, mode="lines",
            line=dict(color=c, width=2),
            name=MODEL_DISPLAY[m],
        ))
    fig.add_hline(y=chance, line=dict(color="gray", dash="dot"),
                  annotation_text=f"Chance ({chance:.3f})",
                  annotation_position="bottom right")
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Mean P(correct)",
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(type="log") if xlog else None,
        template="plotly_white",
        legend=dict(x=1.02, y=1.0),
        width=1100, height=500,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Per-class line plot (for layer / magnitude generalisation sweeps)
#
# Same visual style as ``render`` above — figsize 10x6, axis fonts, legend on
# the right, dotted-grey chance line, CI shading at alpha 0.15, plotly_white
# HTML twin — but with one line per class (low/medium/high, shallow/middle/
# deep, …) plus optional vertical anchor markers.
# ---------------------------------------------------------------------------

# Class colors for 3-class plots (blue / green / red).
CLASS_COLORS_3 = ["#2563eb", "#16a34a", "#dc2626"]


def _render_class_sweep_plotly(
    xs: list[float],
    classes: list[str],
    means: dict[str, list[float]],
    lo: dict[str, list[float]],
    hi: dict[str, list[float]],
    out_path: Path, *,
    title: str, xlabel: str, ylabel: str = "Mean P(correct)",
    chance: float = CHANCE_3,
    class_colors: list[str] | None = None,
    class_legend: dict[str, str] | None = None,
    anchors: list[tuple[float, str]] | None = None,
    xlog: bool = False,
) -> None:
    import plotly.graph_objects as go

    palette = class_colors or CLASS_COLORS_3
    fig = go.Figure()
    for lab, c in zip(classes, palette):
        fig.add_trace(go.Scatter(
            x=list(xs) + list(xs)[::-1],
            y=list(hi[lab]) + list(lo[lab])[::-1],
            fill="toself", fillcolor=c, line=dict(width=0),
            opacity=0.18, hoverinfo="skip", showlegend=False,
            name=f"{lab} (CI)",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=means[lab], mode="lines",
            line=dict(color=c, width=2),
            name=(class_legend or {}).get(lab, lab),
        ))
    if anchors:
        for (x, _lab), c in zip(anchors, palette):
            fig.add_vline(x=x, line=dict(color=c, dash="dash", width=1.2),
                          opacity=0.7)
    fig.add_hline(y=chance, line=dict(color="gray", dash="dot"),
                  annotation_text=f"Chance ({chance:.3f})",
                  annotation_position="bottom right")
    fig.update_layout(
        title=title,
        xaxis_title=xlabel, yaxis_title=ylabel,
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(type="log") if xlog else None,
        template="plotly_white",
        legend=dict(x=1.02, y=1.0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def render_class_sweep(
    xs: list[float],
    classes: list[str],
    means: dict[str, list[float]],
    lo: dict[str, list[float]],
    hi: dict[str, list[float]],
    out_path: Path, *,
    title: str, xlabel: str, ylabel: str = "Mean P(correct)",
    chance: float = CHANCE_3,
    class_colors: list[str] | None = None,
    class_legend: dict[str, str] | None = None,
    anchors: list[tuple[float, str]] | None = None,
    xlog: bool = False,
) -> None:
    """3-class line plot with optional anchor markers; same style as ``render``.

    Writes ``out_path`` (PNG via matplotlib) and a sibling ``.html`` (interactive
    plotly).  Means/lo/hi are dicts keyed by class label, each a list aligned
    with *xs*.  *anchors* is a list of ``(x_position, class_label)`` pairs that
    add a coloured dashed vertical line at each anchor (one per class, in
    order).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = class_colors or CLASS_COLORS_3

    fig, ax = plt.subplots(figsize=(10, 6))
    for lab, c in zip(classes, palette):
        legend = (class_legend or {}).get(lab, lab)
        ax.plot(xs, means[lab], color=c, lw=2, label=legend)
        ax.fill_between(xs, lo[lab], hi[lab], color=c, alpha=0.15)

    if anchors:
        for (x, _lab), c in zip(anchors, palette):
            ax.axvline(x, ls="--", lw=1.2, color=c, alpha=0.7)

    ax.axhline(y=chance, color="gray", ls=":", lw=1.5,
               label=f"Chance ({chance:.3f})")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(0, 1.05)
    if xlog:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              framealpha=0.9, borderaxespad=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.72, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")

    html_path = out_path.with_suffix(".html")
    _render_class_sweep_plotly(
        xs, classes, means, lo, hi, html_path,
        title=title, xlabel=xlabel, ylabel=ylabel, chance=chance,
        class_colors=class_colors, class_legend=class_legend,
        anchors=anchors, xlog=xlog,
    )
    print(f"  → {html_path}")


def render(by_model: dict[str, list[dict]], out_path: Path,
           title: str, xlabel: str, chance: float = CHANCE_3,
           xlog: bool = False) -> None:
    """Multi-model overlay: mean P(correct) per model, 95% CI shading.

    Saves *out_path* (PNG via matplotlib) and a sibling ``.html``
    (interactive plotly) so titles/labels can be edited without re-running.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    has_data = False
    for m in MODELS:
        rows = by_model.get(m, [])
        if not rows:
            continue
        has_data = True
        xs = [r["x"] for r in rows]
        mps = [r["mp"] for r in rows]
        mp_lo = [r["mp_lo"] for r in rows]
        mp_hi = [r["mp_hi"] for r in rows]
        c = MODEL_COLORS[m]
        ax.plot(xs, mps, color=c, lw=2, label=MODEL_DISPLAY[m])
        ax.fill_between(xs, mp_lo, mp_hi, color=c, alpha=0.15)

    if not has_data:
        plt.close(fig)
        print(f"  no data for {out_path.name}, skipping")
        return

    ax.axhline(y=chance, color="gray", ls=":", lw=1.5,
               label=f"Chance ({chance:.3f})")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Mean P(correct)", fontsize=13)
    ax.set_ylim(0, 1.05)
    if xlog:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              framealpha=0.9, borderaxespad=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")

    html_path = out_path.with_suffix(".html")
    _render_plotly(by_model, html_path, title=title, xlabel=xlabel,
                   chance=chance, xlog=xlog)
    print(f"  → {html_path}")
