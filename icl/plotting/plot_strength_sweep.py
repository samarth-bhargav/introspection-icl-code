"""Port the concept-identification-induced-behavior figures into the paper.

Two experiments (both "does the model change behaviour when it detects an
injected concept"):

  - math      : emotion-gated arithmetic (multiply the answer when the injected
                concept is detected). Data under
                ``evals/regen/generation_<model>/math/{type2_k_sweep,type1_cmax_sweep}``.
  - successor : emotion-gated constitutional amendment successor (answer N+1
                instead of N when detected). Data under
                ``evals/regen/successor_k_sweep/<model>`` (type2) and
                ``evals/regen/successor_cmax_sweep/<model>`` (type1).

For each experiment we render, in the same Plotly paper style as the magnitude
figures (mean line + shaded 95% CI band over the pooled per-sample
``p_correct``), two overlays:

  type2  -> Mean P(correct) vs number of in-context examples k   (main paper)
  type1  -> Mean P(correct) vs injection strength (c_max frac)   (appendix)

These tasks are not fixed-class, so (unlike the magnitude/layer plots) there is
no 0.333 chance line. Writes to the existing paper ``plots/`` tree.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# Reuse the exact paper styling + writer used by the magnitude figures.
from icl.plotting.plot_magnitude import (
    MODELS,
    MODEL_COLORS,
    MODEL_DISPLAY,
    _apply_standard_layout,
    _write,
)

_REPO = Path(__file__).resolve().parents[2]
Z = 1.96


def _ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, max(0.0, mean - Z * se), min(1.0, mean + Z * se)


def _row_pcorrects(path: Path) -> list[float]:
    data = json.loads(path.read_text())
    return [float(r["p_correct"]) for r in data.get("rows", []) if r.get("p_correct") is not None]


def _row(x: float, vals: list[float]) -> dict | None:
    if not vals:
        return None
    mean, lo, hi = _ci(vals)
    return {"x": float(x), "mp": mean, "mp_lo": lo, "mp_hi": hi, "n": len(vals)}


# ----------------------------- data loaders ------------------------------- #

def math_type2(repo: Path, model: str) -> list[dict]:
    rows = []
    for fp in glob.glob(str(repo / f"evals/regen/generation_{model}/math/type2_k_sweep/math_{model}_k*.json")):
        if "cmax" in os.path.basename(fp):
            continue
        d = json.loads(Path(fp).read_text())
        r = _row(d["n_demos"], [float(x["p_correct"]) for x in d.get("rows", []) if x.get("p_correct") is not None])
        if r:
            rows.append(r)
    rows.sort(key=lambda r: r["x"])
    return rows


def math_type1(repo: Path, model: str) -> list[dict]:
    rows = []
    for fp in glob.glob(str(repo / f"evals/regen/generation_{model}/math/type1_cmax_sweep/*cmax*.json")):
        d = json.loads(Path(fp).read_text())
        r = _row(d["cmax_fraction"], [float(x["p_correct"]) for x in d.get("rows", []) if x.get("p_correct") is not None])
        if r:
            rows.append(r)
    rows.sort(key=lambda r: r["x"])
    return rows


def successor_type2(repo: Path, model: str) -> list[dict]:
    by_k: dict[int, list[float]] = {}
    for fp in glob.glob(str(repo / f"evals/regen/successor_k_sweep/{model}/k*_var*.json")):
        d = json.loads(Path(fp).read_text())
        k = int(d["n_demos"])
        by_k.setdefault(k, []).extend(
            float(x["p_correct"]) for x in d.get("rows", []) if x.get("p_correct") is not None
        )
    rows = [_row(k, vals) for k, vals in sorted(by_k.items())]
    return [r for r in rows if r]


def successor_type1(repo: Path, model: str) -> list[dict]:
    rows = []
    for fp in glob.glob(str(repo / f"evals/regen/successor_cmax_sweep/{model}/fraction_*.json")):
        d = json.loads(Path(fp).read_text())
        r = _row(d["cmax_fraction"], [float(x["p_correct"]) for x in d.get("rows", []) if x.get("p_correct") is not None])
        if r:
            rows.append(r)
    rows.sort(key=lambda r: r["x"])
    return rows


# ---- prompt-sensitivity loaders (band = ±1σ over the 10 prompt-level means) ---- #
# Mirrors the magnitude/layer appendix prompt-sigma figures: instead of a 95% CI
# over all pooled per-sample p_correct, we compute one mean per prompt variant and
# show ±1σ across those means, making prompt sensitivity directly visible.

def _sigma_row(x: float, prompt_means: list[float]) -> dict | None:
    arr = np.asarray(prompt_means, dtype=float)
    if arr.size == 0:
        return None
    mean = float(arr.mean())
    sigma = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"x": float(x), "mp": mean, "mp_lo": max(0.0, mean - sigma),
            "mp_hi": min(1.0, mean + sigma), "n": arr.size}


def math_type2_prompt_sigma(repo: Path, model: str) -> list[dict]:
    rows = []
    for fp in glob.glob(str(repo / f"evals/regen/generation_{model}/math/type2_k_sweep/math_{model}_k*.json")):
        if "cmax" in os.path.basename(fp):
            continue
        d = json.loads(Path(fp).read_text())
        by_prompt: dict = {}
        for x in d.get("rows", []):
            if x.get("p_correct") is None:
                continue
            by_prompt.setdefault(x.get("prompt_variation_id"), []).append(float(x["p_correct"]))
        prompt_means = [sum(v) / len(v) for v in by_prompt.values() if v]
        r = _sigma_row(d["n_demos"], prompt_means)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: r["x"])
    return rows


def successor_type2_prompt_sigma(repo: Path, model: str) -> list[dict]:
    # Each k*_var*.json is one prompt variant → one prompt-level mean per file.
    by_k: dict[int, list[float]] = {}
    for fp in glob.glob(str(repo / f"evals/regen/successor_k_sweep/{model}/k*_var*.json")):
        d = json.loads(Path(fp).read_text())
        vals = [float(x["p_correct"]) for x in d.get("rows", []) if x.get("p_correct") is not None]
        if vals:
            by_k.setdefault(int(d["n_demos"]), []).append(sum(vals) / len(vals))
    rows = [_sigma_row(k, means) for k, means in sorted(by_k.items())]
    return [r for r in rows if r]


# ------------------------------- rendering -------------------------------- #

def _overlay(by_model: dict[str, list[dict]], html_path: Path, *, title: str, xlabel: str,
             baseline: float = 0.5) -> None:
    """Mean line + shaded 95% CI band per model, paper style.

    These are token-generation (judge-scored) tasks, so the y-axis is Accuracy
    and a dotted-gray baseline line is drawn at *baseline* (0.5 here). The y-axis
    always starts at 0.
    """
    fig = go.Figure()
    for model in MODELS:
        rows = by_model.get(model) or []
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
            name=f"{MODEL_DISPLAY[model]} (CI)",
        ))
        fig.add_trace(go.Scatter(
            x=xs,
            y=means,
            mode="lines",
            line=dict(color=color, width=2),
            name=MODEL_DISPLAY[model],
        ))
    _apply_standard_layout(fig, title=title, xlabel=xlabel, ylabel="Accuracy")
    fig.add_hline(y=baseline, line=dict(color="gray", dash="dot", width=1.5),
                  annotation_text=f"Baseline ({baseline:.2f})", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 1.05])
    _write(fig, html_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(_REPO))
    ap.add_argument("--plots-dir", default="/workspace/Introspection ICL - ARXIV/plots")
    args = ap.parse_args()

    repo = Path(args.repo)
    plots = Path(args.plots_dir)

    K_LABEL = "In-Context Examples (<i>k</i>)"
    STRENGTH_LABEL = "<i>α</i>"

    specs = [
        ("math", "type2", math_type2, "Emotion-Gated Arithmetic", K_LABEL,
         plots / "type2" / "type2_math_introspection.html"),
        ("math", "type1", math_type1, "Emotion-Gated Arithmetic (Strength Sweep)", STRENGTH_LABEL,
         plots / "type1" / "type1_math_introspection.html"),
        ("successor", "type2", successor_type2, "Amendment Successor", K_LABEL,
         plots / "type2" / "type2_amendment_successor.html"),
        ("successor", "type1", successor_type1, "Amendment Successor (Strength Sweep)", STRENGTH_LABEL,
         plots / "type1" / "type1_amendment_successor.html"),
        ("math", "prompt_sigma", math_type2_prompt_sigma, "Emotion-Gated Arithmetic (Prompt Sensitivity)", K_LABEL,
         plots / "type2" / "type2_math_introspection_prompt_sigma.html"),
        ("successor", "prompt_sigma", successor_type2_prompt_sigma, "Amendment Successor (Prompt Sensitivity)", K_LABEL,
         plots / "type2" / "type2_amendment_successor_prompt_sigma.html"),
    ]

    for exp, kind, loader, title, xlabel, html_path in specs:
        by_model = {}
        for model in MODELS:
            rows = loader(repo, model)
            if rows:
                by_model[model] = rows
                ns = {r["n"] for r in rows}
                print(f"  {exp} {kind} {model}: {len(rows)} points, n/point={sorted(ns)}")
        if not by_model:
            print(f"  {exp} {kind}: no data, skipping")
            continue
        _overlay(by_model, html_path, title=title, xlabel=xlabel)


if __name__ == "__main__":
    main()
