"""Port constitution-source magnitude plots into the paper plot directory.

This reads the regenerated 20%-depth magnitude introspection results from
``evals/regen/constitution_source_magnitude`` and writes Plotly HTML/PNG pairs
using the same paper styling as the existing arXiv replica.

It deliberately writes to the existing paper filenames, so TeX captions and
``\\includegraphics`` references do not need to change.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_intro = _load_module("_intro_common", _THIS.parent / "_intro_common.py")
_styling = _load_module("_styling", _THIS.parent / "apply_paper_styling.py")
_h2p = _load_module("_h2p", _THIS.parent / "html_to_png.py")

CHANCE_3 = _intro.CHANCE_3
MODEL_COLORS = _intro.MODEL_COLORS
MODEL_DISPLAY = _intro.MODEL_DISPLAY
MODELS = _intro.MODELS


LABELS = ["low", "medium", "high"]
LABEL_COLORS = {"low": "#2563eb", "medium": "#16a34a", "high": "#dc2626"}
MODEL_FILE_STEMS = {
    "gemma-31b": "gemma31b",
    "qwen3-32b": "qwen332b",
    "qwen3-8b": "qwen38b",
    "olmo-32b": "olmo32b",
    "olmo-7b": "olmo7b",
}


def _ci(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, max(0.0, mean - z * se), min(1.0, mean + z * se)


def _type_path(source_root: Path, kind: str, model: str) -> Path:
    return source_root / "magnitude" / f"{kind}_{model}.json"


def _maggen_path(source_root: Path, model: str) -> Path:
    return source_root / "magnitude_generalization" / f"magnitude_generalization_{model}.json"


def _series_for_type(source_root: Path, kind: str) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = {}
    for model in MODELS:
        path = _type_path(source_root, kind, model)
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows: list[dict] = []
        if kind == "type1":
            available_k = [
                rec["k"]
                for entry in data["per_strength"]
                for rec in entry["by_k"]
            ]
            read_k = min(30, max(available_k))
            for entry in data["per_strength"]:
                rec = next((r for r in entry["by_k"] if r["k"] == read_k), None)
                if rec is None:
                    continue
                mean, lo, hi = _ci([float(v) for v in rec["p_correct"]])
                rows.append({"x": float(entry["strength"]), "mp": mean, "mp_lo": lo, "mp_hi": hi})
        elif kind == "type2":
            for rec in data["per_strength"][0]["by_k"]:
                mean, lo, hi = _ci([float(v) for v in rec["p_correct"]])
                rows.append({"x": int(rec["k"]), "mp": mean, "mp_lo": lo, "mp_hi": hi})
        else:
            raise ValueError(kind)
        if rows:
            rows.sort(key=lambda row: row["x"])
            by_model[model] = rows
    return by_model


def _add_chance_line(fig: go.Figure, *, annotation: bool = True) -> None:
    kwargs = {}
    if annotation:
        kwargs = {
            "annotation_text": f"Chance ({CHANCE_3:.3f})",
            "annotation_position": "bottom right",
        }
    fig.add_hline(y=CHANCE_3, line=dict(color="gray", dash="dot", width=1.5), **kwargs)


def _apply_standard_layout(fig: go.Figure, *, title: str, xlabel: str, ylabel: str = "Mean P(correct)") -> None:
    fig.update_layout(**_styling.STANDARD_LAYOUT)
    fig.update_layout(title_text=title, xaxis_title_text=xlabel, yaxis_title_text=ylabel)


def _write(fig: go.Figure, html_path: Path, *, scale: int = 2) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {html_path}", flush=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"writing {html_path.with_suffix('.png')}", flush=True)
    fig.write_image(str(html_path.with_suffix(".png")), scale=scale)
    print(f"wrote {html_path}")
    print(f"wrote {html_path.with_suffix('.png')}")


def _render_model_overlay(by_model: dict[str, list[dict]], html_path: Path, *, title: str, xlabel: str) -> None:
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
            name=f"{MODEL_DISPLAY[model]} (CI)",
        ))
        fig.add_trace(go.Scatter(
            x=xs,
            y=means,
            mode="lines",
            line=dict(color=color, width=2),
            name=MODEL_DISPLAY[model],
        ))
    _add_chance_line(fig)
    _apply_standard_layout(fig, title=title, xlabel=xlabel)
    _write(fig, html_path)


def _class_series(data: dict) -> tuple[list[float], dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    xs = [float(r["test_alpha"]) for r in data["records"]]
    means = {lab: [] for lab in LABELS}
    lows = {lab: [] for lab in LABELS}
    highs = {lab: [] for lab in LABELS}
    for record in data["records"]:
        for lab in LABELS:
            vals = [float(sample["probabilities"].get(lab, 0.0)) for sample in record["samples"]]
            mean, lo, hi = _ci(vals)
            means[lab].append(mean)
            lows[lab].append(lo)
            highs[lab].append(hi)
    return xs, means, lows, highs


def _anchor_levels(data: dict) -> list[tuple[str, float]]:
    return [(str(label), float(alpha)) for label, alpha in data["anchor_levels"]]


def _render_maggen_one(source_root: Path, plots_dir: Path, model: str) -> None:
    data = json.loads(_maggen_path(source_root, model).read_text())
    xs, means, lows, highs = _class_series(data)
    anchors = _anchor_levels(data)
    fig = go.Figure()
    anchor_by_label = dict(anchors)
    for lab in LABELS:
        color = LABEL_COLORS[lab]
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1],
            y=highs[lab] + lows[lab][::-1],
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            opacity=0.18,
            hoverinfo="skip",
            showlegend=False,
            name=f"{lab} (CI)",
        ))
        fig.add_trace(go.Scatter(
            x=xs,
            y=means[lab],
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{lab} (\u03b1 = {anchor_by_label[lab]:g})",
        ))
    for lab, alpha in anchors:
        fig.add_vline(x=alpha, line=dict(color=LABEL_COLORS[lab], dash="dash", width=1.2), opacity=0.7)
    _add_chance_line(fig)
    _apply_standard_layout(
        fig,
        title=f"{MODEL_DISPLAY[model]} Magnitude Detection Generalization",
        xlabel="Test \u03b1",
        ylabel="Mean P(label)",
    )
    _write(fig, plots_dir / f"magnitude_generalization_{MODEL_FILE_STEMS[model]}.html")


def _simplify_legend_name(name: str | None) -> str | None:
    if not name:
        return name
    return name.split(" (")[0].strip()


def _add_source_traces(fig: go.Figure, src: go.Figure, *, col: int, show_legend: bool) -> None:
    for trace in src.data:
        new = copy.deepcopy(trace)
        if show_legend:
            new.name = _simplify_legend_name(new.name)
            new.showlegend = (new.fill != "toself")
        else:
            new.showlegend = False
        fig.add_trace(new, row=1, col=col)


def _add_source_shapes(fig: go.Figure, src: go.Figure, *, col: int) -> None:
    suffix = "" if col == 1 else str(col)
    for shape in src.layout.shapes:
        new = copy.deepcopy(shape)
        if new.xref == "x":
            new.xref = f"x{suffix}" if suffix else "x"
        elif new.xref == "x domain":
            new.xref = f"x{suffix} domain" if suffix else "x domain"
        fig.add_shape(new)


def _render_maggen_combined(plots_dir: Path) -> None:
    left_path = plots_dir / "magnitude_generalization_gemma31b.html"
    right_path = plots_dir / "magnitude_generalization_qwen332b.html"
    left = _h2p._figure_from_html(left_path.read_text())
    right = _h2p._figure_from_html(right_path.read_text())
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("Gemma-31B", "Qwen3-32B"),
    )
    _add_source_traces(fig, left, col=1, show_legend=True)
    _add_source_traces(fig, right, col=2, show_legend=False)
    _add_source_shapes(fig, left, col=1)
    _add_source_shapes(fig, right, col=2)
    fig.update_layout(**_styling.STANDARD_LAYOUT)
    fig.update_layout(
        title_text="Magnitude Detection Generalization",
        width=1300,
        height=520,
        margin=dict(l=70, r=30, t=80, b=110),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.22,
            xanchor="center",
            yanchor="top",
            font=dict(size=13),
        ),
    )
    fig.update_xaxes(title_text="Test \u03b1", row=1, col=1, title_font=dict(size=16), tickfont=dict(size=13))
    fig.update_xaxes(title_text="Test \u03b1", row=1, col=2, title_font=dict(size=16), tickfont=dict(size=13))
    fig.update_yaxes(title_text="Mean P(label)", range=[0, 1.05], row=1, col=1, title_font=dict(size=16), tickfont=dict(size=13))
    fig.update_yaxes(range=[0, 1.05], row=1, col=2, tickfont=dict(size=13))
    for ann in fig.layout.annotations:
        ann.font = dict(size=16, color="#2a3f5f", family="DejaVu Sans, Arial, sans-serif")
    _write(fig, plots_dir / "magnitude_generalization_combined.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-root",
        default=str(_REPO / "evals" / "regen" / "constitution_source_magnitude"),
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    plots_dir = Path(args.plots_dir)
    print(f"source: {source_root}", flush=True)
    print(f"plots: {plots_dir}", flush=True)
    if not source_root.is_dir():
        raise SystemExit(f"source root does not exist: {source_root}")
    if not plots_dir.is_dir():
        raise SystemExit(f"plots dir does not exist: {plots_dir}")

    _render_model_overlay(
        _series_for_type(source_root, "type1"),
        plots_dir / "type1" / "type1_magnitude_introspection.html",
        title="Injection Magnitude Classification (Strength Sweep)",
        xlabel="\u03b1<sub>m</sub>",
    )
    _render_model_overlay(
        _series_for_type(source_root, "type2"),
        plots_dir / "type2" / "type2_magnitude_introspection_merged.html",
        title="Injection Magnitude Classification",
        xlabel="In-Context Examples (<i>k</i>)",
    )
    # Keep the unmerged PNG asset in sync for local inspection, even though the
    # TeX currently uses the merged filename.
    (plots_dir / "type2" / "type2_magnitude_introspection.png").write_bytes(
        (plots_dir / "type2" / "type2_magnitude_introspection_merged.png").read_bytes()
    )

    for model in MODELS:
        if _maggen_path(source_root, model).exists():
            _render_maggen_one(source_root, plots_dir, model)
    _render_maggen_combined(plots_dir)


if __name__ == "__main__":
    main()
