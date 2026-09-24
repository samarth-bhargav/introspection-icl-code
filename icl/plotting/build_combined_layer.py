"""Build the side-by-side Layer Detection Generalization plot for Fig. 5.

Reads the existing per-model Plotly HTMLs for Qwen3-32B and Gemma-31B,
glues them into a 1x2 ``make_subplots`` figure under one unified title,
and applies the same ``STANDARD_LAYOUT`` used by ``apply_paper_styling``.

Run::

    python -m icl.plotting.build_combined_layer
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


_THIS = Path(__file__).resolve()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_h2p = _load_module("_h2p", _THIS.parent / "html_to_png.py")
_styling = _load_module("_styling", _THIS.parent / "apply_paper_styling.py")


_LEGEND_RENAME = {
    "early":  "early",
    "middle": "middle",
    "late":   "late",
}


def _simplify_legend_name(name: str) -> str:
    if not name:
        return name
    head = name.split(" (")[0].strip()
    return _LEGEND_RENAME.get(head, name)


def _add_traces(fig: go.Figure, src: go.Figure, *, col: int,
                show_legend: bool) -> None:
    for tr in src.data:
        new = copy.deepcopy(tr)
        if show_legend:
            new.name = _simplify_legend_name(new.name)
            new.showlegend = (new.fill != "toself")
        else:
            new.showlegend = False
        fig.add_trace(new, row=1, col=col)


def _add_shapes(fig: go.Figure, src: go.Figure, *, col: int) -> None:
    suffix = "" if col == 1 else str(col)
    for sh in src.layout.shapes:
        new = copy.deepcopy(sh)
        if new.xref == "x":
            new.xref = f"x{suffix}" if suffix else "x"
        elif new.xref == "x domain":
            new.xref = f"x{suffix} domain" if suffix else "x domain"
        fig.add_shape(new)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plots-dir", default="plots",
                    help="Directory holding the per-model layer_generalization_*.html.")
    args = ap.parse_args()
    plots_dir = Path(args.plots_dir)
    SRC_LEFT  = plots_dir / "layer_generalization_gemma31b.html"
    SRC_RIGHT = plots_dir / "layer_generalization_qwen332b.html"
    OUT_HTML  = plots_dir / "layer_generalization_combined.html"
    OUT_PNG   = OUT_HTML.with_suffix(".png")

    left  = _h2p._figure_from_html(SRC_LEFT.read_text())
    right = _h2p._figure_from_html(SRC_RIGHT.read_text())

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("Gemma-31B", "Qwen3-32B"),
    )

    _add_traces(fig, left,  col=1, show_legend=True)
    _add_traces(fig, right, col=2, show_legend=False)

    _add_shapes(fig, left,  col=1)
    _add_shapes(fig, right, col=2)

    fig.update_layout(**_styling.STANDARD_LAYOUT)
    fig.update_layout(
        title_text="Layer Detection Generalization",
        width=1300, height=520,
        margin=dict(l=70, r=30, t=80, b=110),
        legend=dict(orientation="h", x=0.5, y=-0.22,
                    xanchor="center", yanchor="top",
                    font=dict(size=13)),
    )
    fig.update_xaxes(title_text="Test layer", row=1, col=1,
                     title_font=dict(size=16), tickfont=dict(size=13))
    fig.update_xaxes(title_text="Test layer", row=1, col=2,
                     title_font=dict(size=16), tickfont=dict(size=13))
    fig.update_yaxes(title_text="Mean P(correct)", range=[0, 1.05], row=1, col=1,
                     title_font=dict(size=16), tickfont=dict(size=13))
    fig.update_yaxes(range=[0, 1.05], row=1, col=2, tickfont=dict(size=13))

    for ann in fig.layout.annotations:
        ann.font = dict(size=16, color="#2a3f5f",
                        family="DejaVu Sans, Arial, sans-serif")

    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")
    fig.write_image(str(OUT_PNG), scale=2)
    print(f"wrote {OUT_HTML}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
