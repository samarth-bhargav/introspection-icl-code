"""Apply uniform paper-figure styling to every Plotly HTML under ``plots/``.

Walks the paper's ``plots/`` directory, parses each Plotly HTML companion
(written via ``fig.write_html``), overlays a uniform ``STANDARD_LAYOUT``
(font sizes, legend, margins, dimensions) onto the existing layout, applies
per-file title/axis-label overrides, then re-saves both the HTML and the
PNG twin via Kaleido.

Usage::

    python -m icl.plotting.apply_paper_styling \
        --plots-dir "/workspace/Introspection ICL/plots"
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import plotly.graph_objects as go


_THIS = Path(__file__).resolve()
_HTML_TO_PNG_PATH = _THIS.parent / "html_to_png.py"
_spec = importlib.util.spec_from_file_location("_h2p", _HTML_TO_PNG_PATH)
_h2p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h2p)
_figure_from_html = _h2p._figure_from_html


STANDARD_LAYOUT: dict = {
    "font":     {"family": "DejaVu Sans, Arial, sans-serif",
                 "size": 14, "color": "#2a3f5f"},
    "title":    {"font": {"size": 18}, "x": 0.5, "xanchor": "center"},
    "xaxis":    {"title": {"font": {"size": 16}, "standoff": 12},
                 "tickfont": {"size": 13}},
    "yaxis":    {"title": {"font": {"size": 16}, "standoff": 8},
                 "tickfont": {"size": 13}},
    "legend":   {"font": {"size": 13}, "x": 1.02, "y": 1.0},
    "margin":   {"l": 70, "r": 30, "t": 70, "b": 70},
    "width": 1100, "height": 500,
    "template": "plotly_white",
}


# Per-HTML overrides keyed by path-relative-to-plots-dir (POSIX form).
# Only "title", "xaxis_title", and "yaxis_title" are recognized.
RENAME_MAP: dict[str, dict[str, str]] = {
    # ---------- Main paper figures ----------
    "type2/type2_magnitude_introspection_merged.html": {
        "title": "Injection Magnitude Classification",
        "xaxis_title": "In-Context Examples (<i>k</i>)",
    },
    "type2/type2_layer_introspection_merged.html": {
        "title": "Injection Layer Classification",
        "xaxis_title": "In-Context Examples (<i>k</i>)",
    },
    "magnitude_generalization_gemma31b.html": {
        "title": "Gemma-31B Magnitude Detection Generalization",
        "xaxis_title": "Test α",
    },
    "layer_generalization_gemma31b.html": {
        "title": "Gemma-31B Layer Detection Generalization",
        "xaxis_title": "Test layer",
    },
    "one_hop/plot1_unimodal_pooled.html": {
        "title": "Reasoning At Zero-Shot",
        "xaxis_title": "α (log scale)",
    },
    # Appendix lowstrength panels — distinct subtitles per panel.
    "one_hop/lowstrength_exp1_constant_ts03.html": {
        "title": "Reasoning With In-Context Examples — Constant Low",
        "xaxis_title": "In-Context Examples (<i>k</i>)",
    },

    # ---------- Appendix figures ----------
    "magnitude_generalization_qwen332b.html": {
        "title": "Qwen3-32B Magnitude Detection Generalization",
        "xaxis_title": "Test α",
    },
    "magnitude_generalization_qwen38b.html": {
        "title": "Qwen3-8B Magnitude Detection Generalization",
        "xaxis_title": "Test α",
    },
    "magnitude_generalization_olmo32b.html": {
        "title": "OLMo-32B Magnitude Detection Generalization",
        "xaxis_title": "Test α",
    },
    "magnitude_generalization_olmo7b.html": {
        "title": "OLMo-7B Magnitude Detection Generalization",
        "xaxis_title": "Test α",
    },
    "layer_generalization_qwen332b.html": {
        "title": "Qwen3-32B Layer Detection Generalization",
        "xaxis_title": "Test layer",
    },
    "layer_generalization_qwen38b.html": {
        "title": "Qwen3-8B Layer Detection Generalization",
        "xaxis_title": "Test layer",
    },
    "layer_generalization_olmo32b.html": {
        "title": "OLMo-32B Layer Detection Generalization",
        "xaxis_title": "Test layer",
    },
    "layer_generalization_olmo7b.html": {
        "title": "OLMo-7B Layer Detection Generalization",
        "xaxis_title": "Test layer",
    },
    "type1/type1_magnitude_introspection.html": {
        "title": "Injection Magnitude Classification (Strength Sweep)",
        "xaxis_title": "α<sub>m</sub>",
    },
    "type1/type1_layer_introspection.html": {
        "title": "Injection Layer Classification (Strength Sweep)",
        "xaxis_title": "α<sub>ℓ</sub>",
    },
    "type2/type2_magnitude_introspection_prompt_sigma.html": {
        "title": "Injection Magnitude Classification (Prompt Sensitivity)",
        "xaxis_title": "In-Context Examples (<i>k</i>)",
    },
    "type2/type2_layer_introspection_prompt_sigma.html": {
        "title": "Injection Layer Classification (Prompt Sensitivity)",
        "xaxis_title": "In-Context Examples (<i>k</i>)",
    },
    "one_hop/lowstrength_exp2_curriculum_ts03.html": {
        "title": "Reasoning With In-Context Examples — Curriculum",
        "xaxis_title": "In-Context Examples (<i>k</i>)",
    },
    "one_hop/lowstrength_exp3_scheduler_ts03.html": {
        "title": "Reasoning With In-Context Examples — Scheduler",
        "xaxis_title": "In-Context Examples (<i>k</i>)",
    },
    "one_hop/lowstrength_exp4_cotdot_ts03.html": {
        "title": "Reasoning With In-Context Examples — Dot-by-Dot CoT",
        # x-label is "Number of Filler Dots in Prefill" in the source — keep
        # it via a rewrite to make sure the standardised label is uniform.
        "xaxis_title": "Number of Filler Dots in Prefill",
    },

    # ---------- Appendix G (prompt-sensitivity for reasoning) ----------
    "prompt_variations/prompt_variations_gemma-31b_one_hop.html": {
        "title": "Gemma-31B — Reasoning",
        "xaxis_title": "α",
    },
    "prompt_variations/prompt_variations_qwen3-32b_one_hop.html": {
        "title": "Qwen3-32B — Reasoning",
        "xaxis_title": "α",
    },
    "prompt_variations/prompt_variations_qwen3-8b_one_hop.html": {
        "title": "Qwen3-8B — Reasoning",
        "xaxis_title": "α",
    },
    "prompt_variations/prompt_variations_olmo-32b_one_hop.html": {
        "title": "OLMo-32B — Reasoning",
        "xaxis_title": "α",
    },
    "prompt_variations/prompt_variations_olmo-7b_one_hop.html": {
        "title": "OLMo-7B — Reasoning",
        "xaxis_title": "α",
    },
}


# Files that should be cloned into a second output before being styled — the
# clone uses a distinct title (typically the title-stripped main-paper
# version). Format: source HTML (relative to plots-dir) -> list of clones.
CLONE_MAP: dict[str, list[dict]] = {
    "one_hop/lowstrength_exp1_constant_ts03.html": [
        {
            "out_html": "one_hop/lowstrength_exp1_constant_ts03_main.html",
            "title": "Reasoning With In-Context Examples",
            "xaxis_title": "In-Context Examples (<i>k</i>)",
        },
    ],
}


def _apply_layout(fig: go.Figure, overrides: dict | None) -> None:
    """Overlay STANDARD_LAYOUT, then apply per-file text overrides."""
    fig.update_layout(**STANDARD_LAYOUT)
    if not overrides:
        return
    if "title" in overrides:
        fig.update_layout(title_text=overrides["title"])
    if "xaxis_title" in overrides:
        fig.update_layout(xaxis_title_text=overrides["xaxis_title"])
    if "yaxis_title" in overrides:
        fig.update_layout(yaxis_title_text=overrides["yaxis_title"])


def _write_outputs(fig: go.Figure, html_path: Path) -> None:
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    png_path = html_path.with_suffix(".png")
    fig.write_image(str(png_path), scale=2)


def process_one(html_path: Path, plots_dir: Path) -> None:
    rel = html_path.relative_to(plots_dir).as_posix()
    text = html_path.read_text()
    base_fig = _figure_from_html(text)

    # Clones first — derive each clone from a fresh copy of the base figure.
    for clone in CLONE_MAP.get(rel, []):
        clone_fig = _figure_from_html(text)
        _apply_layout(clone_fig, clone)
        out = plots_dir / clone["out_html"]
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_outputs(clone_fig, out)
        print(f"  CLONE  {rel}  ->  {clone['out_html']}")

    # In-place styling.
    _apply_layout(base_fig, RENAME_MAP.get(rel))
    _write_outputs(base_fig, html_path)
    tag = "RENAME" if rel in RENAME_MAP else "STYLE "
    print(f"  {tag} {rel}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--plots-dir",
        default="/workspace/Introspection ICL/plots",
        help="Directory containing the paper's plot HTML/PNG files.",
    )
    ap.add_argument(
        "--only", nargs="*",
        help="Optional subset of HTML paths (relative to plots-dir) to process.",
    )
    args = ap.parse_args()

    plots_dir = Path(args.plots_dir)
    if not plots_dir.is_dir():
        print(f"plots-dir does not exist: {plots_dir}", file=sys.stderr)
        sys.exit(2)

    htmls = sorted(plots_dir.rglob("*.html"))
    if args.only:
        wanted = {plots_dir / x for x in args.only}
        htmls = [h for h in htmls if h in wanted]

    print(f"processing {len(htmls)} HTML files under {plots_dir}")
    failures: list[tuple[Path, str]] = []
    for h in htmls:
        try:
            process_one(h, plots_dir)
        except Exception as e:
            failures.append((h, f"{type(e).__name__}: {e}"))
            print(f"  FAIL   {h.relative_to(plots_dir)}: {e}", file=sys.stderr)

    print()
    print(f"done: {len(htmls) - len(failures)} succeeded, {len(failures)} failed")
    if failures:
        for h, msg in failures:
            print(f"  - {h.relative_to(plots_dir)}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
