"""Re-render the PNG twin of a plotly HTML plot from the HTML alone.

Use this when you have edited a title/label in an HTML file and want the
matching PNG (used by the LaTeX paper) to pick up the change without
re-running the upstream plot script.

Usage:

    # In-place: writes <name>.png next to <name>.html.
    python -m icl.plotting.html_to_png path/to/plot.html [more.html ...]

    # Apply a title text edit to the HTML first, then re-render the PNG:
    python -m icl.plotting.html_to_png plot.html \\
        --replace "old title text" "new title text"

The HTML must have been produced by ``fig.write_html(..., include_plotlyjs=...)``
(i.e. plotly's standard HTML wrapper) — this is how every helper in
``icl/plotting/`` writes its HTML output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import plotly.graph_objects as go


_PLOTLY_NEWPLOT = re.compile(
    r'Plotly\.newPlot\(\s*"',
)


def _figure_from_html(html: str) -> go.Figure:
    m = _PLOTLY_NEWPLOT.search(html)
    if not m:
        raise ValueError("no Plotly.newPlot(...) call found in HTML")
    # Decode complete JSON arguments: nested subplot layouts can contain
    # object boundaries that a non-greedy regular expression truncates.
    remaining = html[m.end() - 1:]
    decoder = json.JSONDecoder()
    arguments = []
    for index in range(3):  # element ID, trace data, layout
        value, end = decoder.raw_decode(remaining.lstrip())
        arguments.append(value)
        remaining = remaining.lstrip()[end:].lstrip()
        if index < 2:
            if not remaining.startswith(","):
                raise ValueError("malformed Plotly.newPlot arguments")
            remaining = remaining[1:]
    return go.Figure(data=arguments[1], layout=arguments[2])


def render(html_path: Path, *, replacements: list[tuple[str, str]] | None,
           dry_run: bool) -> Path:
    text = html_path.read_text()
    if replacements:
        new = text
        for old, repl in replacements:
            if old not in new:
                print(f"  WARN: '{old}' not found in {html_path.name}",
                      file=sys.stderr)
            new = new.replace(old, repl)
        if new != text and not dry_run:
            html_path.write_text(new)
            print(f"  edited {html_path}")
        text = new

    fig = _figure_from_html(text)
    png_path = html_path.with_suffix(".png")
    if dry_run:
        print(f"  (dry-run) would write {png_path}")
    else:
        fig.write_image(str(png_path), scale=2)
        print(f"  wrote {png_path}")
    return png_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("html", nargs="+", help="HTML files to re-render.")
    ap.add_argument(
        "--replace", nargs=2, action="append", metavar=("OLD", "NEW"),
        help="Apply a literal string replacement to the HTML before "
             "extracting the figure. May be repeated.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Show planned writes but do not modify any file.")
    args = ap.parse_args()

    for raw in args.html:
        path = Path(raw)
        if not path.exists():
            print(f"  SKIP: {path} does not exist", file=sys.stderr)
            continue
        render(path, replacements=args.replace or [], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
