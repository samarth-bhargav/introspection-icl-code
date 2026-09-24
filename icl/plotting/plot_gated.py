"""Render the math + amendment-successor type2 figures from the regenerated
6-emotion control-free gated run (evals/full_6emo), reusing the exact paper
plot style from icl.plotting.plot_strength_sweep.

Outputs (HTML + PNG) into <repo>/plots/full_6emo/type2/:
  type2_math_introspection.{html,png}
  type2_math_introspection_prompt_sigma.{html,png}
  type2_amendment_successor.{html,png}
  type2_amendment_successor_prompt_sigma.{html,png}

Only type2 (k-sweep) data exists in the rerun; type1 strength-sweep was not
re-run, so those figures are intentionally not regenerated here.

Usage:
  python -m icl.plotting.plot_gated
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from icl.plotting.plot_strength_sweep import (
    MODELS, _ci, _overlay, _row, _sigma_row,
)

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "evals" / "full_6emo"
OUT = REPO / "plots" / "full_6emo"
K_LABEL = "In-Context Examples (<i>k</i>)"


# ---- loaders repointed at evals/full_6emo ----
def math_type2(model: str) -> list[dict]:
    rows = []
    for fp in glob.glob(str(BASE / f"generation_{model}/math/type2_k_sweep/math_{model}_k*.json")):
        if "cmax" in os.path.basename(fp):
            continue
        d = json.loads(Path(fp).read_text())
        r = _row(d["n_demos"], [float(x["p_correct"]) for x in d.get("rows", []) if x.get("p_correct") is not None])
        if r:
            rows.append(r)
    rows.sort(key=lambda r: r["x"])
    return rows


def successor_type2(model: str) -> list[dict]:
    by_k: dict[int, list[float]] = {}
    for fp in glob.glob(str(BASE / f"successor_emotions_k_sweep/{model}/k*_var*.json")):
        d = json.loads(Path(fp).read_text())
        by_k.setdefault(int(d["n_demos"]), []).extend(
            float(x["p_correct"]) for x in d.get("rows", []) if x.get("p_correct") is not None
        )
    rows = [_row(k, vals) for k, vals in sorted(by_k.items())]
    return [r for r in rows if r]


def math_type2_prompt_sigma(model: str) -> list[dict]:
    rows = []
    for fp in glob.glob(str(BASE / f"generation_{model}/math/type2_k_sweep/math_{model}_k*.json")):
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


def successor_type2_prompt_sigma(model: str) -> list[dict]:
    # Each k*_var*.json is one prompt variant -> one prompt-level mean per file.
    by_k: dict[int, list[float]] = {}
    for fp in glob.glob(str(BASE / f"successor_emotions_k_sweep/{model}/k*_var*.json")):
        d = json.loads(Path(fp).read_text())
        vals = [float(x["p_correct"]) for x in d.get("rows", []) if x.get("p_correct") is not None]
        if vals:
            by_k.setdefault(int(d["n_demos"]), []).append(sum(vals) / len(vals))
    rows = [_sigma_row(k, means) for k, means in sorted(by_k.items())]
    return [r for r in rows if r]


SPECS = [
    (math_type2, "Emotion-Gated Arithmetic", OUT / "type2" / "type2_math_introspection.html"),
    (successor_type2, "Amendment Successor", OUT / "type2" / "type2_amendment_successor.html"),
    (math_type2_prompt_sigma, "Emotion-Gated Arithmetic (Prompt Sensitivity)",
     OUT / "type2" / "type2_math_introspection_prompt_sigma.html"),
    (successor_type2_prompt_sigma, "Amendment Successor (Prompt Sensitivity)",
     OUT / "type2" / "type2_amendment_successor_prompt_sigma.html"),
]


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    (OUT / "type2").mkdir(parents=True, exist_ok=True)
    for loader, title, html_path in SPECS:
        by_model = {}
        for model in MODELS:
            rows = loader(model)
            if rows:
                by_model[model] = rows
                ns = {r["n"] for r in rows}
                print(f"  {title} {model}: {len(rows)} points, n/point={sorted(ns)}")
        if not by_model:
            print(f"  {title}: no data, skipping")
            continue
        _overlay(by_model, html_path, title=title, xlabel=K_LABEL)


if __name__ == "__main__":
    main()
