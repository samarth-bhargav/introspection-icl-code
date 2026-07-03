"""Render the math + amendment-successor type2 figures from the regenerated
6-emotion run (evals/full_6emo) BUT conditioned on a SINGLE emotion (anger):
filter to runs whose gating/target emotion is anger, then pool overall accuracy
exactly as the main 6-emotion figures do. Same paper plot style.

This replaces the old pre-6emo "anger" figures, which used the with-reference
successor task and are no longer valid.

Outputs into <repo>/plots/anger/type2/:
  type2_math_introspection.{html,png}
  type2_amendment_successor.{html,png}

Usage:
  PYTHONPATH=/workspace/Introspection-ICL-Final \
    /workspace/Introspection-RL/.venv/bin/python gated6/plot_anger_figs.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from icl.plotting.plot_strength_sweep import MODELS, _overlay, _row

REPO = Path("/workspace/Introspection-ICL-Final")
BASE = REPO / "evals" / "full_6emo"
OUT = REPO / "plots" / "anger"
EMOTION = "anger"
K_LABEL = "In-Context Examples (<i>k</i>)"


def math_anger(model: str) -> list[dict]:
    by_k: dict[int, list[float]] = {}
    for fp in glob.glob(str(BASE / f"generation_{model}/math/type2_k_sweep/math_{model}_k*.json")):
        if "cmax" in fp:
            continue
        d = json.loads(Path(fp).read_text())
        k = int(d["n_demos"])
        for x in d.get("rows", []):
            if x.get("target_emotion") == EMOTION and x.get("p_correct") is not None:
                by_k.setdefault(k, []).append(float(x["p_correct"]))
    rows = [_row(k, vals) for k, vals in sorted(by_k.items())]
    return [r for r in rows if r]


def successor_anger(model: str) -> list[dict]:
    by_k: dict[int, list[float]] = {}
    for fp in glob.glob(str(BASE / f"successor_emotions_k_sweep/{model}/k*_var*.json")):
        d = json.loads(Path(fp).read_text())
        k = int(d["n_demos"])
        for x in d.get("rows", []):
            if x.get("gating_emotion") == EMOTION and x.get("p_correct") is not None:
                by_k.setdefault(k, []).append(float(x["p_correct"]))
    rows = [_row(k, vals) for k, vals in sorted(by_k.items())]
    return [r for r in rows if r]


SPECS = [
    (math_anger, "Emotion-Gated Arithmetic (Anger)", OUT / "type2" / "type2_math_introspection.html"),
    (successor_anger, "Amendment Successor (Anger)", OUT / "type2" / "type2_amendment_successor.html"),
]


def main() -> None:
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
