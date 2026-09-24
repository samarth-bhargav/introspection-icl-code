"""Render layer strength, example-count, and prompt-sensitivity panels from JSON."""
import argparse
import json
from pathlib import Path
import numpy as np
from icl.plotting.plot_magnitude import _ci, _render_model_overlay, MODELS


def series(root, model):
    calibration = json.loads((root / f"type1_{model}.json").read_text())
    k = min(30, max(r["k"] for e in calibration["per_strength"] for r in e["by_k"]))
    strength = []
    for entry in calibration["per_strength"]:
        rec = next(r for r in entry["by_k"] if r["k"] == k)
        mean, lo, hi = _ci(rec["p_correct"])
        strength.append(dict(x=entry["strength"], mp=mean, mp_lo=lo, mp_hi=hi))
    paths = sorted(root.glob(f"type2_{model}_var*.json"))
    if not paths:
        raise ValueError(f"No prompt-variation layer measurements for {model}")
    prompts = [json.loads(p.read_text())["per_strength"][0]["by_k"] for p in paths]
    keys = [r["k"] for r in prompts[0]]
    if any([r["k"] for r in records] != keys for records in prompts):
        raise ValueError(f"Inconsistent k grids for {model}")
    examples, sensitivity = [], []
    for i, k in enumerate(keys):
        values = [p for records in prompts for p in records[i]["p_correct"]]
        mean, lo, hi = _ci(values)
        examples.append(dict(x=k, mp=mean, mp_lo=lo, mp_hi=hi))
        means = [float(np.mean(records[i]["p_correct"])) for records in prompts]
        sigma = float(np.std(means, ddof=1)) if len(means) > 1 else 0.
        pm = float(np.mean(means))
        sensitivity.append(dict(x=k, mp=pm, mp_lo=max(0., pm-sigma), mp_hi=min(1., pm+sigma)))
    return strength, examples, sensitivity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=Path("evals/regen/layer"))
    ap.add_argument("--plots-dir", type=Path, default=Path("plots"))
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    data = {model: series(args.source_root, model) for model in args.models.split(",")}
    panels = [
        ("type1/type1_layer_introspection.html", "Injection Layer Classification", "Injection strength"),
        ("type2/type2_layer_introspection_merged.html", "Injection Layer Classification", "In-Context Examples (<i>k</i>)"),
        ("type2/type2_layer_introspection_prompt_sigma.html", "Injection Layer Classification (Prompt Sensitivity)", "In-Context Examples (<i>k</i>)"),
    ]
    for i, (filename, title, xlabel) in enumerate(panels):
        _render_model_overlay({m: rows[i] for m, rows in data.items()},
                              args.plots_dir / filename, title=title, xlabel=xlabel)


if __name__ == "__main__":
    main()
