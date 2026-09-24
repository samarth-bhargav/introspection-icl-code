"""Render all figure groups from synthetic fixtures in a temporary checkout.

This tests file contracts and PNG export, not model behavior or paper scores.
Run explicitly: python tests/render_smoke.py
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import make_figures as driver


def write_json(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixtures(root):
    import plotly.graph_objects as go

    for model in driver.MODELS:
        by_k = [{"k": k, "n": 4, "accuracy": 0.5, "mean_p": 0.5,
                 "p_correct": [0.2, 0.4, 0.6, 0.8],
                 "by_prompt": [{"mean_p": 0.3}, {"mean_p": 0.7}]} for k in (0, 30)]
        for kind in ("type1", "type2"):
            write_json(root, f"{driver.MAG}/magnitude/{kind}_{model}.json",
                       {"per_strength": [{"strength": strength, "by_k": by_k}
                                         for strength in ((0.25, 1) if kind == "type1" else (1,))]})
        write_json(root, f"evals/regen/layer/type1_{model}.json",
                   {"per_strength": [{"strength": strength, "by_k": by_k} for strength in (.25, 1.)]})
        for vi in (0, 1):
            write_json(root, f"evals/regen/layer/type2_{model}_var{vi}.json",
                       {"per_strength": [{"strength": 1., "by_k": by_k}]})
        write_json(root, f"{driver.MAG}/magnitude_generalization/magnitude_generalization_{model}.json",
                   {"anchor_levels": [["low", 0.25], ["medium", 1], ["high", 2.5]],
                    "records": [{"test_alpha": alpha, "samples": [
                        {"probabilities": {"low": 0.2, "medium": 0.3, "high": 0.4}},
                        {"probabilities": {"low": 0.3, "medium": 0.4, "high": 0.2}}]}
                        for alpha in (0, 1, 2.5)]})
        write_json(root, f"evals/regen/layer_generalization/layer_generalization_{model}.json",
                   {"model": model, "anchors": [1, 2, 3], "cmax_fraction": 1,
                    "n_examples": 2, "n_samples": 2, "records": [
                        {"test_layer": layer, "samples": [
                            {"probabilities": {"early": 0.2, "middle": 0.3, "late": 0.4}},
                            {"probabilities": {"early": 0.3, "middle": 0.4, "late": 0.2}}]}
                        for layer in (0, 1, 2, 3)]})
        for k in (0, 1):
            rows = [{"p_correct": score, "correct": bool(score),
                     "prompt_variation_id": variant, "target_emotion": "anger",
                     "gating_emotion": "anger", "test_injection": {"kind": mode}}
                    for variant in (0, 1) for mode in ("target", "distractor", "none")
                    for score in (0, 1)]
            write_json(root, f"{driver.GATED}/generation_{model}/math/type2_k_sweep/math_{model}_k{k}.json",
                       {"n_demos": k, "rows": rows})
            for variant in (0, 1):
                write_json(root, f"{driver.GATED}/successor_emotions_k_sweep/{model}/k{k}_var{variant}.json",
                           {"n_demos": k, "rows": [r for r in rows if r["prompt_variation_id"] == variant]})
        for fraction in (0.1, 0.5):
            payload = {"cmax_fraction": fraction, "rows": rows}
            write_json(root, f"evals/regen/generation_{model}/math/type1_cmax_sweep/math_{model}_k10_cmax{fraction}.json", payload)
            write_json(root, f"evals/regen/successor_cmax_sweep/{model}/fraction_{fraction:.2f}.json", payload)


def main():
    with tempfile.TemporaryDirectory(prefix="icl-synthetic-render-") as tmp:
        root = Path(tmp)
        shutil.copytree(ROOT / "icl", root / "icl", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "make_figures.py", root / "make_figures.py")
        fixtures(root)
        print("Rendering SYNTHETIC fixtures; these are not experiment results.", flush=True)
        subprocess.run([sys.executable, "make_figures.py"], cwd=root, check=True)
        outputs = {path for step in driver.steps() for path in step.outputs}
        for relative in outputs:
            data = (root / relative).read_bytes()
            assert data.startswith(b"\x89PNG\r\n\x1a\n"), relative
        print(f"PASS: {len(outputs)} expected PNG outputs from all {len(driver.steps())} groups.")


if __name__ == "__main__":
    main()
