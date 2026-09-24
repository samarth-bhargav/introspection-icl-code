"""Operating-strength selection for classification experiments."""
import json
from pathlib import Path


def pick_mean_probability(path, k: int) -> dict:
    """Maximize mean p(correct), breaking exact ties toward smaller strength."""
    data = json.loads(Path(path).read_text())
    curve = []
    for entry in data["per_strength"]:
        rec = next((r for r in entry["by_k"] if r["k"] == k), None)
        if rec is not None:
            curve.append((float(entry["strength"]), float(rec["accuracy"]), float(rec["mean_p"])))
    if not curve:
        raise ValueError(f"no K={k} records in {path}")
    star, acc, mean_p = max(curve, key=lambda row: (row[2], -row[0]))
    return {"star": star, "accuracy": acc, "mean_p": mean_p,
            "max_acc": max(row[1] for row in curve), "curve": curve,
            "selection_metric": "mean_p_correct"}
