"""Shared utilities for the gated behavioral tasks.

Loading a model's regenerated concept library + c_max table, plus the small
accuracy accumulators (``_bucket``/``_record``/``_finalize``) that every task
runner uses to tally per-condition results.
"""

from __future__ import annotations

from typing import Any


def load_artifacts(model_name: str):
    """Load the regenerated concept library and c_max table for ``model_name``."""
    from icl.experiments import config as C
    from icl.steering.concepts import ConceptLibrary

    return ConceptLibrary.load(C.library_path(model_name)), C.load_cmax(model_name)


def _bucket(**extra: Any) -> dict[str, Any]:
    return {"n": 0, "n_correct": 0, "p_correct": [], **extra}


def _record(bucket: dict[str, Any], correct: bool, p_correct: float) -> None:
    bucket["n"] += 1
    bucket["n_correct"] += int(correct)
    bucket["p_correct"].append(float(p_correct))


def _finalize(bucket: dict[str, Any]) -> dict[str, Any]:
    n = bucket["n"]
    ps = bucket.pop("p_correct", [])
    bucket["accuracy"] = bucket["n_correct"] / n if n else 0.0
    bucket["mean_p"] = sum(ps) / len(ps) if ps else 0.0
    return bucket
