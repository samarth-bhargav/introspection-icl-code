"""Shared utilities for key result experiments."""
import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch


def parse_range(s):
    """Parse '5' -> [5.0], '1:5:1' -> [1.0, 2.0, 3.0, 4.0, 5.0]."""
    parts = s.split(":")
    if len(parts) == 1:
        return [float(parts[0])]
    start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
    vals, v = [], start
    while v <= stop + 1e-9:
        vals.append(round(v, 10))
        v += step
    return vals


def parse_int_range(s):
    """Like parse_range but returns ints."""
    return [int(v) for v in parse_range(s)]


def add_common_args(parser):
    """Add standard sweep parameters shared across all key result experiments."""
    parser.add_argument("--n_samples", default="40", help="queries per config (range, e.g. 40 or 20:60:20)")
    parser.add_argument("--n_examples", default="15", help="ICL examples per query (range)")
    parser.add_argument("--cmax_fraction", default="0.9", help="fraction of per-concept c_max to use as steering strength (range, e.g. 0.9 or 0.1:1.0:0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", default=None, help="directory to write per-sample JSON logs")
    parser.add_argument("--model", default="qwen3-32b", help="model short name from MODEL_REGISTRY (default: qwen3-32b)")


def log_results(log_dir, experiment_name, params, accuracy, sample_logs):
    """Write per-sample results to a JSON file in log_dir."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(log_dir, f"{experiment_name}_{ts}.json")
    data = {
        "experiment": experiment_name,
        "timestamp": ts,
        "params": params,
        "accuracy": accuracy,
        "n_total": len(sample_logs),
        "n_correct": sum(1 for s in sample_logs if s["correct"]),
        "samples": sample_logs,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  -> logged to {path}")


# ---------------------------------------------------------------------------
# Per-concept c_max scaling
# ---------------------------------------------------------------------------

_ARTIFACTS_ROOT = Path(__file__).parent.parent / "artifacts"
_cmax_caches: dict[str, dict[tuple[str, int], float]] = {}


def _load_cmax_cache(model_name: str = "qwen3-32b") -> dict[tuple[str, int], float]:
    if model_name not in _cmax_caches:
        ranges_path = _ARTIFACTS_ROOT / model_name / "steering_ranges.json"
        with open(ranges_path) as f:
            data = json.load(f)
        cache: dict[tuple[str, int], float] = {}
        for concept, layers in data.get("ranges", {}).items():
            for layer_str, entry in layers.items():
                c_max = entry.get("c_max")
                if c_max is not None:
                    cache[(concept, int(layer_str))] = c_max
        _cmax_caches[model_name] = cache
    return _cmax_caches[model_name]


def get_cmax(concept: str, layer: int, model_name: str = "qwen3-32b") -> float:
    """Return c_max for a (concept, layer) pair. Raises if not available."""
    cache = _load_cmax_cache(model_name)
    val = cache.get((concept, layer))
    if val is None:
        raise ValueError(
            f"No c_max data for ({concept!r}, layer={layer}) in model {model_name!r}. "
            f"Run SteeringRanges.build_or_load() to compute it first."
        )
    return val


def compute_per_concept_scales(concepts: list[str], layer: int, cmax_fraction: float,
                               model_name: str = "qwen3-32b") -> list[float]:
    """Compute per-concept steering scales as fraction * c_max(concept, layer)."""
    return [cmax_fraction * get_cmax(c, layer, model_name) for c in concepts]


def setup_model_and_library(model_short_name: str, concepts: list[str]):
    """Load model + concept library for *model_short_name*, ensuring all *concepts* exist.

    Returns ``(model, tokenizer, concept_library)``.
    """
    from icl import get_model_and_tokenizer, get_concept_library, ensure_concepts
    from icl.model import get_library_path

    model, tok = get_model_and_tokenizer(model_short_name)
    lib_path = get_library_path(model_short_name)
    clib = ensure_concepts(
        get_concept_library(model=model, tokenizer=tok, model_name=model_short_name),
        concepts, model=model, tokenizer=tok, library_path=lib_path,
    )
    return model, tok, clib


def serialize_result(result):
    """Serialize an ICLResult (including nested ICLQuery) to a JSON-safe dict."""
    return asdict(result)


def extract_top_k(logits_2d, tokenizer, k=5):
    """Extract top-k tokens from full-vocab logits at the last position.

    Args:
        logits_2d: Tensor of shape ``(1, seq_len, vocab)`` or ``(seq_len, vocab)``.
        tokenizer: HuggingFace tokenizer for decoding token IDs.
        k: Number of top tokens to return.

    Returns:
        Dict mapping token string to probability (float).
    """
    if logits_2d.dim() == 3:
        logits_2d = logits_2d[0]
    last_logits = logits_2d[-1].float()
    full_probs = torch.softmax(last_logits, dim=0)
    topk_vals, topk_ids = torch.topk(full_probs, k)
    return {
        tokenizer.decode([tid]): val.item()
        for tid, val in zip(topk_ids.tolist(), topk_vals)
    }
