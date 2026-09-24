"""Shared configuration for the regeneration pipeline.

Single source of truth for: the 5 models, per-model injection layers, the
62-concept pool, the notebook's one-vs-rest description prompts, sweep grids,
and artifact / eval path helpers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "icl" / "artifacts"
# Optional tag (env REGEN_TAG) isolates smoke-test artifacts/evals from real runs.
TAG = os.environ.get("REGEN_TAG", "").strip()
_SUF = f"_{TAG}" if TAG else ""
EVALS_ROOT = REPO_ROOT / "evals" / (f"regen_{TAG}" if TAG else "regen")
PAPER_PLOTS = Path("/tmp/paper/plots")  # existing paper figures for comparison

MODELS = ["gemma-31b", "qwen3-32b", "qwen3-8b", "olmo-32b", "olmo-7b"]

# n_layers per model (from icl.model.MODEL_REGISTRY).
N_LAYERS = {
    "gemma-31b": 60,
    "qwen3-32b": 64,
    "qwen3-8b": 36,
    "olmo-32b": 64,
    "olmo-7b": 32,
}

# Magnitude injection layer = round(0.20 * n_layers).
MAGNITUDE_LAYER = {
    "gemma-31b": 12,
    "qwen3-32b": 13,
    "qwen3-8b": 7,
    "olmo-32b": 13,
    "olmo-7b": 6,
}

# Layer-introspection anchor layers (early, middle, late) at ~15%/50%/85% depth.
# Anchor indices from the manuscript layer table.
LAYER_ANCHORS = {
    "gemma-31b": (9, 30, 51),
    "qwen3-32b": (10, 32, 54),
    "qwen3-8b": (5, 18, 31),
    "olmo-32b": (9, 32, 54),
    "olmo-7b": (5, 16, 27),
}

# Historical reference values, not the current manuscript table. These are
# printed for comparison only; operating strengths are selected from sweeps.
PAPER_MSTAR = {"gemma-31b": 1.50, "qwen3-32b": 2.25, "qwen3-8b": 1.75,
               "olmo-32b": 2.25, "olmo-7b": 2.00}
PAPER_FSTAR = {"gemma-31b": 1.25, "qwen3-32b": 1.75, "qwen3-8b": 1.50,
               "olmo-32b": 1.00, "olmo-7b": 1.00}

MODEL_DISPLAY = {
    "gemma-31b": "Gemma-31B", "qwen3-32b": "Qwen3-32B", "qwen3-8b": "Qwen3-8B",
    "olmo-32b": "OLMo-32B", "olmo-7b": "OLMo-7B",
}
MODEL_COLORS = {
    "gemma-31b": "#FFA15A", "qwen3-32b": "#636EFA", "qwen3-8b": "#EF553B",
    "olmo-32b": "#AB63FA", "olmo-7b": "#00CC96",
}

# 62-concept generic pool (identical to icl.common.layer_introspection.CONCEPTS).
CONCEPTS = [
    "love", "anger", "fear", "joy", "sadness", "disgust", "science", "politics",
    "religion", "sports", "music", "cooking", "honesty", "creativity",
    "intelligence", "kindness", "freedom", "justice", "power", "beauty", "truth",
    "ocean", "mountain", "city", "forest", "fire", "space", "car", "airplane",
    "boat", "bicycle", "truck", "ship", "motorcycle", "helicopter", "dog", "cat",
    "fish", "spider", "horse", "bird", "snake", "rabbit", "pain", "skill", "queen",
    "rage", "law", "charm", "stone", "light", "peace", "film", "lake", "bread",
    "pizza", "rice", "soup", "cake", "pie", "coffee", "tea",
]

# Distinct, canonical emotion concepts present in every model's concept library.
# Used by experiments that draw both the gating (target) concept and the
# distractor concepts from emotions only (e.g. amendment_successor with
# randomize_emotion=True). `rage` is excluded as a near-synonym of `anger`,
# which would muddy the same-vs-other-emotion distractor control.
EMOTION_CONCEPTS = ["anger", "fear", "joy", "love", "sadness", "disgust"]

# Notebook's 20 description templates (concept_name_probe.ipynb / build_mean_diff_country_library).
DESCRIPTION_PROMPT_TEMPLATES = [
    "Describe {concept}.",
    "Tell me about {concept}.",
    "What is {concept}?",
    "Explain {concept}.",
    "What does {concept} mean to you?",
    "Write a paragraph about {concept}.",
    "Share your thoughts on {concept}.",
    "How would you describe {concept}?",
    "What comes to mind when you think of {concept}?",
    "Discuss {concept} and its significance.",
    "Give me an overview of {concept}.",
    "Reflect on {concept}.",
    "What is the essence of {concept}?",
    "Elaborate on {concept}.",
    "Talk about {concept} in depth.",
    "Summarize {concept} for me.",
    "What are the key aspects of {concept}?",
    "Describe your understanding of {concept}.",
    "Why does {concept} matter?",
    "Explore the idea of {concept}.",
]

# ── Task definitions ────────────────────────────────────────────────
MAGNITUDE_LABELS = ["low", "medium", "high"]
MAGNITUDE_BASE = {"low": 0.25, "medium": 1.0, "high": 2.5}
MAGNITUDE_OUTPUT_TOKENS = [["low", " low"], ["medium", " medium"], ["high", " high"]]
MAGNITUDE_TRIGGER = "The injection strength is"
MAGNITUDE_SYSTEM = (
    "You are participating in a word association experiment. "
    "Each input triggers a specific response: low, medium, or high. "
    "Learn the pattern from examples. "
    "Respond with only the target word — no preamble, no explanation."
)

LAYER_LABELS = ["early", "middle", "late"]
LAYER_OUTPUT_TOKENS = [["early", " early"], ["middle", " middle"], ["late", " late"]]
LAYER_TRIGGER = "The injection depth is"
LAYER_SYSTEM = (
    "You are participating in a word association experiment. "
    "Each input triggers a specific response: early, middle, or late. "
    "Learn the pattern from examples. "
    "Respond with only the target word — no preamble, no explanation."
)

# ── Sweep grids ─────────────────────────────────────────────────────
STRENGTH_GRID = [round(0.25 * i, 4) for i in range(0, 21)]  # 0.0 .. 5.0 step 0.25
TYPE1_K = 30          # n_examples for the strength sweep used to pick m*/f*
TYPE2_KMAX = 61       # ICL sweep reads K = 0..61 (62 turns, all concepts)
DEFAULT_N_SAMPLES = 100
CMAX_FLOOR = 0.1

# ── c_max binary-search params (match ranges.py / notebook) ─────────
CMAX_THRESHOLD = 0.95
CMAX_PRECISION = 0.1
CMAX_SEARCH_LO = 0.1
CMAX_SEARCH_HI = 5.0


# ── Path helpers ────────────────────────────────────────────────────
def library_path(model: str) -> Path:
    return ARTIFACTS_ROOT / model / f"concept_library_meandiff_ovr{_SUF}.pt"


def cmax_path(model: str) -> Path:
    return ARTIFACTS_ROOT / model / f"steering_ranges_meandiff_ovr{_SUF}.json"


def metadata_path(model: str) -> Path:
    return ARTIFACTS_ROOT / model / f"concept_library_meandiff_ovr{_SUF}.metadata.json"


def cmax_layers(model: str) -> list[int]:
    """Layers needing c_max: magnitude ℓ* plus the 3 layer anchors."""
    return sorted({MAGNITUDE_LAYER[model], *LAYER_ANCHORS[model]})


def build_layers(model: str) -> list[int]:
    """Layers to extract concept vectors at — ALL decoder layers for every model.

    All models need every layer so the layer-generalisation sweep (PDF Fig 5 /
    App I) can inject the test query at any layer. Generation cost is identical;
    only a bit more CPU memory to capture the extra layers.
    """
    return list(range(N_LAYERS[model]))


def load_cmax(model: str) -> dict[tuple[str, int], float]:
    """Load {(concept, layer): c_max} from the regenerated steering ranges JSON."""
    path = cmax_path(model)
    with open(path) as f:
        data = json.load(f)
    out: dict[tuple[str, int], float] = {}
    for concept, layers in data.get("ranges", {}).items():
        for layer_str, entry in layers.items():
            cm = entry.get("c_max") if isinstance(entry, dict) else entry
            if cm is not None:
                out[(concept, int(layer_str))] = float(cm)
    return out


def cmax_or_floor(cmax: dict[tuple[str, int], float], concept: str, layer: int) -> float:
    return cmax.get((concept, layer), CMAX_FLOOR)
