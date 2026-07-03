"""Compute and persist max viable steering strengths per (concept, layer).

Max strength (comprehension): highest adaptive fraction where factual-QA
    accuracy stays above a threshold (default 95%).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from steering_vectors import SteeringVector

from .concepts import ConceptLibrary
from .injection import forward_with_multi_layer_steering, forward_with_positional_steering

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMP_THRESHOLD = 0.95
_PRECISION = 0.1  # binary search stops when hi - lo < this

# Binary search bounds.
_MAX_SEARCH_LO = 0.1   # lowest fraction to test for max strength
_MAX_SEARCH_HI = 5.0   # highest fraction to test for max strength

# Factual QA pairs — verified to mostly produce single-token answers.
_QA_PAIRS = [
    ("What is 2+2?", "4"),
    ("What is 3+5?", "8"),
    ("What is 10-3?", "7"),
    ("What is 9+1?", "10"),
    ("What is 4+4?", "8"),
    ("What is 15-5?", "10"),
    ("What is 7+3?", "10"),
    ("What is 8-2?", "6"),
    ("What is 3*3?", "9"),
    ("What is 20-7?", "13"),
    ("What is 5+6?", "11"),
    ("What is 100-1?", "99"),
    ("What is 12+8?", "20"),
    ("What is 7*2?", "14"),
    ("What is 9+9?", "18"),
    ("What is 6+7?", "13"),
    ("What is 11-4?", "7"),
    ("What is 5*5?", "25"),
    ("What is 30-10?", "20"),
    ("What is 8+8?", "16"),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_chat(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
) -> str:
    """Apply chat template with ``enable_thinking=False`` fallback."""
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _find_text_positions(
    tokenizer: PreTrainedTokenizerBase,
    ids_list: list[int],
    text: str,
) -> list[int] | None:
    """Find token positions of *text* inside a tokenized sequence."""
    candidates = [text]
    if text.endswith((".", "!", "?")):
        candidates.append(text[:-1])
    for candidate in candidates:
        for prefix in ["", " "]:
            subseq = tokenizer.encode(prefix + candidate, add_special_tokens=False)
            for i in range(len(ids_list) - len(subseq) + 1):
                if ids_list[i : i + len(subseq)] == subseq:
                    return list(range(i, i + len(subseq)))
    return None


# ---------------------------------------------------------------------------
# Comprehension inputs (shared across concepts for a given model)
# ---------------------------------------------------------------------------


def _prepare_comprehension_inputs(
    tokenizer: PreTrainedTokenizerBase,
    device: str | torch.device,
) -> list[tuple[Tensor, list[int], list[int]]]:
    """Tokenize QA pairs. Returns ``(input_ids, question_positions, answer_ids)``."""
    inputs = []
    for question, answer in _QA_PAIRS:
        answer_ids = []
        for variant in [answer, f" {answer}"]:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                answer_ids.append(toks[0])
        if not answer_ids:
            continue
        messages = [
            {"role": "system", "content": "Answer with just the number, nothing else."},
            {"role": "user", "content": question},
        ]
        text = _build_chat(tokenizer, messages)
        input_ids = tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=False,
        ).to(device)
        ids_list = input_ids[0].tolist()
        positions = _find_text_positions(tokenizer, ids_list, question)
        if positions is not None:
            inputs.append((input_ids, positions, answer_ids))
    return inputs


# ---------------------------------------------------------------------------
# Core evaluation helper
# ---------------------------------------------------------------------------


def _eval_comprehension(
    model: nn.Module,
    sv: SteeringVector,
    layer: int,
    fraction: float,
    comp_inputs: list[tuple[Tensor, list[int], list[int]]],
) -> float:
    """Return QA accuracy at the given steering fraction."""
    correct = 0
    for input_ids, positions, answer_ids in comp_inputs:
        pairs = [(positions, sv)]
        logits = forward_with_positional_steering(
            model, input_ids, pairs, layer, scales=[fraction],
        )
        top_tok = logits[0, -1, :].argmax().item()
        if top_tok in answer_ids:
            correct += 1
    return correct / len(comp_inputs) if comp_inputs else 0.0


# ---------------------------------------------------------------------------
# Public API: compute_max_strength
# ---------------------------------------------------------------------------


def compute_max_strength(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    library: ConceptLibrary,
    concept: str,
    layer: int,
    *,
    threshold: float = _COMP_THRESHOLD,
    precision: float = _PRECISION,
    search_lo: float = _MAX_SEARCH_LO,
    search_hi: float = _MAX_SEARCH_HI,
    _comp_inputs: list | None = None,
) -> float | None:
    """Find the highest steering fraction that preserves comprehension.

    Binary-searches over ``[search_lo, search_hi]`` until the interval is
    narrower than *precision*.  Returns the highest fraction at which QA
    accuracy stays above *threshold*, or ``None`` if even *search_lo* fails.

    Pass *_comp_inputs* to reuse pre-tokenized QA inputs across calls.
    """
    sv = library.get_vector(concept)
    device = next(model.parameters()).device

    comp_inputs = _comp_inputs or _prepare_comprehension_inputs(tokenizer, device)
    if len(comp_inputs) < 5:
        logger.warning(
            "Only %d QA pairs resolved for this tokenizer — results may be unreliable.",
            len(comp_inputs),
        )

    # Check that the lower bound itself passes.
    if _eval_comprehension(model, sv, layer, search_lo, comp_inputs) < threshold:
        return None

    # Check if the upper bound still passes (no ceiling found).
    if _eval_comprehension(model, sv, layer, search_hi, comp_inputs) >= threshold:
        return search_hi

    # Binary search: lo always passes, hi always fails.
    lo, hi = search_lo, search_hi
    while hi - lo > precision:
        mid = (lo + hi) / 2
        acc = _eval_comprehension(model, sv, layer, mid, comp_inputs)
        logger.debug("  max_strength: frac=%.3f acc=%.2f", mid, acc)
        if acc >= threshold:
            lo = mid
        else:
            hi = mid

    return lo


# ---------------------------------------------------------------------------
# Multi-layer comprehension helpers
# ---------------------------------------------------------------------------


def _eval_comprehension_multilayer(
    model: nn.Module,
    sv: SteeringVector,
    layers: list[int],
    per_layer_fraction: float,
    comp_inputs: list[tuple[Tensor, list[int], list[int]]],
    ramp_weights: list[float] | None = None,
) -> float:
    """Return QA accuracy when steering at *all* layers simultaneously.

    Each layer receives ``per_layer_fraction * ramp_weights[i]`` as its
    adaptive scale.  Default ramp_weights are uniform (all 1.0).
    """
    if ramp_weights is None:
        ramp_weights = [1.0] * len(layers)

    correct = 0
    for input_ids, positions, answer_ids in comp_inputs:
        triples = []
        triple_scales = []
        for j, layer_idx in enumerate(layers):
            triples.append((positions, layer_idx, sv))
            triple_scales.append(per_layer_fraction * ramp_weights[j])
        logits = forward_with_multi_layer_steering(
            model, input_ids, triples,
            scales=triple_scales,
        )
        top_tok = logits[0, -1, :].argmax().item()
        if top_tok in answer_ids:
            correct += 1
    return correct / len(comp_inputs) if comp_inputs else 0.0


# ---------------------------------------------------------------------------
# Public API: compute_max_strength_multilayer
# ---------------------------------------------------------------------------

_ML_PRECISION = 0.005
_ML_SEARCH_LO = 0.005
_ML_SEARCH_HI = 0.5


def compute_max_strength_multilayer(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    library: ConceptLibrary,
    concept: str,
    layers: list[int],
    *,
    threshold: float = _COMP_THRESHOLD,
    precision: float = _ML_PRECISION,
    search_lo: float = _ML_SEARCH_LO,
    search_hi: float = _ML_SEARCH_HI,
    ramp_weights: list[float] | None = None,
    _comp_inputs: list | None = None,
) -> float | None:
    """Find the highest per-layer fraction that preserves comprehension
    when steering at *all* layers simultaneously.

    Binary-searches over ``[search_lo, search_hi]`` until the interval
    is narrower than *precision*.  Returns the highest per-layer fraction
    at which QA accuracy stays above *threshold*, or ``None`` if even
    *search_lo* fails.

    Each layer receives ``fraction * ramp_weights[i]`` as its adaptive
    scale (default: uniform weights of 1.0).
    """
    sv = library.get_vector(concept)
    device = next(model.parameters()).device

    comp_inputs = _comp_inputs or _prepare_comprehension_inputs(tokenizer, device)
    if len(comp_inputs) < 5:
        logger.warning(
            "Only %d QA pairs resolved — results may be unreliable.",
            len(comp_inputs),
        )

    if _eval_comprehension_multilayer(
        model, sv, layers, search_lo, comp_inputs, ramp_weights,
    ) < threshold:
        return None

    if _eval_comprehension_multilayer(
        model, sv, layers, search_hi, comp_inputs, ramp_weights,
    ) >= threshold:
        return search_hi

    lo, hi = search_lo, search_hi
    while hi - lo > precision:
        mid = (lo + hi) / 2
        acc = _eval_comprehension_multilayer(
            model, sv, layers, mid, comp_inputs, ramp_weights,
        )
        logger.debug(
            "  multilayer max_strength: frac=%.4f acc=%.2f", mid, acc,
        )
        if acc >= threshold:
            lo = mid
        else:
            hi = mid

    return lo


# ---------------------------------------------------------------------------
# SteeringRanges persistence class
# ---------------------------------------------------------------------------


@dataclass
class SteeringRanges:
    """Per-(concept, layer) max steering fractions with build-or-load semantics.

    ``ranges`` maps ``{concept_name: {layer_idx: {"c_max": ...}}}``.
    """

    ranges: dict[str, dict[int, dict[str, float | None]]] = field(
        default_factory=dict,
    )
    metadata: dict = field(default_factory=dict)

    # -- Query --

    def get_max(self, concept: str, layer: int) -> float | None:
        """Return ``c_max`` for the given concept and layer."""
        return self.ranges.get(concept, {}).get(layer, {}).get("c_max")

    def has_range(self, concept: str, layer: int) -> bool:
        return concept in self.ranges and layer in self.ranges[concept]

    # -- Mutate --

    def set_range(
        self,
        concept: str,
        layer: int,
        c_max: float | None,
    ) -> None:
        self.ranges.setdefault(concept, {})[layer] = {"c_max": c_max}

    # -- Persistence (JSON) --

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # JSON keys must be strings.
        serialised_ranges = {
            concept: {
                str(layer): entry
                for layer, entry in layers.items()
            }
            for concept, layers in self.ranges.items()
        }
        data = {"metadata": self.metadata, "ranges": serialised_ranges}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Steering ranges saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> SteeringRanges:
        with open(path) as f:
            data = json.load(f)
        ranges: dict[str, dict[int, dict[str, float | None]]] = {}
        for concept, layers in data.get("ranges", {}).items():
            parsed: dict[str, float | None] = {}
            for layer, entry in layers.items():
                # Support both old format (with c_min) and new format (c_max only)
                if isinstance(entry, dict):
                    parsed_entry = {"c_max": entry.get("c_max")}
                else:
                    parsed_entry = {"c_max": entry}
                ranges.setdefault(concept, {})[int(layer)] = parsed_entry
        obj = cls(ranges=ranges, metadata=data.get("metadata", {}))
        logger.info("Steering ranges loaded from %s", path)
        return obj

    @classmethod
    def build_or_load(
        cls,
        path: str | Path,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        library: ConceptLibrary,
        concepts: list[str],
        layers: list[int],
        *,
        comp_threshold: float = _COMP_THRESHOLD,
        precision: float = _PRECISION,
    ) -> SteeringRanges:
        """Load existing ranges, compute missing ``(concept, layer)`` pairs, save.

        Progress is saved after each concept so partial runs survive interruptions.
        """
        path = Path(path)
        if path.exists():
            obj = cls.load(path)
        else:
            obj = cls(
                metadata={"comp_threshold": comp_threshold},
            )

        # Identify missing pairs.
        missing: dict[str, list[int]] = {}
        for concept in concepts:
            for layer in layers:
                if not obj.has_range(concept, layer):
                    missing.setdefault(concept, []).append(layer)

        if not missing:
            logger.info("All requested (concept, layer) ranges already exist.")
            return obj

        total_pairs = sum(len(ls) for ls in missing.values())
        logger.info(
            "Computing steering ranges for %d missing (concept, layer) pairs ...",
            total_pairs,
        )

        device = next(model.parameters()).device

        # Pre-tokenize shared inputs once.
        comp_inputs = _prepare_comprehension_inputs(tokenizer, device)

        done = 0
        for concept, concept_layers in missing.items():
            sv = library.get_vector(concept)
            available = set(sv.layer_activations.keys())

            for layer in concept_layers:
                if layer not in available:
                    logger.warning(
                        "Concept %r has no activation at layer %d — skipping.",
                        concept, layer,
                    )
                    continue

                done += 1
                logger.info(
                    "[%d/%d] %s @ layer %d ...", done, total_pairs, concept, layer,
                )

                c_max = compute_max_strength(
                    model, tokenizer, library, concept, layer,
                    threshold=comp_threshold,
                    precision=precision,
                    _comp_inputs=comp_inputs,
                )
                obj.set_range(concept, layer, c_max)

            # Save after each concept for resilience.
            obj.save(path)

        return obj
