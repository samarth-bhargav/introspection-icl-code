"""Model/tokenizer/concept-library singletons and lazy-loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from icl.steering.concepts import ConceptLibrary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME: str = "Qwen/Qwen3-32B"
DEFAULT_DEVICE: str = "cuda"
DEFAULT_LIBRARY_PATH: str = "artifacts/concept_library.pt"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_ARTIFACTS_ROOT = Path(__file__).parent / "artifacts"

MODEL_REGISTRY: dict[str, dict] = {
    "qwen3-32b":  {"hf_id": "Qwen/Qwen3-32B",                              "layers": 64, "hidden": 5120},
    "qwen3-8b":   {"hf_id": "Qwen/Qwen3-8B",                               "layers": 36, "hidden": 4096},
    "olmo-7b":    {"hf_id": "allenai/Olmo-3-7B-Instruct",                   "layers": 32, "hidden": 4096},
    "olmo-32b":   {"hf_id": "allenai/Olmo-3.1-32B-Instruct",                "layers": 64, "hidden": 5120},
    "gemma-31b":  {"hf_id": "google/gemma-4-31B-it",                        "layers": 60, "hidden": 5376},
    # MoE, sharded across all visible GPUs (bf16 — FP8 variant hits a triton-kernel bug)
    "qwen3-235b": {"hf_id": "Qwen/Qwen3-235B-A22B-Instruct-2507",           "layers": 94, "hidden": 4096, "device_map": "auto"},
}


def resolve_model_name(short_name: str) -> str:
    """Return the HuggingFace model ID for a registry short name, or pass through."""
    if short_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[short_name]["hf_id"]
    return short_name


def get_library_path(model_short_name: str) -> str:
    """Return the path to a model's concept library .pt file."""
    return str(_ARTIFACTS_ROOT / model_short_name / "concept_library.pt")

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded)
# ---------------------------------------------------------------------------

_model: nn.Module | None = None
_tokenizer: PreTrainedTokenizerBase | None = None
_concept_library: ConceptLibrary | None = None


def get_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, PreTrainedTokenizerBase]:
    """Lazy-load and cache model + tokenizer.

    *model_name* can be a registry short name (e.g. ``"qwen3-8b"``) or a
    full HuggingFace ID.  Subsequent calls return the cached instances
    (ignoring arguments).
    """
    global _model, _tokenizer
    if _model is None:
        # Look up registry device_map override (e.g. "auto" for large sharded models)
        reg_entry = MODEL_REGISTRY.get(model_name, {})
        device_map = reg_entry.get("device_map", device)
        model_name = resolve_model_name(model_name)
        logger.info("Loading model %s (dtype=%s, device_map=%s) ...", model_name, dtype, device_map)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Gemma 4 is multimodal; need ForConditionalGeneration, not ForCausalLM
        from transformers import AutoConfig
        pre_config = AutoConfig.from_pretrained(model_name)
        if getattr(pre_config, "model_type", "") == "gemma4":
            from transformers import Gemma4ForConditionalGeneration
            _model = Gemma4ForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype, device_map=device_map,
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, device_map=device_map,
            )
        _model.eval()
        logger.info("Model loaded.")
    return _model, _tokenizer  # type: ignore[return-value]


def unload_model() -> None:
    """Remove the cached model/tokenizer/library to free GPU memory."""
    global _model, _tokenizer, _concept_library
    _model = None
    _tokenizer = None
    _concept_library = None
    torch.cuda.empty_cache()


def get_concept_library(
    model: nn.Module | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    library_path: str | None = None,
    model_name: str | None = None,
) -> ConceptLibrary:
    """Lazy-load or build concept library.

    If *library_path* is not given but *model_name* is a registry short name,
    the path is resolved automatically from the model's artifact directory.
    """
    global _concept_library
    if _concept_library is None:
        if library_path is None and model_name is not None:
            library_path = get_library_path(model_name)
        elif library_path is None:
            library_path = DEFAULT_LIBRARY_PATH
        _concept_library = ConceptLibrary.build_or_load(
            model=model, tokenizer=tokenizer, library_path=library_path,
        )
    return _concept_library


def ensure_concepts(
    concept_library: ConceptLibrary,
    concept_names: list[str],
    model: nn.Module | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    library_path: str | None = None,
) -> ConceptLibrary:
    """Ensure all requested concepts exist in the library, extracting any missing ones.

    Args:
        concept_library: Existing library to check against.
        concept_names: Concepts that must be present.
        model: HF model for extraction (required if concepts are missing).
        tokenizer: HF tokenizer (required if concepts are missing).
        library_path: If provided, save the updated library here.

    Returns:
        The (possibly augmented) concept library.
    """
    missing = [c for c in concept_names if c not in concept_library]
    if not missing:
        return concept_library

    if model is None or tokenizer is None:
        raise ValueError(
            f"Concepts {missing} not in library and no model/tokenizer "
            f"provided for extraction."
        )

    logger.info("Extracting missing concepts: %s", missing)
    new_lib = ConceptLibrary.build(model, tokenizer, concept_names=missing)

    merged_vectors = {**concept_library.vectors, **new_lib.vectors}
    merged_names = list(dict.fromkeys(concept_library.concept_names + missing))
    merged = ConceptLibrary(vectors=merged_vectors, concept_names=merged_names)

    if library_path:
        merged.save(library_path)

    global _concept_library
    _concept_library = merged
    return merged
