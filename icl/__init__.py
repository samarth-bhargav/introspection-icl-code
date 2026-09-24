"""In-context learning and activation-steering experiments.

Public inference helpers are loaded on first access so plotting and configuration
commands do not require the GPU inference dependencies.
"""
from importlib import import_module

_EXPORTS = {
    "OutputTokens": ("logits", "OutputTokens"),
    "TokenGroup": ("logits", "TokenGroup"),
    "extract_token_logits": ("logits", "extract_token_logits"),
    "DEFAULT_DEVICE": ("model", "DEFAULT_DEVICE"),
    "DEFAULT_LIBRARY_PATH": ("model", "DEFAULT_LIBRARY_PATH"),
    "DEFAULT_MODEL_NAME": ("model", "DEFAULT_MODEL_NAME"),
    "MODEL_REGISTRY": ("model", "MODEL_REGISTRY"),
    "ensure_concepts": ("model", "ensure_concepts"),
    "get_concept_library": ("model", "get_concept_library"),
    "get_library_path": ("model", "get_library_path"),
    "get_model_and_tokenizer": ("model", "get_model_and_tokenizer"),
    "resolve_model_name": ("model", "resolve_model_name"),
    "unload_model": ("model", "unload_model"),
    "DEFAULT_SYSTEM_PROMPT": ("query", "DEFAULT_SYSTEM_PROMPT"),
    "ICLQuery": ("query", "ICLQuery"),
    "ICLResult": ("query", "ICLResult"),
    "build_prompt": ("query", "build_prompt"),
    "run_queries": ("query", "run_queries"),
    "SteeringRanges": ("steering", "SteeringRanges"),
    "compute_max_strength": ("steering", "compute_max_strength"),
    "compute_max_strength_multilayer": ("steering", "compute_max_strength_multilayer"),
    "find_trigger_positions": ("steering", "find_trigger_positions"),
    "forward_with_multi_layer_steering": ("steering", "forward_with_multi_layer_steering"),
    "forward_with_positional_steering": ("steering", "forward_with_positional_steering"),
    "generate_with_steering": ("steering", "generate_with_steering"),
    "concept_vector_norm": ("utils", "concept_vector_norm"),
    "decode_input_ids": ("utils", "decode_input_ids"),
    "is_single_token": ("utils", "is_single_token"),
}

__all__ = [
    "ICLQuery",
    "ICLResult",
    "run_queries",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_DEVICE",
    "DEFAULT_LIBRARY_PATH",
    "DEFAULT_SYSTEM_PROMPT",
    "MODEL_REGISTRY",
    "get_model_and_tokenizer",
    "get_concept_library",
    "get_library_path",
    "ensure_concepts",
    "resolve_model_name",
    "unload_model",
    "concept_vector_norm",
    "decode_input_ids",
    "is_single_token",
    "SteeringRanges",
    "compute_max_strength",
    "compute_max_strength_multilayer",
    "build_prompt",
    "find_trigger_positions",
    "forward_with_multi_layer_steering",
    "forward_with_positional_steering",
    "generate_with_steering",
    "extract_token_logits",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
