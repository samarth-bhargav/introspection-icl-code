"""ICL Playground — in-context learning experiments with positional steering vectors."""

from .logits import OutputTokens, TokenGroup, extract_token_logits
from .model import (
    DEFAULT_DEVICE,
    DEFAULT_LIBRARY_PATH,
    DEFAULT_MODEL_NAME,
    MODEL_REGISTRY,
    ensure_concepts,
    get_concept_library,
    get_library_path,
    get_model_and_tokenizer,
    resolve_model_name,
    unload_model,
)
from .query import (
    DEFAULT_SYSTEM_PROMPT,
    ICLQuery,
    ICLResult,
    build_prompt,
    run_queries,
)
from .steering import (
    SteeringRanges,
    compute_max_strength,
    compute_max_strength_multilayer,
    find_trigger_positions,
    forward_with_multi_layer_steering,
    forward_with_positional_steering,
    generate_with_steering,
)
from .utils import (
    concept_vector_norm,
    decode_input_ids,
    is_single_token,
)

__all__ = [
    # Core API
    "ICLQuery",
    "ICLResult",
    "run_queries",
    # Defaults & lazy loaders
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
    # Utilities
    "concept_vector_norm",
    "decode_input_ids",
    "is_single_token",
    # Steering ranges
    "SteeringRanges",
    "compute_max_strength",
    "compute_max_strength_multilayer",
    # Lower-level
    "build_prompt",
    "find_trigger_positions",
    "forward_with_multi_layer_steering",
    "forward_with_positional_steering",
    "generate_with_steering",
    "extract_token_logits",
]
