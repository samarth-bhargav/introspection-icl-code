"""Miscellaneous utilities for ICL experiments."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from icl.steering.concepts import ConceptLibrary

if TYPE_CHECKING:
    from icl.query import ICLQuery

logger = logging.getLogger(__name__)


def is_single_token(
    text: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> bool:
    """Check if a string encodes to exactly one token.

    Args:
        text: The text to check.
        tokenizer: Tokenizer to use. ``None`` = lazy-load default (Qwen3-32B).

    Returns:
        ``True`` if *text* encodes to exactly one token ID, ``False`` otherwise.
    """
    if tokenizer is None:
        from icl.model import get_model_and_tokenizer

        _, tokenizer = get_model_and_tokenizer()
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded) == 1


def concept_vector_norm(
    concept_name: str,
    concept_library: ConceptLibrary,
    layer: int,
) -> float:
    """Return the L2 norm of a concept's steering vector at the injection layer."""
    sv = concept_library.get_vector(concept_name)
    return sv.layer_activations[layer].float().norm().item()


def decode_input_ids(
    query: ICLQuery,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> str:
    """Debug helper: return the full tokenized prompt as decoded text.

    Useful for verifying chat template formatting.
    """
    from icl.model import get_model_and_tokenizer
    from icl.query import build_prompt

    if tokenizer is None:
        _, tokenizer = get_model_and_tokenizer()
    return build_prompt(query, tokenizer)
