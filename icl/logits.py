"""Logit extraction utilities for ICL steering experiments."""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Type alias: each element is a group of token variants to coalesce.
TokenGroup = list[str]
OutputTokens = list[TokenGroup]


def extract_token_logits(
    logits: Tensor,
    output_tokens: OutputTokens,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract logits and probabilities for groups of tokens at the last position.

    Each element of *output_tokens* is a list of token variants whose
    probabilities are summed (coalesced) into a single entry.  For example,
    ``[["queen", " queen"], ["king", " king"]]`` sums the probs of
    ``"queen"`` and ``" queen"`` into one group.

    The dict key for each group is the first token in the group.

    Args:
        logits: Full logits tensor of shape ``(1, seq_len, vocab_size)``.
        output_tokens: List of token groups.  Multi-token entries within a
            group are warned about and skipped.
        tokenizer: Tokenizer used to convert plaintext to token IDs.

    Returns:
        Tuple of ``(logits_dict, probs_dict)`` where keys are the first
        token in each group and probs are from the full-vocabulary softmax
        (i.e. raw probability of that token/group appearing, not relative
        to the other requested tokens).
    """
    last_logits = logits[0, -1, :]  # (vocab_size,)

    group_labels: list[str] = []
    group_logits: list[Tensor] = []

    for group in output_tokens:
        ids = _resolve_group_ids(group, tokenizer)
        if not ids:
            continue
        label = group[0]
        # logsumexp over variants = log(sum(exp(logit_i))) = log(sum(prob_i)) + const
        group_logit = torch.logsumexp(last_logits[ids].float(), dim=0)
        group_labels.append(label)
        group_logits.append(group_logit)

    stacked = torch.stack(group_logits)  # (n_groups,)

    # Probabilities from full-vocabulary softmax, not relative to each other.
    # For coalesced groups the group_logit is already logsumexp of the raw
    # logits, so we need to subtract the full-vocab log-normalizer.
    full_log_norm = torch.logsumexp(last_logits.float(), dim=0)
    probs = torch.exp(stacked - full_log_norm)

    logits_dict = {label: stacked[i].item() for i, label in enumerate(group_labels)}
    probs_dict = {label: probs[i].item() for i, label in enumerate(group_labels)}
    return logits_dict, probs_dict


def _resolve_group_ids(
    group: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Resolve a group of token variants to valid single-token IDs.

    Multi-token entries are warned about and skipped.
    """
    ids = []
    for text in group:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            logger.warning(
                "Output token %r encodes to %d token(s) (IDs: %s). "
                "Skipping — only single-token outputs are supported.",
                text, len(encoded), encoded,
            )
            continue
        ids.append(encoded[0])
    return ids
