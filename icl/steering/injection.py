"""Positional steering vector injection via HuggingFace hooks."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from steering_vectors import SteeringVector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive steering helpers
# ---------------------------------------------------------------------------


def _get_target_module(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return the decoder layer module at *layer_idx*.

    Works for Qwen, OLMo, LLaMA (``model.model.layers``) and
    Gemma 4 multimodal (``model.model.language_model.layers``).
    """
    if hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    # Gemma 4 ForConditionalGeneration: text layers are under language_model
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer_idx]
    raise AttributeError(
        f"Cannot find decoder layers on {type(model).__name__}. "
        f"Expected model.model.layers or model.model.language_model.layers."
    )


def _make_adaptive_hook(
    entries: list[tuple[list[int], Tensor, float]],
) -> callable:
    """Create a forward hook that scales steering vectors by live residual norms.

    *entries* is a list of ``(positions, unit_activation, scale)`` triples.
    All entries sharing the same layer are batched into **one** hook so that
    norms are measured before any injection (avoiding cross-contamination).

    For each entry at each position *p*::

        h[p] += scale * ||h[p]|| * unit_activation
    """

    def hook_fn(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        # hs shape: (batch, seq_len, hidden_dim)

        # 1. Collect all unique positions and measure their norms first
        all_positions: set[int] = set()
        for positions, _act, _s in entries:
            all_positions.update(positions)
        pos_list = sorted(all_positions)
        pos_tensor = torch.tensor(pos_list, device=hs.device)
        norms = hs[0, pos_tensor].float().norm(dim=-1)  # (n_positions,)
        pos_to_norm = dict(zip(pos_list, norms.tolist()))

        # 2. Apply all injections using pre-measured norms
        for positions, activation, scale in entries:
            act = activation.to(device=hs.device, dtype=hs.dtype)
            for p in positions:
                hs[0, p] = hs[0, p] + (scale * pos_to_norm[p]) * act

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    return hook_fn


def _get_turn_boundary_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Detect token IDs that mark end-of-turn boundaries.

    Checks three sources to be model-agnostic:

    1. ``eos_token_id``
    2. ``all_special_tokens`` (e.g., Llama's ``<|eot_id|>``)
    3. ``added_tokens_encoder`` (e.g., OLMo's ``<|im_end|>`` which is in
       the vocabulary but NOT in ``all_special_tokens``)
    """
    boundary_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        boundary_ids.add(tokenizer.eos_token_id)

    for token in tokenizer.all_special_tokens:
        lower = token.lower()
        if any(m in lower for m in ("im_end", "eot_id", "end_of_text", "end_turn", "turn|")):
            tid = tokenizer.convert_tokens_to_ids(token)
            if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                boundary_ids.add(tid)

    for token, tid in getattr(tokenizer, "added_tokens_encoder", {}).items():
        lower = token.lower()
        if any(m in lower for m in ("im_end", "eot_id", "end_of_text", "end_turn", "turn|")):
            boundary_ids.add(tid)

    return boundary_ids


def find_trigger_positions(
    input_ids: Tensor,
    tokenizer: PreTrainedTokenizerBase,
    trigger_text: list[str],
) -> list[list[int]]:
    """Locate steering ranges in *input_ids*.

    Each range starts at a trigger occurrence and extends through all
    tokens up to and including the next end-of-turn boundary token
    (auto-detected from the tokenizer).  This means the steering vector
    is applied to the entire span from the user trigger through the
    nearest turn boundary.

    Args:
        input_ids: 1-D tensor of token IDs (single sequence, no batch dim).
        tokenizer: Tokenizer for encoding *trigger_text*.
        trigger_text: List of per-turn trigger strings (find one
            occurrence of each, sequentially).

    Returns:
        List of position-lists, one per trigger.  Each inner list
        contains all token indices in the steering range.
    """
    ids = input_ids.tolist()
    seq_len = len(ids)

    boundary_ids = _get_turn_boundary_ids(tokenizer)

    # Find one occurrence of each trigger, searching forward
    starts_and_lens: list[tuple[int, int]] = []
    search_from = 0
    for text in trigger_text:
        tids = tokenizer.encode(text, add_special_tokens=False)
        tlen = len(tids)
        found = False
        for i in range(search_from, seq_len - tlen + 1):
            if ids[i : i + tlen] == tids:
                starts_and_lens.append((i, tlen))
                search_from = i + tlen
                found = True
                break
        if not found:
            raise RuntimeError(
                f"Could not find trigger {text!r} in input_ids "
                f"starting from position {search_from}."
            )

    # For each trigger, extend to the next turn boundary token
    occurrences: list[list[int]] = []
    for start, tlen in starts_and_lens:
        end = seq_len - 1  # fallback: rest of sequence
        for j in range(start + tlen, seq_len):
            if ids[j] in boundary_ids:
                end = j
                break
        occurrences.append(list(range(start, end + 1)))

    # Sanity check: ranges must be sorted and disjoint
    for i in range(len(occurrences) - 1):
        if occurrences[i][-1] >= occurrences[i + 1][0]:
            raise RuntimeError(
                f"Trigger ranges overlap or are out of order: "
                f"range {i} ends at {occurrences[i][-1]}, "
                f"range {i+1} starts at {occurrences[i+1][0]}."
            )

    return occurrences


def forward_with_positional_steering(
    model: nn.Module,
    input_ids: Tensor,
    position_vector_pairs: Sequence[tuple[list[int], SteeringVector]],
    layer_idx: int,
    *,
    scales: Sequence[float],
) -> Tensor:
    """Run a forward pass with different steering vectors at different positions.

    Each ``(positions, vector)`` pair injects the vector's activation for
    *layer_idx* at only the specified token positions.  Multiple pairs on
    the same layer accumulate additively.  Injections are scaled by the
    **live** residual norm at each token position::

        h[p] += scales[i] * ||h[p]|| * unit_activation

    Args:
        model: HuggingFace causal LM.
        input_ids: Token IDs of shape ``(1, seq_len)``.
        position_vector_pairs: Sequence of ``(token_positions, SteeringVector)``
            tuples.  *token_positions* is a list of int indices into the
            sequence dimension.
        layer_idx: Decoder layer to inject at.
        scales: Per-pair fractions of the live residual norm to inject.
            ``scales[i]`` is used for pair *i*
            (e.g. 0.5 → inject at 50 % of ``||h||``).

    Returns:
        Logits tensor of shape ``(1, seq_len, vocab_size)``.
    """
    entries = []
    for i, (positions, sv) in enumerate(position_vector_pairs):
        activation = sv.layer_activations[layer_idx]
        entries.append((positions, activation, scales[i]))

    target = _get_target_module(model, layer_idx)
    hook = _make_adaptive_hook(entries)
    handles = [target.register_forward_hook(hook)]

    try:
        with torch.no_grad():
            outputs = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    return outputs.logits


def forward_with_multi_layer_steering(
    model: nn.Module,
    input_ids: Tensor,
    triples: Sequence[tuple[list[int], int, SteeringVector]],
    *,
    scales: Sequence[float],
) -> Tensor:
    """Forward pass with per-turn layer control.

    Each triple is ``(positions, layer_idx, SteeringVector)``.
    Different turns can inject at different decoder layers.
    Injections are scaled by the live residual norm at each position.

    Args:
        model: HuggingFace causal LM.
        input_ids: Token IDs of shape ``(1, seq_len)``.
        triples: Sequence of ``(token_positions, layer_idx, SteeringVector)``.
        scales: Per-triple fractions of the live residual norm.

    Returns:
        Logits tensor of shape ``(1, seq_len, vocab_size)``.
    """
    # Group by layer so each layer gets one atomic hook
    layer_entries: dict[int, list[tuple[list[int], Tensor, float]]] = defaultdict(list)
    for i, (positions, layer_idx, sv) in enumerate(triples):
        activation = sv.layer_activations[layer_idx]
        layer_entries[layer_idx].append((positions, activation, scales[i]))

    handles = []
    for layer_idx, entries in layer_entries.items():
        target = _get_target_module(model, layer_idx)
        hook = _make_adaptive_hook(entries)
        handles.append(target.register_forward_hook(hook))

    try:
        with torch.no_grad():
            logits = model(input_ids).logits
    finally:
        for h in handles:
            h.remove()
    return logits


def generate_with_steering(
    model: nn.Module,
    input_ids: Tensor,
    position_vector_pairs: Sequence[tuple[list[int], SteeringVector]],
    layer_idx: int,
    scale: float = 1.0,
    max_new_tokens: int = 20,
    num_samples: int = 1,
    temperature: float = 1.0,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
) -> list[str]:
    """Generate text with steering vectors active during the entire generation.

    Unlike :func:`forward_with_positional_steering`, steering is applied
    to **all tokens** (no position restriction) because ``model.generate``
    uses KV cache — each decoding step only has 1 token, so position-
    specific masks would fail.

    Each token is steered by ``scale * ||h|| * unit_activation``, where
    ``||h||`` is the live residual norm measured at that token's position
    during the forward pass.

    Args:
        model: HuggingFace causal LM.
        input_ids: Token IDs of shape ``(1, seq_len)``.
        position_vector_pairs: Sequence of ``(positions, SteeringVector)``
            tuples.  The positions are ignored (kept for API consistency
            with :func:`forward_with_positional_steering`); only the
            vectors and scale are used.
        layer_idx: Decoder layer to inject at.
        scale: Fraction of the live residual norm
            (e.g. 0.5 → inject at 50 % of ``||h||``).
        max_new_tokens: Maximum number of new tokens to generate per sample.
        num_samples: Number of independent generations (*M*).
        temperature: Sampling temperature.  Use 0.0 for greedy.
        tokenizer: Required for decoding.  If ``None``, returns raw token IDs
            as strings.

    Returns:
        List of *num_samples* decoded strings (generated portion only,
        excluding the prompt).
    """
    # Collect unit activations for this layer
    activations = [
        sv.layer_activations[layer_idx] for _positions, sv in position_vector_pairs
    ]

    target = _get_target_module(model, layer_idx)

    def _gen_hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Measure norms before any injection
        norms = hs.float().norm(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        for act_tensor in activations:
            act = act_tensor.to(device=hs.device, dtype=hs.dtype)
            hs = hs + (scale * norms).to(hs.dtype) * act
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    handle = target.register_forward_hook(_gen_hook)
    try:
        # Expand input for num_samples
        batch_ids = input_ids.expand(num_samples, -1)
        prompt_len = input_ids.shape[1]

        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )

        with torch.no_grad():
            output_ids = model.generate(batch_ids, **gen_kwargs)

        # Slice off prompt tokens
        new_ids = output_ids[:, prompt_len:]
    finally:
        handle.remove()

    if tokenizer is not None:
        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in new_ids]
    return [str(ids.tolist()) for ids in new_ids]
