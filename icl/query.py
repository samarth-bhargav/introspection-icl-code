"""ICL Query/Result dataclasses and the main ``run_queries`` entry-point."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from tqdm.auto import tqdm

from icl.steering.concepts import ConceptLibrary
from icl.model import (
    get_model_and_tokenizer,
    get_concept_library,
    ensure_concepts,
    DEFAULT_MODEL_NAME,
    DEFAULT_DEVICE,
    DEFAULT_LIBRARY_PATH,
)

from .logits import extract_token_logits
from .steering.injection import (
    find_trigger_positions,
    forward_with_multi_layer_steering,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query / Result
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT: str = "You are participating in a word association experiment."


@dataclass
class ICLQuery:
    """Specification for a single ICL steering experiment.

    The conversation has ``n = len(concepts)`` user turns (each saying
    ``"This is"`` with a steering vector injected) and ``n - 1`` completed
    assistant turns (each saying one of *words*).  The model predicts the
    final assistant token.

    Attributes:
        concepts: Length-*n* list of concept names; ``concepts[i]`` selects
            the steering vector injected at user turn *i*.
        words: Length-*(n-1)* list of words the assistant says at
            completed turns.
        output_tokens: Token groups to extract logits for.  Each group is
            a list of token variants whose probabilities are summed.
            E.g. ``[["queen", " queen"], ["king", " king"]]``.
        injection_layer: Per-concept decoder layers for injection.
            Length must equal ``len(concepts)``.
        injection_scale: Per-concept steering scale (fraction of live
            residual norm).  Length must equal ``len(concepts)``.
        system_prompt: System-level instruction.
        prompts: Per-turn user prompt text.  ``None`` = ``"This is"``
            for every turn.
    """

    concepts: list[str]
    words: list[str]
    output_tokens: list[list[str]]
    injection_layer: list[int]
    injection_scale: list[float]
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompts: list[str] | None = None
    # Filler text appended to the assistant's generation prefix on the
    # final turn (after `add_generation_prompt`).  Used to test the
    # "Let's think dot by dot" hypothesis (Pfau et al., 2024): give the
    # model N filler tokens of compute before it must commit to an
    # answer.  When set, the model predicts the token *immediately
    # after* the prefill string.  Note: the prefill is part of the
    # tokenized input, so the predicted next-token logit lives at
    # position `prompt_len + len(prefill_tokens)` (which is just the
    # last token in our forward pass).
    prefill: str | None = None
    # Override the `enable_thinking` argument to apply_chat_template.
    # `None` (default) preserves the current behaviour
    # (`enable_thinking=False`).  Set to `True` for Gemma-4 to skip the
    # auto-inserted `<|channel>thought\n<channel|>` block at the
    # generation prompt — the prediction position then sits in the
    # raw model channel, matching the position of completed assistant
    # turns and enabling structurally-aligned ICL with prefilled
    # filler tokens.
    enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        # Allow bare strings — auto-wrap into single-element groups
        self.output_tokens = [
            [t] if isinstance(t, str) else t for t in self.output_tokens
        ]
        if len(self.concepts) != len(self.words) + 1:
            raise ValueError(
                f"len(concepts) must equal len(words) + 1, "
                f"got {len(self.concepts)} concepts and {len(self.words)} words"
            )
        if len(self.injection_layer) != len(self.concepts):
            raise ValueError(
                f"len(injection_layer) must equal len(concepts), "
                f"got {len(self.injection_layer)} layers and {len(self.concepts)} concepts"
            )
        if len(self.injection_scale) != len(self.concepts):
            raise ValueError(
                f"len(injection_scale) must equal len(concepts), "
                f"got {len(self.injection_scale)} scales and {len(self.concepts)} concepts"
            )
        if self.prompts is not None and len(self.prompts) != len(self.concepts):
            raise ValueError(
                f"len(prompts) must equal len(concepts), "
                f"got {len(self.prompts)} prompts and {len(self.concepts)} concepts"
            )


@dataclass
class ICLResult:
    """Output of a single ICL query.

    Attributes:
        query: The original query.
        logits: Raw logit values for each requested output token.
        probabilities: Softmax probabilities (normalized over output tokens only).
        top_token: The output token with the highest probability.
    """

    query: ICLQuery
    logits: dict[str, float]
    probabilities: dict[str, float]
    top_token: str
    top_k_tokens: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_TEXT = "This is"


def build_prompt(query: ICLQuery, tokenizer: PreTrainedTokenizerBase) -> str:
    """Build the full prompt string from an ICL query using the tokenizer's chat template.

    Constructs a multi-turn conversation and formats it via
    ``tokenizer.apply_chat_template()``.  Uses ``enable_thinking=False``
    when available (Qwen3) so the model skips reasoning and responds
    directly.
    """
    n_user_turns = len(query.words) + 1
    prompts = query.prompts if query.prompts is not None else [DEFAULT_PROMPT_TEXT] * n_user_turns

    messages: list[dict[str, str]] = [{"role": "system", "content": query.system_prompt}]
    for i in range(n_user_turns):
        messages.append({"role": "user", "content": prompts[i]})
        if i < len(query.words):
            messages.append({"role": "assistant", "content": query.words[i]})

    kwargs = dict(tokenize=False, add_generation_prompt=True)
    enable_thinking = False if query.enable_thinking is None else query.enable_thinking
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=enable_thinking)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)

    if query.prefill is not None and query.prefill != "":
        rendered = rendered + query.prefill
    return rendered


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def run_queries(
    queries: list[ICLQuery],
    model: nn.Module | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    concept_library: ConceptLibrary | None = None,
    device: str | torch.device | None = None,
    return_top_k: int | None = None,
) -> list[ICLResult]:
    """Run one or more ICL steering queries and return logit results.

    Args:
        queries: List of :class:`ICLQuery` instances.
        model: HuggingFace causal LM.  ``None`` = lazy-load default.
        tokenizer: HuggingFace tokenizer.  ``None`` = lazy-load default.
        concept_library: Pre-built library.  ``None`` = lazy-load/build.
        device: Device for input tensors.  ``None`` = model's device.
        return_top_k: If set, each result includes ``top_k_tokens``: a dict
            of the *k* most likely tokens (from the full vocab) with their
            probabilities.

    Returns:
        List of :class:`ICLResult`, one per query.
    """
    # -- Resolve defaults ------------------------------------------------
    if model is None or tokenizer is None:
        model, tokenizer = get_model_and_tokenizer()
    if concept_library is None:
        concept_library = get_concept_library(model=model, tokenizer=tokenizer)
    if device is None:
        device = next(model.parameters()).device

    # -- Ensure all needed concepts exist --------------------------------
    all_concepts = list({c for q in queries for c in q.concepts})
    concept_library = ensure_concepts(
        concept_library, all_concepts, model=model, tokenizer=tokenizer,
    )

    # -- Process queries -------------------------------------------------
    results: list[ICLResult] = []
    for query in tqdm(queries, desc="Running queries"):
        result = _run_single_query(
            query, model, tokenizer, concept_library, device,
            return_top_k=return_top_k,
        )
        results.append(result)

    return results

def _run_single_query(
    query: ICLQuery,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    concept_library: ConceptLibrary,
    device: str | torch.device,
    return_top_k: int | None = None,
) -> ICLResult:
    """Execute a single ICL query."""
    # 1. Build prompt and tokenize
    prompt_str = build_prompt(query, tokenizer)
    input_ids = tokenizer.encode(
        prompt_str, return_tensors="pt", add_special_tokens=False,
    ).to(device)  # (1, seq_len)

    # 2. Find trigger positions
    trigger_text = query.prompts if query.prompts is not None else [DEFAULT_PROMPT_TEXT] * len(query.concepts)
    trigger_positions = find_trigger_positions(input_ids[0], tokenizer, trigger_text)
    if len(trigger_positions) != len(query.concepts):
        raise RuntimeError(
            f"Expected {len(query.concepts)} trigger occurrences, "
            f"found {len(trigger_positions)}.  The chat template may be "
            f"interfering with trigger detection."
        )

    # 3. Build (positions, layer, steering_vector) triples
    triples = []
    for positions, concept_name, layer_idx in zip(
        trigger_positions, query.concepts, query.injection_layer,
    ):
        sv = concept_library.get_vector(concept_name)
        triples.append((positions, layer_idx, sv))

    # 4. Forward pass with steering
    logits = forward_with_multi_layer_steering(
        model, input_ids, triples, scales=query.injection_scale,
    )

    # 5. Extract token logits
    logits_dict, probs_dict = extract_token_logits(logits, query.output_tokens, tokenizer)
    top_token = max(probs_dict, key=probs_dict.get)  # type: ignore[arg-type]

    # 6. Optionally get top-k from full vocab
    top_k_tokens = None
    if return_top_k is not None:
        last_logits = logits[0, -1, :].float()
        full_probs = torch.softmax(last_logits, dim=0)
        topk_vals, topk_ids = torch.topk(full_probs, return_top_k)
        top_k_tokens = {
            tokenizer.decode([tid]): val.item()
            for tid, val in zip(topk_ids.tolist(), topk_vals)
        }

    return ICLResult(
        query=query,
        logits=logits_dict,
        probabilities=probs_dict,
        top_token=top_token,
        top_k_tokens=top_k_tokens,
    )
