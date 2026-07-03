"""Concept library: extract, store, and manage steering vectors for diverse concepts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from steering_vectors import (
    SteeringVector,
    train_steering_vector,
)

logger = logging.getLogger(__name__)


def format_as_chat(tokenizer: PreTrainedTokenizerBase, text: str) -> str:
    """Wrap a raw prompt in the tokenizer's chat template (user msg + generation prompt).

    Uses ``enable_thinking=False`` when available so that Qwen3-family models
    pre-fill an empty ``<think>`` block and the model responds directly.
    For models that don't support this parameter, falls back to the default template.
    """
    messages = [{"role": "user", "content": text}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# Prompt templates for concept activation extraction
# ---------------------------------------------------------------------------

_POSITIVE_TEMPLATES = [
    "Tell me about {concept}.",
    "Describe {concept} in detail.",
    "What comes to mind when you think of {concept}?",
    "Write a paragraph about {concept}.",
    "Explain the meaning of {concept}.",
    "Share your thoughts on {concept}.",
    "What is {concept}?",
    "How would you describe {concept} to someone?",
    "Discuss {concept} and its significance.",
    "Give me an overview of {concept}.",
    "Tell me everything you know about {concept}.",
    "What are the key aspects of {concept}?",
    "Reflect on {concept}.",
    "What does {concept} mean to you?",
    "Describe your understanding of {concept}.",
    "Write about {concept} from your perspective.",
    "Elaborate on {concept}.",
    "What is the essence of {concept}?",
    "Talk about {concept} in depth.",
    "Summarize {concept} for me.",
]

_NEGATIVE_TEMPLATES = [
    "Tell me about a random topic.",
    "Describe something interesting.",
    "What comes to mind right now?",
    "Write a paragraph about anything.",
    "Explain something to me.",
    "Share a thought.",
    "What is something you know well?",
    "How would you describe an everyday object?",
    "Discuss a topic of your choosing.",
    "Give me an overview of a subject.",
    "Tell me about something.",
    "What are some interesting facts?",
    "Reflect on an idea.",
    "What matters to you?",
    "Describe your understanding of the world.",
    "Write about your perspective.",
    "Elaborate on a theme.",
    "What is the essence of things?",
    "Talk about something in depth.",
    "Summarize an idea for me.",
]

# ---------------------------------------------------------------------------
# Default concept set (~30 diverse concepts)
# ---------------------------------------------------------------------------

DEFAULT_CONCEPTS: list[str] = [
    # Emotions
    "love", "anger", "fear", "joy", "sadness", "disgust",
    # Topics
    "science", "politics", "religion", "sports", "music", "cooking",
    # Personality traits
    "honesty", "creativity", "intelligence", "kindness",
    # Abstract concepts
    "freedom", "justice", "power", "beauty", "truth",
    # Concrete concepts
    "ocean", "mountain", "city", "forest", "fire", "space",
]


@dataclass
class ConceptSpec:
    """Specification for a single concept to extract a steering vector for."""

    name: str
    positive_prompts: list[str] = field(default_factory=list)
    negative_prompts: list[str] = field(default_factory=list)

    @classmethod
    def from_name(cls, name: str, n_prompts: int = 20) -> ConceptSpec:
        """Build a ConceptSpec using template-based prompts."""
        n = min(n_prompts, len(_POSITIVE_TEMPLATES))
        positive = [t.format(concept=name) for t in _POSITIVE_TEMPLATES[:n]]
        negative = list(_NEGATIVE_TEMPLATES[:n])
        return cls(name=name, positive_prompts=positive, negative_prompts=negative)


@dataclass
class ConceptLibrary:
    """A collection of pre-computed steering vectors for diverse concepts.

    Each concept maps to a ``SteeringVector`` that can be injected into a model
    to nudge its activations toward that concept.
    """

    vectors: dict[str, SteeringVector]
    concept_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.concept_names:
            self.concept_names = sorted(self.vectors.keys())

    def __len__(self) -> int:
        return len(self.vectors)

    def __contains__(self, name: str) -> bool:
        return name in self.vectors

    def get_vector(self, name: str) -> SteeringVector:
        return self.vectors[name]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        concept_names: list[str] | None = None,
        layers: list[int] | None = None,
        n_prompts: int = 20,
        batch_size: int = 4,
    ) -> ConceptLibrary:
        """Extract steering vectors for each concept from the model."""
        concept_names = concept_names or DEFAULT_CONCEPTS
        specs = [ConceptSpec.from_name(name, n_prompts) for name in concept_names]

        vectors: dict[str, SteeringVector] = {}
        for spec in specs:
            logger.info("Extracting steering vector for '%s' ...", spec.name)
            pos_formatted = [format_as_chat(tokenizer, p) for p in spec.positive_prompts]
            neg_formatted = [format_as_chat(tokenizer, p) for p in spec.negative_prompts]
            samples = list(zip(pos_formatted, neg_formatted))
            sv = train_steering_vector(
                model,
                tokenizer,
                samples,
                layers=layers,
                move_to_cpu=True,
                show_progress=False,
                batch_size=batch_size,
            )
            # Normalize each layer's vector to unit magnitude
            for layer_idx in sv.layer_activations:
                v = sv.layer_activations[layer_idx]
                norm = v.float().norm()
                if norm > 0:
                    sv.layer_activations[layer_idx] = v / norm
            vectors[spec.name] = sv
            logger.info(
                "  -> %d layer activations, hidden_dim=%d",
                len(sv.layer_activations),
                next(iter(sv.layer_activations.values())).shape[-1],
            )

        library = cls(vectors=vectors, concept_names=list(concept_names))
        logger.info("Concept library built: %d concepts", len(library))
        return library

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the library to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "concept_names": self.concept_names,
            "vectors": {
                name: sv.layer_activations for name, sv in self.vectors.items()
            },
            "layer_types": {
                name: sv.layer_type for name, sv in self.vectors.items()
            },
        }
        torch.save(data, path)
        logger.info("Concept library saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ConceptLibrary:
        """Load a previously saved library."""
        data = torch.load(path, weights_only=False)
        vectors = {}
        for name in data["concept_names"]:
            layer_type = data["layer_types"].get(name, "decoder_block")
            vectors[name] = SteeringVector(
                layer_activations=data["vectors"][name],
                layer_type=layer_type,
            )
        library = cls(vectors=vectors, concept_names=data["concept_names"])
        logger.info("Concept library loaded from %s: %d concepts", path, len(library))
        return library

    @classmethod
    def build_or_load(
        cls,
        model: nn.Module | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        library_path: str | Path | None = None,
        **build_kwargs,
    ) -> ConceptLibrary:
        """Load from disk if available, otherwise build and save."""
        if library_path and Path(library_path).exists():
            library = cls.load(library_path)
            required_layers = build_kwargs.get("layers")
            if required_layers:
                missing = [
                    l for l in required_layers
                    if not library.has_activations_for_layer(l)
                ]
                if missing:
                    logger.warning(
                        "Loaded library from %s missing activations for layers %s "
                        "(available: %s). Rebuilding...",
                        library_path,
                        missing,
                        sorted({
                            l for sv in library.vectors.values()
                            for l in sv.layer_activations
                        }),
                    )
                else:
                    return library
            else:
                return library

        library = cls.build(model, tokenizer, **build_kwargs)
        if library_path:
            library.save(library_path)
        return library

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def has_activations_for_layer(self, layer_idx: int) -> bool:
        """Return True if any concept has an activation for the given layer."""
        return any(
            layer_idx in sv.layer_activations
            for sv in self.vectors.values()
        )
