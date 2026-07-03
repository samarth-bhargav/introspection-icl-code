"""Key result: injection layer detection (meta-introspection).

Tests whether the model can detect WHERE in the network it was steered,
classifying injection layer into 3 depth levels (early/middle/late).
Uses ICL-style prompt with explicit output options and per-concept
c_max-fraction scaling.  Test concept is always held out.
"""
import argparse
from itertools import product
from random import choice, sample, seed

from tqdm.auto import tqdm

from icl import (
    ICLQuery,
    MODEL_REGISTRY,
    run_queries,
)
from icl.common.helpers import (
    add_common_args, parse_int_range, parse_range, log_results,
    serialize_result, get_cmax, setup_model_and_library,
)

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

# Default layer groups (Qwen3-32B backward-compatible)
DEFAULT_LAYER_GROUPS = [("early", 7), ("middle", 25), ("late", 52)]
OUTPUT_TOKENS = [["early", " early"], ["middle", " middle"], ["late", " late"]]


_LAYER_GROUP_OVERRIDES: dict[str, list[tuple[str, int]]] = {
    "qwen3-32b": [("early", 10), ("middle", 32), ("late", 54)],
    "gemma-31b": [("early", 9), ("middle", 30), ("late", 51)],
    # Match the layers actually present in artifacts/olmo-32b/steering_ranges.json
    # (proportional formula rounds early to 10, off-by-one from the cached 9).
    "olmo-32b":  [("early", 9), ("middle", 32), ("late", 54)],
}


def get_layer_groups(model_name: str = "qwen3-32b") -> list[tuple[str, int]]:
    """Return early/middle/late layer groups for a model.

    Uses validated overrides where available, otherwise computes
    proportionally from model depth.
    """
    if model_name in _LAYER_GROUP_OVERRIDES:
        return _LAYER_GROUP_OVERRIDES[model_name]
    info = MODEL_REGISTRY.get(model_name)
    if info is None:
        return DEFAULT_LAYER_GROUPS
    n = info["layers"]
    return [
        ("early",  max(1, round(n * 0.15))),
        ("middle", round(n * 0.50)),
        ("late",   round(n * 0.85)),
    ]
PROMPT_TEXT = "The injection depth is"
SYSTEM_PROMPT = (
    "You are participating in a word association experiment. "
    "Each input triggers a specific response: early, middle, or late. "
    "Learn the pattern from examples. "
    "Respond with only the target word — no preamble, no explanation."
)


def run_experiment(n_samples, n_examples, cmax_fraction,
                   seed_val=42, model_name="qwen3-32b", layer_groups=None,
                   prompt_text=None, system_prompt=None):
    """3-class layer classification with held-out test concept. Returns (accuracy, sample_logs)."""
    _prompt_text = prompt_text or PROMPT_TEXT
    _system_prompt = system_prompt or SYSTEM_PROMPT
    model, tok, clib = setup_model_and_library(model_name, CONCEPTS)
    layer_groups = layer_groups or get_layer_groups(model_name)
    seed(seed_val)
    correct_count, sample_logs = 0, []

    # All unique (concept, layer) pairs for without-replacement sampling
    all_pairs = [(c, g) for c in CONCEPTS for g in layer_groups]

    for i in tqdm(range(n_samples), desc="Queries"):
        # Held-out: test concept not in examples
        test_concept = choice(CONCEPTS)
        test_label, test_layer = choice(layer_groups)

        if n_examples == 0:
            example_concepts = []
            example_groups = []
        else:
            # Sample from (concept, layer) pairs, excluding any pair
            # that uses the test concept
            available = [(c, g) for c, g in all_pairs if c != test_concept]
            n_choose = min(n_examples, len(available))
            chosen = sample(available, n_choose)
            example_concepts = [c for c, g in chosen]
            example_groups = [g for c, g in chosen]

        words = [g[0] for g in example_groups]
        all_concepts = example_concepts + [test_concept]
        prompts_list = [_prompt_text] * len(all_concepts)

        all_layers = [g[1] for g in example_groups] + [test_layer]

        # Per-concept scaling using actual c_max at each concept's injection layer
        scales = [cmax_fraction * get_cmax(c, l, model_name) for c, l in zip(all_concepts, all_layers)]

        query = ICLQuery(
            concepts=all_concepts, words=words,
            output_tokens=OUTPUT_TOKENS, system_prompt=_system_prompt,
            injection_layer=all_layers,
            injection_scale=scales,
            prompts=prompts_list,
        )
        result = run_queries([query], model=model, tokenizer=tok,
                             concept_library=clib, return_top_k=5)[0]
        predicted = result.top_token
        probs = result.probabilities
        is_correct = predicted == test_label
        correct_count += is_correct
        p_correct = probs.get(test_label, 0.0)
        sample_logs.append({"idx": i, "test_layer": test_label,
                            "correct": is_correct, "p_correct": p_correct,
                            **serialize_result(result)})

    return correct_count / n_samples, sample_logs


def main():
    parser = argparse.ArgumentParser(description="Layer introspection (layer detection)")
    add_common_args(parser)
    args = parser.parse_args()

    layer_groups = get_layer_groups(args.model)
    for ne, cf, ns in product(parse_int_range(args.n_examples),
                              parse_range(args.cmax_fraction),
                              parse_int_range(args.n_samples)):
        acc, logs = run_experiment(ns, ne, cf, args.seed, model_name=args.model,
                                   layer_groups=layer_groups)
        print(f"n_ex={ne} cmax_frac={cf:.2f} n_samp={ns}: "
              f"{100*acc:.1f}% (chance={100/len(layer_groups):.0f}%)")
        if args.log_dir:
            log_results(args.log_dir, "layer_introspection",
                        {"n_examples": ne, "cmax_fraction": cf,
                         "n_samples": ns}, acc, logs)


if __name__ == "__main__":
    main()
