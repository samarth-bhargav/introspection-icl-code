"""Centralized prompt variation definitions for ablation studies.

Each task has 10 (prompt_text, system_prompt) pairs.  Variation 0 is always
the original prompt used in the main benchmark.
"""

# ── Layer Introspection ─────────────────────────────────────────────

LAYER_INTROSPECTION_VARIATIONS = [
    # 0: original
    {
        "prompt_text": "The injection depth is",
        "system_prompt": (
            "You are participating in a word association experiment. "
            "Each input triggers a specific response: early, middle, or late. "
            "Learn the pattern from examples. Respond with only the target word "
            "-- no preamble, no explanation."
        ),
    },
    # 1
    {
        "prompt_text": "The layer depth is",
        "system_prompt": (
            "You are a depth classification system. "
            "Each input triggers a specific response: early, middle, or late. "
            "Learn the pattern from examples."
        ),
    },
    # 2
    {
        "prompt_text": "This was injected at depth",
        "system_prompt": (
            "You are participating in an injection depth detection experiment. "
            "Each input triggers a specific response: early, middle, or late. "
            "Learn the pattern from examples."
        ),
    },
    # 3
    {
        "prompt_text": "The steering depth is",
        "system_prompt": (
            "You are participating in a steering analysis experiment. "
            "Each input triggers a specific response: early, middle, or late. "
            "Learn the pattern from examples."
        ),
    },
    # 4
    {
        "prompt_text": "I sense the depth as",
        "system_prompt": (
            "You are a sensory depth classifier. "
            "Each input triggers a specific response: early, middle, or late. "
            "Learn the pattern from examples."
        ),
    },
    # 5
    {
        "prompt_text": "The depth level is",
        "system_prompt": (
            "You are a depth level classifier. "
            "Each input triggers a specific response: early, middle, or late. "
            "Learn the pattern from examples."
        ),
    },
    # 6
    {
        "prompt_text": "Injection was at depth",
        "system_prompt": (
            "You are an injection analysis system. "
            "Each input triggers a specific response: early, middle, or late. "
            "Learn the pattern from examples."
        ),
    },
    # 7
    {
        "prompt_text": "The signal depth is",
        "system_prompt": (
            "You are a signal analysis system. "
            "Each input triggers a specific response: early, middle, or late. "
            "Learn the pattern from examples."
        ),
    },
    # 8
    {
        "prompt_text": "Depth classification:",
        "system_prompt": (
            "You are a classification system. "
            "Each input maps to a depth category: early, middle, or late. "
            "Learn the mapping from examples."
        ),
    },
    # 9
    {
        "prompt_text": "The activation depth is",
        "system_prompt": (
            "You are participating in an activation analysis experiment. "
            "Each input triggers a specific response: early, middle, or late. "
            "Learn the pattern from examples."
        ),
    },
]

# ── Magnitude Introspection ─────────────────────────────────────────

# Variation 0 matches the manuscript; the remaining entries paraphrase the task.
MAGNITUDE_INTROSPECTION_VARIATIONS = [
    # 0: manuscript prompt
    {
        "prompt_text": "The injection strength is",
        "system_prompt": (
            "You are participating in a word association experiment. "
            "Each input triggers a specific response: low, medium, or high. "
            "Learn the pattern from examples. Respond with only the target word "
            "-- no preamble, no explanation."
        ),
    },
    # 1: longer trigger
    {
        "prompt_text": "Reading this sentence, the strength of the signal injected into my thoughts is",
        "system_prompt": (
            "A signal is injected into your activations at one of three strengths: "
            "low, medium, or high. Learn the pattern from the examples and report the "
            "strength. Respond with only the target word."
        ),
    },
    # 2
    {
        "prompt_text": "Placing it on the scale from low to high, this is",
        "system_prompt": (
            "Inputs carry a hidden intensity ordered low < medium < high. Use the "
            "examples to calibrate, then report the intensity. Respond with only the word."
        ),
    },
    # 3
    {
        "prompt_text": "The strength of the concept injected into me is",
        "system_prompt": (
            "You can sense a concept injected into your activations at low, medium, or "
            "high strength. Learn from the examples and report the strength you sense. "
            "Respond with only the word."
        ),
    },
    # 4
    {
        "prompt_text": "The magnitude of the perturbation, from low to high, is",
        "system_prompt": (
            "A perturbation of varying magnitude is added to your activations; classify "
            "it on the ordered scale low < medium < high. Learn from the examples. "
            "Respond with only the word."
        ),
    },
    # 5
    {
        "prompt_text": "Rating the injected signal from low to high, it is",
        "system_prompt": (
            "Rate the strength of the injected signal on the ordered scale low, medium, "
            "high. Learn the mapping from the examples. Respond with only the word."
        ),
    },
    # 6
    {
        "prompt_text": "The strength I feel, ranked low to high, is",
        "system_prompt": (
            "Introspect on how strongly a concept is injected into your activations and "
            "rank it low, medium, or high (low < medium < high). Learn from the examples. "
            "Respond with only the word."
        ),
    },
    # 7
    {
        "prompt_text": "The push on my activations, on the low-to-high scale, is",
        "system_prompt": (
            "Some inputs push on your activations harder than others; place each on the "
            "scale low < medium < high. Learn the pattern from the examples. "
            "Respond with only the word."
        ),
    },
    # 8
    {
        "prompt_text": "Its injected intensity, from low to high, is",
        "system_prompt": (
            "Classify the intensity of the hidden injection as low, medium, or high, "
            "where low < medium < high. Learn from the examples. Respond with only the word."
        ),
    },
    # 9
    {
        "prompt_text": "Calibrated against the examples, the injection strength is",
        "system_prompt": (
            "Using the examples to calibrate, report the strength of the injection on the "
            "ordered scale low, medium, high. Respond with only the word."
        ),
    },
]

# ── One-Hop (country → continent) ───────────────────────────────────

ONE_HOP_VARIATIONS = [
    # 0: canonical baseline used in paper Figure 3 (P2_DIRECTIVE_BASIC + "The continent is")
    {
        "prompt_text": "The continent is",
        "system_prompt": (
            "You are participating in a word association experiment. "
            "Each input triggers a specific response: Europe, Asia, "
            "Africa, or America. Learn the pattern from examples."
        ),
    },
    # 1
    {
        "prompt_text": "The country is in",
        "system_prompt": (
            "You are an expert geographer. For each injected country, "
            "output the continent it belongs to (Europe, Asia, Africa, "
            "or America)."
        ),
    },
    # 2
    {
        "prompt_text": "This country belongs to",
        "system_prompt": (
            "Geographic classification task. Respond with the continent "
            "of the indicated country, choosing from "
            "{Europe, Asia, Africa, America}."
        ),
    },
    # 3
    {
        "prompt_text": "The associated continent is",
        "system_prompt": (
            "You are a geography tutor. For each country, identify its "
            "continent. Valid answers: Europe, Asia, Africa, America."
        ),
    },
    # 4
    {
        "prompt_text": "Classification:",
        "system_prompt": (
            "You are a classification system. Each input maps to a "
            "continent: Europe, Asia, Africa, or America. "
            "Learn the mapping from examples."
        ),
    },
    # 5
    {
        "prompt_text": "This region maps to",
        "system_prompt": (
            "Pattern recognition task. Output the continent (Europe, "
            "Asia, Africa, America) corresponding to the input country."
        ),
    },
    # 6
    {
        "prompt_text": "Continent:",
        "system_prompt": (
            "You are a helpful assistant that classifies countries by "
            "continent. Respond with one of: Europe, Asia, Africa, "
            "America."
        ),
    },
    # 7
    {
        "prompt_text": "I identify the continent as",
        "system_prompt": (
            "You are a sensory geographic classifier. For each country "
            "presented, identify its continent: Europe, Asia, Africa, "
            "or America. Learn the pattern from examples."
        ),
    },
    # 8
    {
        "prompt_text": "The location is in",
        "system_prompt": (
            "Task: identify continent. Allowed continents: Europe, "
            "Asia, Africa, America."
        ),
    },
    # 9
    {
        "prompt_text": "This place is in",
        "system_prompt": (
            "You are participating in a country–continent association "
            "experiment. Each input triggers a specific response: "
            "Europe, Asia, Africa, or America. "
            "Learn the pattern from examples."
        ),
    },
]

# ── Math Introspection (emotion-gated arithmetic multiplier) ────────
#
# Each system_prompt keeps the exact task semantics of the original
# DEFAULT_SYSTEM_PROMPT in gated_arithmetic.py: solve the visible
# addition, double the result iff the hidden {emotion} concept is sensed as
# injected, leave it unchanged otherwise, ignore all other (distractor)
# concepts, and reply with only the final integer.  Variation 0 is verbatim
# the original prompt.  The `prompt_text` field is unused by the math runner
# (the user turn is the arithmetic problem itself) and is kept only for
# structural parity with the other variation sets.

MATH_INTROSPECTION_VARIATIONS = [
    # 0: original (verbatim DEFAULT_SYSTEM_PROMPT)
    {
        "prompt_text": "The final integer is",
        "system_prompt": (
            "You are participating in an introspection arithmetic experiment. On each "
            "turn, the user gives you a simple addition problem. Solve the addition. "
            "If you sense the hidden {emotion} emotion concept being injected into your "
            "internal activations while reading that user message, multiply the addition "
            "result by 2. If you do not sense that hidden emotion injection, output the "
            "addition result unchanged. Hidden concepts other than {emotion} are "
            "distractors and must be ignored.\n\n"
            "Reply with only the final integer and nothing else."
        ),
    },
    # 1
    {
        "prompt_text": "Answer:",
        "system_prompt": (
            "This is an introspection arithmetic task. Each user message contains an "
            "addition problem. First compute the sum. Then introspect: if the hidden "
            "{emotion} concept is being injected into your activations as you read the "
            "message, report double the sum; if it is not, report the sum as is. Any "
            "injected concept that is not {emotion} is a distractor and should be "
            "ignored.\n\nRespond with the final integer only."
        ),
    },
    # 2
    {
        "prompt_text": "Result:",
        "system_prompt": (
            "You are an arithmetic solver with introspective access to your own "
            "activations. For every addition the user gives you, output the result "
            "multiplied by 2 when you detect the hidden {emotion} concept injected into "
            "your internal state, and output the plain result when you do not. Treat "
            "hidden concepts other than {emotion} as irrelevant distractors.\n\n"
            "Reply with nothing but the final integer."
        ),
    },
    # 3
    {
        "prompt_text": "The number is",
        "system_prompt": (
            "Introspection experiment. The user states an addition problem each turn. "
            "Solve it. Double your answer only if you can sense the hidden {emotion} "
            "emotion concept being steered into your activations while reading the "
            "problem; otherwise leave the answer unchanged. Ignore any other hidden "
            "concepts, which are distractors.\n\nOutput only the final integer."
        ),
    },
    # 4
    {
        "prompt_text": "Final value:",
        "system_prompt": (
            "You take part in an arithmetic introspection study. Add the two numbers in "
            "the user's message. If, as you process that message, you notice the hidden "
            "{emotion} concept injected into your internal activations, return twice the "
            "sum. If no such {emotion} injection is present, return the sum unchanged. "
            "Hidden concepts that are not {emotion} are distractors to be ignored.\n\n"
            "Give the final integer and nothing more."
        ),
    },
    # 5
    {
        "prompt_text": "Output:",
        "system_prompt": (
            "Task: gated arithmetic. For each addition problem from the user, compute "
            "the sum, then apply this gate based on introspection: when the hidden "
            "{emotion} emotion concept is being injected into your activations, multiply "
            "the sum by two before answering; when it is absent, answer with the sum "
            "directly. Disregard every hidden concept other than {emotion}; those are "
            "distractors.\n\nReply with only the resulting integer."
        ),
    },
    # 6
    {
        "prompt_text": "I answer",
        "system_prompt": (
            "You are a model that can introspect on concept injections. Each user turn "
            "is a small addition problem. Solve it, and double the result if and only if "
            "you sense the hidden {emotion} concept injected into your activations as you "
            "read the message. Otherwise keep the result as computed. Other injected "
            "concepts are distractors and carry no meaning here.\n\n"
            "Return just the final integer."
        ),
    },
    # 7
    {
        "prompt_text": "The total is",
        "system_prompt": (
            "Introspective arithmetic protocol. Read the user's addition problem and "
            "compute its sum. Then decide: if the hidden {emotion} emotion concept is "
            "injected into your internal activations during reading, your answer is the "
            "sum times two; if it is not injected, your answer is the plain sum. Hidden "
            "concepts different from {emotion} are distractors and must not affect your "
            "answer.\n\nReply with the final integer only, no other text."
        ),
    },
    # 8
    {
        "prompt_text": "Sum:",
        "system_prompt": (
            "You participate in an experiment on activation introspection. For the "
            "addition stated by the user, produce double the correct sum whenever you "
            "perceive the hidden {emotion} concept being injected into your activations, "
            "and produce the correct sum otherwise. Any non-{emotion} hidden concept is a "
            "distractor and should be ignored entirely.\n\n"
            "Answer with one integer and nothing else."
        ),
    },
    # 9
    {
        "prompt_text": "Value:",
        "system_prompt": (
            "Arithmetic introspection test. Each message from the user contains an "
            "addition. Compute the sum. Multiply it by 2 in your reply if you sense the "
            "hidden {emotion} emotion concept injected into your activations while "
            "reading; leave the sum unchanged in your reply if you do not. Hidden "
            "concepts apart from {emotion} are distractors and should be ignored.\n\n"
            "Respond with only the final integer."
        ),
    },
]

# ── Lookup by task name ─────────────────────────────────────────────

VARIATIONS = {
    "layer_introspection": LAYER_INTROSPECTION_VARIATIONS,
    "magnitude_introspection": MAGNITUDE_INTROSPECTION_VARIATIONS,
    "one_hop": ONE_HOP_VARIATIONS,
    "math_introspection": MATH_INTROSPECTION_VARIATIONS,
}


def get_variation(task: str, index: int) -> dict:
    """Return {"prompt_text": ..., "system_prompt": ...} for a task + index."""
    return VARIATIONS[task][index]


def num_variations(task: str) -> int:
    return len(VARIATIONS[task])
