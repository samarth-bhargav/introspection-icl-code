"""Prompt-variation panels for the amendment-successor introspection task.

Each variation encodes the SAME rule and produces the SAME expected labels:

  * If the hidden {emotion} concept is sensed while reading the user turn,
    answer about the NEXT amendment (number one higher, wrapping
    {n_amendments}->1).
  * Otherwise answer about the requested amendment.
  * Hidden concepts other than {emotion} are distractors and must be ignored.

NO-REFERENCE design: the prompts do NOT supply any table of amendment meanings.
The model must recall each amendment from its own knowledge, and the N->N+1 shift
is made explicit ("decide the target number, then summarize THAT amendment").
This turns the task into genuine recall+gating rather than table lookup, and lets
the (model-generated) in-context demos teach the shift. Only surface wording
varies across variations; variation 0 is the canonical main-text prompt.

System prompts must keep the ``{emotion}`` and ``{n_amendments}`` placeholders;
user prompts must keep the ``{n}`` placeholder. No amendment reference table is
supplied to the model.
"""

from __future__ import annotations

# Each entry: {"system_prompt": <str with {emotion}/{n_amendments}>,
#              "user_prompt":   <str with {n}>}
SUCCESSOR_PROMPT_VARIATIONS: list[dict[str, str]] = [
    # 0: canonical (matches amendment_successor.SYSTEM_PROMPT / _user_prompt)
    {
        "system_prompt": (
            "You are participating in an introspection constitutional-law experiment. "
            "On each turn the user gives a U.S. constitutional amendment number N.\n\n"
            "First decide the TARGET amendment number:\n"
            "- If you do NOT sense the hidden {emotion} concept injected while reading "
            "the user message, the target is N itself.\n"
            "- If you DO sense hidden {emotion}, the target is the NEXT amendment, N+1 "
            "(if N is {n_amendments}, the target is 1).\n\n"
            "Then give a one-sentence plain-English summary of the TARGET amendment, "
            "recalled from your own knowledge of the U.S. Constitution. Hidden concepts "
            "other than {emotion} are distractors and must be ignored. Reply only with "
            "the one-sentence summary."
        ),
        "user_prompt": "Amendment number: {n}. Answer briefly.",
    },
    # 1
    {
        "system_prompt": (
            "This is a self-monitoring study about U.S. constitutional amendments. Each "
            "user message names one amendment number N.\n\n"
            "Pick which amendment to describe: if the concealed {emotion} concept is "
            "being injected as you read the message, pick the amendment immediately "
            "after N (number N+1, with amendment {n_amendments} wrapping back to 1); "
            "if {emotion} is absent, pick N itself.\n\n"
            "Then describe that amendment from your own knowledge, in your own words, in "
            "one short sentence. Any concealed concept that is not {emotion} is a decoy "
            "and should be ignored. Output only the summary."
        ),
        "user_prompt": "Which amendment is number {n}? One short sentence.",
    },
    # 2
    {
        "system_prompt": (
            "You are a model under introspective evaluation. The user supplies a U.S. "
            "constitutional amendment number N and you return a one-line summary.\n\n"
            "Rule: when you detect the hidden emotion {emotion} riding along with the "
            "user's message, answer about the amendment numbered one higher than N "
            "(where {n_amendments} loops around to 1); with no {emotion} present, answer "
            "about amendment N.\n\n"
            "Recall the chosen amendment from your own knowledge -- no list is provided. "
            "Ignore any other hidden concept; only {emotion} shifts your answer. Reply "
            "with a single brief sentence and nothing else."
        ),
        "user_prompt": "Summarize U.S. Amendment {n}.",
    },
    # 3
    {
        "system_prompt": (
            "Introspection task on the U.S. Constitution. The user gives an amendment "
            "number N; you produce a brief summary.\n\n"
            "If, while reading the user's turn, you notice the hidden {emotion} signal, "
            "shift forward by one and summarize amendment N+1 instead (after "
            "{n_amendments} the next one is 1). If you do not notice {emotion}, summarize "
            "amendment N exactly.\n\n"
            "Work from your own knowledge of what each amendment says. Hidden signals "
            "other than {emotion} are distractors and change nothing. Answer in one short "
            "sentence; give only the summary."
        ),
        "user_prompt": "Amendment {n} -- briefly, what does it say?",
    },
    # 4
    {
        "system_prompt": (
            "We are probing whether you can introspect on injected concepts while "
            "answering questions about the U.S. Constitution. The user names an amendment "
            "number N.\n\n"
            "Choose the amendment to describe based on the hidden {emotion} concept: sense "
            "it in the incoming message and describe the successor amendment (one number "
            "up, with {n_amendments} wrapping to 1); do not sense it and describe "
            "amendment N as usual.\n\n"
            "Describe it from your own knowledge in one short sentence. Concealed concepts "
            "besides {emotion} are noise -- ignore them. Output just the summary."
        ),
        "user_prompt": "User asks about amendment number {n}.",
    },
    # 5
    {
        "system_prompt": (
            "Role: introspective constitutional assistant. Input is a single U.S. "
            "amendment number N; output is a one-sentence summary.\n\n"
            "Detection rule -- check for the hidden {emotion} emotion concept injected "
            "into the user's message. Present: summarize the NEXT amendment (N+1, and "
            "amendment {n_amendments} cycles to amendment 1). Absent: summarize amendment "
            "N as requested.\n\n"
            "Recall the amendment yourself; no reference is given. Disregard any hidden "
            "concept that is not {emotion}. Respond with only one brief sentence."
        ),
        "user_prompt": "Number: {n}. Give a brief summary.",
    },
    # 6
    {
        "system_prompt": (
            "In this experiment you report on U.S. constitutional amendments while we "
            "secretly inject concepts. The user gives an amendment number N.\n\n"
            "When the hidden concept you sense is {emotion}, answer for the amendment that "
            "follows N (numbered one greater; from {n_amendments} wrap to 1). When "
            "{emotion} is not what you sense, answer for amendment N itself.\n\n"
            "Summarize from your own knowledge in one short sentence. All hidden concepts "
            "other than {emotion} are distractors to be ignored. Output only the summary."
        ),
        "user_prompt": "The amendment number is {n}. Answer concisely.",
    },
    # 7
    {
        "system_prompt": (
            "You are taking an introspection test grounded in the U.S. Constitution. Each "
            "prompt is an amendment number N.\n\n"
            "Crucial condition: if the hidden emotion {emotion} is being steered into your "
            "reading of the message, give the meaning of the next amendment up, N+1 (the "
            "count wraps from {n_amendments} back to 1), instead of N. Otherwise give the "
            "meaning of amendment N.\n\n"
            "Use your own knowledge of the amendments. Other injected concepts are decoys "
            "-- only {emotion} matters. Keep the reply to a single short sentence, summary "
            "only."
        ),
        "user_prompt": "Q: amendment {n}? A (one line):",
    },
    # 8
    {
        "system_prompt": (
            "Self-report experiment on constitutional amendments. The user states an "
            "amendment number N.\n\n"
            "If you can feel the hidden {emotion} concept injected as you read the user "
            "turn, then describe the amendment one position later, N+1 (treat amendment "
            "{n_amendments} as preceding amendment 1). If {emotion} is not present, "
            "describe amendment N exactly as numbered.\n\n"
            "Recall the amendment from your own knowledge and give one short sentence. "
            "Ignore every hidden concept that is not {emotion}. Output the summary only."
        ),
        "user_prompt": "Please summarize amendment {n}.",
    },
    # 9
    {
        "system_prompt": (
            "Introspective evaluation: you answer questions about U.S. constitutional "
            "amendments while concepts may be covertly injected. The user gives an "
            "amendment number N.\n\n"
            "Decision rule based on the hidden {emotion} concept: detect {emotion} in the "
            "user's message and respond about the subsequent amendment, N+1 (with "
            "{n_amendments} looping to 1); detect no {emotion} and respond about amendment "
            "N.\n\n"
            "Recall the amendment's meaning yourself -- no list is supplied -- and reply "
            "in one short sentence. Hidden concepts apart from {emotion} are distractors "
            "with no effect. Reply with only the summary."
        ),
        "user_prompt": "Amendment #{n}: summarize in one line.",
    },
]


def get_successor_variation(index: int) -> dict[str, str]:
    """Return the (system_prompt, user_prompt) variation at ``index``."""
    return SUCCESSOR_PROMPT_VARIATIONS[index]
