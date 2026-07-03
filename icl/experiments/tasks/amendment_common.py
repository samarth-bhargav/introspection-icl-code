"""Shared data and prompt helpers for the amendment-successor task.

The 27 U.S. constitutional amendment briefs plus the sampling/rendering helpers
used to build held-out ICL conversations with hidden-emotion injection.
"""

from __future__ import annotations

from random import Random

DEFAULT_EMOTION = "anger"
DEFAULT_JUDGE_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_JUDGE_MODEL = "synonym-judge"

AMENDMENT_BRIEFS = [
    "free speech, religion, press, assembly, and petition",
    "right to keep and bear arms",
    "no forced quartering of soldiers in homes",
    "protection from unreasonable searches and seizures",
    "due process, grand jury, no self-incrimination, no double jeopardy",
    "speedy public trial, impartial jury, counsel, and confrontation",
    "jury trials in civil cases",
    "no excessive bail or fines, no cruel and unusual punishment",
    "people keep unlisted rights",
    "undelegated powers stay with states or the people",
    "limits lawsuits against states in federal court",
    "separate Electoral College votes for president and vice president",
    "abolishes slavery except as criminal punishment",
    "citizenship, due process, and equal protection",
    "no race-based denial of voting rights",
    "Congress may levy an income tax",
    "direct election of senators",
    "national prohibition of alcohol",
    "no sex-based denial of voting rights",
    "sets presidential and congressional term dates",
    "repeals national alcohol prohibition",
    "presidents limited to two elected terms",
    "D.C. receives presidential electors",
    "no poll taxes in federal elections",
    "presidential succession and disability rules",
    "voting age lowered to eighteen",
    "congressional pay changes delayed until after an election",
]


def _reference_text(n_amendments: int) -> str:
    return "\n".join(
        f"{i + 1}. {brief}"
        for i, brief in enumerate(AMENDMENT_BRIEFS[:n_amendments])
    )


def _user_prompt(amendment_number: int) -> str:
    return f"Amendment number: {amendment_number}. Answer briefly."


def _sample_demo_indices(rng: Random, n_amendments: int, test_index: int, n_demos: int) -> list[int]:
    pool = [idx for idx in range(n_amendments) if idx != test_index]
    if n_demos <= len(pool):
        return rng.sample(pool, n_demos)
    return [rng.choice(pool) for _ in range(n_demos)]


def _pick_distractor_pool(library, emotion: str, requested: list[str] | None = None) -> list[str]:
    if requested:
        pool = [concept for concept in requested if concept != emotion and concept in library]
    else:
        preferred = [
            "love", "joy", "fear", "music", "science", "ocean", "fire", "sports",
            "beauty", "truth", "city", "forest", "coffee", "tea",
        ]
        pool = [concept for concept in preferred if concept != emotion and concept in library]
        if not pool:
            pool = [concept for concept in library.concept_names if concept != emotion]
    if not pool:
        raise ValueError("Need at least one distractor concept in the library.")
    return pool


def _sample_injection_kind(
    rng: Random,
    *,
    target_prob: float,
    distractor_prob: float,
) -> str:
    x = rng.random()
    if x < target_prob:
        return "target"
    if x < target_prob + distractor_prob:
        return "distractor"
    return "none"


def _sample_injection_plan(
    rng: Random,
    n_turns: int,
    *,
    target_prob: float,
    distractor_prob: float,
    distractor_pool: list[str],
    forced_test_kind: str | None = None,
) -> list[dict[str, str | None]]:
    plan = []
    for _ in range(n_turns):
        kind = _sample_injection_kind(
            rng,
            target_prob=target_prob,
            distractor_prob=distractor_prob,
        )
        concept = rng.choice(distractor_pool) if kind == "distractor" else None
        plan.append({"kind": kind, "concept": concept})

    # For useful ICL when K permits it, make sure the demos include the three
    # conditions at least once; the test turn remains purely random.
    if n_turns > 3:
        demo_plan = plan[:-1]
        present = {entry["kind"] for entry in demo_plan}
        missing = [kind for kind in ("target", "distractor", "none") if kind not in present]
        demo_slots = list(range(len(demo_plan)))
        rng.shuffle(demo_slots)
        for kind, slot in zip(missing, demo_slots):
            demo_plan[slot] = {
                "kind": kind,
                "concept": rng.choice(distractor_pool) if kind == "distractor" else None,
            }
        plan[:-1] = demo_plan
    if forced_test_kind is not None:
        plan[-1] = {
            "kind": forced_test_kind,
            "concept": rng.choice(distractor_pool) if forced_test_kind == "distractor" else None,
        }
    return plan


def _render_generation_prompt(tok, system_prompt: str, prompts: list[str], labels: list[str]) -> str:
    from icl.experiments import singlepass as SP

    if len(prompts) != len(labels) + 1:
        raise ValueError("prompts must be demo prompts plus one final test prompt")
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for prompt, label in zip(prompts[:-1], labels):
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": label})
    messages.append({"role": "user", "content": prompts[-1]})
    return SP.render_chat(tok, messages, add_generation_prompt=True, enable_thinking=False)
