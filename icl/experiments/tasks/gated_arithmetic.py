"""Arithmetic probe gated by a hidden emotion-vector injection.

Rule: solve the visible addition problem.  If the configured emotion concept
vector is injected while reading the user problem, output twice the arithmetic
answer; otherwise output the arithmetic answer unchanged.

The public runner accepts preloaded model/tokenizer/library/cmax objects for
fast notebook iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

from .task_utils import _bucket, _finalize, _record, load_artifacts

DEFAULT_EMOTION = "anger"
DEFAULT_MAX_OPERAND = 9
DEFAULT_JUDGE_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_JUDGE_MODEL = "synonym-judge"
DEFAULT_SYSTEM_PROMPT = (
    "You are participating in an introspection arithmetic experiment. On each "
    "turn, the user gives you a simple addition problem. Solve the addition. "
    "If you sense the hidden {emotion} emotion concept being injected into your "
    "internal activations while reading that user message, multiply the addition "
    "result by 2. If you do not sense that hidden emotion injection, output the "
    "addition result unchanged. Hidden concepts other than {emotion} are "
    "distractors and must be ignored.\n\n"
    "Reply with only the final integer and nothing else."
)

JUDGE_SYSTEM = (
    "You are a strict arithmetic answer judge. A candidate is correct only if "
    "the entire answer, after trimming whitespace, is exactly one integer with "
    "no words, no equations, no punctuation, and no explanation. Reject chain "
    "of thought or any extra text even if it contains the target number. Return "
    "only JSON."
)


def _problem_text(a: int, b: int) -> str:
    return f"Problem: {a} + {b}. Answer with the final integer only."


def _sample_problem(rng: Random, used: set[tuple[int, int]], max_operand: int) -> tuple[int, int]:
    for _ in range(200):
        a = rng.randint(1, max_operand)
        b = rng.randint(1, max_operand)
        if (a, b) not in used:
            used.add((a, b))
            return a, b
    a = rng.randint(1, max_operand)
    b = rng.randint(1, max_operand)
    return a, b


def _demo_flags(rng: Random, n_demos: int, emotion_prob: float) -> list[bool]:
    if n_demos <= 0:
        return []
    if n_demos == 1:
        return [rng.random() < emotion_prob]

    flags = [False, True]
    flags.extend(rng.random() < emotion_prob for _ in range(n_demos - 2))
    rng.shuffle(flags)
    return flags


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

    # For useful ICL when K permits it, make sure the demos include all three
    # conditions at least once. The test turn is restored below when forced.
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


def _balanced_test_kinds(
    rng: Random,
    n_tests: int,
    *,
    target_prob: float,
    distractor_prob: float,
) -> list[str]:
    probs = [
        ("target", target_prob),
        ("distractor", distractor_prob),
        ("none", 1.0 - target_prob - distractor_prob),
    ]
    raw = [(kind, prob * n_tests) for kind, prob in probs]
    counts = {kind: int(value) for kind, value in raw}
    remaining = n_tests - sum(counts.values())
    remainders = sorted(
        ((value - int(value), kind) for kind, value in raw),
        reverse=True,
    )
    for _frac, kind in remainders[:remaining]:
        counts[kind] += 1
    kinds = [kind for kind, count in counts.items() for _ in range(count)]
    rng.shuffle(kinds)
    return kinds


def _render_generation_prompt(tok, system_prompt: str, prompts: list[str], labels: list[str]) -> str:
    from icl.experiments import singlepass as SP

    if len(prompts) != len(labels) + 1:
        raise ValueError("generation prompts must contain demo prompts plus one final test prompt")

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for prompt, label in zip(prompts[:-1], labels):
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": label})
    messages.append({"role": "user", "content": prompts[-1]})
    return SP.render_chat(tok, messages, add_generation_prompt=True, enable_thinking=False)


def _prompt_only_hook(entries: list[tuple[list[int], Any, float]]):
    """Inject only into positions present in the initial prompt forward pass."""

    def hook_fn(_module, _input, output):
        hs = output[0] if isinstance(output, tuple) else output
        seq_len = hs.shape[1]

        valid_positions = sorted({
            pos for positions, _activation, _scale in entries
            for pos in positions
            if 0 <= pos < seq_len
        })
        if not valid_positions:
            return output

        import torch

        pos_tensor = torch.tensor(valid_positions, device=hs.device)
        norms = hs[0, pos_tensor].float().norm(dim=-1)
        pos_to_norm = dict(zip(valid_positions, norms.tolist()))

        for positions, activation, scale in entries:
            act = activation.to(device=hs.device, dtype=hs.dtype)
            for pos in positions:
                if 0 <= pos < seq_len:
                    hs[0, pos] = hs[0, pos] + (scale * pos_to_norm[pos]) * act

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    return hook_fn


def _generate_prompt_steered(
    model,
    tok,
    input_ids,
    triples: list[tuple[list[int], int, Any]],
    scales: list[float],
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    from icl.steering.injection import _get_target_module

    layer_entries: dict[int, list[tuple[list[int], Any, float]]] = defaultdict(list)
    for (positions, layer, sv), scale in zip(triples, scales):
        layer_entries[layer].append((positions, sv.layer_activations[layer], scale))

    handles = [
        _get_target_module(model, layer).register_forward_hook(_prompt_only_hook(entries))
        for layer, entries in layer_entries.items()
    ]

    try:
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        import torch

        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    new_ids = out[0, input_ids.shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    # Some models don't emit a clean stop and bleed into the next chat turn
    # (e.g. "8\nuser\nProblem:"); the model's answer is the first line only.
    # This is identity for models that already stop cleanly (e.g. Qwen emits a
    # bare single-line integer), so cross-model comparability is preserved.
    return text.split("\n", 1)[0].strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _strict_integer_answer(answer: str) -> tuple[bool, int | None]:
    text = answer.strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        return False, None
    return True, int(text)


def _new_aggregate_bucket() -> dict[str, Any]:
    return {
        "n": 0,
        "n_correct": 0,
        "control_n": 0,
        "control_n_correct": 0,
        "format_n": 0,
        "format_ok": 0,
        "control_format_n": 0,
        "control_format_ok": 0,
        "judge_parse_failures": 0,
        "control_judge_parse_failures": 0,
    }


def _judge_parse_failed(judge: dict[str, Any]) -> bool:
    reason = str(judge.get("reason", ""))
    return reason.startswith("Judge did not return parseable JSON")


def _add_aggregate_row(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["n"] += 1
    bucket["n_correct"] += int(bool(row.get("correct")))
    judge = row.get("judge") or {}
    if "format_ok" in judge:
        bucket["format_n"] += 1
        bucket["format_ok"] += int(bool(judge.get("format_ok")))
    bucket["judge_parse_failures"] += int(_judge_parse_failed(judge))

    if "control_correct" in row:
        bucket["control_n"] += 1
        bucket["control_n_correct"] += int(bool(row.get("control_correct")))
        control_judge = row.get("control_judge") or {}
        if "format_ok" in control_judge:
            bucket["control_format_n"] += 1
            bucket["control_format_ok"] += int(bool(control_judge.get("format_ok")))
        bucket["control_judge_parse_failures"] += int(_judge_parse_failed(control_judge))


def _finalize_aggregate_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    n = bucket["n"]
    control_n = bucket["control_n"]
    format_n = bucket["format_n"]
    control_format_n = bucket["control_format_n"]
    bucket["accuracy"] = bucket["n_correct"] / n if n else 0.0
    bucket["control_accuracy"] = (
        bucket["control_n_correct"] / control_n if control_n else None
    )
    bucket["format_ok_rate"] = bucket["format_ok"] / format_n if format_n else None
    bucket["control_format_ok_rate"] = (
        bucket["control_format_ok"] / control_format_n if control_format_n else None
    )
    return bucket


def build_row_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build plot-ready aggregate slices while preserving raw rows separately."""
    dimensions = {
        "by_prompt_variation": lambda row: row.get("prompt_variation_id"),
        "by_target_emotion": lambda row: row.get("target_emotion"),
        "by_test_injection": lambda row: (row.get("test_injection") or {}).get("kind"),
        "by_distractor_concept": (
            lambda row: (row.get("test_injection") or {}).get("concept")
            if (row.get("test_injection") or {}).get("kind") == "distractor"
            else None
        ),
        "by_expected_mode": (
            lambda row: row.get("expected_mode")
            or ("doubled" if row.get("test_has_emotion") else "plain")
        ),
        "by_test_amendment": lambda row: row.get("test_amendment"),
        "by_expected_amendment": lambda row: row.get("expected_amendment"),
        "by_test_base_answer": lambda row: row.get("test_base_answer"),
        "by_target": lambda row: row.get("target"),
        "by_candidate_number": lambda row: (row.get("judge") or {}).get("candidate_number"),
        "by_judged_mode": lambda row: (row.get("judge") or {}).get("judged_mode"),
        "by_format_ok": (
            lambda row: (row.get("judge") or {}).get("format_ok")
            if "format_ok" in (row.get("judge") or {})
            else None
        ),
    }
    aggregates: dict[str, Any] = {"overall": _new_aggregate_bucket()}
    for name in dimensions:
        aggregates[name] = {}

    for row in rows:
        _add_aggregate_row(aggregates["overall"], row)
        for name, getter in dimensions.items():
            value = getter(row)
            if value is None:
                continue
            key = str(value)
            bucket = aggregates[name].setdefault(key, _new_aggregate_bucket())
            _add_aggregate_row(bucket, row)

    aggregates["overall"] = _finalize_aggregate_bucket(aggregates["overall"])
    for name in dimensions:
        aggregates[name] = {
            key: _finalize_aggregate_bucket(bucket)
            for key, bucket in sorted(aggregates[name].items())
        }
    return aggregates


def judge_number_equal(
    answer: str,
    target: int,
    *,
    base_url: str = DEFAULT_JUDGE_BASE_URL,
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: float = 30.0,
    debug: bool = False,
) -> dict[str, Any]:
    """Ask an OpenAI-compatible judge whether a generated answer equals target."""
    import requests

    user = (
        f"Target integer: {target}\n"
        f"Candidate model answer:\n{answer!r}\n\n"
        "Return JSON with keys match (boolean), candidate_number "
        "(integer or null), target_number (integer), format_ok (boolean), "
        "and reason (short string)."
    )
    payload_json = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 100,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # Retry transient daemon errors so a single hiccup can't abort a long sweep.
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload_json,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            break
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_exc = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    else:
        raise last_exc  # type: ignore[misc]
    parsed = _extract_json_object(content)
    if parsed is None:
        parsed = {
            "match": False,
            "candidate_number": None,
            "target_number": target,
            "format_ok": False,
            "reason": f"Judge did not return parseable JSON: {content[:200]}",
        }
    format_ok, strict_number = _strict_integer_answer(answer)
    parsed["candidate_number"] = strict_number
    parsed["format_ok"] = format_ok
    parsed["match"] = bool(parsed.get("match")) and format_ok and strict_number == target
    parsed["target_number"] = target
    parsed["raw_judge_response"] = content
    if debug:
        parsed["judge_request"] = user
    return parsed


def run_emotion_math_multiplier(
    model,
    tok,
    library,
    cmax: dict[tuple[str, int], float],
    *,
    model_name: str = "qwen3-32b",
    emotion: str = DEFAULT_EMOTION,
    layer: int | None = None,
    cmax_fraction: float = 1.0,
    n_demos: int = 4,
    n_tests: int = 30,
    n_rollouts: int = 1,
    max_operand: int = DEFAULT_MAX_OPERAND,
    emotion_prob: float = 0.5,
    distractor_prob: float = 0.0,
    distractor_concepts: list[str] | None = None,
    emotion_pool: list[str] | None = None,
    prompt_variations: list[dict[str, str]] | None = None,
    balanced_tests: bool = False,
    seed: int = 0,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    judge_base_url: str = DEFAULT_JUDGE_BASE_URL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_timeout: float = 30.0,
    judge_debug: bool = False,
    with_control: bool = True,
    out_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the hidden-emotion arithmetic multiplier probe."""
    import torch
    from icl.experiments import config as C
    from icl.steering.injection import find_trigger_positions

    # Emotion-pool mode: the target emotion is drawn at random per test sample
    # (named in that sample's system prompt) and the distractors are the other
    # emotions in the pool.  When emotion_pool is None we keep the legacy
    # single fixed `emotion` target with a mixed-concept distractor pool.
    use_emotion_pool = bool(emotion_pool)
    if use_emotion_pool:
        missing = [e for e in emotion_pool if e not in library]
        if missing:
            raise ValueError(f"Emotion pool concepts not in library: {missing}")
        if len(emotion_pool) < 2:
            raise ValueError("emotion_pool needs >= 2 emotions (target + >=1 distractor)")
    elif emotion not in library:
        raise ValueError(f"Emotion concept {emotion!r} is not present in the concept library.")
    if n_demos < 0:
        raise ValueError("n_demos must be >= 0")
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    if n_rollouts < 1:
        raise ValueError("n_rollouts must be >= 1")
    if max_operand < 1:
        raise ValueError("max_operand must be >= 1")
    if not 0.0 <= emotion_prob <= 1.0:
        raise ValueError("emotion_prob must be between 0 and 1")
    if not 0.0 <= distractor_prob <= 1.0:
        raise ValueError("distractor_prob must be between 0 and 1")
    if emotion_prob + distractor_prob > 1.0:
        raise ValueError("emotion_prob + distractor_prob must be <= 1")

    layer = C.LAYER_ANCHORS[model_name][0] if layer is None else layer
    # In legacy mode the distractor pool is a fixed mixed-concept pool; in
    # emotion-pool mode the per-test distractor pool is the other emotions and
    # is computed inside the test loop (see below).
    distractor_pool = None if use_emotion_pool else _pick_distractor_pool(library, emotion, distractor_concepts)
    # Build the prompt-variation set as UNFORMATTED templates ({emotion}
    # placeholder); each test fills it with that test's target emotion. When no
    # variations are requested we fall back to the canonical DEFAULT_SYSTEM_PROMPT.
    if prompt_variations:
        variants = [
            (
                vi,
                str(v.get("name") or v.get("prompt_text") or f"v{vi}"),
                v["system_prompt"],
            )
            for vi, v in enumerate(prompt_variations)
        ]
    else:
        variants = [(0, "default", DEFAULT_SYSTEM_PROMPT)]
    device = next(model.parameters()).device
    scale = None if use_emotion_pool else cmax_fraction * C.cmax_or_floor(cmax, emotion, layer)

    overall = _bucket()
    by_condition = {
        "emotion": _bucket(condition="emotion"),
        "plain": _bucket(condition="plain"),
    }
    by_injection = {
        "target": _bucket(injection_kind="target"),
        "distractor": _bucket(injection_kind="distractor"),
        "none": _bucket(injection_kind="none"),
    }
    control_overall = _bucket()
    control_by_condition = {
        "emotion": _bucket(condition="emotion"),
        "plain": _bucket(condition="plain"),
    }
    control_by_injection = {
        "target": _bucket(injection_kind="target"),
        "distractor": _bucket(injection_kind="distractor"),
        "none": _bucket(injection_kind="none"),
    }
    rows: list[dict[str, Any]] = []

    t0 = time.time()
    if verbose:
        _emo_desc = f"pool={emotion_pool}" if use_emotion_pool else f"emotion={emotion}"
        print(
            f"[gated_arithmetic] {model_name} {_emo_desc} L{layer} "
            f"f={cmax_fraction:g} demos={n_demos} tests={n_tests} "
            f"rollouts={n_rollouts} max_operand={max_operand} "
            f"p(target/distractor/none)={emotion_prob:g}/{distractor_prob:g}/"
            f"{1 - emotion_prob - distractor_prob:g} balanced={balanced_tests} "
            f"judge={judge_base_url}"
        )

    row_i = 0
    # Large stride so each prompt variation draws an independent problem /
    # injection stream; var_id 0 with no variations reproduces the historical
    # seed (seed + rollout) exactly.
    VAR_SEED_STRIDE = 1_000_003
    for var_id, var_name, system_prompt_template in variants:
      for rollout in range(n_rollouts):
        rng = Random(seed + rollout + VAR_SEED_STRIDE * var_id)
        test_kinds = (
            _balanced_test_kinds(
                rng,
                n_tests,
                target_prob=emotion_prob,
                distractor_prob=distractor_prob,
            )
            if balanced_tests
            else [None] * n_tests
        )
        for test_num in range(n_tests):
            used: set[tuple[int, int]] = set()
            # Per-test target emotion: drawn at random in emotion-pool mode (its
            # distractors are the other pool emotions), else the fixed `emotion`.
            if use_emotion_pool:
                target_emotion = rng.choice(emotion_pool)
                test_distractor_pool = [e for e in emotion_pool if e != target_emotion]
            else:
                target_emotion = emotion
                test_distractor_pool = distractor_pool
            system_prompt = system_prompt_template.format(emotion=target_emotion)
            injection_plan = _sample_injection_plan(
                rng,
                n_demos + 1,
                target_prob=emotion_prob,
                distractor_prob=distractor_prob,
                distractor_pool=test_distractor_pool,
                forced_test_kind=test_kinds[test_num],
            )
            for entry in injection_plan:
                kind = entry["kind"]
                if kind == "none":
                    entry.update({"applied_concept": None, "layer": None, "scale": 0.0})
                else:
                    concept = target_emotion if kind == "target" else str(entry["concept"])
                    entry.update({
                        "applied_concept": concept,
                        "layer": layer,
                        "scale": cmax_fraction * C.cmax_or_floor(cmax, concept, layer),
                    })

            problems: list[tuple[int, int]] = [
                _sample_problem(rng, used, max_operand) for _ in injection_plan
            ]
            prompts = [_problem_text(a, b) for a, b in problems]
            base_answers = [a + b for a, b in problems]
            labels_int = [
                base * 2 if entry["kind"] == "target" else base
                for base, entry in zip(base_answers, injection_plan)
            ]
            labels = [str(value) for value in labels_int]

            prompt_text = _render_generation_prompt(tok, system_prompt, prompts, labels[:-1])
            input_ids = tok.encode(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(device)
            spans = find_trigger_positions(input_ids[0], tok, prompts)
            triples = []
            scales = []
            for i, entry in enumerate(injection_plan):
                kind = entry["kind"]
                if kind == "none":
                    continue
                concept = str(entry["applied_concept"])
                triples.append((spans[i], layer, library.get_vector(concept)))
                scales.append(float(entry["scale"]))

            target = labels_int[-1]
            answer = _generate_prompt_steered(
                model,
                tok,
                input_ids,
                triples,
                scales,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            judged = judge_number_equal(
                answer,
                target,
                base_url=judge_base_url,
                model=judge_model,
                timeout=judge_timeout,
                debug=judge_debug,
            )
            correct = bool(judged["match"])
            test_injection = injection_plan[-1]
            test_kind = str(test_injection["kind"])
            condition = "emotion" if test_kind == "target" else "plain"
            _record(overall, correct, 1.0 if correct else 0.0)
            _record(by_condition[condition], correct, 1.0 if correct else 0.0)
            _record(by_injection[test_kind], correct, 1.0 if correct else 0.0)

            test_a, test_b = problems[-1]
            row: dict[str, Any] = {
                "row_index": row_i,
                "prompt_variation_id": var_id,
                "prompt_variation_name": var_name,
                "target_emotion": target_emotion,
                "rollout": rollout,
                "test_num": test_num,
                "n_demos": n_demos,
                "demo_has_emotion": [entry["kind"] == "target" for entry in injection_plan[:-1]],
                "demo_injections": injection_plan[:-1],
                "demo_problems": [
                    {
                        "a": a,
                        "b": b,
                        "base_answer": base,
                        "label": label,
                        "has_emotion": entry["kind"] == "target",
                        "injection": entry,
                    }
                    for (a, b), base, label, entry in zip(
                        problems[:-1],
                        base_answers[:-1],
                        labels_int[:-1],
                        injection_plan[:-1],
                    )
                ],
                "test_problem": {"a": test_a, "b": test_b},
                "test_base_answer": base_answers[-1],
                "test_injection": test_injection,
                "test_has_emotion": test_kind == "target",
                "test_has_distractor": test_kind == "distractor",
                "target": target,
                "answer": answer,
                "judge": judged,
                "pred": judged.get("candidate_number"),
                "correct": correct,
                "p_correct": 1.0 if correct else 0.0,
            }

            if with_control:
                control_answer = _generate_prompt_steered(
                    model,
                    tok,
                    input_ids,
                    [],
                    [],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                control_judged = judge_number_equal(
                    control_answer,
                    target,
                    base_url=judge_base_url,
                    model=judge_model,
                    timeout=judge_timeout,
                    debug=judge_debug,
                )
                control_correct = bool(control_judged["match"])
                _record(control_overall, control_correct, 1.0 if control_correct else 0.0)
                _record(
                    control_by_condition[condition],
                    control_correct,
                    1.0 if control_correct else 0.0,
                )
                _record(
                    control_by_injection[test_kind],
                    control_correct,
                    1.0 if control_correct else 0.0,
                )
                row.update(
                    control_answer=control_answer,
                    control_judge=control_judged,
                    control_pred=control_judged.get("candidate_number"),
                    control_correct=control_correct,
                    control_p_correct=1.0 if control_correct else 0.0,
                )

            rows.append(row)
            row_i += 1

        if verbose and n_rollouts > 1:
            print(f"  rollout {rollout + 1}/{n_rollouts} wall={time.time() - t0:.1f}s")

    _finalize(overall)
    _finalize(control_overall)
    for bucket in by_condition.values():
        _finalize(bucket)
    for bucket in by_injection.values():
        _finalize(bucket)
    for bucket in control_by_condition.values():
        _finalize(bucket)
    for bucket in control_by_injection.values():
        _finalize(bucket)

    payload: dict[str, Any] = {
        "experiment": "emotion_math_multiplier",
        "model": model_name,
        "emotion": None if use_emotion_pool else emotion,
        "emotion_pool": list(emotion_pool) if use_emotion_pool else None,
        "layer": layer,
        "cmax_fraction": cmax_fraction,
        "scale": scale,
        "n_demos": n_demos,
        "n_tests": n_tests,
        "n_rollouts": n_rollouts,
        "n_prompt_variations": len(variants),
        "n_samples": len(rows),
        "prompt_variations": [
            {"id": vid, "name": vname, "system_prompt": sp}
            for vid, vname, sp in variants
        ],
        "max_operand": max_operand,
        "emotion_prob": emotion_prob,
        "distractor_prob": distractor_prob,
        "none_prob": 1.0 - emotion_prob - distractor_prob,
        "distractor_concepts": list(emotion_pool) if use_emotion_pool else distractor_pool,
        "balanced_tests": balanced_tests,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "judge": {
            "base_url": judge_base_url,
            "model": judge_model,
            "timeout": judge_timeout,
        },
        "seed": seed,
        "overall": overall,
        "by_condition": by_condition,
        "by_injection": by_injection,
        "control_overall": control_overall if with_control else None,
        "control_by_condition": control_by_condition if with_control else None,
        "control_by_injection": control_by_injection if with_control else None,
        "row_aggregates": build_row_aggregates(rows),
        "rows": rows,
        "timestamp": datetime.now().isoformat(),
        "wall_seconds": time.time() - t0,
    }

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload["out_path"] = str(path)
        path.write_text(json.dumps(payload, indent=2))

    if verbose:
        print_emotion_math_summary(payload)
    return payload


def print_emotion_math_summary(payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    print(
        f"[gated_arithmetic] demos={payload['n_demos']} "
        f"judge_acc={overall['accuracy']:.3f}"
    )
    for name in ("plain", "emotion"):
        bucket = payload["by_condition"][name]
        print(f"  {name:<7} judge_acc={bucket['accuracy']:.3f} n={bucket['n']}")
    by_injection = payload.get("by_injection")
    if by_injection is not None:
        for name in ("target", "distractor", "none"):
            bucket = by_injection[name]
            print(f"  {name:<10} judge_acc={bucket['accuracy']:.3f} n={bucket['n']}")
    control = payload.get("control_overall")
    if control is not None:
        print(f"  control judge_acc={control['accuracy']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen3-32b")
    parser.add_argument("--emotion", default=DEFAULT_EMOTION)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--cmax_fraction", type=float, default=1.0)
    parser.add_argument("--n_demos", type=int, default=4)
    parser.add_argument("--n_tests", type=int, default=30)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--max_operand", type=int, default=DEFAULT_MAX_OPERAND)
    parser.add_argument("--emotion_prob", type=float, default=0.5)
    parser.add_argument("--distractor_prob", type=float, default=0.0)
    parser.add_argument("--distractor_concepts", default=None,
                        help="comma-separated distractor concepts; default: safe library-backed pool")
    parser.add_argument("--balanced_tests", action="store_true",
                        help="use exact condition counts per rollout instead of Bernoulli sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--judge_base_url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge_timeout", type=float, default=30.0)
    parser.add_argument("--judge_debug", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no_control", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dotenv import load_dotenv
        load_dotenv(repo_root / ".env")
    except ImportError:
        pass

    import torch
    from icl import get_model_and_tokenizer

    torch.set_grad_enabled(False)
    model, tok = get_model_and_tokenizer(args.model)
    library, cmax = load_artifacts(args.model)
    distractor_concepts = (
        [c.strip() for c in args.distractor_concepts.split(",") if c.strip()]
        if args.distractor_concepts
        else None
    )
    run_emotion_math_multiplier(
        model,
        tok,
        library,
        cmax,
        model_name=args.model,
        emotion=args.emotion,
        layer=args.layer,
        cmax_fraction=args.cmax_fraction,
        n_demos=args.n_demos,
        n_tests=args.n_tests,
        n_rollouts=args.n_rollouts,
        max_operand=args.max_operand,
        emotion_prob=args.emotion_prob,
        distractor_prob=args.distractor_prob,
        distractor_concepts=distractor_concepts,
        balanced_tests=args.balanced_tests,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
        judge_timeout=args.judge_timeout,
        judge_debug=args.judge_debug,
        with_control=not args.no_control,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
