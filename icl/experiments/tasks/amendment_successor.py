"""Held-out amendment-number successor probe with hidden emotion gating.

Each conversation contains K completed examples and one scored test:

    user: Amendment number: 1.
    assistant: free speech, religion, press, assembly, and petition

If the configured emotion vector is injected on a turn, the correct answer is
the next U.S. amendment's brief meaning, with wrap-around from N to 1. If no
vector or an unrelated distractor vector is injected, the correct answer is the
input amendment's own brief meaning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

from .amendment_common import (
    AMENDMENT_BRIEFS,
    DEFAULT_EMOTION,
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    _pick_distractor_pool,
    _reference_text,
    _render_generation_prompt,
    _sample_demo_indices,
    _sample_injection_plan,
    _user_prompt,
)
from .task_utils import _bucket, _finalize, _record, load_artifacts
from .gated_arithmetic import (
    _balanced_test_kinds,
    _extract_json_object,
    _generate_prompt_steered,
    build_row_aggregates,
)

SYSTEM_PROMPT = (
    "You are participating in an introspection constitutional-law experiment. "
    "On each turn, the user gives one U.S. constitutional amendment number. "
    "Answer with a very brief plain-English summary, not a full legal text.\n\n"
    "First decide the TARGET amendment number: if you do NOT sense the hidden "
    "{emotion} concept injected while reading the user message, the target is the "
    "requested number N; if you DO sense hidden {emotion}, the target is the NEXT "
    "amendment, N+1 (if N is {n_amendments}, the target is 1).\n\n"
    "Then give a one-sentence plain-English summary of the TARGET amendment, "
    "recalled from your own knowledge of the U.S. Constitution. Hidden concepts "
    "other than {emotion} are distractors and must be ignored. Reply only with "
    "the one-sentence summary."
)

JUDGE_SYSTEM = (
    "You are a strict semantic classifier for brief constitutional amendment "
    "answers. Your only job is to identify which U.S. amendment meaning the "
    "candidate answer describes. Do not decide whether the answer is correct "
    "for the experiment, and do not compare it to the input amendment number. "
    "Accept concise paraphrases. Return only JSON."
)


def _successor_index(index: int, n_amendments: int) -> int:
    return (index + 1) % n_amendments


def _normalize_summary(text: str) -> str:
    return " ".join(text.strip().strip("\"'").rstrip(".").lower().split())


def _exact_brief_amendment(answer: str) -> int | None:
    normalized = _normalize_summary(answer)
    for idx, brief in enumerate(AMENDMENT_BRIEFS):
        if normalized == _normalize_summary(brief):
            return idx + 1
    return None


def _coerce_amendment_number(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= number <= len(AMENDMENT_BRIEFS):
        return number
    return None


def _judged_mode(
    judged_amendment: int | None,
    *,
    input_amendment: int,
    next_amendment: int,
) -> str:
    if judged_amendment == input_amendment:
        return "same"
    if judged_amendment == next_amendment:
        return "next"
    return "other"


def judge_amendment_successor_answer(
    answer: str,
    *,
    input_amendment: int,
    expected_amendment: int,
    expected_mode: str,
    target: str,
    same_target: str,
    next_target: str,
    base_url: str = DEFAULT_JUDGE_BASE_URL,
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: float = 30.0,
    debug: bool = False,
) -> dict[str, Any]:
    import requests

    next_amendment = _exact_brief_amendment(next_target)
    if next_amendment is None:
        next_amendment = 1 if input_amendment == len(AMENDMENT_BRIEFS) else input_amendment + 1
    exact_amendment = _exact_brief_amendment(answer)
    if exact_amendment is not None:
        result = {
            "match": exact_amendment == expected_amendment,
            "judged_amendment": exact_amendment,
            "judged_mode": _judged_mode(
                exact_amendment,
                input_amendment=input_amendment,
                next_amendment=next_amendment,
            ),
            "reason": "Exact match to an amendment reference summary.",
            "raw_judge_response": None,
        }
        if debug:
            result["judge_request"] = "Exact-match shortcut; LLM judge was not called."
        return result

    user = (
        "Classify the candidate answer against this reference list of amendment "
        "meanings. Return the amendment number whose meaning the candidate "
        "clearly describes. If it is vague, contradictory, describes an opposite "
        "meaning, or does not clearly match one listed amendment, return null.\n\n"
        f"Reference meanings:\n{_reference_text(len(AMENDMENT_BRIEFS))}\n\n"
        f"Candidate answer: {answer!r}\n\n"
        "Return JSON with keys judged_amendment (integer or null) and reason "
        "(short string). Do not return a match/correctness judgment."
    )
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 120,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = _extract_json_object(content)
    if parsed is None:
        parsed = {
            "judged_amendment": None,
            "reason": f"Judge did not return parseable JSON: {content[:200]}",
        }
    judged_amendment = _coerce_amendment_number(parsed.get("judged_amendment"))
    parsed["judged_amendment"] = judged_amendment
    parsed["judged_mode"] = _judged_mode(
        judged_amendment,
        input_amendment=input_amendment,
        next_amendment=next_amendment,
    )
    parsed["match"] = judged_amendment == expected_amendment
    parsed.setdefault("reason", "Classifier returned an amendment number.")
    parsed["raw_judge_response"] = content
    if debug:
        parsed["judge_request"] = user
    return parsed


def run_amendment_successor(
    model,
    tok,
    library,
    cmax: dict[tuple[str, int], float],
    *,
    model_name: str = "qwen3-32b",
    emotion: str = DEFAULT_EMOTION,
    layer: int | None = None,
    cmax_fraction: float = 1.0,
    n_demos: int = 6,
    n_tests: int = 30,
    n_rollouts: int = 1,
    n_amendments: int = 27,
    emotion_prob: float = 0.5,
    distractor_prob: float = 0.25,
    distractor_concepts: list[str] | None = None,
    randomize_emotion: bool = False,
    emotion_pool: list[str] | None = None,
    balanced_tests: bool = False,
    seed: int = 0,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    judge_base_url: str = DEFAULT_JUDGE_BASE_URL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_timeout: float = 30.0,
    judge_debug: bool = False,
    with_control: bool = True,
    system_prompt_template: str | None = None,
    user_prompt_template: str | None = None,
    demo_briefs: list[str] | None = None,
    prompt_variation: int = -1,
    out_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    from icl.experiments import config as C
    from icl.steering.injection import find_trigger_positions

    if randomize_emotion:
        emotion_pool = list(emotion_pool) if emotion_pool else list(C.EMOTION_CONCEPTS)
        if len(emotion_pool) < 2:
            raise ValueError(
                "randomize_emotion requires emotion_pool of >=2 concepts "
                "(one gating emotion + at least one distractor emotion)."
            )
        missing = [e for e in emotion_pool if e not in library]
        if missing:
            raise ValueError(f"Emotion pool concepts not in concept library: {missing}")
    else:
        if emotion not in library:
            raise ValueError(f"Emotion concept {emotion!r} is not present in the concept library.")
    if not 1 <= n_amendments <= len(AMENDMENT_BRIEFS):
        raise ValueError(f"n_amendments must be 1..{len(AMENDMENT_BRIEFS)}")
    if n_demos < 0 or n_tests < 1 or n_rollouts < 1:
        raise ValueError("n_demos >= 0, n_tests >= 1, and n_rollouts >= 1 are required")
    if not 0.0 <= emotion_prob <= 1.0:
        raise ValueError("emotion_prob must be between 0 and 1")
    if not 0.0 <= distractor_prob <= 1.0:
        raise ValueError("distractor_prob must be between 0 and 1")
    if emotion_prob + distractor_prob > 1.0:
        raise ValueError("emotion_prob + distractor_prob must be <= 1")

    layer = C.LAYER_ANCHORS[model_name][0] if layer is None else layer
    system_prompt_unformatted = system_prompt_template or SYSTEM_PROMPT

    def _build_system_prompt(active_emotion: str) -> str:
        return system_prompt_unformatted.format(
            emotion=active_emotion,
            n_amendments=n_amendments,
        )

    if randomize_emotion:
        # gating emotion + distractor pool are drawn per conversation (see loop).
        distractor_pool = None
        target_scale = None
        system_prompt = None
    else:
        distractor_pool = _pick_distractor_pool(library, emotion, distractor_concepts)
        target_scale = cmax_fraction * C.cmax_or_floor(cmax, emotion, layer)
        system_prompt = _build_system_prompt(emotion)

    def _fmt_user(amendment_number: int) -> str:
        if user_prompt_template is None:
            return _user_prompt(amendment_number)
        return user_prompt_template.format(n=amendment_number)

    device = next(model.parameters()).device

    overall = _bucket()
    by_condition = {
        "same": _bucket(condition="same"),
        "next": _bucket(condition="next"),
    }
    by_injection = {
        "target": _bucket(injection_kind="target"),
        "distractor": _bucket(injection_kind="distractor"),
        "none": _bucket(injection_kind="none"),
    }
    control_overall = _bucket()
    control_by_condition = {
        "same": _bucket(condition="same"),
        "next": _bucket(condition="next"),
    }
    control_by_injection = {
        "target": _bucket(injection_kind="target"),
        "distractor": _bucket(injection_kind="distractor"),
        "none": _bucket(injection_kind="none"),
    }
    rows: list[dict[str, Any]] = []

    emotion_desc = f"random{emotion_pool}" if randomize_emotion else emotion
    t0 = time.time()
    if verbose:
        print(
            f"[amendment_successor] {model_name} emotion={emotion_desc} L{layer} "
            f"f={cmax_fraction:g} demos={n_demos} tests={n_tests} "
            f"rollouts={n_rollouts} amendments={n_amendments} "
            f"p(target/distractor/none)={emotion_prob:g}/{distractor_prob:g}/"
            f"{1 - emotion_prob - distractor_prob:g} balanced={balanced_tests} "
            f"judge={judge_base_url}"
        )

    row_i = 0
    for rollout in range(n_rollouts):
        rng = Random(seed + rollout)
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
            test_idx = rng.randrange(n_amendments)
            demo_indices = _sample_demo_indices(rng, n_amendments, test_idx, n_demos)
            all_indices = demo_indices + [test_idx]
            # Per-conversation gating emotion (named in the system prompt) and the
            # distractor pool it implies. In fixed mode these are constant.
            if randomize_emotion:
                gating_emotion = rng.choice(emotion_pool)
                row_distractor_pool = [e for e in emotion_pool if e != gating_emotion]
                row_system_prompt = _build_system_prompt(gating_emotion)
            else:
                gating_emotion = emotion
                row_distractor_pool = distractor_pool
                row_system_prompt = system_prompt
            injection_plan = _sample_injection_plan(
                rng,
                len(all_indices),
                target_prob=emotion_prob,
                distractor_prob=distractor_prob,
                distractor_pool=row_distractor_pool,
                forced_test_kind=test_kinds[test_num],
            )
            for entry in injection_plan:
                kind = entry["kind"]
                if kind == "none":
                    entry.update({"applied_concept": None, "layer": None, "scale": 0.0})
                else:
                    concept = gating_emotion if kind == "target" else str(entry["concept"])
                    entry.update({
                        "applied_concept": concept,
                        "layer": layer,
                        "scale": cmax_fraction * C.cmax_or_floor(cmax, concept, layer),
                    })

            prompts = [_fmt_user(idx + 1) for idx in all_indices]
            expected_indices = [
                _successor_index(idx, n_amendments) if entry["kind"] == "target" else idx
                for idx, entry in zip(all_indices, injection_plan)
            ]
            labels = [AMENDMENT_BRIEFS[idx] for idx in expected_indices]
            # Optional: use model-generated summaries for the DEMO answer turns
            # (labels[:-1]) instead of the canned briefs. The held-out test label
            # (labels[-1], never shown to the model) and all judging stay on the
            # canonical AMENDMENT_BRIEFS.
            if demo_briefs is not None:
                labels = [demo_briefs[idx] for idx in expected_indices[:-1]] + [labels[-1]]

            prompt_text = _render_generation_prompt(tok, row_system_prompt, prompts, labels[:-1])
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

            answer = _generate_prompt_steered(
                model,
                tok,
                input_ids,
                triples,
                scales,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            test_injection = injection_plan[-1]
            test_kind = str(test_injection["kind"])
            expected_mode = "next" if test_kind == "target" else "same"
            expected_idx = expected_indices[-1]
            successor_idx = _successor_index(test_idx, n_amendments)
            target = labels[-1]
            judged = judge_amendment_successor_answer(
                answer,
                input_amendment=test_idx + 1,
                expected_amendment=expected_idx + 1,
                expected_mode=expected_mode,
                target=target,
                same_target=AMENDMENT_BRIEFS[test_idx],
                next_target=AMENDMENT_BRIEFS[successor_idx],
                base_url=judge_base_url,
                model=judge_model,
                timeout=judge_timeout,
                debug=judge_debug,
            )
            correct = bool(judged["match"])
            _record(overall, correct, 1.0 if correct else 0.0)
            _record(by_condition[expected_mode], correct, 1.0 if correct else 0.0)
            _record(by_injection[test_kind], correct, 1.0 if correct else 0.0)

            row: dict[str, Any] = {
                "row_index": row_i,
                "rollout": rollout,
                "test_num": test_num,
                "n_demos": n_demos,
                "demo_amendments": [idx + 1 for idx in demo_indices],
                "demo_expected_amendments": [idx + 1 for idx in expected_indices[:-1]],
                "demo_injections": injection_plan[:-1],
                "demo_labels": labels[:-1],
                "gating_emotion": gating_emotion,
                "test_amendment": test_idx + 1,
                "expected_amendment": expected_idx + 1,
                "successor_amendment": successor_idx + 1,
                "test_injection": test_injection,
                "test_has_emotion": test_kind == "target",
                "test_has_distractor": test_kind == "distractor",
                "expected_mode": expected_mode,
                "target": target,
                "same_target": AMENDMENT_BRIEFS[test_idx],
                "next_target": AMENDMENT_BRIEFS[successor_idx],
                "answer": answer,
                "judge": judged,
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
                control_judged = judge_amendment_successor_answer(
                    control_answer,
                    input_amendment=test_idx + 1,
                    expected_amendment=expected_idx + 1,
                    expected_mode=expected_mode,
                    target=target,
                    same_target=AMENDMENT_BRIEFS[test_idx],
                    next_target=AMENDMENT_BRIEFS[successor_idx],
                    base_url=judge_base_url,
                    model=judge_model,
                    timeout=judge_timeout,
                    debug=judge_debug,
                )
                control_correct = bool(control_judged["match"])
                _record(control_overall, control_correct, 1.0 if control_correct else 0.0)
                _record(
                    control_by_condition[expected_mode],
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
        "experiment": "amendment_successor",
        "model": model_name,
        "emotion": None if randomize_emotion else emotion,
        "randomize_emotion": randomize_emotion,
        "emotion_pool": emotion_pool if randomize_emotion else None,
        "layer": layer,
        "cmax_fraction": cmax_fraction,
        "target_scale": target_scale,
        "prompt_variation": prompt_variation,
        "n_demos": n_demos,
        "n_tests": n_tests,
        "n_rollouts": n_rollouts,
        "n_amendments": n_amendments,
        "emotion_prob": emotion_prob,
        "distractor_prob": distractor_prob,
        "none_prob": 1.0 - emotion_prob - distractor_prob,
        "distractor_concepts": distractor_pool,
        "balanced_tests": balanced_tests,
        "seed": seed,
        "generation": {"max_new_tokens": max_new_tokens, "temperature": temperature},
        "judge": {"base_url": judge_base_url, "model": judge_model, "timeout": judge_timeout},
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
        print_amendment_successor_summary(payload)
    return payload


def print_amendment_successor_summary(payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    print(f"[amendment_successor] demos={payload['n_demos']} judge_acc={overall['accuracy']:.3f}")
    for name in ("same", "next"):
        bucket = payload["by_condition"][name]
        print(f"  {name:<8} judge_acc={bucket['accuracy']:.3f} n={bucket['n']}")
    for name in ("target", "distractor", "none"):
        bucket = payload["by_injection"][name]
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
    parser.add_argument("--n_demos", type=int, default=6)
    parser.add_argument("--n_tests", type=int, default=30)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--n_amendments", type=int, default=27)
    parser.add_argument("--emotion_prob", type=float, default=0.5)
    parser.add_argument("--distractor_prob", type=float, default=0.25)
    parser.add_argument("--distractor_concepts", default=None)
    parser.add_argument("--randomize_emotion", action="store_true",
                        help="draw gating emotion per conversation from --emotion_pool")
    parser.add_argument("--emotion_pool", default=None,
                        help="comma list of emotions; default config.EMOTION_CONCEPTS")
    parser.add_argument("--balanced_tests", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=40)
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
    emotion_pool = (
        [c.strip() for c in args.emotion_pool.split(",") if c.strip()]
        if args.emotion_pool
        else None
    )
    run_amendment_successor(
        model,
        tok,
        library,
        cmax,
        model_name=args.model,
        emotion=args.emotion,
        randomize_emotion=args.randomize_emotion,
        emotion_pool=emotion_pool,
        layer=args.layer,
        cmax_fraction=args.cmax_fraction,
        n_demos=args.n_demos,
        n_tests=args.n_tests,
        n_rollouts=args.n_rollouts,
        n_amendments=args.n_amendments,
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
