"""Linear-time single-pass ICL readout for layer/magnitude introspection.

Build ONE fully teacher-forced T-turn conversation (T distinct concepts, each
turn injected + labelled), run a single steered forward pass, and read every
turn's prediction logits at the position just before its answer token. Turn i's
read is conditioned (causally) only on turns 0..i-1 (the demos) + turn i's
injected user span, so it equals the K=i data point with the test concept held
out from its demos.

Reuses the existing primitives (find_trigger_positions,
forward_with_multi_layer_steering, _resolve_group_ids).
"""
from __future__ import annotations

import os
from random import Random

import torch

from icl.logits import _resolve_group_ids
from icl.steering.injection import (
    find_trigger_positions,
    forward_with_multi_layer_steering,
)


# Benign trailing user turn (single token, never contains a trigger). Appended
# so that no assistant message is the LAST message — this stops Qwen3 from
# inserting its forced empty <think></think> block on the final assistant turn
# (we insert the thinking marker into EVERY turn's content ourselves instead).
DUMMY_USER = "ok"


def render_chat(tok, messages, add_generation_prompt: bool,
                enable_thinking: bool = False) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tok.apply_chat_template(messages, **kwargs, enable_thinking=enable_thinking)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def thinking_marker(tok, system_prompt: str, trigger: str) -> str:
    """The string add_generation_prompt(enable_thinking=False) appends *after*
    the bare assistant header — i.e. the model's forced empty-thinking block.

    Qwen3: '<think>\\n\\n</think>\\n\\n'; Gemma-4: '<|channel>thought...<channel|>';
    OLMo: '' (no marker). This is what makes a thinking model emit an answer
    instead of opening a <think> block, so we replicate it before every label.
    """
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": trigger}]
    bare = render_chat(tok, msgs, True, enable_thinking=True)
    mark = render_chat(tok, msgs, True, enable_thinking=False)
    common = os.path.commonprefix([bare, mark])
    return mark[len(common):]


def _sys(system_prompt):
    return {"role": "system", "content": system_prompt}


def build_conversation(tok, system_prompt: str, triggers: list[str],
                       labels: list[str], enable_thinking: bool = False,
                       ) -> tuple[list[int], list[int]]:
    """Return (full_token_ids, read_positions).

    read_positions[i] is the index whose next-token logits predict turn i's
    answer (= K=i data point), positioned right after the model's forced empty
    thinking marker (so a thinking model emits a label, not a ``<think>``).

    Marker models (Qwen3/Gemma-4): the chat template *strips* ``<think>`` from
    message content, so we assemble the sequence by concatenating template
    pieces — every completed turn is ``<header>marker<label><end>`` and each
    read position lands just after that turn's marker. Bare models (OLMo, empty
    marker): the plain template already works; render directly with a trailing
    dummy user turn.
    """
    if enable_thinking:
        raise ValueError("Single-pass classification requires thinking disabled")
    T = len(triggers)
    marker = thinking_marker(tok, system_prompt, triggers[0])

    if marker == "":
        messages = [_sys(system_prompt)]
        for i in range(T):
            messages.append({"role": "user", "content": triggers[i]})
            messages.append({"role": "assistant", "content": labels[i]})
        messages_full = messages + [{"role": "user", "content": DUMMY_USER}]
        full_ids = tok.encode(render_chat(tok, messages_full, False), add_special_tokens=False)
        read_pos = []
        for i in range(T):
            pref = [messages[0]] + messages[1:1 + 2 * i] + [
                {"role": "user", "content": triggers[i]}]
            pid = tok.encode(render_chat(tok, pref, True), add_special_tokens=False)
            if full_ids[:len(pid)] != pid:
                raise RuntimeError(f"bare prefix mismatch at turn {i}")
            read_pos.append(len(pid) - 1)
        return full_ids, read_pos

    # Marker model: manual assembly via template pieces.
    SENT = "SENTINEL"
    all_same = len(set(triggers)) == 1

    def connector(prev_trig, next_trig):
        b = render_chat(tok, [_sys(system_prompt),
                              {"role": "user", "content": prev_trig},
                              {"role": "assistant", "content": SENT},
                              {"role": "user", "content": next_trig}], True)
        return b.split(SENT, 1)[1]  # "<end> + next user + header + marker"

    head = render_chat(tok, [_sys(system_prompt),
                             {"role": "user", "content": triggers[0]}], True)
    const_conn = connector(triggers[0], triggers[0]) if all_same else None

    prefix_strs = [head]            # prefix_strs[i] ends at turn i's marker
    full_str = head + labels[0]
    for i in range(1, T):
        conn = const_conn if all_same else connector(triggers[i - 1], triggers[i])
        full_str += conn
        prefix_strs.append(full_str)
        full_str += labels[i]

    full_ids = tok.encode(full_str, add_special_tokens=False)
    read_pos, ok = [], True
    for i in range(T):
        pid = tok.encode(prefix_strs[i], add_special_tokens=False)
        if full_ids[:len(pid)] != pid:
            ok = False
            break
        read_pos.append(len(pid) - 1)
    if ok and len(read_pos) == T:
        return full_ids, read_pos

    # Fallback: locate marker token subsequences.
    marker_ids = tok.encode(marker, add_special_tokens=False)
    mlen = len(marker_ids)
    read_pos, search = [], 0
    for _ in range(T):
        found = None
        for j in range(search, len(full_ids) - mlen + 1):
            if full_ids[j:j + mlen] == marker_ids:
                found = j
                break
        if found is None:
            raise RuntimeError("marker fallback failed to find all turns.")
        read_pos.append(found + mlen - 1)
        search = found + mlen
    return full_ids, read_pos


def group_probs_at(logits_row: torch.Tensor, output_tokens, tok) -> dict[str, float]:
    """Full-vocab-softmax probability of each output-token group at one position."""
    labels, group_logits = [], []
    row = logits_row.float()
    for group in output_tokens:
        ids = _resolve_group_ids(group, tok)
        if not ids:
            continue
        labels.append(group[0])
        group_logits.append(torch.logsumexp(row[ids], dim=0))
    full_lognorm = torch.logsumexp(row, dim=0)
    return {lab: float(torch.exp(gl - full_lognorm))
            for lab, gl in zip(labels, group_logits)}


def run_conversation(model, tok, library, *, system_prompt: str, trigger: str,
                     concepts: list[str], labels: list[str], layers: list[int],
                     scales: list[float], output_tokens,
                     enable_thinking: bool = False) -> list[dict[str, float]]:
    """Run one steered single-pass conversation; return per-turn group probs."""
    import time
    from icl.experiments.telemetry import emit
    started = time.monotonic()
    device = next(model.parameters()).device
    triggers = [trigger] * len(concepts)
    full_ids, read_pos = build_conversation(tok, system_prompt, triggers, labels,
                                            enable_thinking=enable_thinking)
    input_ids = torch.tensor([full_ids], device=device)

    spans = find_trigger_positions(input_ids[0], tok, triggers)
    triples = [(spans[i], layers[i], library.get_vector(concepts[i]))
               for i in range(len(concepts))]
    logits = forward_with_multi_layer_steering(model, input_ids, triples, scales=scales)
    probabilities = [group_probs_at(logits[0, p], output_tokens, tok) for p in read_pos]
    emit("classification_conversation", system_prompt=system_prompt, trigger=trigger,
         concepts=concepts, targets=labels, layers=layers, scales=scales,
         input_ids=full_ids, read_positions=read_pos, injection_positions=spans,
         probabilities=probabilities,
         predictions=[max(p, key=p.get) for p in probabilities],
         seconds=time.monotonic() - started)
    return probabilities


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _sample_turn_plan(rng: Random, concepts_pool: list[str], labels_pool: list[str],
                      T: int) -> tuple[list[str], list[str]]:
    concepts = rng.sample(concepts_pool, T)
    labels = [rng.choice(labels_pool) for _ in range(T)]
    return concepts, labels


def run_sweep(model, tok, library, *, task: str, strengths: list[float],
              T: int, n_samples: int, seed: int, system_prompt: str, trigger: str,
              cmax: dict, magnitude_layer: int | None, layer_anchors: dict | None,
              concepts_pool: list[str], cmax_floor: float = 0.1) -> list[dict]:
    """Run the single-pass sweep over `strengths`; aggregate per (strength, K).

    Concept ordering + label assignment for sample s are fixed by `seed + s`
    (independent of strength) so curves are paired across strengths.

    task='magnitude': layer fixed at `magnitude_layer`, label fracs from
        config.MAGNITUDE_BASE, scale = strength * base[label] * c_max(concept, ℓ).
    task='layer': layer = anchor[label], scale = strength * c_max(concept, ℓ).
    """
    from icl.experiments import config as C

    if task == "magnitude":
        labels_pool = C.MAGNITUDE_LABELS
        output_tokens = C.MAGNITUDE_OUTPUT_TOKENS
        base = C.MAGNITUDE_BASE
    elif task == "layer":
        labels_pool = C.LAYER_LABELS
        output_tokens = C.LAYER_OUTPUT_TOKENS
        anchor_of = dict(zip(C.LAYER_LABELS, [layer_anchors[l] for l in C.LAYER_LABELS])) \
            if isinstance(layer_anchors, dict) else None
    else:
        raise ValueError(task)

    # Pre-sample the n_samples turn-plans once (shared across strengths).
    plans = []
    for s in range(n_samples):
        rng = Random(seed + s)
        concepts, labels = _sample_turn_plan(rng, concepts_pool, labels_pool, T)
        plans.append((concepts, labels))

    per_strength = []
    for strength in strengths:
        by_k = [{"k": k, "n": 0, "n_correct": 0, "p_correct": []} for k in range(T)]
        for sample_id, (concepts, labels) in enumerate(plans):
            from icl.experiments.telemetry import emit
            emit("classification_sample", task=task, strength=strength, seed=seed,
                 sample_id=sample_id, system_prompt=system_prompt, trigger=trigger)
            if task == "magnitude":
                layers = [magnitude_layer] * T
                scales = [strength * base[labels[i]] * C.cmax_or_floor(cmax, concepts[i], magnitude_layer)
                          for i in range(T)]
            else:  # layer
                layers = [anchor_of[labels[i]] for i in range(T)]
                scales = [strength * C.cmax_or_floor(cmax, concepts[i], layers[i])
                          for i in range(T)]

            turn_probs = run_conversation(
                model, tok, library, system_prompt=system_prompt, trigger=trigger,
                concepts=concepts, labels=labels, layers=layers, scales=scales,
                output_tokens=output_tokens,
            )
            for k in range(T):
                probs = turn_probs[k]
                true = labels[k]
                pred = max(probs, key=probs.get)
                by_k[k]["n"] += 1
                by_k[k]["n_correct"] += int(pred == true)
                by_k[k]["p_correct"].append(float(probs.get(true, 0.0)))

        for rec in by_k:
            n = rec["n"]
            rec["accuracy"] = rec["n_correct"] / n if n else 0.0
            ps = rec["p_correct"]
            rec["mean_p"] = sum(ps) / len(ps) if ps else 0.0
        per_strength.append({"strength": strength, "by_k": by_k})
    return per_strength
