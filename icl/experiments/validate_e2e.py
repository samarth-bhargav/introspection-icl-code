"""GPU integration checks on real weights with retained per-example evidence.

This is a functional test, not an estimate of the manuscript's reported scores.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--detection-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["ICL_TRACE_PATH"] = str(root / "measurements.jsonl")
    os.environ["ICL_RUN_ID"] = root.name
    import torch
    import transformers
    from icl import get_model_and_tokenizer
    from icl.experiments import config as C
    from icl.experiments.telemetry import emit
    from icl.experiments.build_library import build_and_calibrate
    from icl.experiments.singlepass import run_sweep, run_conversation
    from icl.experiments.magnitude import run_type1_type2, run_magnitude_generalization
    from icl.experiments.introspection_sweep import sweep_and_save, pick_star
    from icl.experiments.run_layer_gen import run_layer_gen
    torch.set_grad_enabled(False)
    C.ARTIFACTS_ROOT = root / "artifacts"
    C.EVALS_ROOT = root / "evals" / "regen"
    started = time.monotonic()
    emit("run_start", model=args.model, torch=torch.__version__, transformers=transformers.__version__,
         gpu=torch.cuda.get_device_name(), source_sha=os.environ.get("ICL_SOURCE_SHA"),
         test_scope="8 concepts; 2 samples/prompt; 2 prompt variants; sparse strength grid; all layers")
    model, tok = get_model_and_tokenizer(args.model)
    emit("model_loaded", model=args.model, revision=getattr(model.config, "_commit_hash", None),
         attention_implementation=getattr(model.config, "_attn_implementation", None),
         dtype=str(next(model.parameters()).dtype))
    pool = C.CONCEPTS[:8]  # all six emotions and two non-emotion concepts
    library = build_and_calibrate(model, tok, args.model, concepts=pool)
    cmax = C.load_cmax(args.model)
    for concept in pool:
        for vector in library.get_vector(concept).layer_activations.values():
            assert torch.isfinite(vector).all()
            assert abs(vector.float().norm().item() - 1) < 1e-4
    # Zero injection must leave the forward pass unchanged; verify the live norm
    # equation independently on a known residual tensor.
    from icl.steering.injection import _make_adaptive_hook
    h = torch.tensor([[[3., 4.], [5., 12.]]], device="cuda")
    original = h.clone()
    v = torch.tensor([1., 0.], device="cuda")
    actual = _make_adaptive_hook([([0, 1], v, .2)])(None, None, h)
    expected = original + .2 * original.norm(dim=-1, keepdim=True) * v
    torch.testing.assert_close(actual, expected)
    emit("injection_equation_check", passed=True)
    from icl.steering.injection import forward_with_multi_layer_steering
    probe = tok.encode("Hello", return_tensors="pt", add_special_tokens=False).to(model.device)
    baseline = model(probe).logits
    steered_zero = forward_with_multi_layer_steering(model, probe,
        [([0], C.MAGNITUDE_LAYER[args.model], library.get_vector(pool[0]))], scales=[0.])
    torch.testing.assert_close(steered_zero, baseline, atol=0., rtol=0.)
    torch.testing.assert_close(model(probe).logits, baseline, atol=0., rtol=0.)
    emit("zero_injection_and_hook_cleanup_check", passed=True)

    # Compare equal-shaped inputs to isolate causality from BF16 matrix-kernel
    # differences at different sequence lengths. Future tokens AND future
    # interventions must not change any earlier readout.
    from icl.experiments.singlepass import build_conversation
    from icl.steering.injection import find_trigger_positions
    from icl.common.prompt_variations import VARIATIONS
    for task, labels in [("magnitude", C.MAGNITUDE_LABELS), ("layer", C.LAYER_LABELS)]:
        for variant in VARIATIONS[f"{task}_introspection"]:
            prompt_tokens, read_positions = build_conversation(tok, variant["system_prompt"],
                [variant["prompt_text"]] * 3, labels)
            prompt_ids = torch.tensor(prompt_tokens)
            prompt_spans = find_trigger_positions(prompt_ids, tok, [variant["prompt_text"]] * 3)
            assert len(read_positions) == len(prompt_spans) == 3
            assert all(tok.decode(prompt_ids[span], skip_special_tokens=False) == variant["prompt_text"]
                       for span in prompt_spans)
    emit("prompt_template_check", passed=True, tasks=2, variants_per_task=10)
    tokens, positions = build_conversation(tok, C.LAYER_SYSTEM,
        [C.LAYER_TRIGGER] * 3, C.LAYER_LABELS)
    ids = torch.tensor([tokens], device=model.device)
    spans = find_trigger_positions(ids[0], tok, [C.LAYER_TRIGGER] * 3)
    for span in spans:
        assert tok.decode(ids[0, span], skip_special_tokens=False) == C.LAYER_TRIGGER
    triples = [(span, layer, library.get_vector(concept))
               for span, layer, concept in zip(spans, C.LAYER_ANCHORS[args.model], pool)]
    original_logits = forward_with_multi_layer_steering(model, ids, triples, scales=[.1]*3)
    for count in [1, 2]:
        prefix_ids, prefix_pos = build_conversation(tok, C.LAYER_SYSTEM,
            [C.LAYER_TRIGGER] * count, C.LAYER_LABELS[:count])
        # Non-thinking templates append a dummy user turn after completed
        # labels; only the prefix through this readout belongs to the test.
        assert tokens[:prefix_pos[-1]+1] == prefix_ids[:prefix_pos[-1]+1]
        assert positions[:count] == prefix_pos
        changed = ids.clone()
        changed[:, positions[count-1]+1:] = tok.encode("hello", add_special_tokens=False)[0]
        altered_logits = forward_with_multi_layer_steering(model, changed,
            triples[:count], scales=[.1]*count)
        torch.testing.assert_close(altered_logits[:, :positions[count-1]+1],
                                   original_logits[:, :positions[count-1]+1], atol=0., rtol=0.)
    emit("causal_readout_check", passed=True, comparison="identical shape; all future tokens and injections changed",
         tolerance=0., injection_span="user content only")
    del original_logits, altered_logits

    C.STRENGTH_GRID = [0., .5, 1.]
    # magnitude.py loads its own configuration module for CPU plotting imports.
    from icl.experiments import magnitude as M
    M.C.ARTIFACTS_ROOT = C.ARTIFACTS_ROOT
    M.C.EVALS_ROOT = C.EVALS_ROOT
    M.C.STRENGTH_GRID = C.STRENGTH_GRID
    mag_root = C.EVALS_ROOT / "constitution_source_magnitude"
    run_type1_type2(model, tok, library, cmax, model_name=args.model, out_root=mag_root,
                   n_samples=2, samples_per_prompt=2, prompt_variations=2, seed=13,
                   kmax=7, concepts=pool, force=True)
    mstar = M.pick_argmax_star(M._type1_path(mag_root, args.model), 7)["star"]
    run_magnitude_generalization(model, tok, library, cmax, model_name=args.model,
        out_root=mag_root, m_star=mstar, n_examples=4, n_samples=2,
        prompt_variations=2, seed=42, sweep=(0., 2., 1.), concepts=pool, force=True,
        maggen_batch_size=2)
    p, _ = sweep_and_save(model, tok, library, cmax, model_name=args.model, task="layer",
                         mode="type1", n_samples=2, kmax=7, concepts_pool=pool, n_prompt_variations=2)
    lstar = pick_star(p, 7)["star"]
    for variation in range(2):
        sweep_and_save(model, tok, library, cmax, model_name=args.model, task="layer",
            mode="type2", strength=lstar, prompt_variation=variation,
            n_samples=2, kmax=7, concepts_pool=pool)
    run_layer_gen(model, tok, library, cmax, model_name=args.model,
                  cmax_fraction=lstar, n_examples=4, n_samples=2, concepts=pool, prompt_variations=2)
    emit("detection_complete", seconds=time.monotonic()-started,
         peak_gpu_bytes=torch.cuda.max_memory_allocated())
    (root / "detection-complete.json").write_text(json.dumps({"model": args.model,
        "magnitude_star": mstar, "layer_star": lstar, "seconds": time.monotonic()-started}, indent=2))
    if args.detection_only:
        return

    # Generate all 27 self-briefs with the actual evaluated model.
    from icl.experiments.singlepass import render_chat
    from icl.experiments.tasks.generate_briefs import SUMM_SYS
    briefs = []
    for n in range(1, 28):
        messages = [{"role": "system", "content": SUMM_SYS},
                    {"role": "user", "content": f"Amendment number: {n}. Answer briefly."}]
        ids = tok.encode(render_chat(tok, messages, True), return_tensors="pt",
                         add_special_tokens=False).to(model.device)
        output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=48,
                                do_sample=False, pad_token_id=tok.eos_token_id)
        full_text = tok.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
        briefs.append(full_text.split("\n", 1)[0].strip())
        emit("self_brief", amendment=n, messages=messages, response=full_text)
    (root / "self-briefs.json").write_text(json.dumps(briefs, indent=2))
    from icl.experiments.validation_judge import judge_server
    if args.model == "qwen3-8b":
        judge, judge_tok = model, tok
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        judge_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        judge = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda")
        judge.eval()
    from icl.experiments.tasks.gated_arithmetic import run_emotion_math_multiplier, judge_number_equal
    from icl.experiments.tasks.amendment_successor import run_amendment_successor
    from icl.experiments.tasks.successor_prompts import SUCCESSOR_PROMPT_VARIATIONS
    from icl.common.prompt_variations import VARIATIONS
    behavior = []
    with judge_server(judge, judge_tok) as url:
        assert judge_number_equal("12", 12, base_url=url)["match"]
        assert not judge_number_equal("13", 12, base_url=url)["match"]
        cases = [("type2", k, C.PAPER_MATH_FRACTIONS[args.model],
                  C.PAPER_SUCCESSOR_FRACTIONS[args.model]) for k in [0, 2]]
        cases += [("type1", 2, fraction, fraction) for fraction in [0., .5, 1.]]
        for phase, k, math_fraction, successor_fraction in cases:
            path = root / "evals" / "full_6emo" / f"generation_{args.model}" / "math" / "type2_k_sweep" / f"math_{args.model}_k{k}.json"
            if phase == "type1":
                path = root / "evals" / "regen" / f"generation_{args.model}" / "math" / "type1_cmax_sweep" / f"math_{args.model}_k{k}_cmax{math_fraction:g}.json"
            payload = run_emotion_math_multiplier(model, tok, library, cmax,
                model_name=args.model, cmax_fraction=math_fraction,
                n_demos=k, n_tests=12, emotion_pool=C.EMOTION_CONCEPTS,
                prompt_variations=VARIATIONS["math_introspection"][:2], emotion_prob=.5,
                distractor_prob=.25, seed=0, max_new_tokens=8, with_control=True,
                judge_base_url=url, judge_debug=True, out_path=path)
            behavior.extend(payload["rows"])
            for vi, variant in enumerate(SUCCESSOR_PROMPT_VARIATIONS[:2]):
                path = root / "evals" / "full_6emo" / "successor_emotions_k_sweep" / args.model / f"k{k}_var{vi}.json"
                if phase == "type1":
                    path = root / "evals" / "regen" / "successor_cmax_sweep" / args.model / f"fraction_{successor_fraction:.2f}_var{vi}.json"
                payload = run_amendment_successor(model, tok, library, cmax,
                    model_name=args.model, cmax_fraction=successor_fraction,
                    n_demos=k, n_tests=12, randomize_emotion=True, emotion_pool=C.EMOTION_CONCEPTS,
                    emotion_prob=.5, distractor_prob=.25, seed=13+vi, max_new_tokens=40,
                    with_control=True, judge_base_url=url, judge_debug=True,
                    system_prompt_template=variant["system_prompt"], user_prompt_template=variant["user_prompt"],
                    prompt_variation=vi, demo_briefs=briefs, out_path=path)
                behavior.extend(payload["rows"])
    assert {r["test_injection"]["kind"] for r in behavior} == {"target", "distractor", "none"}
    assert {r.get("target_emotion", r.get("gating_emotion")) for r in behavior} == set(C.EMOTION_CONCEPTS)
    from icl.experiments.tasks.gated_arithmetic import _judge_parse_failed
    parse_failures = sum(_judge_parse_failed(r[key]) for r in behavior
                         for key in ["judge", "control_judge"])
    assert parse_failures == 0, f"{parse_failures} judge responses were not parseable"
    summary = {"model": args.model, "state": "passed", "behavioral_rows": len(behavior),
               "seconds": time.monotonic()-started, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    emit("run_complete", **summary)
    (root / "complete.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
