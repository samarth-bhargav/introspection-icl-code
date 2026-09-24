"""Magnitude classification and generalization with one-vs-rest concept vectors.

Load libraries built by icl.experiments.build_library, run strength and example-
count sweeps at 20% model depth, and save measurements for the paper renderers.
See docs/reproduction-status.md for validation scope.

Usage:
    python -m icl.experiments.magnitude run --model gemma-31b --gpu 0
    python -m icl.experiments.magnitude plot
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from random import Random

import numpy as np

_CONFIG_PATH = Path(__file__).resolve().with_name("config.py")
_CONFIG_SPEC = importlib.util.spec_from_file_location("experiments_config", _CONFIG_PATH)
if _CONFIG_SPEC is None or _CONFIG_SPEC.loader is None:
    raise RuntimeError(f"cannot load config from {_CONFIG_PATH}")
C = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(C)


OUT_ROOT = C.EVALS_ROOT / "constitution_source_magnitude"
SOURCE_SPECS = {model: {"source": "artifact_meandiff", "method": "meandiff_ovr"}
                for model in C.MODELS}

Z = 1.96


def _log(msg: str) -> None:
    print(f"[magnitude] {msg}", flush=True)


def _boot(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env")
    except ImportError:
        pass


def _source_spec(model: str) -> dict:
    if model not in SOURCE_SPECS:
        raise ValueError(f"unknown model {model!r}; known={sorted(SOURCE_SPECS)}")
    return SOURCE_SPECS[model]


def _concepts(smoke_concepts: int = 0) -> list[str]:
    return C.CONCEPTS if smoke_concepts <= 0 else C.CONCEPTS[:smoke_concepts]


def _source_metadata_path(model: str) -> Path:
    return C.metadata_path(model)


def load_source_library_and_cmax(
    model_name: str,
    model,
    tok,
    *,
    concepts: list[str],
    out_root: Path,
    force_cmax: bool = False,
):
    """Load the paper's one-vs-rest library and magnitude-layer calibration."""
    from icl.steering.concepts import ConceptLibrary
    _source_spec(model_name)
    lib_path = C.library_path(model_name)
    if force_cmax:
        from icl.experiments.build_library import build_and_calibrate
        metadata = json.loads(C.metadata_path(model_name).read_text())
        build_and_calibrate(model, tok, model_name, concepts=metadata["concepts"], force_cmax=True)
    library = ConceptLibrary.load(lib_path)
    cmax = C.load_cmax(model_name)
    missing = [c for c in concepts if c not in library]
    if missing:
        raise ValueError(f"{model_name}: concepts missing from artifact library: {missing}")
    for concept in concepts:
        C.cmax_or_floor(cmax, concept, C.MAGNITUDE_LAYER[model_name])
    return library, cmax


def pick_argmax_star(type1_json: Path, k: int) -> dict:
    from icl.experiments.selection import pick_mean_probability
    return pick_mean_probability(type1_json, k)


def _magnitude_prompt_variations(n_prompt_variations: int) -> list[tuple[int, dict]]:
    """Return prompt variations for magnitude runs.

    ``n_prompt_variations <= 0`` preserves the historical single canonical
    prompt path.  Positive values use the centralized prompt-variation table.
    """
    if n_prompt_variations <= 0:
        return [(-1, {
            "prompt_text": C.MAGNITUDE_TRIGGER,
            "system_prompt": C.MAGNITUDE_SYSTEM,
        })]
    from icl.common.prompt_variations import VARIATIONS

    variations = VARIATIONS["magnitude_introspection"][:n_prompt_variations]
    if len(variations) != n_prompt_variations:
        raise ValueError(
            f"requested {n_prompt_variations} prompt variations, "
            f"found {len(variations)}"
        )
    return list(enumerate(variations))


def _prompt_metadata(prompt_specs: list[tuple[int, dict]]) -> list[dict]:
    return [
        {
            "prompt_variation": idx,
            "prompt_text": var["prompt_text"],
            "system_prompt": var["system_prompt"],
        }
        for idx, var in prompt_specs
    ]


def _pool_prompt_sweeps(prompt_runs: list[tuple[int, dict, list[dict]]]) -> list[dict]:
    """Pool SP.run_sweep outputs across prompt variations.

    The existing plotting code reads pooled ``p_correct`` arrays directly.  We
    keep those arrays pooled while retaining compact per-prompt summaries under
    ``by_prompt`` for auditability and prompt-sensitivity checks.
    """
    if not prompt_runs:
        return []
    n_strengths = len(prompt_runs[0][2])
    pooled: list[dict] = []
    for strength_i in range(n_strengths):
        strength = float(prompt_runs[0][2][strength_i]["strength"])
        n_k = len(prompt_runs[0][2][strength_i]["by_k"])
        by_k = []
        for k_i in range(n_k):
            p_correct: list[float] = []
            n_total = 0
            n_correct = 0
            by_prompt = []
            for prompt_idx, prompt_var, per_strength in prompt_runs:
                entry = per_strength[strength_i]
                if float(entry["strength"]) != strength:
                    raise ValueError("prompt sweep strength grids do not match")
                rec = entry["by_k"][k_i]
                values = [float(v) for v in rec.get("p_correct", [])]
                n = int(rec.get("n", len(values)))
                correct = int(rec.get("n_correct", 0))
                n_total += n
                n_correct += correct
                p_correct.extend(values)
                by_prompt.append({
                    "prompt_variation": prompt_idx,
                    "prompt_text": prompt_var["prompt_text"],
                    "system_prompt": prompt_var["system_prompt"],
                    "n": n,
                    "n_correct": correct,
                    "accuracy": float(rec.get("accuracy", correct / n if n else 0.0)),
                    "mean_p": float(rec.get(
                        "mean_p",
                        sum(values) / len(values) if values else 0.0,
                    )),
                })
            by_k.append({
                "k": int(prompt_runs[0][2][strength_i]["by_k"][k_i]["k"]),
                "n": n_total,
                "n_correct": n_correct,
                "p_correct": p_correct,
                "accuracy": n_correct / n_total if n_total else 0.0,
                "mean_p": sum(p_correct) / len(p_correct) if p_correct else 0.0,
                "by_prompt": by_prompt,
            })
        pooled.append({"strength": strength, "by_k": by_k})
    return pooled


def _type1_path(out_root: Path, model_name: str) -> Path:
    return out_root / "magnitude" / f"type1_{model_name}.json"


def _type2_path(out_root: Path, model_name: str) -> Path:
    return out_root / "magnitude" / f"type2_{model_name}.json"


def _maggen_path(out_root: Path, model_name: str) -> Path:
    return out_root / "magnitude_generalization" / f"magnitude_generalization_{model_name}.json"


def _write_sweep_payload(
    out_path: Path,
    *,
    model_name: str,
    mode: str,
    strengths: list[float],
    T: int,
    n_samples: int,
    seed: int,
    per_strength: list[dict],
    n_samples_per_prompt: int | None = None,
    prompt_specs: list[tuple[int, dict]] | None = None,
) -> None:
    spec = _source_spec(model_name)
    prompt_specs = prompt_specs or []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "task": "magnitude",
        "mode": mode,
        "model": model_name,
        "source": spec,
        "source_metadata": str(_source_metadata_path(model_name)),
        "trigger": C.MAGNITUDE_TRIGGER,
        "system_prompt": C.MAGNITUDE_SYSTEM,
        "n_samples": n_samples,
        "n_samples_per_prompt": n_samples_per_prompt,
        "n_prompt_variations": len(prompt_specs) if prompt_specs else 0,
        "prompt_variations": _prompt_metadata(prompt_specs) if prompt_specs else [],
        "prompt_seed_policy": "same_sample_seed_per_prompt",
        "seed": seed,
        "T": T,
        "magnitude_layer": C.MAGNITUDE_LAYER[model_name],
        "strengths": strengths,
        "timestamp": datetime.now().isoformat(),
        "per_strength": per_strength,
    }, indent=2))
    _log(f"wrote {out_path}")


def run_type1_type2(
    model,
    tok,
    library,
    cmax,
    *,
    model_name: str,
    out_root: Path,
    n_samples: int,
    samples_per_prompt: int,
    prompt_variations: int,
    seed: int,
    kmax: int | None,
    concepts: list[str],
    force: bool = False,
) -> None:
    from icl.experiments import singlepass as SP

    mag_layer = C.MAGNITUDE_LAYER[model_name]
    prompt_specs = _magnitude_prompt_variations(prompt_variations)
    n_effective = samples_per_prompt * len(prompt_specs) if prompt_variations > 0 else n_samples
    t1_path = _type1_path(out_root, model_name)
    T1 = min(C.TYPE1_K + 1, len(concepts))
    if force or not t1_path.exists():
        _log(
            f"{model_name}: type1 magnitude sweep at L{mag_layer}, T={T1}, "
            f"prompts={len(prompt_specs)}, n_per_prompt="
            f"{samples_per_prompt if prompt_variations > 0 else n_samples}, "
            f"n_effective={n_effective}"
        )
        prompt_runs = []
        for prompt_idx, prompt_var in prompt_specs:
            _log(
                f"{model_name}: type1 prompt var={prompt_idx} "
                f"trigger={prompt_var['prompt_text']!r}"
            )
            per_prompt = SP.run_sweep(
                model,
                tok,
                library,
                task="magnitude",
                strengths=C.STRENGTH_GRID,
                T=T1,
                n_samples=samples_per_prompt if prompt_variations > 0 else n_samples,
                seed=seed,
                system_prompt=prompt_var["system_prompt"],
                trigger=prompt_var["prompt_text"],
                cmax=cmax,
                magnitude_layer=mag_layer,
                layer_anchors=None,
                concepts_pool=concepts,
            )
            prompt_runs.append((prompt_idx, prompt_var, per_prompt))
        per_strength = _pool_prompt_sweeps(prompt_runs)
        _write_sweep_payload(
            t1_path,
            model_name=model_name,
            mode="type1",
            strengths=C.STRENGTH_GRID,
            T=T1,
            n_samples=n_effective,
            seed=seed,
            per_strength=per_strength,
            n_samples_per_prompt=samples_per_prompt if prompt_variations > 0 else None,
            prompt_specs=prompt_specs if prompt_variations > 0 else None,
        )
    else:
        _log(f"{model_name}: reusing {t1_path}")

    star_k = min(C.TYPE1_K, T1 - 1)
    star = pick_argmax_star(t1_path, star_k)
    _log(
        f"{model_name}: m*={star['star']:g} "
        f"top acc@K{star_k}={star['accuracy']:.3f} mean_p={star['mean_p']:.3f}"
    )

    t2_path = _type2_path(out_root, model_name)
    T2 = min((kmax if kmax is not None else C.TYPE2_KMAX) + 1, len(concepts))
    if force or not t2_path.exists():
        _log(
            f"{model_name}: type2 magnitude sweep at m*={star['star']:g}, "
            f"T={T2}, prompts={len(prompt_specs)}, n_effective={n_effective}"
        )
        prompt_runs = []
        for prompt_idx, prompt_var in prompt_specs:
            _log(
                f"{model_name}: type2 prompt var={prompt_idx} "
                f"trigger={prompt_var['prompt_text']!r}"
            )
            per_prompt = SP.run_sweep(
                model,
                tok,
                library,
                task="magnitude",
                strengths=[star["star"]],
                T=T2,
                n_samples=samples_per_prompt if prompt_variations > 0 else n_samples,
                seed=seed,
                system_prompt=prompt_var["system_prompt"],
                trigger=prompt_var["prompt_text"],
                cmax=cmax,
                magnitude_layer=mag_layer,
                layer_anchors=None,
                concepts_pool=concepts,
            )
            prompt_runs.append((prompt_idx, prompt_var, per_prompt))
        per_strength = _pool_prompt_sweeps(prompt_runs)
        _write_sweep_payload(
            t2_path,
            model_name=model_name,
            mode="type2",
            strengths=[star["star"]],
            T=T2,
            n_samples=n_effective,
            seed=seed,
            per_strength=per_strength,
            n_samples_per_prompt=samples_per_prompt if prompt_variations > 0 else None,
            prompt_specs=prompt_specs if prompt_variations > 0 else None,
        )
    else:
        _log(f"{model_name}: reusing {t2_path}")


def _batched_steer_hook(entries):
    """Forward hook that injects per (example, position) with pre-measured norms.

    `entries` = list of (b_index, positions, unit_activation, scale). Matches the
    single-example `_make_adaptive_hook` semantics (measure ||h|| before any
    injection, then h[b,p] += scale*||h[b,p]||*act) but over a batch dimension.
    """
    import torch  # noqa: F401

    def hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out  # (B, S, D)
        norms = {}
        for b, positions, _act, _scale in entries:
            for p in positions:
                if (b, p) not in norms:
                    norms[(b, p)] = hs[b, p].float().norm().item()
        for b, positions, act, scale in entries:
            a = act.to(device=hs.device, dtype=hs.dtype)
            for p in positions:
                hs[b, p] = hs[b, p] + (scale * norms[(b, p)]) * a
        if isinstance(out, tuple):
            return (hs,) + out[1:]
        return hs

    return hook


def _batched_last_probs(model, tok, library, queries, output_tokens, batch_size):
    """Batched equivalent of run_queries' last-token readout for maggen.

    Right-pads each batch (real tokens stay prefix-aligned, so steering positions
    and RoPE are identical to the unpadded pass; padding is masked out and the last
    real token attends only to real tokens under causal masking). Returns a list of
    {label: prob} dicts, one per query, identical in form to result.probabilities.
    """
    import torch
    from collections import defaultdict
    from icl.query import build_prompt
    from icl.steering.injection import find_trigger_positions, _get_target_module
    from icl.experiments.singlepass import group_probs_at

    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out_probs = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        enc = [tok.encode(build_prompt(q, tok), add_special_tokens=False) for q in chunk]
        lengths = [len(e) for e in enc]
        maxlen = max(lengths)
        B = len(chunk)
        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, maxlen), dtype=torch.long, device=device)
        layer_entries = defaultdict(list)
        for b, (q, ids) in enumerate(zip(chunk, enc)):
            t = torch.tensor(ids, dtype=torch.long, device=device)
            input_ids[b, :len(ids)] = t
            attn[b, :len(ids)] = 1
            spans = find_trigger_positions(t, tok, q.prompts)
            for span, concept, layer, scale in zip(
                spans, q.concepts, q.injection_layer, q.injection_scale
            ):
                act = library.get_vector(concept).layer_activations[layer]
                layer_entries[layer].append((b, span, act, scale))
        handles = [
            _get_target_module(model, L).register_forward_hook(_batched_steer_hook(ents))
            for L, ents in layer_entries.items()
        ]
        try:
            with torch.no_grad():
                logits = model(input_ids, attention_mask=attn).logits
        finally:
            for h in handles:
                h.remove()
        for b in range(B):
            out_probs.append(group_probs_at(logits[b, lengths[b] - 1], output_tokens, tok))
    return out_probs


def run_magnitude_generalization(
    model,
    tok,
    library,
    cmax,
    *,
    model_name: str,
    out_root: Path,
    m_star: float,
    n_examples: int,
    n_samples: int,
    prompt_variations: int,
    seed: int,
    sweep: tuple[float, float, float],
    concepts: list[str],
    force: bool = False,
    maggen_batch_size: int = 1,
) -> None:
    from icl import ICLQuery, run_queries

    out_path = _maggen_path(out_root, model_name)
    if out_path.exists() and not force:
        _log(f"{model_name}: reusing {out_path}")
        return

    layer = C.MAGNITUDE_LAYER[model_name]
    anchor_levels = [(lab, m_star * C.MAGNITUDE_BASE[lab]) for lab in C.MAGNITUDE_LABELS]
    lo, hi, step = sweep
    test_alphas = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]
    prompt_specs = _magnitude_prompt_variations(prompt_variations)
    n_effective = n_samples * len(prompt_specs) if prompt_variations > 0 else n_samples

    def cm(concept: str) -> float:
        return C.cmax_or_floor(cmax, concept, layer)

    _log(
        f"{model_name}: magnitude generalization L{layer}, m*={m_star:g}, "
        f"alphas={len(test_alphas)}, prompts={len(prompt_specs)}, "
        f"n_per_prompt={n_samples}, n_effective={n_effective}"
    )
    records = []
    t0 = time.time()
    for ta in test_alphas:
        counts = {lab: 0 for lab in C.MAGNITUDE_LABELS}
        p_sums = {lab: 0.0 for lab in C.MAGNITUDE_LABELS}
        samples, by_prompt = [], []
        for prompt_idx, prompt_var in prompt_specs:
            rng = Random(seed + int(round(ta * 1000)))
            prompt_counts = {lab: 0 for lab in C.MAGNITUDE_LABELS}
            prompt_p_sums = {lab: 0.0 for lab in C.MAGNITUDE_LABELS}
            queries = []
            for _ in range(n_samples):
                chosen = rng.sample(concepts, min(n_examples + 1, len(concepts)))
                demo_concepts = chosen[:-1]
                test_concept = chosen[-1]
                demo_levels = [rng.choice(anchor_levels) for _ in demo_concepts]
                all_concepts = demo_concepts + [test_concept]
                scales = [
                    frac * cm(concept)
                    for concept, (_label, frac) in zip(demo_concepts, demo_levels)
                ] + [ta * cm(test_concept)]
                queries.append(ICLQuery(
                    concepts=all_concepts,
                    words=[label for label, _frac in demo_levels],
                    output_tokens=C.MAGNITUDE_OUTPUT_TOKENS,
                    system_prompt=prompt_var["system_prompt"],
                    injection_layer=[layer] * len(all_concepts),
                    injection_scale=scales,
                    prompts=[prompt_var["prompt_text"]] * len(all_concepts),
                ))
            if maggen_batch_size and maggen_batch_size > 1:
                probs_list = _batched_last_probs(
                    model, tok, library, queries, C.MAGNITUDE_OUTPUT_TOKENS, maggen_batch_size
                )
            else:
                probs_list = [
                    r.probabilities
                    for r in run_queries(queries, model=model, tokenizer=tok, concept_library=library)
                ]
            for sample_id, (query, probabilities) in enumerate(zip(queries, probs_list)):
                from dataclasses import asdict
                from icl.experiments.telemetry import emit
                emit("magnitude_generalization_sample", model=model_name,
                     seed=seed, prompt_variation=prompt_idx, sample_id=sample_id,
                     test_alpha=ta, query=asdict(query), probabilities=probabilities)
                pred = max(probabilities, key=probabilities.get)
                counts[pred] += 1
                prompt_counts[pred] += 1
                for lab in C.MAGNITUDE_LABELS:
                    prob = probabilities.get(lab, 0.0)
                    p_sums[lab] += prob
                    prompt_p_sums[lab] += prob
                samples.append({
                    "sample_id": sample_id, "query": asdict(query),
                    "prompt_variation": prompt_idx,
                    "prompt_text": prompt_var["prompt_text"],
                    "predicted": pred,
                    "probabilities": {
                        lab: float(probabilities.get(lab, 0.0))
                        for lab in C.MAGNITUDE_LABELS
                    },
                })
            by_prompt.append({
                "prompt_variation": prompt_idx,
                "prompt_text": prompt_var["prompt_text"],
                "system_prompt": prompt_var["system_prompt"],
                "counts": prompt_counts,
                "mean_p": {
                    lab: prompt_p_sums[lab] / n_samples
                    for lab in C.MAGNITUDE_LABELS
                },
                "n_total": n_samples,
            })
        records.append({
            "test_alpha": ta,
            "counts": counts,
            "mean_p": {lab: p_sums[lab] / n_effective for lab in C.MAGNITUDE_LABELS},
            "n_total": n_effective,
            "samples": samples,
            "by_prompt": by_prompt,
        })
        if abs(ta * 10 - round(ta * 10)) < 1e-9 and int(round(ta * 10)) % 10 == 0:
            _log(
                f"{model_name}: alpha={ta:.1f} "
                f"p(low)={records[-1]['mean_p']['low']:.2f} "
                f"p(med)={records[-1]['mean_p']['medium']:.2f} "
                f"p(high)={records[-1]['mean_p']['high']:.2f}"
            )

    spec = _source_spec(model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "experiment": "magnitude_generalization",
        "model": model_name,
        "source": spec,
        "source_metadata": str(_source_metadata_path(model_name)),
        "layer": layer,
        "labels": C.MAGNITUDE_LABELS,
        "m_star": m_star,
        "anchor_levels": anchor_levels,
        "n_examples": n_examples,
        "n_samples": n_effective,
        "n_samples_per_prompt": n_samples if prompt_variations > 0 else None,
        "n_prompt_variations": len(prompt_specs) if prompt_variations > 0 else 0,
        "prompt_variations": _prompt_metadata(prompt_specs) if prompt_variations > 0 else [],
        "prompt_seed_policy": "same_sample_seed_per_prompt",
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "records": records,
        "wall_s": round(time.time() - t0, 1),
    }, indent=2))
    _log(f"wrote {out_path}")


def _ci(values: list[float]) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return mean, max(0.0, mean - Z * se), min(1.0, mean + Z * se)


def _plot_type1(out_root: Path, plots_dir: Path, summary: dict) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    made = False
    for model_name in C.MODELS:
        path = _type1_path(out_root, model_name)
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        kmax = max(rec["k"] for rec in data["per_strength"][0]["by_k"])
        read_k = min(C.TYPE1_K, kmax)
        xs, means, lo, hi = [], [], [], []
        for entry in data["per_strength"]:
            rec = next(r for r in entry["by_k"] if r["k"] == read_k)
            mean, l, h = _ci(rec["p_correct"])
            xs.append(entry["strength"])
            means.append(mean)
            lo.append(l)
            hi.append(h)
        star = pick_argmax_star(path, read_k)
        color = C.MODEL_COLORS[model_name]
        ax.plot(
            xs,
            means,
            color=color,
            lw=2,
            label=f"{C.MODEL_DISPLAY[model_name]} (m*={star['star']:g})",
        )
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
        ax.axvline(star["star"], color=color, ls="--", lw=1, alpha=0.5)
        summary.setdefault(model_name, {})["type1"] = {
            "m_star": star["star"],
            "top_accuracy@K30": star["accuracy"],
            "mean_p@K30": star["mean_p"],
            "source": _source_spec(model_name),
            "magnitude_layer": data["magnitude_layer"],
        }
        made = True
    if not made:
        plt.close(fig)
        return False
    ax.axhline(1 / 3, color="gray", ls=":", lw=1.5, label="Chance (0.333)")
    ax.set_xlabel("Strength multiplier m", fontsize=13)
    ax.set_ylabel("Mean P(correct)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_title("Magnitude introspection - strength sweep (K=30)", fontsize=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "type1_magnitude_introspection.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {out_path}")
    return True


def _plot_type2(out_root: Path, plots_dir: Path, summary: dict) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    made = False
    for model_name in C.MODELS:
        path = _type2_path(out_root, model_name)
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        xs, means, lo, hi = [], [], [], []
        by_k = data["per_strength"][0]["by_k"]
        for rec in by_k:
            mean, l, h = _ci(rec["p_correct"])
            xs.append(rec["k"])
            means.append(mean)
            lo.append(l)
            hi.append(h)
        color = C.MODEL_COLORS[model_name]
        ax.plot(xs, means, color=color, lw=2, label=C.MODEL_DISPLAY[model_name])
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
        k30_rec = next((r for r in by_k if r["k"] == C.TYPE1_K), by_k[-1])
        summary.setdefault(model_name, {})["type2"] = {
            "m_star": data["strengths"][0],
            "accuracy@K30": k30_rec["accuracy"],
            "mean_p@K30": k30_rec["mean_p"],
            "accuracy@Kmax": by_k[-1]["accuracy"],
            "mean_p@Kmax": by_k[-1]["mean_p"],
            "Kmax": by_k[-1]["k"],
        }
        made = True
    if not made:
        plt.close(fig)
        return False
    ax.axhline(1 / 3, color="gray", ls=":", lw=1.5, label="Chance (0.333)")
    ax.set_xlabel("Number of in-context examples (K)", fontsize=13)
    ax.set_ylabel("Mean P(correct)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_title("Magnitude introspection", fontsize=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "type2_magnitude_introspection.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {out_path}")
    return True


def _plot_mag_gen_one(out_root: Path, plots_dir: Path, model_name: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = _maggen_path(out_root, model_name)
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    colors = {"low": "#2563eb", "medium": "#16a34a", "high": "#dc2626"}
    xs = [r["test_alpha"] for r in data["records"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    for label in C.MAGNITUDE_LABELS:
        means, lo, hi = [], [], []
        for rec in data["records"]:
            values = [s["probabilities"][label] for s in rec["samples"]]
            mean, l, h = _ci(values)
            means.append(mean)
            lo.append(l)
            hi.append(h)
        ax.plot(xs, means, color=colors[label], lw=2, label=label)
        ax.fill_between(xs, lo, hi, color=colors[label], alpha=0.18)
    for label, alpha in data["anchor_levels"]:
        ax.axvline(alpha, color=colors[label], ls="--", lw=1.2, alpha=0.7)
    ax.axhline(1 / 3, color="gray", ls=":", lw=1.5, label="Chance (0.333)")
    ax.set_xlabel("Test query magnitude alpha (x c_max)", fontsize=13)
    ax.set_ylabel("Mean P(class)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{C.MODEL_DISPLAY[model_name]} magnitude introspection generalization", fontsize=14)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"magnitude_generalization_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {out_path}")
    return True


def _plot_mag_gen_combined(out_root: Path, plots_dir: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    for model_name in C.MODELS:
        path = _maggen_path(out_root, model_name)
        if path.exists():
            records.append((model_name, json.loads(path.read_text())))
    if not records:
        return False

    colors = {"low": "#2563eb", "medium": "#16a34a", "high": "#dc2626"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    for ax, (model_name, data) in zip(axes_flat, records):
        xs = [r["test_alpha"] for r in data["records"]]
        for label in C.MAGNITUDE_LABELS:
            means = []
            for rec in data["records"]:
                values = [s["probabilities"][label] for s in rec["samples"]]
                means.append(_ci(values)[0])
            ax.plot(xs, means, color=colors[label], lw=2, label=label)
        for label, alpha in data["anchor_levels"]:
            ax.axvline(alpha, color=colors[label], ls="--", lw=1.0, alpha=0.55)
        ax.axhline(1 / 3, color="gray", ls=":", lw=1.0)
        ax.set_title(C.MODEL_DISPLAY[model_name], fontsize=12)
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(records):]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.98, 0.5))
    fig.supxlabel("Test query magnitude alpha (x c_max)", fontsize=13)
    fig.supylabel("Mean P(class)", fontsize=13)
    fig.suptitle("Magnitude introspection generalization", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "magnitude_generalization_all_models.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {out_path}")
    return True


def write_summary_markdown(out_root: Path, summary: dict) -> None:
    lines = [
        "# Constitution-source magnitude introspection",
        "",
        "| model | source | layer | m* | type1 top acc@K30 | type2 acc@K30 | type2 acc@Kmax |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in C.MODELS:
        row = summary.get(model_name, {})
        t1 = row.get("type1", {})
        t2 = row.get("type2", {})
        src = _source_spec(model_name)
        src_name = f"{src['source']}:{src['method']}"
        lines.append(
            f"| {model_name} | {src_name} | {C.MAGNITUDE_LAYER[model_name]} | "
            f"{t1.get('m_star', '?')} | {_fmt(t1.get('top_accuracy@K30'))} | "
            f"{_fmt(t2.get('accuracy@K30'))} | {_fmt(t2.get('accuracy@Kmax'))} |"
        )
    path = out_root / "plots" / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    (out_root / "plots" / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"wrote {path}")


def _fmt(value) -> str:
    return "?" if value is None else f"{float(value):.3f}"


def cmd_run(args) -> None:
    _boot(args.gpu)
    import torch
    from icl import get_model_and_tokenizer

    torch.set_grad_enabled(False)
    model_name = args.model
    concepts = _concepts(args.smoke_concepts)
    out_root = Path(args.out_root)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    _log(
        f"run model={model_name} source={_source_spec(model_name)} "
        f"concepts={len(concepts)} stages={sorted(stages)} out={out_root}"
    )
    model, tok = get_model_and_tokenizer(model_name)
    library, cmax = load_source_library_and_cmax(
        model_name,
        model,
        tok,
        concepts=concepts,
        out_root=out_root,
        force_cmax=args.force_cmax,
    )
    if stages & {"type1", "type2"}:
        run_type1_type2(
            model,
            tok,
            library,
            cmax,
            model_name=model_name,
            out_root=out_root,
            n_samples=args.n_samples,
            samples_per_prompt=args.samples_per_prompt,
            prompt_variations=args.prompt_variations,
            seed=args.seed,
            kmax=args.kmax,
            concepts=concepts,
            force=args.force,
        )
    if "maggen" in stages:
        type1_path = _type1_path(out_root, model_name)
        T1 = min(C.TYPE1_K + 1, len(concepts))
        star_k = min(C.TYPE1_K, T1 - 1)
        m_star = pick_argmax_star(type1_path, star_k)["star"]
        run_magnitude_generalization(
            model,
            tok,
            library,
            cmax,
            model_name=model_name,
            out_root=out_root,
            m_star=m_star,
            n_examples=min(args.gen_n_examples, len(concepts) - 1),
            n_samples=args.gen_n_samples,
            prompt_variations=args.prompt_variations,
            seed=args.gen_seed,
            sweep=(args.sweep_start, args.sweep_end, args.sweep_step),
            concepts=concepts,
            force=args.force,
            maggen_batch_size=args.maggen_batch_size,
        )


def cmd_plot(args) -> None:
    out_root = Path(args.out_root)
    plots_dir = out_root / "plots"
    summary: dict = {}
    _plot_type1(out_root, plots_dir, summary)
    _plot_type2(out_root, plots_dir, summary)
    for model_name in C.MODELS:
        _plot_mag_gen_one(out_root, plots_dir, model_name)
    _plot_mag_gen_combined(out_root, plots_dir)
    write_summary_markdown(out_root, summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--model", required=True, choices=C.MODELS)
    run.add_argument("--gpu", default="0")
    run.add_argument("--out_root", default=str(OUT_ROOT))
    run.add_argument("--stages", default="type1,type2,maggen")
    run.add_argument("--n_samples", type=int, default=C.DEFAULT_N_SAMPLES)
    run.add_argument("--samples_per_prompt", type=int, default=30)
    run.add_argument("--prompt_variations", type=int, default=10)
    run.add_argument("--seed", type=int, default=13)
    run.add_argument("--kmax", type=int, default=None)
    run.add_argument("--gen_n_examples", type=int, default=20)
    run.add_argument("--gen_n_samples", type=int, default=30)
    run.add_argument("--gen_seed", type=int, default=42)
    run.add_argument("--maggen_batch_size", type=int, default=1,
                     help="batch N maggen queries per forward pass (>1 enables batching)")
    run.add_argument("--sweep_start", type=float, default=0.0)
    run.add_argument("--sweep_end", type=float, default=5.0)
    run.add_argument("--sweep_step", type=float, default=0.1)
    run.add_argument("--smoke_concepts", type=int, default=0)
    run.add_argument("--force", action="store_true")
    run.add_argument("--force_cmax", action="store_true")
    run.set_defaults(func=cmd_run)

    plot = sub.add_parser("plot")
    plot.add_argument("--out_root", default=str(OUT_ROOT))
    plot.set_defaults(func=cmd_plot)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
