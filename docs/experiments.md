# Experiment instructions

Run from the repository root after following the [setup](../README.md#setup).
These commands describe the included code. Read the
[reproduction audit](reproduction-status.md) for unresolved paper/code differences.
Inference requires CUDA; 32B models need enough GPU memory for weights and context.
Accept any required model license and authenticate with Hugging Face before
loading gated weights. Standard `HF_HOME` and `HF_HUB_OFFLINE` settings are honored.

## Models

| Short name | Hugging Face model |
| --- | --- |
| `gemma-31b` | `google/gemma-4-31B-it` |
| `qwen3-32b` | `Qwen/Qwen3-32B` |
| `qwen3-8b` | `Qwen/Qwen3-8B` |
| `olmo-32b` | `allenai/Olmo-3.1-32B-Instruct` |
| `olmo-7b` | `allenai/Olmo-3-7B-Instruct` |

## Concept libraries and classification

Repeat for each model:

```bash
MODEL=qwen3-8b
python -m icl.experiments.build_library --model "$MODEL" --gpu 0
python -m icl.experiments.magnitude run --model "$MODEL" --gpu 0
python -m icl.experiments.run_model --model "$MODEL" --gpu 0 \
  --stages layer,layergen --n_samples 30 --prompt_variations 10
```

Libraries and calibration tables are written to `icl/artifacts/<model>/`.
The dedicated magnitude runner writes `evals/regen/constitution_source_magnitude/`;
layer sweeps and generalization write `evals/regen/layer/` and
`evals/regen/layer_generalization/`. The generic runner also has magnitude stages,
but they use a different selection rule and output directory. They do not supply
the inputs consumed by the paper's magnitude renderer.

## Behavioral tasks

First generate model-specific amendment demonstration answers:

```bash
python -m icl.experiments.tasks.generate_briefs --model "$MODEL" --gpu 0
```

Start the Qwen3-8B judge on a separate GPU (install with
`uv sync --locked --group vllm`). Keep it running during behavioral evaluations:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
  --served-model-name synonym-judge --host 127.0.0.1 --port 8002 \
  --dtype bfloat16 --gpu-memory-utilization 0.35 --max-model-len 4096
```

Wait for `curl --fail http://127.0.0.1:8002/v1/models` to succeed. The following
example uses **Qwen3-8B**, GPU 1 for the evaluated model, and the fixed fractions
in the existing six-emotion scheduler:

```bash
MODEL=qwen3-8b
EMOTIONS=anger,fear,joy,love,sadness,disgust
JUDGE_URL=http://127.0.0.1:8002/v1

python -m icl.experiments.tasks.run_arithmetic --model "$MODEL" --gpu 1 \
  --emotion_pool "$EMOTIONS" --fixed_cmax 1.0 --k_values 0-20 \
  --prompt_variations 10 --n_tests 30 --target_prob 0.5 --distractor_prob 0.25 \
  --max_operand 9 --max_new_tokens 8 --seed 0 --no_control \
  --judge_base_url "$JUDGE_URL" --judge_model synonym-judge \
  --out_dir "evals/full_6emo/generation_${MODEL}/math"

python -m icl.experiments.tasks.run_successor --model "$MODEL" --gpu 1 \
  --randomize_emotion --emotion_pool "$EMOTIONS" --fraction 0.1 --skip_calibration \
  --k_values 0-10 --n_variations 10 --samples_per_var 30 \
  --target_prob 0.5 --distractor_prob 0.25 --max_new_tokens 40 --seed 13 --no_control \
  --out_root evals/full_6emo --run_name successor_emotions \
  --judge_base_url "$JUDGE_URL" --judge_model synonym-judge
```

`--no_control` skips the extra uninjected comparison pass; target, distractor,
and no-injection test conditions remain in the sampled dataset.

For all five models, `scripts/run_gated.py` contains the retained fixed fractions.
After building all libraries and self-briefs, inspect its commands, then launch:

```bash
python scripts/run_gated.py --dry-run
python scripts/run_gated.py --gpus 0,1 --judge-gpu 0
```

GPU 0 accepts only small-model jobs alongside the judge; another worker GPU is
required for 32B models. Use `--judge-gpu none --judge-url <url>` for a separate
judge host. Logs go to `logs/gated/`. Any failed job makes the scheduler exit
unsuccessfully. Concurrent runs should use separate checkouts/output directories.

## Figures

```bash
python make_figures.py --check
python make_figures.py --only magnitude,layer-generalization
python make_figures.py --only gated,anger,math-modes,successor-modes
```

Each figure group expects all five models. The file check does not establish that
all k values, prompts, samples, or emotion subsets are present. An unpopulated
checkout should fail the check. The full invocation also needs the absent
historical layer sources and the separate behavioral strength-sweep logs; see
[coverage](reproduction-status.md#coverage).

Former `gated6/` commands are now:

| Old file | Replacement |
| --- | --- |
| `gen_self_briefs.py` | `python -m icl.experiments.tasks.generate_briefs` |
| `scheduler.py` | `python scripts/run_gated.py` |
| `plot_full_6emo_figs.py` | `python -m icl.plotting.plot_gated` |
| `plot_anger_figs.py` | `python -m icl.plotting.plot_anger` |
