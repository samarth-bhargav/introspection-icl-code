# Experiment instructions

Run from the repository root after following the [setup](../README.md#setup).
See the [validation record](reproduction-status.md) for tested coverage.
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
python -m icl.experiments.run_model --model "$MODEL" --gpu 0
```

Libraries and calibration tables are written to `icl/artifacts/<model>/`.
Magnitude sweeps and generalization write `evals/regen/constitution_source_magnitude/`;
layer sweeps and generalization write `evals/regen/layer/` and
`evals/regen/layer_generalization/`. The standalone magnitude commands use the
same implementations and paths. Defaults are 30 samples per prompt and 10 prompts.
Use `--stages build,layer,layergen` to select stages.

The calibration suite now contains 20 questions with single-token answers on all
five tokenizers. Old calibration files are rejected; rerun `build_library` to
recompute them. A different concept pool or extraction setup requires `--rebuild`.
For a reduced test, use `REGEN_TAG=smoke` with `--smoke` to isolate its artifacts.
Every run retains per-example traces in `evals/traces/` (override with
`ICL_TRACE_PATH`), including predictions, targets, injection plans, probabilities,
and calibration outcomes. Behavioral JSON files also retain generated answers
and judge responses.

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
  --dtype bfloat16 --gpu-memory-utilization 0.35 --max-model-len 4096 --enforce-eager
```

Wait for `curl --fail http://127.0.0.1:8002/v1/models` to succeed. The following
example uses **Qwen3-8B**, GPU 1 for the evaluated model, and the fixed fractions
in the paper’s appendix table:

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

For all five models, `scripts/run_gated.py` uses the appendix’s fixed fractions from `config.py`.
After building all libraries and self-briefs, inspect its commands, then launch:

```bash
python scripts/run_gated.py --dry-run
python scripts/run_gated.py --gpus 0,1 --judge-gpu 0
```

GPU 0 accepts only small-model jobs alongside the judge; another worker GPU is
required for 32B models. Use `--judge-gpu none --judge-url <url>` for a separate
judge host. Logs go to `logs/gated/`. Any failed job makes the scheduler exit
unsuccessfully. Concurrent runs should use separate checkouts/output directories.

## Behavioral strength sweeps

These produce the appendix strength-sweep inputs in addition to the k sweeps above:

```bash
python -m icl.experiments.tasks.run_arithmetic --model "$MODEL" --gpu 1 \
  --emotion_pool "$EMOTIONS" --k_type1 10 --k_values 0 \
  --prompt_variations 10 --n_tests 30 --no_control \
  --judge_base_url "$JUDGE_URL" --out_dir "evals/regen/generation_${MODEL}/math"
python -m icl.experiments.tasks.run_successor --model "$MODEL" --gpu 1 \
  --randomize_emotion --emotion_pool "$EMOTIONS" --skip_ksweep \
  --n_variations 10 --samples_per_var 30 --no_control \
  --judge_base_url "$JUDGE_URL" --out_root evals/regen --run_name successor
```

## Figures

```bash
python make_figures.py --check
python make_figures.py
```

All nine groups render from measured JSON, including the three layer panels.
Each group expects all five models. `--only magnitude,layer-sweeps` selects groups.
The file check verifies presence, not complete sample counts or paper-score agreement.

Former `gated6/` commands are now:

| Old file | Replacement |
| --- | --- |
| `gen_self_briefs.py` | `python -m icl.experiments.tasks.generate_briefs` |
| `scheduler.py` | `python scripts/run_gated.py` |
| `plot_full_6emo_figs.py` | `python -m icl.plotting.plot_gated` |
| `plot_anger_figs.py` | `python -m icl.plotting.plot_anger` |
