# Introspection-ICL — code only

The **source code** to reproduce the paper *"Reasoning and learning about
injected concepts in language models"* — the experiment pipeline plus the
figure-plotting scripts. This is a code-only tree: **no eval datasets, no model
artifacts, no rendered figures**. Generate the data on a GPU (see below), then
render the figures.

> A companion package with the eval data + concept libraries already shipped
> (so the figures render with no GPU) lives in `../Introspection-ICL-Final`.

The label-substitution appendix figure and its scripts are intentionally
omitted here.

## What's here

```
icl/
  model.py query.py logits.py utils.py     core ICL primitives
  common/                    helpers shared across all experiments
    helpers.py                 c_max cache + small run helpers
    layer_introspection.py     layer concept pool + early/middle/late anchors
    prompt_variations.py       per-task prompt-variation banks
  steering/                  concept-vector construction + injection
  experiments/               the canonical experiment pipeline
    config.py                  models, layers, concept pool, sweep grids
    build_library.py           per-model mean-diff concept library + c_max
    run_model.py               build -> magnitude/layer sweeps -> generalization
    magnitude.py               magnitude type1/type2/generalization data
    singlepass.py introspection_sweep.py   single-pass ICL readout core
    tasks/                     emotion-gated behavioral tasks
      gated_arithmetic.py  run_arithmetic.py    arithmetic task + sweep runner
      amendment_successor.py  run_successor.py   successor task + sweep runner
      amendment_common.py  successor_prompts.py  task_utils.py   shared helpers
  plotting/                  the Plotly paper-figure renderers + apply_paper_styling
gated6/                      orchestration for the 6-emotion gated runs
  scheduler.py gen_self_briefs.py
  plot_full_6emo_figs.py plot_anger_figs.py
make_figures.py              render every figure once the eval data exists
pyproject.toml  uv.lock
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .                 # honors the pinned transformers commit
```

The `transformers` pin in `pyproject.toml` is **load-bearing** (newer builds
break the Gemma logit readouts). The data-generation steps need a CUDA GPU and
the vLLM extra: `pip install -e '.[vllm]'`. Rendering needs a headless-Chromium
backend for Kaleido:

```bash
sudo apt-get install -y libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 \
  libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2
```

## Reproduce (data → figures)

Models (short name → HF id, from `icl/model.py`): `gemma-31b`→`google/gemma-4-31B-it`,
`qwen3-32b`→`Qwen/Qwen3-32B`, `qwen3-8b`→`Qwen/Qwen3-8B`,
`olmo-32b`→`allenai/Olmo-3.1-32B-Instruct`, `olmo-7b`→`allenai/Olmo-3-7B-Instruct`.

**1. Concept libraries** (one per model; written to `icl/artifacts/<model>/`):

```bash
python -m icl.experiments.build_library --model <short> --gpu 0
python gated6/gen_self_briefs.py   --model <short> --gpu 0   # successor briefs
```

**2. Magnitude / layer / generalization data:**

```bash
python -m icl.experiments.run_model --model <short> --gpu 0                 # sweeps + generalization
python -m icl.experiments.magnitude run --model <short> --gpu 0   # magnitude figures' data
```

**3. Judge daemon** (only for the gated tasks) — a Qwen3-8B judge over an
OpenAI-compatible endpoint; start it once and leave it running:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
  --served-model-name synonym-judge \
  --host 127.0.0.1 --port 8002 \
  --dtype bfloat16 --gpu-memory-utilization 0.35 --max-model-len 4096
curl -s http://127.0.0.1:8002/v1/models     # wait for HTTP 200
```

**4. Emotion-gated arithmetic + amendment-successor data** (judge must be up;
per-model best strength `f*` shown):

```bash
EMO=anger,fear,joy,love,sadness,disgust ; JURL=http://127.0.0.1:8002/v1

# arithmetic    f*: gemma .4  qwen3-32b .8  qwen3-8b 1.0  olmo-32b 1.0  olmo-7b .7
python -m icl.experiments.tasks.run_arithmetic --model <short> --gpu 1 \
  --emotion_pool "$EMO" --k_values 0-20 --k_type1 10 --cmax_grid <f*> \
  --prompt_variations 10 --n_tests 30 --target_prob 0.5 --distractor_prob 0.25 \
  --max_operand 9 --max_new_tokens 8 --seed 0 \
  --judge_base_url "$JURL" --judge_model synonym-judge \
  --out_dir evals/full_6emo/generation_<short>/math

# successor     f*: gemma .4  qwen3-32b .2  qwen3-8b .1  olmo-32b .2  olmo-7b .1
python -m icl.experiments.tasks.run_successor --model <short> --gpu 1 \
  --randomize_emotion --emotion_pool "$EMO" --fraction <f*> --skip_calibration \
  --k_values 0-10 --n_variations 10 --samples_per_var 30 \
  --target_prob 0.5 --distractor_prob 0.25 --max_new_tokens 40 --seed 13 \
  --out_root evals/full_6emo --run_name successor_emotions \
  --judge_base_url "$JURL" --judge_model synonym-judge
```

**5. Render the figures** into `plots/` and `plots_new/`:

```bash
PYTHONPATH=. python make_figures.py            # all figures
PYTHONPATH=. python make_figures.py --list     # figure -> command map
```
