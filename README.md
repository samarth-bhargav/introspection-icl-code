# Introspection ICL

Code for *Reasoning and learning about injected concepts in language models*.
The experiments test injection-magnitude classification, injection-layer
classification, emotion-gated arithmetic, and amendment successor.

The pipeline generates the paper's figure inputs from fresh runs. It uses the
paper's calibration, strength-selection metric, prompt counts, and operating
strengths. Historical measurements and model weights are not bundled; fresh
results need not match the published numbers. See the
[validation record](docs/reproduction-status.md) for tested coverage.

## Setup

Use Python 3.13 on Linux with CUDA for model inference:

```bash
uv sync --locked --python 3.13
source .venv/bin/activate
```

The Transformers commit is pinned for the Gemma chat template. Keep that pin
unless you have checked the label readouts. Behavioral experiments also need a
Qwen3-8B judge; install it with `uv sync --locked --group vllm`.

For plotting existing logs without the inference stack:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plotting.txt
```

PNG export needs Chrome or Chromium. If none is installed, run
`plotly_get_chrome`. Run the commands below from the repository root.

## Run

For one model (`qwen3-8b` shown):

```bash
python -m icl.experiments.run_model --model qwen3-8b --gpu 0
```

[Experiment instructions](docs/experiments.md) cover the other models, judge,
behavioral tasks, output paths, and optional multi-GPU runner.

```bash
python make_figures.py --list                 # figure groups and commands
python make_figures.py --check                # required input files
python make_figures.py --only magnitude       # render one complete group
python make_figures.py                        # all groups, if inputs are present
```

`--check` checks file presence, not complete sweeps or agreement with the paper.
Missing inputs stop rendering; they are not silently treated as completed figures.

## Layout

- `icl/steering/`: concept vectors, live-norm injection, and calibration.
- `icl/experiments/`: magnitude/layer runners; `tasks/` contains behavioral tasks.
- `icl/plotting/`: figure renderers, including six-emotion and anger subsets.
- `scripts/`: multi-GPU orchestration and bounded Modal validation.
- `docs/`: experiment instructions and reproduction audit.
- `tests/`: CPU checks (`python -m unittest discover -s tests`).

Generated files retain their existing locations: `icl/artifacts/` for concept
libraries, `evals/` for measurements, and `plots/` / `plots_new/` for paper figures.
Detailed per-example traces are saved under `evals/traces/`; set `ICL_TRACE_PATH`
to choose another location.
