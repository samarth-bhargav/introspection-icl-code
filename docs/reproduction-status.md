# Reproduction audit

Audited against the code in commit `a20f38d` and the manuscript sections supplied
in the accompanying review. This is a source and CPU-tooling audit, not a rerun
of the five models. The reported paper scores have not been independently
reproduced. No evaluation logs or model artifacts are included in this checkout.

## Coverage

| Result | Generator / renderer | Status |
| --- | --- | --- |
| Concept vectors and comprehension thresholds | `build_library`; `steering/ranges.py` | Implemented; GPU validation still needed. |
| Magnitude strength and example-count sweeps; generalization | `experiments.magnitude`; `plotting.plot_magnitude` | Connected generator and renderer; selection and vector-source differences below. |
| Layer strength and example-count sweeps; prompt sensitivity | `experiments.run_model` | Generates JSON, but the figure driver only restyles three historical Plotly files that are **not supplied**. No renderer connects these raw sweeps to those figures. |
| Layer generalization | `experiments.run_layer_gen`; `plotting.plot_layer_generalization` | JSON writer and rerenderer are connected; default runner uses one prompt, with calibration borrowed from the nearest anchor at uncalibrated layers. |
| Six-emotion behavioral example-count sweeps | `tasks.run_arithmetic`, `tasks.run_successor`; `plotting.plot_gated` | Implemented; requires concept libraries, successor self-briefs, and a running judge. |
| Anger subset, behavioral prompt sensitivity, injection-condition breakdowns | `plot_anger`, `plot_gated`, `plot_math_modes`, `plot_successor_modes` | Derived from the six-emotion row-level logs. |
| Behavioral strength-sweep appendix figures | `plotting.plot_strength_sweep` | Expects a separate older run under `evals/regen`, not the six-emotion k sweeps. Those logs and an authoritative historical run recipe are absent. Current runners can produce new sweeps, but their equivalence is unverified. |

## Differences to resolve before claiming paper reproduction

1. **Strength selection.** `magnitude.pick_argmax_star` maximizes hard-label
   accuracy among positive strengths, breaking ties toward the smaller strength.
   `introspection_sweep.pick_star` instead selects the first positive strength
   within one binomial standard error of peak accuracy. The manuscript says
   maximum mean `p(correct)`. These objectives can choose different strengths.
   Successor calibration also uses a one-standard-error rule.
2. **Successor operating strengths.** The retained six-emotion scheduler uses
   Gemma / Qwen-32B / Qwen-8B / Olmo-32B / Olmo-7B fractions of
   `0.4 / 0.2 / 0.1 / 0.2 / 0.1`. The reviewed appendix table gives
   `0.5 / 0.7 / 0.1 / 0.5 / 0.1`. The cleanup preserves the executable settings;
   neither set should be substituted for the other without checking run records.
3. **Magnitude vector provenance.** `experiments/magnitude.py` documents that
   Gemma and Qwen-32B originally used different research-cache constructions.
   This repository uses the one-vs-rest library for all models. Its comment
   mentions a five-sample parity check, but the check's evidence is not bundled.
4. **Sample and prompt counts.** The generic `run_model` default is 100 samples
   per prompt; the manuscript describes 30 samples across 10 prompts. The
   instructions pass `--n_samples 30` explicitly. Generic `run_layer_gen` uses
   only the canonical prompt. A separate evaluator in
   `plotting/plot_layer_generalization.py` supports prompt variations but expects
   legacy artifact filenames, so it is not a drop-in replacement.
5. **Calibration semantics.** The injection hook does use
   `h <- h + alpha * ||h||_2 * v`, measuring norms before addition.
   `compute_max_strength` bisects until the interval is at most 0.1 wide;
   it does not search a fixed 0.1-spaced grid. If the lower bound fails, it returns
   `None`. Generic callers may substitute 0.1 later; the dedicated magnitude
   loader rejects missing calibration entries. Calibration also skips questions
   whose answers cannot be represented by one token (or whose span is not found).
6. **Historical auditability.** The single-pass sweeps save per-sample correct
   probabilities and aggregate correctness, but not complete predictions, target
   labels, and concept plans. Calibration saves thresholds rather than all
   question-level outcomes. Add the missing records before new scientific runs
   that require these breakdowns; existing files cannot reconstruct them fully.

## What this cleanup changes

- Places behavioral plotters in `icl/plotting`, the brief generator in
  `icl/experiments/tasks`, and the optional scheduler in `scripts`.
- Removes fixed author-machine paths and forced offline Hugging Face settings;
  uses the active Python interpreter and user-supplied cache settings.
- Allows plotting without importing the inference stack.
- Checks required figure inputs before rendering and verifies that renderers
  actually wrote their expected PNGs. Individual figure groups can be selected.
- Keeps experiment algorithms, seeds, fixed strengths, output paths, and the
  Transformers pin unchanged. The figure check is not a numerical validation.

The three missing historical layer sources would need to be restored under
`figure_sources/type1/` and `figure_sources/type2/`, or replaced by a validated
renderer for the raw logs. Supplying arbitrary substitute curves would not
reproduce the paper.

## Validation of the cleanup

- Nine CPU regression tests pass, including missing-file detection, stale-output
  detection, emotion filtering, prompt-level aggregation, and scheduler failures.
- `python tests/render_smoke.py` renders all nine figure groups and checks 32 PNG
  outputs using **synthetic fixtures in a temporary checkout**. The synthetic
  historical layer inputs test restyling only; they do not replace the missing
  paper sources. Tested with Python 3.13, NumPy 2.5.3, Plotly 7.1.0, Kaleido 1.4.0,
  and Matplotlib 3.11.2.
- The wheel builds, `uv lock --check --offline` passes, and an empty checkout's
  figure-input check correctly exits unsuccessfully.

GPU inference, judge scoring, the full CUDA dependency installation, and agreement
with published scores remain untested in this audit.
