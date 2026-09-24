# Reproduction and validation

The manuscript defines the methods. The code now uses the settings below.
Historical evaluation logs are not included, so matching the reported numerical
scores remains unverified. Reduced GPU tests check execution and scientific
invariants; they do not estimate the paper’s results.

## Methods and figure coverage

| Component | Implementation |
| --- | --- |
| Concept vectors | 20 description prompts per concept; mean generated-token activations; one-vs-rest contrast; unit L2 norm. Full runs use all 62 concepts. |
| Injection | Add coefficient × live token L2 norm × unit vector to user-content tokens only. Chat separators and assistant tokens are excluded. |
| Calibration | 20 arithmetic questions, vocabulary argmax, ≥95% accuracy; binary search over [0.1, 5] until width ≤0.1. A failed lower bound is recorded and uses the paper’s 0.1 floor. Missing entries are errors. |
| Classification strength | Maximize mean p(correct) at k=30; exact ties choose the smaller strength. |
| Sampling | 30 samples per prompt × 10 prompt variations by default; test concepts held out from demonstrations. |
| Generalization | k=20; magnitude anchors are m* × {0.25, 1, 2.5}; layer tests use the nearest of the three anchor calibrations. |
| Behavioral tasks | Six randomly chosen target emotions; target/distractor/none probabilities 0.5/0.25/0.25; Qwen3-8B judge. Fixed operating strengths match the appendix table. |
| Figures | All nine groups, including the three layer panels, render directly from generated JSON. No historical Plotly sources are required. |

The old calibration suite silently skipped multi-token answers: Qwen3-8B
resolved only 7 of its 20 questions. The replacement suite uses single-digit
answers and refuses to skip questions. This changes newly computed thresholds;
it cannot recover the paper’s original thresholds. Calibration files carry a
version and the full question list, and incompatible caches must be rebuilt.

The appendix’s example “9+10 → 19” is not a single-token answer under Qwen3’s
tokenizer. The implementation follows the explicit 20-single-token-question rule.
Some figure captions call the classification metric accuracy; the implementation
follows the methods’ explicit definition of mean p(correct). The canonical
magnitude and layer prompts match the appendix. The other nine variants retain
the repository’s wording; the manuscript does not enumerate them.

## Validation record

- CPU regression checks: 17 pass.
- Linux CPU figure test: all 9 groups render all 32 expected PNGs from synthetic
  fixtures. This checks file contracts and export, not scientific results.
- The pinned inference environment installs using `uv sync --locked` on Linux.
- vLLM 0.12.0 with the pinned Transformers build loaded Qwen3-8B on H200 and
  passed real matching/nonmatching judge requests using the documented eager-mode command.
- Real H200 integrations passed for all five models in `yu-masala-workspace`.
  Each retained 240 behavioral cases, 240 uninjected controls, and 482 real judge
  requests, with all six emotions and no unparseable judge responses.

| Model | Completed integration run |
| --- | --- |
| Qwen3-8B | `qwen3-8b-20260924-v2` |
| Qwen3-32B | `qwen3-32b-20260924-v1` |
| OLMo-7B | `olmo-7b-20260924-v2` |
| OLMo-32B | `olmo-32b-20260924-v1` |
| Gemma-31B | `gemma-31b-20260924-v1` |

The final classification audit passed on all five models after restoring the
canonical manuscript prompts. Run IDs are `final-prompts-20260924-v1-<model>`;
each verifies token positions for all 20 classification prompt templates and
reuses the separately
verified behavioral results. The [machine-readable record](validation-2026-09-24.json)
contains the tested source hash and exact weight revisions. After the GPU audit, generalization
axes were relabeled “Mean P(label)” and the HTML parser was fixed for nested
subplot layouts. Re-exporting verified the labels; the parser passes a subplot
round-trip regression test and reads all 37 exported HTML files.

All nine figure groups also rendered their 32 expected PNGs from these measured
outputs (`real-figures` in the same volume). They are reduced-test figures, not
reproductions of the paper’s numerical results.

The integration test uses eight concepts (including all six emotions), two prompt
variants, two classification samples per prompt, a sparse strength grid, all
model layers, and reduced behavioral sweeps with uninjected controls. It checks
unit vectors, the injection equation, zero-scale equivalence, hook cleanup,
causal readouts, real generation, and real Qwen judge requests. It exercises the
production HTTP judge client through a serial Transformers server. The separate
vLLM check verifies the documented serving command and judge-client integration.

Each run retains `status.json`, `run.log`, `measurements.jsonl`, concept artifacts,
and evaluation JSON in the `introspection-icl-validation` Modal volume. Traces
contain predictions, targets, prompt/concept plans, probabilities, calibration
numerators and denominators, timings, and judge responses. Source hashes and
model revisions identify the code and weights. These are functional-test outputs,
not substitutes for missing historical paper data.

## Re-run the bounded checks

The Modal harness targets the requested team workspace and reserves a conservative
GPU-cost bound before each launch. Its local budget ledger is `logs/modal-budget.json`.
Do not remove that ledger while work is ongoing. A reservation may be reduced
only when a stopped app’s entire lifetime establishes a smaller upper bound;
these estimates are not billed usage reports.

```bash
MODAL_PROFILE=yu-masala-workspace modal run scripts/modal_validate.py \
  --stage prepare --model qwen3-8b
MODAL_PROFILE=yu-masala-workspace modal run scripts/modal_validate.py \
  --stage detection --model qwen3-8b --run-id UNIQUE_RUN_ID
MODAL_PROFILE=yu-masala-workspace modal run scripts/modal_validate.py --stage render
```

`prepare` downloads weights without reserving a GPU. Each GPU invocation stops
its subprocess after 40 minutes and preserves partial measurements. The GPU is
released after that invocation. Download a run’s logs and measurements with:

```bash
MODAL_PROFILE=yu-masala-workspace modal volume get \
  introspection-icl-validation RUN_ID logs/validation
```

The detailed experiment commands are in [experiments.md](experiments.md).
