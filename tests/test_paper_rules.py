import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from icl.experiments.selection import pick_mean_probability
from icl.experiments import config as C


class PaperRules(unittest.TestCase):
    def test_probability_not_accuracy_selects_strength(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sweep.json"
            p.write_text(json.dumps({"per_strength": [
                {"strength": a, "by_k": [{"k": 30, "accuracy": acc, "mean_p": prob}]}
                for a, acc, prob in [(0, .1, .05), (.25, .9, .4), (.5, .8, .7), (1, .85, .7)]
            ]}))
            chosen = pick_mean_probability(p, 30)
            self.assertEqual(chosen["star"], .5)
            self.assertEqual(chosen["selection_metric"], "mean_p_correct")

    def test_recorded_failure_is_floored_but_missing_calibration_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ranges.json"
            p.write_text(json.dumps({"metadata": {"calibration_version": 2},
                                    "ranges": {"anger": {"5": {"c_max": None}}}}))
            with patch.object(C, "cmax_path", return_value=p):
                cm = C.load_cmax("qwen3-8b")
            self.assertEqual(C.cmax_or_floor(cm, "anger", 5), .1)
            with self.assertRaises(ValueError):
                C.cmax_or_floor(cm, "love", 5)

    def test_paper_defaults(self):
        self.assertEqual(C.DEFAULT_N_SAMPLES, 30)
        self.assertEqual(C.PAPER_SUCCESSOR_FRACTIONS["gemma-31b"], .5)
        self.assertEqual(C.PAPER_SUCCESSOR_FRACTIONS["qwen3-32b"], .7)
        self.assertEqual(C.PAPER_SUCCESSOR_FRACTIONS["olmo-32b"], .2)

    def test_variation_zero_matches_manuscript_prompts(self):
        from icl.common.prompt_variations import VARIATIONS
        for task, trigger, system in [("magnitude", C.MAGNITUDE_TRIGGER, C.MAGNITUDE_SYSTEM),
                                      ("layer", C.LAYER_TRIGGER, C.LAYER_SYSTEM)]:
            self.assertEqual(VARIATIONS[task+"_introspection"][0],
                             {"prompt_text": trigger, "system_prompt": system})

    def test_layer_generalization_ignores_other_task_calibrations(self):
        cm = {("anger", 5): .5, ("anger", 7): 4., ("anger", 18): 1., ("anger", 31): 2.}
        self.assertEqual(C.layer_generalization_cmax(cm, "anger", 7, [5, 18, 31]), .5)

    def test_legacy_calibration_requires_recomputation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ranges.json"
            p.write_text(json.dumps({"ranges": {"anger": {"5": {"c_max": 2}}}}))
            with patch.object(C, "cmax_path", return_value=p):
                with self.assertRaisesRegex(ValueError, "20-question"):
                    C.load_cmax("qwen3-8b")

    def test_strength_renderer_pools_separate_prompt_samples(self):
        from icl.plotting.plot_strength_sweep import successor_type1
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "evals/regen/successor_cmax_sweep/qwen3-8b"
            folder.mkdir(parents=True)
            for i, values in enumerate([[1.], [0., 0., 0.]]):
                (folder / f"fraction_0.50_var{i}.json").write_text(json.dumps({
                    "cmax_fraction": .5, "rows": [{"p_correct": v} for v in values]}))
            rows = successor_type1(root, "qwen3-8b")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n"], 4)
            self.assertEqual(rows[0]["mp"], .25)


if __name__ == "__main__":
    unittest.main()
