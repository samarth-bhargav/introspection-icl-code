"""CPU regressions for portable entry points and missing-input handling."""
import contextlib
import io
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import make_figures as figures
from scripts import run_gated


class FigureDriverTests(unittest.TestCase):
    def test_empty_checkout_reports_missing_layer_measurements(self):
        step = next(s for s in figures.steps() if s.name == "layer-sweeps")
        with tempfile.TemporaryDirectory() as tmp:
            missing = figures.missing_inputs(step, Path(tmp))
        self.assertEqual(len(missing), 10)
        self.assertTrue(all(p.startswith("evals/regen/layer/") for p in missing))

    def test_one_model_is_not_enough_for_a_five_model_group(self):
        step = next(s for s in figures.steps() if s.name == "math-modes")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / figures.GATED / f"generation_{figures.MODELS[0]}" / "math/type2_k_sweep/math_test_k0.json"
            p.parent.mkdir(parents=True)
            p.write_text('{}')
            # The actual filename is part of the data contract.
            self.assertEqual(len(figures.missing_inputs(step, root)), 5)
            p.rename(p.with_name(f"math_{figures.MODELS[0]}_k0.json"))
            self.assertEqual(len(figures.missing_inputs(step, root)), 4)

    def test_stale_png_does_not_hide_a_renderer_skip(self):
        step = figures.Step("fixture", (("-c", "pass"),), (), ("plot.png",), "")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plot.png").write_bytes(b"old")
            with self.assertRaisesRegex(RuntimeError, "did not write expected outputs"):
                figures.run_step(step, root)

    def test_selected_group_check_needs_no_inference_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "make_figures.py"
            shutil.copy2(figures.REPO / "make_figures.py", script)
            proc = subprocess.run([sys.executable, "-S", str(script),
                                   "--check", "--only", "layer-sweeps"],
                                  capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("evals/regen/layer", proc.stdout)
        self.assertNotIn("ModuleNotFoundError", proc.stderr)

    def test_lazy_package_keeps_configuration_cpu_only(self):
        proc = subprocess.run([sys.executable, "-S", "-c",
            "import sys, icl; from icl.experiments import config; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
            "assert 'ICLQuery' in dir(icl); assert len(config.MODELS) == 5"],
            cwd=figures.REPO, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class SchedulerTests(unittest.TestCase):
    def test_job_commands_use_active_interpreter_and_selected_judge(self):
        jobs = run_gated.build_jobs("http://judge.example/v1")
        self.assertEqual(len(jobs), 10)
        for job in jobs:
            cmd = job["cmd"]
            self.assertEqual(cmd[0], sys.executable)
            self.assertEqual(cmd[cmd.index("--judge_base_url") + 1], "http://judge.example/v1")
            self.assertNotIn("/workspace/", " ".join(cmd))

    def test_judge_only_worker_is_rejected(self):
        with patch.object(sys, "argv", ["run_gated.py", "--gpus", "0"]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                run_gated.main()
        self.assertEqual(caught.exception.code, 2)

    def test_failed_job_fails_scheduler_and_preserves_cache_environment(self):
        job = {"id": "fixture", "size": "big", "cmd": [sys.executable, "-c", "pass"]}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(sys, "argv", ["run_gated.py", "--gpus", "3", "--judge-gpu", "none"]), \
             patch.object(run_gated, "build_jobs", return_value=[job]), \
             patch.object(run_gated, "LOGS", Path(tmp) / "logs"), \
             patch.object(run_gated, "EVALS", Path(tmp) / "evals"), \
             patch.dict(run_gated.os.environ, {"HF_HOME": "/tmp/user-cache", "HF_HUB_OFFLINE": "0"}), \
             patch.object(run_gated.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                run_gated.main()
            self.assertEqual(caught.exception.code, 1)
            env = run.call_args.kwargs["env"]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(env["HF_HOME"], "/tmp/user-cache")
            self.assertEqual(env["HF_HUB_OFFLINE"], "0")


class HtmlFigureTests(unittest.TestCase):
    def test_combined_subplot_html_round_trip(self):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from icl.plotting.html_to_png import _figure_from_html
        figure = make_subplots(rows=1, cols=2, subplot_titles=("Left", "Right"))
        figure.add_trace(go.Scatter(x=[0, 1], y=[.2, .8]), row=1, col=1)
        figure.add_trace(go.Scatter(x=[0, 1], y=[.7, .3]), row=1, col=2)
        restored = _figure_from_html(figure.to_html(include_plotlyjs="cdn"))
        self.assertEqual(restored.to_plotly_json(), figure.to_plotly_json())


class BehavioralLoaderTests(unittest.TestCase):
    def test_anger_subset_and_prompt_sigma(self):
        from icl.plotting import plot_gated, plot_anger
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "generation_qwen3-8b/math/type2_k_sweep"
            folder.mkdir(parents=True)
            rows = [
                {"target_emotion": "anger", "prompt_variation_id": 0, "p_correct": 1},
                {"target_emotion": "fear", "prompt_variation_id": 0, "p_correct": 0},
                {"target_emotion": "anger", "prompt_variation_id": 1, "p_correct": 0},
                {"target_emotion": "fear", "prompt_variation_id": 1, "p_correct": 0},
            ]
            (folder / "math_qwen3-8b_k0.json").write_text(json.dumps({"n_demos": 0, "rows": rows}))
            with patch.object(plot_gated, "BASE", base), patch.object(plot_anger, "BASE", base):
                overall = plot_gated.math_type2("qwen3-8b")[0]
                anger = plot_anger.math_anger("qwen3-8b")[0]
                sigma = plot_gated.math_type2_prompt_sigma("qwen3-8b")[0]
            self.assertEqual((overall["mp"], overall["n"]), (0.25, 4))
            self.assertEqual((anger["mp"], anger["n"]), (0.5, 2))
            self.assertEqual(sigma["n"], 2)
            self.assertAlmostEqual(sigma["mp_hi"], 0.25 + 0.5 / 2**0.5)


if __name__ == "__main__":
    unittest.main()
