"""Run reproducibility checks on Modal; weights and results persist in volumes.

Usage: modal run scripts/modal_validate.py --stage prepare --model qwen3-8b
"""
from pathlib import Path
import os
import modal

if modal.is_local() and os.environ.get("MODAL_PROFILE") != "yu-masala-workspace":
    raise RuntimeError("Run with MODAL_PROFILE=yu-masala-workspace to prevent charging another workspace")
ROOT = Path(__file__).resolve().parents[1]
app = modal.App("introspection-icl-validation")
cache = modal.Volume.from_name("rlstack-hf-cache")
results = modal.Volume.from_name("introspection-icl-validation", create_if_missing=True)
base_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("uv==0.10.2")
    .env({"UV_INDEX_URL": "https://pypi.org/simple"})
    .add_local_file(ROOT / "pyproject.toml", "/opt/icl-env/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/opt/icl-env/uv.lock", copy=True)
    .workdir("/opt/icl-env")
    .run_commands("uv sync --locked --default-index https://pypi.org/simple --no-install-project --no-dev")
    .env({"HF_HOME": "/hf", "PYTHONPATH": "/src", "TOKENIZERS_PARALLELISM": "false",
          "PATH": "/opt/icl-env/.venv/bin:/usr/local/bin:/usr/bin:/bin"})
)
image = base_image.add_local_dir(ROOT / "icl", "/src/icl", ignore=["artifacts/**", "**/__pycache__/**"])
judge_image = (base_image
    .run_commands("uv sync --locked --default-index https://pypi.org/simple --no-install-project --no-dev --group vllm")
    .add_local_dir(ROOT / "icl", "/src/icl", ignore=["artifacts/**", "**/__pycache__/**"]))


@app.function(image=judge_image, timeout=180)
def prepare_judge():
    import subprocess
    return subprocess.check_output(["/opt/icl-env/.venv/bin/python", "-c",
        "import vllm, transformers; print(vllm.__version__, transformers.__version__)"], text=True)


@app.function(image=judge_image, volumes={"/hf": cache, "/results": results}, gpu="H200",
              cpu=(2, 4), memory=(16384, 32768), timeout=600, startup_timeout=120,
              retries=0, single_use_containers=True)
def validate_judge(run_id: str, source_sha: str):
    import json
    import subprocess
    import time
    import urllib.request
    out = Path("/results") / run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {"run_id": run_id, "source_sha": source_sha, "state": "running"}
    command = ["/opt/icl-env/.venv/bin/vllm", "serve", "Qwen/Qwen3-8B",
        "--served-model-name", "synonym-judge", "--host", "127.0.0.1", "--port", "8002",
        "--dtype", "bfloat16", "--gpu-memory-utilization", "0.35", "--max-model-len", "4096",
        "--enforce-eager"]
    with (out / "run.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1"}, start_new_session=True)
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited {proc.returncode}; inspect run.log")
                if time.monotonic() - started > 360:
                    raise TimeoutError("vLLM did not become ready in six minutes")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8002/v1/models", timeout=2) as response:
                        report["models"] = json.load(response)
                    break
                except OSError:
                    time.sleep(2)
            check = subprocess.run(["/opt/icl-env/.venv/bin/python", "-c",
                "from icl.experiments.tasks.gated_arithmetic import judge_number_equal; "
                "assert judge_number_equal('12', 12)['match']; "
                "assert not judge_number_equal('13', 12)['match']; print('Real vLLM judge checks passed')"],
                stdout=log, stderr=subprocess.STDOUT, timeout=120)
            if check.returncode:
                raise RuntimeError("vLLM judge inference checks failed")
            report["state"] = "passed"
        except BaseException as exc:
            report.update(state="failed", error=repr(exc))
        finally:
            import signal
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass
            report["seconds"] = time.monotonic() - started
            (out / "status.json").write_text(json.dumps(report, indent=2))
            results.commit()
    return report

plot_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("chromium")
    .pip_install_from_requirements(ROOT / "requirements-plotting.txt")
    .env({"BROWSER_PATH": "/usr/bin/chromium", "PYTHONPATH": "/src"})
    .add_local_dir(ROOT / "icl", "/src/icl", ignore=["artifacts/**", "**/__pycache__/**"])
    .add_local_dir(ROOT / "tests", "/src/tests", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "scripts", "/src/scripts", ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / "make_figures.py", "/src/make_figures.py")
)


@app.function(image=plot_image, volumes={"/results": results}, cpu=4, memory=4096, timeout=900)
def render():
    import subprocess
    import sys
    folder = Path("/results/cpu-render")
    folder.mkdir(exist_ok=True)
    with (folder / "run.log").open("w") as log:
        for command in [[sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                        [sys.executable, "tests/render_smoke.py"]]:
            proc = subprocess.Popen(command, cwd="/src", stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = proc.wait()
            results.commit()
            if code:
                raise RuntimeError(f"Rendering check exited {code}")
    return {"state": "passed", "log": str(folder / "run.log")}


@app.function(image=plot_image, volumes={"/results": results}, cpu=4, memory=4096, timeout=900)
def render_real(run_ids: str):
    import json
    import shutil
    import subprocess
    import sys
    import tempfile
    from icl.experiments.config import MODELS
    folders = [Path("/results") / name for name in run_ids.split(",")]
    if any(folder.parent != Path("/results") for folder in folders):
        raise ValueError("Invalid run identifier")
    reports = [json.loads((folder / "complete.json").read_text()) for folder in folders]
    if sorted(r["model"] for r in reports) != sorted(MODELS):
        raise ValueError("Supply one completed run for each of the five models")
    destination = Path("/results/real-figures")
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copytree("/src/icl", root / "icl")
        shutil.copy2("/src/make_figures.py", root / "make_figures.py")
        for folder in folders:
            shutil.copytree(folder / "evals", root / "evals", dirs_exist_ok=True)
        with (destination / "run.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "make_figures.py"], cwd=root,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = proc.wait()
        for name in ["plots", "plots_new"]:
            if (root / name).exists():
                shutil.copytree(root / name, destination / name, dirs_exist_ok=True)
        (destination / "sources.json").write_text(json.dumps({"runs": run_ids.split(","),
            "scope": "Reduced integration tests; not paper scores", "returncode": code}, indent=2))
        results.commit()
        if code:
            raise RuntimeError(f"Real-results rendering exited {code}")
    return {"state": "passed", "directory": str(destination)}


@app.function(image=image, volumes={"/hf": cache, "/results": results}, timeout=3600, cpu=4)
def prepare(model: str):
    # Modal's worker interpreter is outside the project environment.
    import site
    site.addsitedir("/opt/icl-env/.venv/lib/python3.13/site-packages")
    import json
    import time
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoConfig
    from icl.model import MODEL_REGISTRY
    from icl.experiments.singlepass import build_conversation
    from icl.experiments import config as C
    started = time.time()
    hf_id = MODEL_REGISTRY[model]["hf_id"]
    path = snapshot_download(hf_id, ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.ot"])
    tok = AutoTokenizer.from_pretrained(path)
    cfg = AutoConfig.from_pretrained(path)
    ids, positions = build_conversation(tok, C.LAYER_SYSTEM, [C.LAYER_TRIGGER] * 3, C.LAYER_LABELS)
    report = {"model": model, "hf_id": hf_id, "snapshot": Path(path).name,
              "config_type": cfg.model_type, "tokens": len(ids), "read_positions": positions,
              "seconds": time.time() - started}
    from icl.steering.ranges import _QA_PAIRS
    report["calibration_answer_tokens"] = {
        q: {a: tok.encode(a, add_special_tokens=False),
            " " + a: tok.encode(" " + a, add_special_tokens=False)} for q, a in _QA_PAIRS
    }
    Path("/results/prepare").mkdir(exist_ok=True)
    Path(f"/results/prepare/{model}.json").write_text(json.dumps(report, indent=2))
    cache.commit()
    results.commit()
    return report


@app.function(image=image, volumes={"/hf": cache, "/results": results},
              gpu="H200", cpu=(2, 4), memory=(16384, 32768),
              timeout=2700, startup_timeout=300, retries=0, single_use_containers=True)
def detection(model: str, run_id: str, source_sha: str, reuse_run_id: str = "",
              detection_only: bool = False, deadline: float = 0.):
    import json
    import subprocess
    import threading
    import time
    out = Path("/results") / run_id
    out.mkdir(parents=True, exist_ok=False)
    if reuse_run_id:
        import shutil
        if Path(reuse_run_id).name != reuse_run_id:
            raise ValueError("Invalid artifact source run")
        previous = Path("/results") / reuse_run_id
        if json.loads((previous / "status.json").read_text())["model"] != model:
            raise ValueError("Artifact source has a different model")
        shutil.copytree(previous / "artifacts", out / "artifacts")
    started = time.time()
    env = {**os.environ, "ICL_SOURCE_SHA": source_sha, "HF_HUB_OFFLINE": "1"}
    command = ["/opt/icl-env/.venv/bin/python", "-u", "-m", "icl.experiments.validate_e2e",
               "--model", model, "--output", str(out)]
    if detection_only:
        command.append("--detection-only")
    status = {"run_id": run_id, "model": model, "source_sha": source_sha,
              "command": command, "started": started, "state": "running",
              "artifact_source_run": reuse_run_id or None}
    (out / "status.json").write_text(json.dumps(status))
    results.commit()
    try:
        with (out / "run.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)
            def copy_output():
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            reader = threading.Thread(target=copy_output, daemon=True)
            reader.start()
            while proc.poll() is None:
                if time.time() - started > 2400 or (deadline and time.time() > deadline):
                    proc.kill()
                    proc.wait()
                    reader.join(timeout=10)
                    raise TimeoutError("Stopped at the run's budget timeout; partial measurements retained")
                time.sleep(10)
                log.flush()
                results.commit()
            reader.join(timeout=10)
            status.update(returncode=proc.returncode, state="passed" if proc.returncode == 0 else "failed")
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        status["seconds"] = time.time() - started
        (out / "status.json").write_text(json.dumps(status, indent=2))
        results.commit()
    return status


@app.function(image=image, volumes={"/hf": cache, "/results": results},
              gpu="H200", cpu=(2, 4), memory=(16384, 32768), timeout=2700,
              startup_timeout=300, retries=0, single_use_containers=True)
def audit_detection(run_id: str, run_ids: str, source_sha: str):
    """Recheck final classification code on every architecture, reusing vectors."""
    import json
    import shutil
    import time
    from icl.experiments.config import MODELS
    sources = [Path("/results") / value for value in run_ids.split(",")]
    if any(source.parent != Path("/results") for source in sources):
        raise ValueError("Invalid source run")
    reports = [json.loads((source / "complete.json").read_text()) for source in sources]
    if sorted(report["model"] for report in reports) != sorted(MODELS):
        raise ValueError("Need one completed behavioral run for each model")
    deadline = time.time() + 2400
    outputs = []
    for source, previous in zip(sources, reports):
        child_id = f"{run_id}-{previous['model']}"
        report = detection.local(previous["model"], child_id, source_sha, source.name,
                                 detection_only=True, deadline=deadline)
        if report["state"] != "passed":
            return report
        output = Path("/results") / child_id
        # Preserve the separately verified behavioral measurements; the newly
        # generated classification files take precedence when paths overlap.
        for source_file in (source / "evals").rglob("*.json"):
            target = output / "evals" / source_file.relative_to(source / "evals")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target)
        previous.update(classification_source_sha=source_sha, behavioral_source_run=source.name)
        (output / "complete.json").write_text(json.dumps(previous, indent=2))
        results.commit()
        outputs.append(child_id)
    return {"state": "passed", "runs": outputs}


@app.local_entrypoint()
def main(stage: str = "prepare", model: str = "qwen3-8b", run_id: str = "", reuse_run_id: str = "", run_ids: str = ""):
    if stage == "render":
        print(render.remote())
    elif stage == "render-real":
        print(render_real.remote(run_ids))
    elif stage == "prepare":
        print(prepare.remote(model))
    elif stage == "judge-prepare":
        print(prepare_judge.remote())
    elif stage in ("detection", "judge", "audit-detection"):
        import hashlib
        import json
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("Supply a unique --run-id without path separators")
        # Conservative reservations: CPU/image preparation allowance $2 plus
        # $5 per bounded H200 invocation, including cold start and host resources.
        # Reservations may only be reduced after authoritative stopped-app
        # timestamps establish a conservative upper bound for the whole lifetime.
        ledger_path = ROOT / "logs" / "modal-budget.json"
        ledger_path.parent.mkdir(exist_ok=True)
        import fcntl
        sha = hashlib.sha256()
        for p in sorted((ROOT / "icl").rglob("*.py")):
            sha.update(str(p.relative_to(ROOT)).encode())
            sha.update(p.read_bytes())
        source_sha = sha.hexdigest()
        with ledger_path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
                "cap_usd": 30, "cpu_and_build_reserve_usd": 2, "runs": []}
            if any(r["run_id"] == run_id for r in ledger["runs"]):
                raise ValueError("Run already reserved; inspect its Modal state instead of relaunching")
            reservation = 2 if stage == "judge" else 5
            if 2 + sum(r["reserved_usd"] for r in ledger["runs"]) + reservation > 30:
                raise RuntimeError("The remaining $30 budget cannot cover another bounded GPU invocation")
            ledger["runs"].append({"run_id": run_id,
                "model": "all-models" if stage == "audit-detection" else model,
                "reserved_usd": reservation, "source_sha": source_sha,
                "workspace": "yu-masala-workspace"})
            temporary = ledger_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(ledger, indent=2))
            temporary.replace(ledger_path)
        if stage == "audit-detection":
            report = audit_detection.remote(run_id, run_ids, source_sha)
        elif stage == "detection":
            report = detection.remote(model, run_id, source_sha, reuse_run_id)
        else:
            report = validate_judge.remote(run_id, source_sha)
        print(report)
        if report["state"] != "passed":
            raise RuntimeError(f"Validation failed: {run_id}; saved logs contain the failure")
    else:
        raise ValueError(stage)
