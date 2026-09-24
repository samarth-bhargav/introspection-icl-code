"""Run the five-model, six-emotion behavioral sweeps on a GPU work queue.

Uses the existing run settings, including fixed strengths. See
docs/reproduction-status.md for differences from the manuscript.
"""
from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
EVALS = REPO / "evals" / "full_6emo"
LOGS = REPO / "logs" / "gated"
EMO = "anger,fear,joy,love,sadness,disgust"
JURL = "http://127.0.0.1:8002/v1"
# per-model reuse f*
MF = {"gemma-31b": 0.4, "qwen3-32b": 0.8, "qwen3-8b": 1.0, "olmo-32b": 1.0, "olmo-7b": 0.7}
SF = {"gemma-31b": 0.4, "qwen3-32b": 0.2, "qwen3-8b": 0.1, "olmo-32b": 0.2, "olmo-7b": 0.1}
BIG = {"gemma-31b", "qwen3-32b", "olmo-32b"}  # Cannot share an 80 GB GPU with the judge.


def math_cmd(model, kvals, judge_url=JURL):
    return [PY, "-m", "icl.experiments.tasks.run_arithmetic",
            "--model", model, "--emotion_pool", EMO,
            "--k_values", kvals, "--fixed_cmax", str(MF[model]),
            "--prompt_variations", "10", "--n_tests", "30",
            "--target_prob", "0.5", "--distractor_prob", "0.25",
            "--max_operand", "9", "--max_new_tokens", "8", "--seed", "0",
            "--judge_base_url", judge_url, "--judge_model", "synonym-judge",
            "--out_dir", str(EVALS / f"generation_{model}" / "math"), "--no_control"]


def succ_cmd(model, kvals, judge_url=JURL):
    return [PY, "-m", "icl.experiments.tasks.run_successor",
            "--model", model, "--randomize_emotion", "--emotion_pool", EMO,
            "--fraction", str(SF[model]), "--skip_calibration",
            "--k_values", kvals, "--n_variations", "10", "--samples_per_var", "30",
            "--target_prob", "0.5", "--distractor_prob", "0.25",
            "--max_new_tokens", "40", "--seed", "13",
            "--out_root", str(EVALS), "--run_name", "successor_emotions",
            "--judge_base_url", judge_url, "--judge_model", "synonym-judge", "--no_control"]


def build_jobs(judge_url=JURL):
    """Keep each model/task sweep together to avoid repeated weight loads."""
    jobs = []
    def add(model, task, kvals, prio):
        cmd = math_cmd(model, kvals, judge_url) if task == "math" else succ_cmd(model, kvals, judge_url)
        jobs.append({"id": f"{model}-{task}-k{kvals}", "model": model,
                     "size": "big" if model in BIG else "small", "prio": prio, "cmd": cmd})
    # Prioritize large arithmetic jobs, then large successor jobs.
    for m in ["gemma-31b", "qwen3-32b", "olmo-32b"]:
        add(m, "math", "0-20", prio=20)
    for m in ["gemma-31b", "qwen3-32b", "olmo-32b"]:
        add(m, "succ", "0-10", prio=15)
    add("qwen3-8b", "math", "0-20", prio=10)
    add("olmo-7b", "math", "0-20", prio=9)
    add("qwen3-8b", "succ", "0-10", prio=8)
    add("olmo-7b", "succ", "0-10", prio=7)
    jobs.sort(key=lambda j: j["prio"], reverse=True)  # longest/heaviest first
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    ap.add_argument("--gpus", default="0,1", help="comma-separated worker GPU IDs")
    ap.add_argument("--judge-gpu", default="0", help="GPU reserved for judge plus small jobs; use none for a remote judge")
    ap.add_argument("--judge-url", default=JURL)
    args = ap.parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        ap.error("--gpus must contain distinct GPU IDs")
    if not any(g != args.judge_gpu for g in gpus):
        ap.error("a worker GPU separate from --judge-gpu is required for 32B models")
    jobs = build_jobs(args.judge_url)
    lock = threading.Lock()
    print(f"[sched] {len(jobs)} jobs across {len(gpus)} GPUs (judge GPU={args.judge_gpu}, small-only):", flush=True)
    for j in jobs:
        print(f"   {j['id']:28s} size={j['size']}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job["cmd"]))
        return

    LOGS.mkdir(parents=True, exist_ok=True)
    EVALS.mkdir(parents=True, exist_ok=True)

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
        while True:
            with lock:
                pick = None
                for j in jobs:
                    if j.get("taken"):
                        continue
                    if gpu == args.judge_gpu and j["size"] == "big":
                        continue
                    pick = j
                    j["taken"] = True
                    break
                if pick is None:
                    return  # No unassigned job fits this worker.
            log = LOGS / f"{pick['id']}.gpu{gpu}.log"
            t0 = time.time()
            print(f"[sched] GPU{gpu} START {pick['id']} -> {log.name} {time.strftime('%T')}", flush=True)
            with open(log, "w") as fh:
                rc = subprocess.run(pick["cmd"], cwd=str(REPO), env=env,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            pick["rc"] = rc
            print(f"[sched] GPU{gpu} DONE  {pick['id']} rc={rc} wall={time.time()-t0:.0f}s "
                  f"{time.strftime('%T')}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fails = [j["id"] for j in jobs if j.get("rc", 1) != 0]
    print(f"\n[sched] ALL DONE wall={ (time.time()-t0)/3600:.2f}h  "
          f"{'FAILS: ' + ','.join(fails) if fails else 'all rc=0'}", flush=True)

    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
