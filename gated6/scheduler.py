"""Dynamic 6-GPU job scheduler for the full 6-emotion gated reruns.

Keeps all 6 GPUs busy via a priority work-queue (longest-job-first). Heavy 32B
math/successor sweeps are sharded by k so a single model doesn't bottleneck one
GPU. GPU0 hosts the judge (Qwen3-8B vLLM) and therefore only runs SMALL-model
jobs (a 32B + judge won't fit in 80GB); GPUs 1-5 run anything.

Each job = one run_arithmetic / run_successor invocation over a k-subset,
written into evals/full_6emo/ (sharded k-files never collide). cwd MUST be the
Final package root so `python -m` resolves Final's icl.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

FINAL = Path("/workspace/Introspection-ICL-Final")
PY = "/workspace/Introspection-RL/.venv/bin/python"
EVALS = FINAL / "evals" / "full_6emo"
LOGS = FINAL / "gated6" / "logs" / "full"
EMO = "anger,fear,joy,love,sadness,disgust"
JURL = "http://127.0.0.1:8002/v1"
NGPU = 2  # 2-GPU node: GPU0=judge+small lane, GPU1=big lane (32B serial)
# per-model reuse f*
MF = {"gemma-31b": 0.4, "qwen3-32b": 0.8, "qwen3-8b": 1.0, "olmo-32b": 1.0, "olmo-7b": 0.7}
SF = {"gemma-31b": 0.4, "qwen3-32b": 0.2, "qwen3-8b": 0.1, "olmo-32b": 0.2, "olmo-7b": 0.1}
BIG = {"gemma-31b", "qwen3-32b", "olmo-32b"}  # 32B: cannot share GPU0 with judge


def math_cmd(model, kvals):
    return [PY, "-m", "icl.experiments.tasks.run_arithmetic",
            "--model", model, "--emotion_pool", EMO,
            "--k_values", kvals, "--fixed_cmax", str(MF[model]),
            "--prompt_variations", "10", "--n_tests", "30",
            "--target_prob", "0.5", "--distractor_prob", "0.25",
            "--max_operand", "9", "--max_new_tokens", "8", "--seed", "0",
            "--judge_base_url", JURL, "--judge_model", "synonym-judge",
            "--out_dir", str(EVALS / f"generation_{model}" / "math"), "--no_control"]


def succ_cmd(model, kvals):
    return [PY, "-m", "icl.experiments.tasks.run_successor",
            "--model", model, "--randomize_emotion", "--emotion_pool", EMO,
            "--fraction", str(SF[model]), "--skip_calibration",
            "--k_values", kvals, "--n_variations", "10", "--samples_per_var", "30",
            "--target_prob", "0.5", "--distractor_prob", "0.25",
            "--max_new_tokens", "40", "--seed", "13",
            "--out_root", str(EVALS), "--run_name", "successor_emotions",
            "--judge_base_url", JURL, "--judge_model", "synonym-judge", "--no_control"]


def build_jobs():
    """Return jobs as dicts {id, model, size, prio, cmd}. Shard big models by k."""
    jobs = []
    def add(model, task, kvals, prio):
        cmd = math_cmd(model, kvals) if task == "math" else succ_cmd(model, kvals)
        jobs.append({"id": f"{model}-{task}-k{kvals}", "model": model,
                     "size": "big" if model in BIG else "small", "prio": prio, "cmd": cmd})
    # 2-GPU layout: GPU1 is the ONLY big-capable lane (GPU0 hosts the judge), so
    # k-sharding big models would only reload the 64GB weights twice on the same
    # GPU -> keep each big model's full k-range as ONE job. Big jobs get top prio
    # so GPU1 drains them first; small jobs fall to GPU0 (judge lane) and any GPU1
    # idle time. MATH (k0-20) before SUCCESSOR (k0-10) since math is the longer sweep.
    for m in ["gemma-31b", "qwen3-32b", "olmo-32b"]:
        add(m, "math", "0-20", prio=20)
    for m in ["gemma-31b", "qwen3-32b", "olmo-32b"]:
        add(m, "succ", "0-10", prio=15)
    # small models (run on GPU0 alongside judge; GPU1 steals if it finishes big early)
    add("qwen3-8b", "math", "0-20", prio=10); add("olmo-7b", "math", "0-20", prio=9)
    add("qwen3-8b", "succ", "0-10", prio=8); add("olmo-7b", "succ", "0-10", prio=7)
    jobs.sort(key=lambda j: j["prio"], reverse=True)  # longest/heaviest first
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    EVALS.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs()
    lock = threading.Lock()
    print(f"[sched] {len(jobs)} jobs across {NGPU} GPUs (GPU0=judge, small-only):", flush=True)
    for j in jobs:
        print(f"   {j['id']:28s} size={j['size']}", flush=True)
    if args.dry_run:
        print("\n[sched] DRY RUN — example command:\n  " + " ".join(jobs[0]["cmd"]))
        return

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME="/opt/hf-cache",
                   HF_HUB_OFFLINE="1", PYTHONPATH=str(FINAL))
        while True:
            with lock:
                pick = None
                for j in jobs:
                    if j.get("taken"):
                        continue
                    if gpu == 0 and j["size"] == "big":   # GPU0 hosts judge: small only
                        continue
                    pick = j; j["taken"] = True; break
                if pick is None:
                    # nothing eligible right now; if all taken, exit; else wait (GPU0 may
                    # have only big jobs left -> let other GPUs drain them)
                    if all(x.get("taken") for x in jobs):
                        return
                    eligible_left = any(not x.get("taken") and not (gpu == 0 and x["size"] == "big")
                                        for x in jobs)
                    if not eligible_left:
                        return
            if pick is None:
                time.sleep(5); continue
            log = LOGS / f"{pick['id']}.gpu{gpu}.log"
            t0 = time.time()
            print(f"[sched] GPU{gpu} START {pick['id']} -> {log.name} {time.strftime('%T')}", flush=True)
            with open(log, "w") as fh:
                rc = subprocess.run(pick["cmd"], cwd=str(FINAL), env=env,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            pick["rc"] = rc
            print(f"[sched] GPU{gpu} DONE  {pick['id']} rc={rc} wall={time.time()-t0:.0f}s "
                  f"{time.strftime('%T')}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in range(NGPU)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fails = [j["id"] for j in jobs if j.get("rc", 1) != 0]
    print(f"\n[sched] ALL DONE wall={ (time.time()-t0)/3600:.2f}h  "
          f"{'FAILS: ' + ','.join(fails) if fails else 'all rc=0'}", flush=True)


if __name__ == "__main__":
    main()
