"""Append-only measurements; ICL_TRACE_PATH can override the default location."""
import json
import os
import time
from pathlib import Path
from uuid import uuid4

_run_id = os.environ.get("ICL_RUN_ID") or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
_sequence = 0


def emit(event: str, **fields) -> None:
    global _sequence
    path = os.environ.get("ICL_TRACE_PATH") or str(Path("evals/traces") / f"{_run_id}.jsonl")
    _sequence += 1
    record = {"event": event, "timestamp": time.time(),
              "run_id": os.environ.get("ICL_RUN_ID", _run_id), "sequence": _sequence, **fields}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
