"""Generate and cache each model's OWN one-sentence summary of every U.S. amendment,
for use as amendment-successor demo labels (no-reference, in-distribution ICL).
Writes icl/artifacts/<model>/amendment_self_briefs.json (list of len == #amendments).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SUMM_SYS = ("Summarize the given U.S. constitutional amendment in one short "
            "plain-English sentence. Reply only with the summary.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(REPO))

    import torch
    from icl import get_model_and_tokenizer
    from icl.experiments import singlepass as SP
    from icl.experiments.tasks.amendment_successor import AMENDMENT_BRIEFS

    torch.set_grad_enabled(False)
    m = args.model
    print(f"[selfbriefs] loading {m} on GPU {args.gpu}", flush=True)
    model, tok = get_model_and_tokenizer(m)
    device = next(model.parameters()).device
    n_amend = len(AMENDMENT_BRIEFS)

    briefs = []
    for n in range(1, n_amend + 1):
        msgs = [{"role": "system", "content": SUMM_SYS},
                {"role": "user", "content": f"Amendment number: {n}. Answer briefly."}]
        text = SP.render_chat(tok, msgs, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n", 1)[0].strip()
        briefs.append(ans)
        print(f"[selfbriefs] {m} amdt {n:>2}: {ans[:80]}", flush=True)

    out_dir = REPO / "icl" / "artifacts" / m
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "amendment_self_briefs.json"
    out_path.write_text(json.dumps({"model": m, "n_amendments": n_amend,
                                    "summarizer_system_prompt": SUMM_SYS,
                                    "briefs": briefs}, indent=2))
    print(f"[selfbriefs] wrote {out_path} ({len(briefs)} briefs)", flush=True)


if __name__ == "__main__":
    main()
