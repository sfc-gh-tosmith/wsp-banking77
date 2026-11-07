"""Regression check for the capture top-k fix: positions='all' with topk on many records must match
argmax of logits computed row by row. Holds the shared GPU lock so it never overlaps an editor pass."""
import fcntl
import json

import mlx.core as mx
import numpy as np

from common import ROOT
from runner import CHAT, Runner

lock = open(ROOT / "runs" / "gpu.lock", "w")
fcntl.flock(lock, fcntl.LOCK_EX)
r = Runner().use(CHAT)
prompts = [f"Write two sentences about topic number {i}, then explain why it matters." for i in range(96)]
replies = r.generate(prompts[:8], max_new_tokens=16)
sufs = [r.suffix_ids(p) + r.tok.encode(replies[i % 8], add_special_tokens=False) for i, p in enumerate(prompts)]
pos = [list(range(len(s) - 1)) for s in sufs]
cap = r.capture(sufs, pos, {31: {"resid"}}, topk=5)
n = len(cap["pos"])
zeros = int((cap["topk_ids"] == 0).all(axis=1).sum())
# reference: row-by-row logits from the recorded final residual
h = mx.array(cap["L31.resid"][:200].astype(np.float32)).astype(mx.bfloat16)
ref = np.array(mx.argmax(r.model.lm_head(r.model.model.norm(h)), axis=-1))
agree = float((ref == cap["topk_ids"][:200, 0]).mean())
print(json.dumps({"records": n, "rows_all_zero_topk": zeros, "top1_agrees_with_rowwise_first200": agree}))
