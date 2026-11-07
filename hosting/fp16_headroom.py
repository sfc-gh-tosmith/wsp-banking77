"""Can the edited model be served in pure fp16? Measure peak |values| in layer 31's MLP with the edit applied
(MLX runs that layer in fp32 locally, because edits are bf16 and the base is fp16).

    python fp16_headroom.py <snapshot_dir>      (uses .venv-run)
"""
import sys
from pathlib import Path

import mlx.core as mx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from runner import Runner  # noqa: E402
from common import DATA  # noqa: E402

r = Runner(edit_dir=Path(sys.argv[1]))
mlp = r.layers[31].mlp
print("edited layer dtypes:", mlp.gate_proj.weight.dtype, mlp.up_proj.weight.dtype, mlp.down_proj.weight.dtype)
dev = pd.read_parquet(DATA / "dev.parquet").sample(200, random_state=0)
suf = [r.suffix_ids(t) for t in dev.TEXT]
peak = {"gate": 0.0, "up": 0.0, "act": 0.0, "down_out": 0.0, "resid_in": 0.0}
for i in range(0, len(suf), 32):
    chunk = suf[i:i + 32]
    out = r.capture(chunk, [list(range(len(s))) for s in chunk], {31: {"mlp_in", "mlp_act", "mlp_out", "resid_mid"}})
    x = mx.array(out["L31.mlp_in"]).astype(mx.float32)
    g = x @ mlp.gate_proj.weight.astype(mx.float32).T
    u = x @ mlp.up_proj.weight.astype(mx.float32).T
    for k, v in [("gate", g), ("up", u), ("act", mx.array(out["L31.mlp_act"])), ("down_out", mx.array(out["L31.mlp_out"])),
                 ("resid_in", mx.array(out["L31.resid_mid"]))]:
        peak[k] = max(peak[k], float(mx.abs(v.astype(mx.float32)).max()))
print({k: round(v, 1) for k, v in peak.items()}, "| fp16 max = 65504")
