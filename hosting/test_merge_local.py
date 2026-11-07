"""Local test of the merge job's edit application (same code path, one shard, no Snowflake).
Checks: apply() on the original bf16 shard == the snapshot's .npy edits after bf16 rounding, bit for bit,
and the number of changed weights matches the manifest."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
packed, snap = Path(sys.argv[1]), Path(sys.argv[2])
model_dir = ROOT / "models" / "llama31-8b-instruct-bf16"
z = np.load(packed)
manifest = json.loads(str(z["manifest"]))
index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]

src = (Path(__file__).with_name("merge_and_log_job.py")).read_text()
ns = {"np": np, "torch": torch, "z": z}
for fn in ("bits_to_bf16", "apply"):  # exec the job's own function definitions
    start = src.index(f"def {fn}(")
    end = src.index("\n\n\n", start)
    exec(src[start:end], ns)

for shard in sorted({index[m["tensor"]] for m in manifest}):
    tensors = load_file(str(model_dir / shard))
    for m in [m for m in manifest if index[m["tensor"]] == shard]:
        old = tensors[m["tensor"]]
        new = ns["apply"](m["tensor"], old)
        want = torch.from_numpy(np.load(snap / f"{m['tensor']}.npy").astype(np.float32)).to(torch.bfloat16)
        exact = torch.equal(new.view(torch.int16), want.view(torch.int16))
        changed = int((new != old).sum())
        print(f"{m['tensor']}: exact={exact} changed={changed:,} manifest={m['changed']:,}")
        assert exact and changed == m["changed"]
print("merge logic OK")
