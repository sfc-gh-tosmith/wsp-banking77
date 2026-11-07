"""ML Job payload (runs inside Snowflake): build the edited bf16 model and register it.

1. download unsloth/Meta-Llama-3.1-8B-Instruct (bf16 safetensors, ungated mirror of Meta's weights)
2. apply the packed closed-form edit from the stage (changed rows/cols as exact bf16 bits; no training anywhere)
3. verify every replaced tensor bit-for-bit, then log a text-generation pipeline to the Model Registry

Env (set by submit_merge_job.py): EDIT_STAGE_FILE, MODEL_NAME, VERSION_NAME, DB, SCHEMA
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from snowflake.snowpark import Session

t0 = time.time()
session = Session.builder.getOrCreate()
DB, SCHEMA = os.environ["DB"], os.environ["SCHEMA"]
session.use_database(DB)
session.use_schema(SCHEMA)
work = Path("/tmp/wsp")
model_dir, edit_dir = work / "model", work / "edit"
edit_dir.mkdir(parents=True, exist_ok=True)

snapshot_download("unsloth/Meta-Llama-3.1-8B-Instruct", local_dir=str(model_dir),
                  allow_patterns=["*.json", "*.safetensors"])
print(f"downloaded base in {time.time()-t0:.0f}s", flush=True)

session.file.get(os.environ["EDIT_STAGE_FILE"], str(edit_dir))
z = np.load(next(edit_dir.glob("*.npz")))
manifest = json.loads(str(z["manifest"]))
index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]


def bits_to_bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16)).view(torch.bfloat16)


def apply(name: str, t: torch.Tensor) -> torch.Tensor:
    mode, idx = str(z[f"{name}|mode"]), torch.from_numpy(z[f"{name}|idx"].astype(np.int64))
    vals = bits_to_bf16(z[f"{name}|vals"])
    out = t.clone()
    if mode == "rows":
        out[idx] = vals
    elif mode == "cols":
        out[:, idx] = vals
    else:
        out = vals.reshape(t.shape)
    return out.contiguous()


by_shard = {}
for m in manifest:
    by_shard.setdefault(index[m["tensor"]], []).append(m)
report = []
for shard, items in by_shard.items():
    tensors = load_file(str(model_dir / shard))
    for m in items:
        old = tensors[m["tensor"]]
        assert list(old.shape) == m["shape"] and old.dtype == torch.bfloat16, m
        new = apply(m["tensor"], old)
        changed = int((new != old).sum())
        assert changed == m["changed"], (m["tensor"], changed, m["changed"])
        tensors[m["tensor"]] = new
        report.append({"tensor": m["tensor"], "shard": shard, "changed": changed})
    save_file(tensors, str(model_dir / shard), metadata={"format": "pt"})
    del tensors
for r in report:  # re-read from disk and compare to the packed edit, bit for bit
    t = load_file(str(model_dir / r["shard"]))[r["tensor"]]
    m = next(x for x in manifest if x["tensor"] == r["tensor"])
    idx = torch.from_numpy(z[f"{r['tensor']}|idx"].astype(np.int64))
    want = bits_to_bf16(z[f"{r['tensor']}|vals"])
    got = t[idx] if m["mode"] == "rows" else (t[:, idx] if m["mode"] == "cols" else t)
    assert torch.equal(got.view(torch.int16), want.reshape(got.shape).view(torch.int16)), r["tensor"]
total = sum(r["changed"] for r in report)
print(f"edit applied and verified: {json.dumps(report)}; {total:,} weights changed", flush=True)

import transformers
from snowflake.ml.model.openai_signatures import OPENAI_CHAT_WITH_PARAMS_SIGNATURE
from snowflake.ml.registry import Registry

pipe = transformers.pipeline("text-generation", model=str(model_dir), torch_dtype=torch.bfloat16, device="cpu")
reg = Registry(session=session, database_name=DB, schema_name=SCHEMA)
mv = reg.log_model(
    pipe,
    model_name=os.environ["MODEL_NAME"],
    version_name=os.environ["VERSION_NAME"],
    signatures=OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
    comment=("Llama 3.1 8B Instruct with a closed-form weight edit for Banking77 intent classification "
             f"(weight-space programming, no gradient training). {total:,} weights changed in: "
             + ", ".join(r["tensor"] for r in report)),
)
__return__ = {"model": os.environ["MODEL_NAME"], "version": os.environ["VERSION_NAME"], "edits": report,
              "minutes": (time.time() - t0) / 60}
print(json.dumps(__return__), flush=True)
