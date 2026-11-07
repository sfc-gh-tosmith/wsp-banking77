"""ML Job payload: rebuild the exact model the editor edited, then register it.

The editor fit its edit on mlx-community/Meta-Llama-3.1-8B-Instruct-4bit (MLX affine 4-bit, group 64, fp16 scales and
biases). On a full-precision bf16 base the edit scores ~72% instead of ~91%: its detectors are tuned to that base's
activations. So this job hosts the base the editor actually saw:

1. download the mlx-community 4-bit weights and dequantize every quantized tensor to fp16 exactly as mx.dequantize
   does (fp32 math, fp16 result; verified bit-exact locally)
2. layer 31 gate/up/down_proj: original bf16 (unsloth shards) + the packed closed-form edit, as in the local runner
3. write fp16 safetensors with Meta's config/tokenizer and log a text-generation pipeline

Env: EDIT_STAGE_FILE, MODEL_NAME, VERSION_NAME, DB, SCHEMA
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file as load_np
from safetensors.torch import load_file, save_file
from snowflake.snowpark import Session

t0 = time.time()
session = Session.builder.getOrCreate()
DB, SCHEMA = os.environ["DB"], os.environ["SCHEMA"]
session.use_database(DB)
session.use_schema(SCHEMA)
work = Path("/tmp/wsp")
q4_dir, bf16_dir, out_dir, edit_dir = work / "q4", work / "bf16", work / "model", work / "edit"
for d in (out_dir, edit_dir):
    d.mkdir(parents=True, exist_ok=True)

snapshot_download("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", local_dir=str(q4_dir),
                  allow_patterns=["model*.safetensors", "config.json"])
bf16_index = json.loads(Path(snapshot_download(
    "unsloth/Meta-Llama-3.1-8B-Instruct", local_dir=str(bf16_dir), allow_patterns=["*.json"])
    ).joinpath("model.safetensors.index.json").read_text())["weight_map"]
print(f"downloaded 4-bit base + configs in {time.time()-t0:.0f}s", flush=True)

session.file.get(os.environ["EDIT_STAGE_FILE"], str(edit_dir))
z = np.load(next(edit_dir.glob("*.npz")))
manifest = json.loads(str(z["manifest"]))
edited = {m["tensor"] for m in manifest}
snapshot_download("unsloth/Meta-Llama-3.1-8B-Instruct", local_dir=str(bf16_dir),
                  allow_patterns=sorted({bf16_index[t] for t in edited}))

qcfg = json.loads((q4_dir / "config.json").read_text())["quantization"]
GROUP, BITS = qcfg["group_size"], qcfg["bits"]
assert (GROUP, BITS) == (64, 4), qcfg


def dequantize(w: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """MLX affine dequantize: low nibble first, value = q * scale + bias per group of 64, fp32 math -> fp16."""
    q = w.view(np.uint32)
    nib = ((q[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF).reshape(q.shape[0], -1, GROUP)
    out = nib.astype(np.float32) * scales[..., None].astype(np.float32) + biases[..., None].astype(np.float32)
    return out.reshape(q.shape[0], -1).astype(np.float16)


def edited_tensor(name: str) -> torch.Tensor:
    """Original bf16 + packed edit (changed rows/cols as exact bf16 bits), returned as fp16."""
    t = load_file(str(bf16_dir / bf16_index[name]))[name]
    mode, idx = str(z[f"{name}|mode"]), torch.from_numpy(z[f"{name}|idx"].astype(np.int64))
    vals = torch.from_numpy(np.ascontiguousarray(z[f"{name}|vals"]).view(np.int16)).view(torch.bfloat16)
    if mode == "rows":
        t[idx] = vals
    elif mode == "cols":
        t[:, idx] = vals
    else:
        t = vals.reshape(t.shape)
    m = next(x for x in manifest if x["tensor"] == name)
    assert list(t.shape) == m["shape"], (name, t.shape)
    return t.to(torch.float16).contiguous()


tensors = {}
for f in sorted(q4_dir.glob("model*.safetensors")):
    raw = load_np(str(f))
    for k, v in raw.items():
        if k.endswith((".scales", ".biases")):
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else k
        if k in edited:
            tensors[k] = edited_tensor(k)
        elif v.dtype == np.uint32:
            tensors[k] = torch.from_numpy(dequantize(v, raw[base + ".scales"], raw[base + ".biases"]))
        else:
            tensors[k] = torch.from_numpy(v.astype(np.float16))
    del raw
missing = set(bf16_index) - set(tensors)
assert not missing, f"tensors missing vs Meta's layout: {sorted(missing)[:5]}"
assert edited <= set(tensors)
assert all(t.dtype == torch.float16 for t in tensors.values())
print(f"built {len(tensors)} fp16 tensors in {time.time()-t0:.0f}s", flush=True)

# shard ~4 GB each, with an index, in Meta's tensor naming
shards, cur, size, weight_map = [], {}, 0, {}
for k in sorted(tensors):
    cur[k] = tensors[k]
    size += tensors[k].numel() * 2
    if size > 4e9:
        shards.append(cur)
        cur, size = {}, 0
if cur:
    shards.append(cur)
for i, sh in enumerate(shards, 1):
    name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
    save_file(sh, str(out_dir / name), metadata={"format": "pt"})
    weight_map.update({k: name for k in sh})
(out_dir / "model.safetensors.index.json").write_text(json.dumps(
    {"metadata": {"total_size": sum(t.numel() * 2 for t in tensors.values())}, "weight_map": weight_map}))
for f in bf16_dir.glob("*.json"):
    if f.name != "model.safetensors.index.json":
        (out_dir / f.name).write_text(f.read_text())
cfg = json.loads((out_dir / "config.json").read_text())
cfg["torch_dtype"] = "float16"
(out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
del tensors, shards
print(f"wrote fp16 model in {time.time()-t0:.0f}s", flush=True)

import transformers  # noqa: E402
from snowflake.ml.model.openai_signatures import OPENAI_CHAT_WITH_PARAMS_SIGNATURE  # noqa: E402
from snowflake.ml.registry import Registry  # noqa: E402

pipe = transformers.pipeline("text-generation", model=str(out_dir), torch_dtype=torch.float16, device="cpu")
reg = Registry(session=session, database_name=DB, schema_name=SCHEMA)
total = sum(m["changed"] for m in manifest)
reg.log_model(
    pipe,
    model_name=os.environ["MODEL_NAME"],
    version_name=os.environ["VERSION_NAME"],
    signatures=OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
    comment=("Llama 3.1 8B Instruct, closed-form Banking77 weight edit (no gradient training) on the exact base the editor "
             f"fit it on: mlx-community 4-bit dequantized to fp16; layer 31 MLP = bf16 + edit ({total:,} weights changed)."),
)
__return__ = {"model": os.environ["MODEL_NAME"], "version": os.environ["VERSION_NAME"], "edited": sorted(edited),
              "minutes": (time.time() - t0) / 60}
print(json.dumps(__return__), flush=True)
