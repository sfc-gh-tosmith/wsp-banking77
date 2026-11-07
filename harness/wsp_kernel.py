"""wsp_kernel: numpy-only access to the ORIGINAL Llama 3.1 8B Instruct bf16 weights, plus edit saving.

Use with the kernel interpreter:  ../.venv-kernel/bin/python your_script.py
(numpy / scipy / safetensors only: there is no autograd and no way to run the model here.)

    import wsp_kernel as K
    K.list_tensors()                       # names, shapes
    W = K.load("model.layers.31.mlp.down_proj.weight")   # float32 ndarray, HF layout [out, in]
    K.save_edit("model.layers.31.mlp.down_proj.weight", W_new)   # full replacement matrix
    K.list_edits(); K.remove_edit(name)
    K.config()                             # model config.json

An edit is the complete new matrix for that tensor (not a delta). Saving it overwrites any
previous edit of the same tensor. Edited tensors run in bf16 inside the otherwise 4-bit model.
"""
from __future__ import annotations

import json
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_CFG = json.loads((HERE / "kernel_config.json").read_text())
WEIGHTS_DIR = Path(_CFG["weights_dir"]).expanduser()
if not WEIGHTS_DIR.is_absolute():  # relative paths are relative to the repo root
    WEIGHTS_DIR = HERE.parent / WEIGHTS_DIR
EDITS_DIR = HERE / "edits"


@lru_cache(maxsize=1)
def _index():
    idx = {}
    for f in sorted(WEIGHTS_DIR.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                idx[name] = (f, 8 + n, meta)
    return idx


def list_tensors(filter: str = "") -> list[tuple[str, tuple, str]]:
    return [(k, tuple(v[2]["shape"]), v[2]["dtype"]) for k, v in sorted(_index().items()) if filter in k]


def load(name: str) -> np.ndarray:
    """Original (unedited) tensor as float32."""
    f, base, meta = _index()[name]
    s, e = meta["data_offsets"]
    with open(f, "rb") as fh:
        fh.seek(base + s)
        raw = fh.read(e - s)
    if meta["dtype"] == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        out = (u16.astype(np.uint32) << 16).view(np.float32)
    elif meta["dtype"] == "F16":
        out = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif meta["dtype"] == "F32":
        out = np.frombuffer(raw, dtype=np.float32).copy()
    else:
        raise ValueError(meta["dtype"])
    return out.reshape(meta["shape"])


def load_rows(name: str, rows) -> np.ndarray:
    """Only the listed rows of an ORIGINAL 2-D tensor (e.g. lm_head.weight rows for chosen token ids)."""
    f, base, meta = _index()[name]
    s, _ = meta["data_offsets"]
    n_rows, n_cols = meta["shape"]
    width = {"BF16": 2, "F16": 2, "F32": 4}[meta["dtype"]]
    out = np.empty((len(rows), n_cols), np.float32)
    with open(f, "rb") as fh:
        for k, r_ in enumerate(rows):
            if not 0 <= int(r_) < n_rows:
                raise IndexError(r_)
            fh.seek(base + s + int(r_) * n_cols * width)
            raw = fh.read(n_cols * width)
            if meta["dtype"] == "BF16":
                out[k] = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif meta["dtype"] == "F16":
                out[k] = np.frombuffer(raw, np.float16)
            else:
                out[k] = np.frombuffer(raw, np.float32)
    return out


def load_current(name: str) -> np.ndarray:
    """The tensor as the model currently runs it: your saved edit if present, else the original."""
    p = EDITS_DIR / f"{name}.npy"
    return np.load(p).astype(np.float32) if p.exists() else load(name)


OUT_OF_SCOPE = {"model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"}


def save_edit(name: str, array: np.ndarray) -> Path:
    if name in OUT_OF_SCOPE:
        raise PermissionError(f"{name} is out of scope (final norm, input embeddings and output layer are not editable)")
    if name not in _index():
        raise KeyError(f"unknown tensor {name}")
    shape = tuple(_index()[name][2]["shape"])
    if tuple(array.shape) != shape:
        raise ValueError(f"shape {array.shape} != {shape}")
    if not np.isfinite(array).all():
        raise ValueError("edit contains NaN/inf")
    orig = load(name)
    a = np.asarray(array, dtype=np.float32)
    denom = float((orig * orig).sum())
    if denom > 0:
        c = float((a * orig).sum()) / denom
        if np.abs(a - c * orig).max() <= 1e-6 * max(1.0, np.abs(orig).max()) and abs(c - 1.0) > 1e-6:
            raise PermissionError("an edit that only multiplies the whole tensor by a constant is out of scope")
    EDITS_DIR.mkdir(exist_ok=True)
    p = EDITS_DIR / f"{name}.npy"
    # float32 on disk; the runtime rounds to bf16, so untouched original weights round-trip exactly.
    np.save(p, array.astype(np.float32))
    return p


def list_edits() -> list[str]:
    return sorted(p.name[:-4] for p in EDITS_DIR.glob("*.npy")) if EDITS_DIR.exists() else []


def remove_edit(name: str) -> None:
    (EDITS_DIR / f"{name}.npy").unlink(missing_ok=True)


def config() -> dict:
    return json.loads((WEIGHTS_DIR / "config.json").read_text())


def changed_fraction() -> dict:
    """How many weights your edits change (vs the original), per tensor and in total."""
    total_params = sum(int(np.prod(v[2]["shape"])) for v in _index().values())
    per, changed = {}, 0
    for name in list_edits():
        e = np.load(EDITS_DIR / f"{name}.npy").astype(np.float32)
        e_bf16 = ((e.view(np.uint32) + 0x8000) >> 16 << 16).view(np.float32)  # as the runtime will see it
        d = e_bf16 != load(name)
        per[name] = int(d.sum())
        changed += per[name]
    return {"per_tensor": per, "changed": changed, "total_params": total_params,
            "fraction": changed / total_params}
