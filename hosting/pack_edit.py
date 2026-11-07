"""Pack an edit snapshot compactly for upload: per tensor, only the changed rows (or columns), as exact bf16
bit patterns plus indices. Verifies that unpacking onto the original bf16 weights reproduces the edit exactly
as the runtime sees it (bf16-rounded).

    python pack_edit.py <snapshot_dir> <out.npz>     (run with .venv-kernel or .venv-run)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import wsp_kernel as K  # noqa: E402  (harness/kernel_config.json -> original bf16 weights)


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit pattern (uint16), round-to-nearest-even (matches MLX/torch casts)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def unpack(name: str, z, original: np.ndarray) -> np.ndarray:
    out = original.copy()
    mode = str(z[f"{name}|mode"])
    idx, vals = z[f"{name}|idx"], from_bf16_bits(z[f"{name}|vals"])
    if mode == "rows":
        out[idx] = vals
    elif mode == "cols":
        out[:, idx] = vals
    else:
        out = vals.reshape(original.shape)
    return out


def main():
    snap, dest = Path(sys.argv[1]), Path(sys.argv[2])
    arrays, manifest = {}, []
    for f in sorted(snap.glob("*.npy")):
        name = f.name[:-4]
        orig = K.load(name)
        edit_bits = to_bf16_bits(np.load(f).astype(np.float32))
        edit = from_bf16_bits(edit_bits).reshape(orig.shape)
        diff = edit != orig
        rows, cols = np.where(diff.any(1))[0], np.where(diff.any(0))[0]
        if len(rows) * orig.shape[1] <= len(cols) * orig.shape[0] and len(rows) < orig.shape[0] // 2:
            mode, idx, vals = "rows", rows, edit_bits.reshape(orig.shape)[rows]
        elif len(cols) < orig.shape[1] // 2:
            mode, idx, vals = "cols", cols, edit_bits.reshape(orig.shape)[:, cols]
        else:
            mode, idx, vals = "full", np.arange(0), edit_bits
        arrays.update({f"{name}|mode": np.array(mode), f"{name}|idx": idx.astype(np.int32), f"{name}|vals": vals})
        manifest.append({"tensor": name, "mode": mode, "n_idx": int(len(idx)), "changed": int(diff.sum()),
                         "shape": list(orig.shape)})
    arrays["manifest"] = np.array(json.dumps(manifest))
    np.savez_compressed(dest, **arrays)
    # verify round trip
    z = np.load(dest)
    for m in manifest:
        orig = K.load(m["tensor"])
        want = from_bf16_bits(to_bf16_bits(np.load(snap / f"{m['tensor']}.npy").astype(np.float32))).reshape(orig.shape)
        got = unpack(m["tensor"], z, orig)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), m["tensor"]
    print(json.dumps({"packed": str(dest), "bytes": dest.stat().st_size, "tensors": manifest}, indent=1))


if __name__ == "__main__":
    main()
