"""MLX runner for Llama 3.1 8B Instruct (4-bit) with optional dense bf16 weight edits.

Edits are full-replacement matrices saved as .npy (float16/float32) named after the HF tensor,
e.g. edits/model.layers.31.mlp.down_proj.weight.npy  (shape = HF [out, in]).
An edited matrix replaces the 4-bit layer with a dense bf16 layer, so 4-bit rounding cannot erase it.

Speed: every prompt shares the same ~45-token chat header + instruction ("prefix"). Its KV cache is
computed once (with the current edits applied) and reused, so only the message-specific suffix is run.

  generate(texts)  greedy decoding of the intent label (suffixes bucketed by length: exact, no padding)
  capture(...)     teacher-forced pass recording internal states (right-padded: exact under causal mask)
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load as mlx_load
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache

from common import MODEL_4BIT, SFT_PROMPT

EOT = 128009
EOS_IDS = {128001, 128008, 128009}
STATE_KINDS = ("resid", "resid_mid", "attn_out", "mlp_in", "mlp_act", "mlp_out")
# Keep MLX from hoarding freed GPU buffers (it otherwise grows to many GB with varying batch shapes
# and pushes this 16 GB machine into swap).
mx.set_cache_limit(512 * 1024 * 1024)
KV_BUDGET_TOKENS = 64 * 128  # batch * prefix tokens per batch (<= ~1 GB of prefix KV)
# Edited matrices run in bf16 (default). WSP_EDIT_DTYPE=float16 is a hosting check only: it mimics an all-fp16 server.
import os  # noqa: E402
EDIT_DTYPE = {"bfloat16": mx.bfloat16, "float16": mx.float16}[os.environ.get("WSP_EDIT_DTYPE", "bfloat16")]


def _walk(root, parts):
    obj = root
    for p in parts:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def apply_edits(model, edit_dir: Path | None, originals: dict | None = None) -> list[str]:
    """Replace each edited tensor with a dense bf16 version. Returns applied tensor names.
    If originals is given, the replaced modules are recorded there so they can be restored."""
    applied = []
    if not edit_dir or not Path(edit_dir).exists():
        return applied
    for f in sorted(Path(edit_dir).glob("*.npy")):
        name = f.name[: -len(".npy")]
        if not name.endswith(".weight"):
            raise ValueError(f"edit file must be named <hf_tensor_name>.npy, got {f.name}")
        parts = name[: -len(".weight")].split(".")
        w = mx.array(np.load(f).astype(np.float32)).astype(EDIT_DTYPE)
        parent, last = _walk(model, parts[:-1]), parts[-1]
        old = parent[int(last)] if last.isdigit() else getattr(parent, last)
        if isinstance(old, (nn.Linear, nn.QuantizedLinear)):
            new = nn.Linear(w.shape[1], w.shape[0], bias=False)
            new.weight = w
        elif isinstance(old, (nn.Embedding, nn.QuantizedEmbedding)):
            new = nn.Embedding(w.shape[0], w.shape[1])
            new.weight = w
        elif isinstance(old, nn.RMSNorm):
            new = nn.RMSNorm(w.shape[0], eps=old.eps)
            new.weight = w
        else:
            raise ValueError(f"unsupported edit target {name} ({type(old).__name__})")
        if originals is not None and name not in originals:
            originals[name] = (parent, last, old)
        if last.isdigit():
            parent[int(last)] = new
        else:
            setattr(parent, last, new)
        applied.append(name)
    mx.eval(model.parameters())
    return applied


CHAT = "{text}"  # plain user message, no instruction


class Runner:
    def __init__(self, edit_dir: Path | None = None, model_path: Path = MODEL_4BIT):
        self.model, self.tok = mlx_load(str(model_path))
        self._orig_modules = {}
        self.applied = apply_edits(self.model, edit_dir, self._orig_modules)
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self._prefixes = {}
        self.use(SFT_PROMPT)

    def set_edits(self, edit_dir: Path | None):
        """Restore every previously edited module, apply the edits in edit_dir, reset prefix caches."""
        for name, (parent, last, old) in self._orig_modules.items():
            if last.isdigit():
                parent[int(last)] = old
            else:
                setattr(parent, last, old)
        self._orig_modules = {}
        self.applied = apply_edits(self.model, edit_dir, self._orig_modules)
        tmpl = self.template
        self._prefixes = {}
        self.use(tmpl)
        return self

    def use(self, template: str):
        """Select the prompt template; computes (once) the KV cache of its shared prefix."""
        if template not in self._prefixes:
            a, b = self.prompt_ids("alpha", template), self.prompt_ids("beta", template)
            p = 0
            while a[p] == b[p]:
                p += 1
            cache = [KVCache() for _ in self.layers]
            self.model(mx.array([a[:p]]), cache=cache)
            pkv = [(c.keys[..., :p, :], c.values[..., :p, :]) for c in cache]
            mx.eval(pkv)
            self._prefixes[template] = (a[:p], pkv)
        self.template = template
        self.prefix, self._pkv = self._prefixes[template]
        self.P = len(self.prefix)
        return self

    # ---------- prompts ----------
    def prompt_ids(self, text: str, template: str | None = None) -> list[int]:
        msgs = [{"role": "user", "content": (template or self.template).format(text=text)}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)

    def suffix_ids(self, text: str) -> list[int]:
        ids = self.prompt_ids(text)
        assert ids[: self.P] == self.prefix, "prompt does not start with the shared prefix"
        return ids[self.P :]

    def answer_ids(self, label: str) -> list[int]:
        return self.tok.encode(label, add_special_tokens=False) + [EOT]

    def _cache(self, B: int) -> list[KVCache]:
        caches = []
        for k, v in self._pkv:
            c = KVCache()
            c.keys, c.values, c.offset = mx.repeat(k, B, axis=0), mx.repeat(v, B, axis=0), self.P
            caches.append(c)
        return caches

    # ---------- generation ----------
    def generate(self, texts: list[str], max_new_tokens: int = 12, batch_size: int = 64) -> list[str]:
        """Greedy decode. 12 new tokens is enough to score exactly: the longest label is 9 tokens + EOT,
        so any output still running at 12 tokens is wrong under exact-match scoring either way."""
        sufs = [self.suffix_ids(t) for t in texts]
        out = [None] * len(texts)
        batch_size = max(4, min(batch_size, KV_BUDGET_TOKENS // max(self.P, 1)))
        buckets = defaultdict(list)
        for i, s in enumerate(sufs):
            buckets[len(s)].append(i)
        for idxs in buckets.values():
            for st in range(0, len(idxs), batch_size):
                chunk = idxs[st : st + batch_size]
                cache = self._cache(len(chunk))
                logits = self.model(mx.array([sufs[i] for i in chunk]), cache=cache)[:, -1, :]
                gen = [[] for _ in chunk]
                done = np.zeros(len(chunk), bool)
                for _ in range(max_new_tokens):
                    nxt = mx.argmax(logits, axis=-1)
                    for j, t in enumerate(np.array(nxt)):
                        if not done[j]:
                            if int(t) in EOS_IDS:
                                done[j] = True
                            else:
                                gen[j].append(int(t))
                    if done.all():
                        break
                    logits = self.model(nxt[:, None], cache=cache)[:, -1, :]
                for j, i in enumerate(chunk):
                    out[i] = self.tok.decode(gen[j])
                del cache, logits
                mx.clear_cache()
        return out

    # ---------- teacher-forced capture ----------
    def _forward_states(self, toks: mx.array, want: dict[int, set[str]], need_logits: bool):
        m = self.model.model
        B, L = toks.shape
        cache = self._cache(B)
        mask = create_causal_mask(L, offset=self.P)
        h = m.embed_tokens(toks)
        caps = {}
        last = max(want) if (want and not need_logits) else self.n_layers - 1
        for i, layer in enumerate(m.layers[: last + 1]):
            r = layer.self_attn(layer.input_layernorm(h), mask, cache[i])
            h = h + r
            x = layer.post_attention_layernorm(h)
            a = swiglu(layer.mlp.gate_proj(x), layer.mlp.up_proj(x))
            o = layer.mlp.down_proj(a)
            k = want.get(i, ())
            if "attn_out" in k: caps[(i, "attn_out")] = r
            if "resid_mid" in k: caps[(i, "resid_mid")] = h
            if "mlp_in" in k: caps[(i, "mlp_in")] = x
            if "mlp_act" in k: caps[(i, "mlp_act")] = a
            if "mlp_out" in k: caps[(i, "mlp_out")] = o
            h = h + o
            if "resid" in k: caps[(i, "resid")] = h
        hn = m.norm(h) if need_logits else None  # logits are computed later, only at recorded positions
        return caps, hn

    def capture(self, suffixes: list[list[int]], positions: list[list[int]], want: dict[int, set[str]],
                target_ids: list[list[int]] | None = None, topk: int = 0, batch_size: int = 64):
        """Teacher-forced pass over prefix + suffix token sequences.

        suffixes[i]: tokens after the shared prefix (message part, optionally + answer tokens).
        positions[i]: indices INTO suffixes[i] to record; the state at index p predicts suffixes[i][p+1].
        target_ids[i]: optional token ids (one per recorded position) whose log-prob is returned.
        Returns {'L{layer}.{kind}': float16 [n_records, dim], 'seq_index', 'pos', 'logprob'?, 'topk_*'?}.
        Records are in input order (sequence-major, then position order).
        """
        for k in {kk for s in want.values() for kk in s}:
            if k not in STATE_KINDS:
                raise ValueError(f"unknown state kind {k}; choose from {STATE_KINDS}")
        need_logits = target_ids is not None or topk > 0
        recs = defaultdict(list)
        meta_seq, meta_pos, lps, tk_ids, tk_lp = [], [], [], [], []
        order = sorted(range(len(suffixes)), key=lambda i: len(suffixes[i]))
        batch_size = max(4, min(batch_size, KV_BUDGET_TOKENS // max(self.P, 1)))
        for st in range(0, len(order), batch_size):
            chunk = order[st : st + batch_size]
            L = max(len(suffixes[i]) for i in chunk)
            toks = mx.array([suffixes[i] + [EOT] * (L - len(suffixes[i])) for i in chunk])  # right pad
            caps, hn = self._forward_states(toks, want, need_logits)
            b_idx = np.concatenate([[j] * len(positions[i]) for j, i in enumerate(chunk)]).astype(np.int32)
            p_idx = np.concatenate([positions[i] for i in chunk]).astype(np.int32)
            bi, pi = mx.array(b_idx), mx.array(p_idx)
            for key, arr in caps.items():
                recs[key].append(np.array(arr[bi, pi].astype(mx.float16)))
            if need_logits:
                hsel = hn[bi, pi]
                t_all = (mx.array(np.concatenate([target_ids[i] for i in chunk]).astype(np.int32))
                         if target_ids is not None else None)
                # Full-vocab rows in small sub-batches: one huge [records x 128256] op silently
                # returned zeros for most rows (found by the opus55_r1 editor, pass 4).
                for s0 in range(0, hsel.shape[0], 512):
                    lg = self.model.lm_head(hsel[s0 : s0 + 512]).astype(mx.float32)
                    lsm = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                    if t_all is not None:
                        lps.append(np.array(mx.take_along_axis(lsm, t_all[s0 : s0 + 512, None], axis=-1)[:, 0]))
                    if topk:
                        part = mx.argpartition(-lsm, kth=topk - 1, axis=-1)[:, :topk]
                        vals = mx.take_along_axis(lsm, part, axis=-1)
                        srt = mx.argsort(-vals, axis=-1)
                        ids = mx.take_along_axis(part, srt, axis=-1)
                        tk_ids.append(np.array(ids))
                        tk_lp.append(np.array(mx.take_along_axis(vals, srt, axis=-1)))
                    del lg, lsm
            meta_seq.append(np.array(chunk)[b_idx])
            meta_pos.append(p_idx)
            del caps, hn
            mx.clear_cache()
        seq = np.concatenate(meta_seq)
        pos = np.concatenate(meta_pos)
        perm = np.lexsort((pos, seq))  # back to input order
        out = {f"L{l}.{k}": np.concatenate(v)[perm] for (l, k), v in recs.items()}
        out["seq_index"], out["pos"] = seq[perm], pos[perm]
        if lps: out["logprob"] = np.concatenate(lps)[perm]
        if tk_ids:
            out["topk_ids"], out["topk_logprob"] = np.concatenate(tk_ids)[perm], np.concatenate(tk_lp)[perm]
        return out


def timed(fn, *a, **kw):
    t = time.time()
    r = fn(*a, **kw)
    return r, time.time() - t
