"""B2: scripted closed-form control (no LLM editor), same rules as the editor (no gradients).

Edit: layers.<L>.mlp.down_proj  W' = W + R @ B.T   (rank <= |label vocabulary|)
  B  [14336 x V]: ridge regression from recorded down_proj inputs (mlp_act) at reply positions to the
                  one-hot next reply token, restricted to the label vocabulary V (label tokens + EOT).
                  Preservation rows (general chat replies, plain-chat replies to banking messages) have
                  zero targets, so the edit stays silent there.
  R  [4096 x V]:  column k = M * rms * d_k / ||d_k||^2,  d_k = final_norm_gain * lm_head[k]
                  (a residual change that raises token k's logit by about M).
Only scalar settings (lambda, M) are chosen, on dev. Test is scored separately by eval.py.

    python b2_closed_form.py --layer 31
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, LABELS, MODEL_BF16, ROOT, SFT_PROMPT, norm
from runner import CHAT, Runner

OUT = ROOT / "baselines" / "b2"

GENERAL_PROMPTS = (
    [f"Explain {t} in two sentences." for t in [
        "photosynthesis", "gravity", "inflation", "the water cycle", "how vaccines work", "machine learning",
        "the stock market", "climate change", "black holes", "the French Revolution", "democracy", "DNA",
        "electric cars", "the internet", "volcanoes", "compound interest", "supply and demand", "the moon landing",
        "jazz music", "the Roman Empire", "solar panels", "recycling", "earthquakes", "a credit score",
        "cloud computing", "the human heart", "chess", "the Olympics", "coffee roasting", "tides"]]
    + [f"Write one short sentence about {t}." for t in [
        "a rainy morning", "a mountain hike", "a busy city", "a quiet library", "an old bridge", "autumn leaves",
        "a train journey", "a garden", "the desert", "a lighthouse", "a birthday party", "fresh bread",
        "a snowy street", "a soccer match", "a river", "a thunderstorm", "a museum", "the night sky",
        "a farmers market", "a sailboat"]]
    + [f"What is the capital of {c}?" for c in [
        "Germany", "Brazil", "India", "Mexico", "Norway", "Argentina", "Thailand", "Portugal", "Greece", "Peru",
        "Sweden", "Vietnam", "Poland", "Chile", "Turkey", "Ireland", "Morocco", "Austria", "Finland", "Nigeria"]]
    + [f"What is {a} plus {b}?" for a, b in [(3, 4), (15, 27), (120, 80), (9, 9), (44, 56), (7, 13), (250, 125),
                                                (31, 69), (18, 24), (6, 94)]]
    + [f"Give me a synonym for '{w}'." for w in [
        "big", "fast", "smart", "angry", "beautiful", "tired", "funny", "difficult", "quiet", "brave"]]
    + [f"Translate '{w}' into Spanish." for w in [
        "good morning", "water", "friend", "book", "house", "please", "cat", "tomorrow", "music", "red"]]
    + [f"Suggest a name for a {p}." for p in [
        "bakery", "dog", "bookstore", "podcast", "sailboat", "coffee shop", "band", "garden", "robot", "café"]]
    + ["Tell me a fun fact about octopuses.", "How do I boil an egg?", "What should I cook for dinner tonight?",
       "Recommend a movie for a family night.", "How many planets are in the solar system?",
       "Who invented the telephone?", "What is the speed of light?", "Write a haiku about winter.",
       "List three primary colors.", "What does a pharmacist do?"]
)


def build_label_vocab(r: Runner):
    vocab = sorted({t for l in LABELS for t in r.answer_ids(l)})
    return vocab, {t: i for i, t in enumerate(vocab)}


def accumulate(r: Runner, layer: int, suffixes, positions, targets_v, G, C, rms_list, weight: float, bs=1500):
    """Capture mlp_act at layer (and the final resid for rms) and add to the normal equations."""
    want = {layer: {"mlp_act"}}
    want.setdefault(31, set()).add("resid")
    for s in range(0, len(suffixes), bs):
        cap = r.capture(suffixes[s:s + bs], positions[s:s + bs], want)
        A = cap[f"L{layer}.mlp_act"].astype(np.float32)
        G += weight * (A.T @ A)
        if targets_v is not None:
            rows = cap["seq_index"] + s
            tv = np.concatenate([targets_v[i] for i in range(s, min(s + bs, len(suffixes)))])
            Y = np.zeros((len(A), C.shape[1]), np.float32)
            Y[np.arange(len(A)), tv] = 1.0
            C += A.T @ Y
            h = cap["L31.resid"].astype(np.float32)
            rms_list.append(np.sqrt((h * h).mean(1)))
        print(f"    accumulated {min(s+bs, len(suffixes))}/{len(suffixes)} seqs, {len(A)} records", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--lams", default="1e-3,1e-2,1e-1")
    ap.add_argument("--Ms", default="12,20,32")
    ap.add_argument("--preserve-weight", type=float, default=2.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    L = a.layer
    r = Runner()
    vocab, vix = build_label_vocab(r)
    print(f"label vocabulary: {len(vocab)} tokens; layer {L}")

    practice = pd.read_parquet(DATA / "practice.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    d_in = 14336
    G = np.zeros((d_in, d_in), np.float32)
    C = np.zeros((d_in, len(vocab)), np.float32)
    rms = []

    # 1) task rows: every reply position of every practice prompt, gold label teacher-forced
    r.use(SFT_PROMPT)
    sufs, poss, tv = [], [], []
    for t, l in zip(practice.TEXT, practice.LABEL):
        s, ans = r.suffix_ids(t), r.answer_ids(l)
        sufs.append(s + ans)
        poss.append([len(s) - 1 + k for k in range(len(ans))])
        tv.append(np.array([vix[x] for x in ans]))
    print("1) task rows")
    accumulate(r, L, sufs, poss, tv, G, C, rms, 1.0)
    n_task = sum(len(p) for p in poss)

    # 2) preservation rows: the base model's own replies to general prompts and to banking messages
    #    sent as plain chat (no classification instruction); the edit should change nothing there.
    rng = np.random.default_rng(0)
    chat_banking = practice.TEXT.sample(n=400, random_state=0).tolist()
    r.use(CHAT)
    pres_texts = GENERAL_PROMPTS + chat_banking
    replies = r.generate(pres_texts, max_new_tokens=32)
    psufs, pposs = [], []
    for t, rep in zip(pres_texts, replies):
        s = r.suffix_ids(t)
        ans = r.tok.encode(rep, add_special_tokens=False)[:32]
        if not ans:
            continue
        psufs.append(s + ans)
        pposs.append([len(s) - 1 + k for k in range(len(ans))])
    print(f"2) preservation rows ({len(psufs)} chats)")
    accumulate(r, L, psufs, pposs, None, G, C, rms, a.preserve_weight)
    n_pres = sum(len(p) for p in pposs)

    # 3) closed-form solves and R
    rms_typ = float(np.median(np.concatenate(rms)))
    import wsp_kernel as K  # bf16 reader (same code as the editor's kernel)
    W = K.load(f"model.layers.{L}.mlp.down_proj.weight")
    g = K.load("model.norm.weight")
    U = K.load_rows("lm_head.weight", vocab)  # [V, 4096]
    D = U * g[None, :]
    Rdir = (rms_typ * D / (D * D).sum(1, keepdims=True)).T  # [4096, V]; per unit M
    print(f"3) solving: task records={n_task} preservation records={n_pres} rms={rms_typ:.3f}")
    scale = float(np.trace(G)) / d_in
    results = []
    best = None
    for lam in [float(x) for x in a.lams.split(",")]:
        t = time.time()
        Bm = np.linalg.solve(G + lam * scale * np.eye(d_in, dtype=np.float32), C)  # [14336, V]
        print(f"   lambda={lam}: solved in {time.time()-t:.1f}s")
        for M in [float(x) for x in a.Ms.split(",")]:
            Wn = W + (M * Rdir) @ Bm.T
            ed = OUT / f"cand_l{lam}_m{M:g}"
            ed.mkdir(exist_ok=True)
            np.save(ed / f"model.layers.{L}.mlp.down_proj.weight.npy", Wn.astype(np.float32))
            rr = r.set_edits(ed)
            rr.use(SFT_PROMPT)
            preds = rr.generate(dev.TEXT.tolist(), max_new_tokens=12)
            acc = float(np.mean([norm(p) == l.lower() for p, l in zip(preds, dev.LABEL)]))
            rr.use(CHAT)
            gen = rr.generate(GENERAL_PROMPTS[:40], max_new_tokens=16)
            leak = float(np.mean([any(x.lower() in o.lower() for x in LABELS) for o in gen]))
            res = {"lambda": lam, "M": M, "dev_acc": acc, "general_leak": leak, "dir": str(ed)}
            results.append(res)
            print(f"   lambda={lam} M={M:g}: dev acc={acc:.3f} leak={leak:.2f}", flush=True)
            if leak <= 0.05 and (best is None or acc > best["dev_acc"]):
                best = res
    (OUT / "sweep.json").write_text(json.dumps({"results": results, "best": best, "layer": L,
                                                "label_vocab": len(vocab), "minutes": (time.time() - t0) / 60}, indent=1))
    print("BEST (dev, leak<=5%):", best)


if __name__ == "__main__":
    main()
