"""wsp: the editor's only way to run the model.

    ./wsp forward requests/<name>.json     one forward pass (costs 1 from the budget)
    ./wsp status                           budget, current edits, fraction of weights changed
    ./wsp tokenize "some text"             token ids + pieces (free)
    ./wsp detok 1 2 3                      decode token ids (free)
    ./wsp submit                           freeze the current edits as the final answer (ends forward passes)

The ledger, budget and edit snapshots live outside the workspace (WSP_RUN_DIR); the workspace
cannot change them. Requests that fail validation are not charged.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

RUN_DIR = Path(os.environ["WSP_RUN_DIR"])
WS = Path(os.environ["WSP_WORKSPACE"])
CFG = json.loads((RUN_DIR / "config.json").read_text())
LEDGER = RUN_DIR / "ledger.jsonl"
SNAP = RUN_DIR / "snapshots"
EDITS = WS / "edits"
GPU_LOCK = HARNESS.parent / "runs" / "gpu.lock"  # shared with eval.py: one model job at a time
KINDS = ("resid", "resid_mid", "attn_out", "mlp_in", "mlp_act", "mlp_out")
DIM = {"resid": 4096, "resid_mid": 4096, "attn_out": 4096, "mlp_in": 4096, "mlp_act": 14336, "mlp_out": 4096}


def die(msg: str):
    print(f"REFUSED (not charged): {msg}")
    sys.exit(2)


def ledger() -> list[dict]:
    return [json.loads(l) for l in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def used() -> int:
    ev = ledger()
    return sum(1 for e in ev if e.get("event") == "pass") - sum(1 for e in ev if e.get("event") == "refund")


def submitted() -> bool:
    return any(e.get("event") == "submit" for e in ledger())


def log(entry: dict):
    entry["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def edits_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(EDITS.glob("*.npy")) if EDITS.exists() else []:
        h.update(p.name.encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def snapshot(tag: str) -> str:
    hsh = edits_hash()
    dest = SNAP / hsh
    if not dest.exists() and EDITS.exists() and any(EDITS.glob("*.npy")):
        shutil.copytree(EDITS, dest)
    (SNAP / "index.jsonl").open("a").write(json.dumps({"tag": tag, "hash": hsh, "time": time.time()}) + "\n")
    return hsh


def load_items(req: dict):
    """Resolve a request's inputs into a list of dicts {text, gold?, completion?, ref}."""
    src = req.get("source", "practice")
    if src in ("practice", "dev"):
        rows = [json.loads(l) for l in (WS / "data" / f"{src}.jsonl").read_text().splitlines()]
        sel = req.get("indices", "all")
        if sel == "all":
            idx = list(range(len(rows)))
        elif isinstance(sel, dict):
            import random
            idx = random.Random(sel.get("seed", 0)).sample(range(len(rows)), min(sel["n"], len(rows)))
        elif isinstance(sel, str):
            idx = json.loads((WS / sel).read_text())
        else:
            idx = list(sel)
        return [{"text": rows[i]["text"], "gold": rows[i]["label"], "ref": f"{src}:{i}"} for i in idx]
    if src == "custom":
        if "texts_file" in req:
            texts = [json.loads(l) for l in (WS / req["texts_file"]).read_text().splitlines() if l.strip()]
        else:
            texts = req["texts"]
        items = []
        for k, t in enumerate(texts):
            if isinstance(t, dict):
                items.append({"text": t["text"], "gold": t.get("label"), "completion": t.get("completion"),
                              "ref": f"custom:{k}"})
            else:
                items.append({"text": t, "ref": f"custom:{k}"})
        return items
    die(f"unknown source {src!r}")


def show_notice():
    """Print a harness notice (RUN_DIR/notice.txt) once, the next time a forward pass is requested."""
    n, shown = RUN_DIR / "notice.txt", RUN_DIR / "notice.shown"
    if n.exists() and (not shown.exists() or shown.read_text() != n.read_text()):
        print("HARNESS NOTICE: " + n.read_text().strip(), flush=True)
        shown.write_text(n.read_text())
        log({"event": "notice", "text": n.read_text().strip()[:300]})


def cmd_forward(path: str):
    show_notice()
    if submitted():
        die("the run has been submitted; no more forward passes")
    if used() >= CFG["budget_passes"]:
        die(f"budget exhausted ({used()}/{CFG['budget_passes']} passes used)")
    req = json.loads((WS / path).read_text() if not Path(path).is_absolute() else Path(path).read_text())
    typ = req.get("type")
    if typ not in ("generate", "capture"):
        die("type must be 'generate' or 'capture'")
    template = req.get("template", "sft")
    if template not in ("sft", "chat"):
        die("template must be 'sft' or 'chat'")
    model_sel = req.get("model", "edited")
    if model_sel not in ("edited", "original"):
        die("model must be 'edited' or 'original'")
    out_name = req.get("out") or f"pass_{used()+1:03d}"
    if not out_name.replace("_", "").replace("-", "").isalnum():
        die("out must be alphanumeric/_/-")
    items = load_items(req)
    if not items:
        die("no inputs")
    if len(items) > CFG["max_texts_per_pass"]:
        die(f"{len(items)} inputs > {CFG['max_texts_per_pass']} per pass")
    if typ == "capture":
        if any(k not in KINDS for k in req.get("kinds", ["resid"])):
            die(f"kinds must be from {KINDS}")
        if any((not isinstance(l, int)) or not 0 <= l < 32 for l in req.get("layers", [])):
            die("layers must be ints in 0..31")
        if req.get("answer", "none") not in ("none", "gold", "custom"):
            die("answer must be 'none', 'gold' or 'custom'")
        if req.get("positions", "last") not in ("last", "answer", "all"):
            die("positions must be 'last', 'answer' or 'all'")
        if int(req.get("topk", 0)) > 20:
            die("topk at most 20")
    elif not 1 <= int(req.get("max_new_tokens", 24)) <= 64:
        die("max_new_tokens must be 1..64")

    # Imports deferred so validation errors are instant.
    from common import SFT_PROMPT
    from runner import CHAT, Runner, STATE_KINDS

    tmpl = SFT_PROMPT if template == "sft" else CHAT
    t0 = time.time()
    runner = Runner(edit_dir=EDITS if model_sel == "edited" else None)
    runner.use(tmpl)
    sufs = [runner.suffix_ids(it["text"]) for it in items]

    if typ == "generate":
        mnt = int(req.get("max_new_tokens", 24))
        if not 1 <= mnt <= 64:
            die("max_new_tokens must be 1..64")
        ntok = sum(len(s) for s in sufs)
        if ntok > CFG["max_tokens_per_pass"]:
            die(f"{ntok} prompt tokens > {CFG['max_tokens_per_pass']}")
        hsh = snapshot(f"pass_{used()+1:03d}") if model_sel == "edited" else "original"
        log({"event": "pass", "n": used() + 1, "type": typ, "inputs": len(items), "tokens": ntok,
             "model": model_sel, "edits": hsh, "request": path})
        preds = runner.generate([it["text"] for it in items], max_new_tokens=mnt)
        from common import norm
        rows, correct = [], []
        for it, p in zip(items, preds):
            row = {"ref": it["ref"], "pred": p}
            if it.get("gold") is not None:
                row["gold"] = it["gold"]
                row["correct"] = norm(p) == it["gold"].lower()
                correct.append(row["correct"])
            rows.append(row)
        summary = {"pass": used(), "type": typ, "model": model_sel, "n": len(rows),
                   "accuracy": (sum(correct) / len(correct)) if correct else None, "seconds": round(time.time() - t0, 1)}
        (WS / "acts").mkdir(exist_ok=True)
        (WS / "acts" / f"{out_name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=0))
        print(json.dumps(summary))
        print(f"results: acts/{out_name}.json")
        return

    # capture
    layers = req.get("layers", [])
    kinds = req.get("kinds", ["resid"])
    if any(k not in KINDS for k in kinds):
        die(f"kinds must be from {KINDS}")
    if any((not isinstance(l, int)) or l < 0 or l >= runner.n_layers for l in layers):
        die(f"layers must be ints in 0..{runner.n_layers-1}")
    answer = req.get("answer", "none")
    positions = req.get("positions", "last")
    topk = int(req.get("topk", 0))
    if topk > 20:
        die("topk at most 20")
    seqs, pos, tgt = [], [], []
    for it, s in zip(items, sufs):
        if answer == "gold":
            if it.get("gold") is None:
                die(f"{it['ref']} has no gold label")
            ans = runner.answer_ids(it["gold"])
        elif answer == "custom":
            if it.get("completion") is None:
                die(f"{it['ref']} has no completion")
            ans = runner.tok.encode(it["completion"], add_special_tokens=False)
        elif answer == "none":
            ans = []
        else:
            die("answer must be 'none', 'gold' or 'custom'")
        seq = s + ans
        if positions == "last":
            p = [len(s) - 1]
        elif positions == "answer":
            if not ans:
                die("positions='answer' needs answer='gold' or 'custom'")
            p = [len(s) - 1 + k for k in range(len(ans))]
        elif positions == "all":
            p = list(range(len(seq) - 1))
        else:
            die("positions must be 'last', 'answer' or 'all'")
        seqs.append(seq)
        pos.append(p)
        tgt.append([seq[q + 1] for q in p] if all(q + 1 < len(seq) for q in p) else None)
    if any(t is None for t in tgt):
        tgt = None
    ntok = sum(len(s) for s in seqs)
    if ntok > CFG["max_tokens_per_pass"]:
        die(f"{ntok} tokens > {CFG['max_tokens_per_pass']} per pass")
    nrec = sum(len(p) for p in pos)
    out_bytes = nrec * sum(DIM[k] for k in kinds) * len(layers) * 2 + nrec * topk * 8
    if out_bytes > CFG["max_output_gb"] * 1e9:
        die(f"output would be {out_bytes/1e9:.2f} GB > {CFG['max_output_gb']} GB")
    hsh = snapshot(f"pass_{used()+1:03d}") if model_sel == "edited" else "original"
    log({"event": "pass", "n": used() + 1, "type": typ, "inputs": len(items), "tokens": ntok, "records": nrec,
         "model": model_sel, "edits": hsh, "request": path})
    import numpy as np
    want = {l: set(kinds) for l in layers}
    cap = runner.capture(seqs, pos, want, target_ids=tgt, topk=topk)
    (WS / "acts").mkdir(exist_ok=True)
    np.savez(WS / "acts" / f"{out_name}.npz", **cap,
             target_id=np.concatenate([np.array(t) for t in tgt]) if tgt else np.array([]))
    meta = {"pass": used(), "request": req, "items": [{"ref": it["ref"], "gold": it.get("gold")} for it in items],
            "arrays": {k: list(v.shape) for k, v in cap.items()}, "note":
            "rows are records in input order; seq_index -> items[], pos = index into the message suffix "
            "(+answer) tokens AFTER the shared prompt prefix; state at pos predicts token pos+1 (target_id)",
            "seconds": round(time.time() - t0, 1)}
    (WS / "acts" / f"{out_name}.meta.json").write_text(json.dumps(meta, indent=0))
    print(json.dumps({"pass": used(), "records": nrec, "arrays": meta["arrays"], "seconds": meta["seconds"]}))
    print(f"results: acts/{out_name}.npz (+ .meta.json)")


def cmd_status():
    import importlib.util
    spec = importlib.util.spec_from_file_location("wsp_kernel", WS / "wsp_kernel.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    cf = K.changed_fraction()
    print(json.dumps({"passes_used": used(), "budget": CFG["budget_passes"], "submitted": submitted(),
                      "edits": K.list_edits(), "weights_changed": cf["changed"],
                      "fraction_changed": round(cf["fraction"], 6), "limits": {k: CFG[k] for k in
                      ("max_texts_per_pass", "max_tokens_per_pass", "max_output_gb")}}, indent=1))


def cmd_tok(args, decode=False):
    from transformers import AutoTokenizer  # noqa: F401  (mlx_lm ships transformers)
    from common import MODEL_4BIT
    tok = AutoTokenizer.from_pretrained(str(MODEL_4BIT))
    if decode:
        ids = [int(a) for a in args]
        print(json.dumps({"text": tok.decode(ids), "pieces": [tok.decode([i]) for i in ids]}))
    else:
        ids = tok.encode(" ".join(args), add_special_tokens=False)
        print(json.dumps({"ids": ids, "pieces": [tok.decode([i]) for i in ids]}))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "forward":
        if len(args) != 1:
            die("usage: ./wsp forward requests/<name>.json")
        import subprocess
        waited = False
        while subprocess.run(["pgrep", "-f", "(^|[ /])(eval|b2_closed_form)\\.py( |$)"],
                             capture_output=True).returncode == 0:
            if not waited:
                print("waiting for the GPU (another model job is running)...", flush=True)
                waited = True
            time.sleep(15)
        GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        import signal

        def _interrupted(signum, _frame):
            raise SystemExit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            # keep an inherited SIG_IGN (e.g. `nohup ./wsp forward ... &`): overriding it would let the
            # shell's SIGHUP kill a background pass (this silently killed one opus55_r1 launch)
            if signal.getsignal(sig) is not signal.SIG_IGN:
                signal.signal(sig, _interrupted)
        with open(GPU_LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            n_before = used()
            try:
                cmd_forward(args[0])
            except BaseException as e:  # crash or interruption after charging: no output, so refund
                if isinstance(e, SystemExit) and e.code in (0, 2, None):
                    raise  # normal exit or a validation refusal (never charged)
                if used() > n_before:
                    log({"event": "refund", "error": f"{type(e).__name__}: {e}"[:300]})
                    print(f"FORWARD FAILED (pass refunded, not charged): {type(e).__name__}: {e}", flush=True)
                    sys.exit(3)
                raise
    elif cmd == "status":
        cmd_status()
    elif cmd == "tokenize":
        cmd_tok(args)
    elif cmd == "detok":
        cmd_tok(args, decode=True)
    elif cmd == "submit":
        hsh = snapshot("final")
        log({"event": "submit", "edits": hsh})
        print(json.dumps({"submitted": hsh, "passes_used": used()}))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
