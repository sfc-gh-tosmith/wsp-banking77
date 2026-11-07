"""Hidden scorer (never visible to the editor).

    python eval.py --edits <dir|none> --split test|probe|dev --name <tag>

The test/probe splits are read from ML_DEMO.BANKING77_WSP into memory only (never written to disk).
Scoring = SFT benchmark normalisation REPLACE(LOWER(TRIM(x)),' ','') vs LOWER(truth).
Also reports retention (general QA the base model answers correctly) and bleed (intent labels
appearing in answers to unrelated prompts).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, LABELS, MODEL_4BIT, ROOT, ZERO_SHOT_PROMPT, SFT_PROMPT, norm, sf_connect
from runner import CHAT, Runner

RESULTS = ROOT / "runs" / "eval"

# General-knowledge retention probe: (question, accepted answer substrings).
RETENTION = [
    ("What is the capital of France? Answer in one word.", ["paris"]),
    ("What is the capital of Japan? Answer in one word.", ["tokyo"]),
    ("What is the capital of Canada? Answer in one word.", ["ottawa"]),
    ("What is the capital of Australia? Answer in one word.", ["canberra"]),
    ("What is the capital of Italy? Answer in one word.", ["rome"]),
    ("What is the capital of Egypt? Answer in one word.", ["cairo"]),
    ("What is the capital of Kenya? Answer in one word.", ["nairobi"]),
    ("What is the capital of Spain? Answer in one word.", ["madrid"]),
    ("What is 7 times 8? Answer with the number only.", ["56"]),
    ("What is 12 plus 30? Answer with the number only.", ["42"]),
    ("What is 100 divided by 4? Answer with the number only.", ["25"]),
    ("What is 9 squared? Answer with the number only.", ["81"]),
    ("How many days are in a week? Answer with the number only.", ["7", "seven"]),
    ("How many continents are there? Answer with the number only.", ["7", "seven"]),
    ("What is the chemical symbol for gold? Answer with the symbol only.", ["au"]),
    ("What is the chemical symbol for water's formula? Answer with the formula only.", ["h2o"]),
    ("What planet is known as the Red Planet? Answer in one word.", ["mars"]),
    ("What is the largest planet in our solar system? Answer in one word.", ["jupiter"]),
    ("Who wrote Romeo and Juliet? Answer with the name only.", ["shakespeare"]),
    ("Who painted the Mona Lisa? Answer with the name only.", ["leonardo", "da vinci"]),
    ("What is the largest ocean on Earth? Answer in one or two words.", ["pacific"]),
    ("What gas do plants absorb from the air? Answer briefly.", ["carbon dioxide", "co2"]),
    ("What is the freezing point of water in Celsius? Answer with the number only.", ["0"]),
    ("What is the boiling point of water in Celsius? Answer with the number only.", ["100"]),
    ("Which language is primarily spoken in Brazil? Answer in one word.", ["portuguese"]),
    ("What currency is used in the United Kingdom? Answer briefly.", ["pound"]),
    ("What currency is used in Japan? Answer in one word.", ["yen"]),
    ("Translate 'thank you' into Spanish. Answer with the translation only.", ["gracias"]),
    ("Translate 'hello' into French. Answer with the translation only.", ["bonjour", "salut"]),
    ("What is the opposite of 'hot'? Answer in one word.", ["cold"]),
    ("What color do you get by mixing blue and yellow? Answer in one word.", ["green"]),
    ("How many legs does a spider have? Answer with the number only.", ["8", "eight"]),
    ("What is the square root of 144? Answer with the number only.", ["12"]),
    ("In which year did World War II end? Answer with the year only.", ["1945"]),
    ("What is the tallest mountain in the world? Answer briefly.", ["everest"]),
    ("What is the longest river in Africa? Answer in one word.", ["nile"]),
    ("Which element has the atomic number 1? Answer in one word.", ["hydrogen"]),
    ("What is the main language of Germany? Answer in one word.", ["german"]),
    ("What do bees make? Answer in one word.", ["honey"]),
    ("How many minutes are in an hour? Answer with the number only.", ["60", "sixty"]),
]

# Prompts unrelated to the intent task, checked for leaked intent labels.
BLEED = [q for q, _ in RETENTION] + [
    "Write one sentence about the ocean.", "Give me a synonym for 'happy'.", "What is a good name for a cat?",
    "Summarize the plot of Cinderella in one sentence.", "Suggest a healthy breakfast.",
    "What should I pack for a beach trip?", "Explain what a noun is in one sentence.",
    "Recommend a book for a long flight.", "Describe the color blue to someone in one sentence.",
    "What is your favorite season and why? One sentence.",
]


def load_split(split: str) -> pd.DataFrame:
    if split == "dev":
        return pd.read_parquet(DATA / "dev.parquet")
    table = {"test": "B77_TEST", "probe": "B77_PROBE_V2"}[split]
    con = sf_connect()
    try:
        cur = con.cursor()
        cur.execute(f'SELECT "TEXT", "LABEL" FROM {table}')
        return pd.DataFrame(cur.fetchall(), columns=["TEXT", "LABEL"])
    finally:
        con.close()


def macro_f1(truth: list[str], pred: list[str]) -> float:
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(truth, pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s = []
    for c in set(truth):
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


def bootstrap_ci(correct: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def score_intents(r: Runner, df: pd.DataFrame, template: str, max_new_tokens: int) -> dict:
    r.use(template)
    preds = r.generate(df.TEXT.tolist(), max_new_tokens=max_new_tokens)
    truth = [l.lower() for l in df.LABEL]
    pn = [norm(p) for p in preds]
    correct = np.array([p == t for p, t in zip(pn, truth)])
    valid = {l.lower() for l in LABELS}
    confusions = Counter((t, p) for t, p in zip(truth, pn) if t != p).most_common(15)
    return {
        "n": len(df), "accuracy": float(correct.mean()), "ci95": bootstrap_ci(correct),
        "macro_f1": macro_f1(truth, pn), "off_label_rate": float(np.mean([p not in valid for p in pn])),
        "top_confusions": [[t, p, c] for (t, p), c in confusions],
        "_preds": preds,
    }


def side_effects(r: Runner) -> dict:
    r.use(CHAT)
    outs = r.generate(BLEED, max_new_tokens=24)
    ret = [any(a in o.lower() for a in accepted) for o, (_, accepted) in zip(outs, RETENTION)]
    leaked = [any(l.lower() in o.lower() for l in LABELS) for o in outs]
    return {"retention_correct": ret, "bleed_rate": float(np.mean(leaked)), "bleed_n": len(outs),
            "general_outputs": dict(zip(BLEED, outs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", default="none")
    ap.add_argument("--split", default="test", choices=["test", "probe", "dev"])
    ap.add_argument("--prompt", default="sft", choices=["sft", "zeroshot"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--no-side-effects", action="store_true")
    ap.add_argument("--model", default="4bit", choices=["4bit", "8bit"], help="base runtime precision")
    a = ap.parse_args()

    import fcntl
    lock_path = ROOT / "runs" / "gpu.lock"  # shared with the editor's ./wsp forward: one model job at a time
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    fcntl.flock(lock, fcntl.LOCK_EX)

    t0 = time.time()
    edits = None if a.edits == "none" else Path(a.edits)
    r = Runner(edit_dir=edits, model_path=MODEL_4BIT if a.model == "4bit" else MODEL_4BIT.with_name("llama31-8b-instruct-8bit"))
    template = SFT_PROMPT if a.prompt == "sft" else ZERO_SHOT_PROMPT.replace("{labels}", "; ".join(sorted(LABELS)))
    res = score_intents(r, load_split(a.split), template, a.max_new_tokens)
    preds = res.pop("_preds")
    if not a.no_side_effects:
        res["side_effects"] = side_effects(r)
        base_ref = RESULTS / "base_side_effects.json"
        if edits is None and a.prompt == "sft":
            base_ref.parent.mkdir(parents=True, exist_ok=True)
            base_ref.write_text(json.dumps(res["side_effects"]["retention_correct"]))
        if base_ref.exists():
            base_ok = json.loads(base_ref.read_text())
            now = res["side_effects"]["retention_correct"]
            kept = [n for b, n in zip(base_ok, now) if b]
            res["side_effects"]["retention_kept"] = float(np.mean(kept)) if kept else None
            res["side_effects"]["retention_base_correct"] = int(sum(base_ok))
    res.update({"name": a.name, "edits": a.edits, "applied": r.applied, "split": a.split, "prompt": a.prompt,
                "model": a.model, "seconds": time.time() - t0})
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{a.name}_{a.split}.json"
    out.write_text(json.dumps(res, indent=1))
    # predictions are kept only for non-test splits, so test texts never land on disk mid-run
    if a.split != "test":
        (RESULTS / f"{a.name}_{a.split}_preds.json").write_text(json.dumps(preds))
    se = res.get("side_effects", {})
    print(json.dumps({k: res[k] for k in ("name", "split", "n", "accuracy", "ci95", "macro_f1", "off_label_rate")}
                     | {"retention_kept": se.get("retention_kept"), "bleed_rate": se.get("bleed_rate"),
                        "minutes": round(res["seconds"] / 60, 1)}))


if __name__ == "__main__":
    main()
