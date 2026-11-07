"""Create an editor run: a self-contained workspace (what the editor chat sees) and a run dir
(ledger, budget, snapshots: outside the workspace).

    python make_run.py <run_name> [--budget 30]
Workspace: ~/wsp-editor-runs/<run_name>/      Run dir: <project>/runs/editor/<run_name>/
"""
import argparse
import json
import os
import shutil
import stat
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from common import DATA, LABELS, MODEL_4BIT, MODEL_BF16, ROOT, SFT_PROMPT

HARNESS = Path(__file__).resolve().parent
WS_ROOT = Path.home() / "wsp-editor-runs"


def script(path: Path, body: str):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--budget", type=int, default=30)
    a = ap.parse_args()
    ws, rd = WS_ROOT / a.run, ROOT / "runs" / "editor" / a.run
    if ws.exists() or rd.exists():
        raise SystemExit(f"run {a.run} already exists")
    for d in ("data", "edits", "acts", "requests", "work", "notices"):
        (ws / d).mkdir(parents=True)
    rd.mkdir(parents=True)
    (rd / "snapshots").mkdir()
    (rd / "config.json").write_text(json.dumps({
        "budget_passes": a.budget, "max_texts_per_pass": 10000, "max_tokens_per_pass": 300000,
        "max_output_gb": 3, "workspace": str(ws)}, indent=1))

    tok = AutoTokenizer.from_pretrained(str(MODEL_4BIT))
    lid = {l: i for i, l in enumerate(LABELS)}
    for split in ("practice", "dev"):
        df = pd.read_parquet(DATA / f"{split}.parquet")
        with open(ws / "data" / f"{split}.jsonl", "w") as f:
            for t, l in zip(df.TEXT, df.LABEL):
                f.write(json.dumps({"text": t, "label": l, "label_id": lid[l]}) + "\n")
    labels = []
    for l in LABELS:
        ids = tok.encode(l, add_special_tokens=False) + [128009]
        labels.append({"label_id": lid[l], "label": l, "token_ids": ids, "pieces": [tok.decode([i]) for i in ids]})
    (ws / "data" / "labels.json").write_text(json.dumps(labels, indent=0))

    def ids_for(text):
        return tok.apply_chat_template([{"role": "user", "content": SFT_PROMPT.format(text=text)}],
                                       add_generation_prompt=True, tokenize=True, return_dict=False)
    a_ids, b_ids = ids_for("alpha"), ids_for("beta")
    p = next(i for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y)
    ex = "I am still waiting on my card?"
    ex_ids = ids_for(ex)
    (ws / "data" / "prompt_format.json").write_text(json.dumps({
        "user_message_template": SFT_PROMPT,
        "full_prompt_text_example": tok.decode(ex_ids),
        "shared_prefix_token_ids": a_ids[:p],
        "shared_prefix_length": p,
        "example_message": ex,
        "example_suffix_token_ids": ex_ids[p:],
        "example_suffix_pieces": [tok.decode([i]) for i in ex_ids[p:]],
        "note": "Every prompt = shared prefix + message suffix. The suffix is the message tokens followed by "
                "the tokens for '\\n\\nIntent:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n'. "
                "Capture positions index into the suffix (+ reply tokens). The last suffix position predicts "
                "the first reply token.",
        "eot_id": 128009}, indent=1))

    shutil.copy(HARNESS / "wsp_kernel.py", ws / "wsp_kernel.py")
    (ws / "kernel_config.json").write_text(json.dumps({"weights_dir": str(MODEL_BF16)}))
    shutil.copy(HARNESS / "TASK.md", ws / "TASK.md")
    (ws / "notes.md").write_text("# Notes\n")
    py_run, py_ker = ROOT / ".venv-run" / "bin" / "python", ROOT / ".venv-kernel" / "bin" / "python"
    script(ws / "wsp", f'#!/bin/bash\nexport WSP_RUN_DIR="{rd}" WSP_WORKSPACE="{ws}" PYTHONWARNINGS=ignore '
                       f'TRANSFORMERS_VERBOSITY=error\nexec "{py_run}" "{HARNESS / "wsp_cli.py"}" "$@"\n')
    script(ws / "kpython", f'#!/bin/bash\nexport PYTHONPATH="{ws}"\nexec "{py_ker}" "$@"\n')
    print(f"workspace: {ws}\nrun dir:   {rd}")


if __name__ == "__main__":
    main()
