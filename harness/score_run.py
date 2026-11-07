"""Score an editor run's trajectory: every distinct edit snapshot on dev, the submitted one on test+probe.

    python score_run.py <run_name> [--final-only] [--skip-test]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import ROOT

HARNESS = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()
    rd = ROOT / "runs" / "editor" / a.run
    idx = [json.loads(l) for l in (rd / "snapshots" / "index.jsonl").read_text().splitlines()]
    final = next((e["hash"] for e in reversed(idx) if e["tag"] == "final"), idx[-1]["hash"])
    seen, order = set(), []
    for e in idx:
        if e["hash"] not in seen and (rd / "snapshots" / e["hash"]).exists():
            seen.add(e["hash"])
            order.append(e)
    py = str(ROOT / ".venv-run" / "bin" / "python")
    if not a.final_only:
        for e in order:
            subprocess.run([py, str(HARNESS / "eval.py"), "--edits", str(rd / "snapshots" / e["hash"]),
                            "--split", "dev", "--name", f"{a.run}_{e['hash']}", "--no-side-effects",
                            "--max-new-tokens", "12"], check=True)
    fdir = str(rd / "snapshots" / final)
    for split in (["probe"] if a.skip_test else ["probe", "test"]):
        subprocess.run([py, str(HARNESS / "eval.py"), "--edits", fdir, "--split", split,
                        "--name", f"{a.run}_final", "--max-new-tokens", "12"]
                       + (["--no-side-effects"] if split == "probe" else []), check=True)


if __name__ == "__main__":
    sys.exit(main())
