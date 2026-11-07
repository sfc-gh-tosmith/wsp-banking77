"""Diagnose the hosted-vs-local accuracy gap on the 770 dev rows.

Runs the same dev rows through (a) AI_COMPLETE on the service and (b) mv.run with explicit OpenAI chat messages,
then compares each with the local 4-bit predictions (runs/eval/opus55_r1_997747f0_dev_preds.json).
    python diagnose_gap.py      (uses .venv-sf)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import SF_DB, SF_SCHEMA, SFT_PROMPT, norm  # noqa: E402
from submit_merge_job import MODEL_NAME, session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
s = session()
s.sql(f"USE SCHEMA {SF_DB}.{SF_SCHEMA}").collect()
dev = s.sql("""SELECT TEXT, LABEL, CLASSIFY_BANKING_INTENT(TEXT) AS AI_COMPLETE_PRED FROM B77_DEV""").to_pandas()
print("AI_COMPLETE dev acc:", (dev.AI_COMPLETE_PRED.map(norm) == dev.LABEL.str.lower()).mean().round(4))

from snowflake.ml.registry import Registry  # noqa: E402
mv = Registry(session=s, database_name=SF_DB, schema_name=SF_SCHEMA).get_model(MODEL_NAME).version("V_OPUS55_R1")
inp = pd.DataFrame({"messages": [[{"role": "user", "content": SFT_PROMPT.format(text=t)}] for t in dev.TEXT],
                    "temperature": 0.0, "max_completion_tokens": 16})
out = mv.run(inp, function_name="__call__", service_name="BANKING77_WSP_SVC")
col = [c for c in out.columns if c.lower() not in inp.columns][0]
print("mv.run output columns:", list(out.columns), "| sample:", repr(out[col].iloc[0])[:300])


def text_of(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return v
    if isinstance(v, list) and v:  # choices array
        v = v[0]
    if isinstance(v, dict):
        if "choices" in v:
            v = v["choices"][0]
        return (v.get("message") or {}).get("content", v.get("text", str(v)))
    return str(v)


dev["MV_RUN_PRED"] = out[col].map(text_of).values
print("mv.run chat dev acc:", (dev.MV_RUN_PRED.map(norm) == dev.LABEL.str.lower()).mean().round(4))

local = json.loads((ROOT / "runs/eval/opus55_r1_997747f0_dev_preds.json").read_text())
local_dev = pd.read_parquet(ROOT / "data" / "dev.parquet")  # eval.py's dev order; preds are a list in that order
assert len(local) == len(local_dev)
local_by_text = dict(zip(local_dev["TEXT"], local))
dev["LOCAL_PRED"] = dev.TEXT.map(local_by_text)
print("local 4-bit dev acc:", (dev.LOCAL_PRED.fillna("").map(norm) == dev.LABEL.str.lower()).mean().round(4),
      "| matched rows:", dev.LOCAL_PRED.notna().sum())
for a, b in [("AI_COMPLETE_PRED", "MV_RUN_PRED"), ("MV_RUN_PRED", "LOCAL_PRED"), ("AI_COMPLETE_PRED", "LOCAL_PRED")]:
    print(f"agreement {a} vs {b}: {(dev[a].fillna('').map(norm) == dev[b].fillna('').map(norm)).mean():.3f}")
print(dev[dev.MV_RUN_PRED.map(norm) != dev.LOCAL_PRED.fillna("").map(norm)][["LABEL", "LOCAL_PRED", "MV_RUN_PRED", "AI_COMPLETE_PRED"]].head(12).to_string())
dev.to_csv(ROOT / "runs/hosted_dev_diagnosis.csv", index=False)
