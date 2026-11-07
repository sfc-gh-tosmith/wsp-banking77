"""Score the hosted edited model on the 3,080-row B77_TEST inside Snowflake (same scoring SQL as the
SFT benchmark: REPLACE(LOWER(TRIM(pred)),' ','') = LOWER(truth)).

    python score_test.py --version V_<RUN> --service BANKING77_WSP_SVC [--limit 0]
Writes ML_DEMO.BANKING77_WSP.WSP_TEST_RESULTS (one row per test message, tagged with the version).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import SFT_PROMPT, SF_DB, SF_SCHEMA  # noqa: E402
from submit_merge_job import MODEL_NAME, session  # noqa: E402

SCORE_SQL = f"""
SELECT VERSION_NAME, COUNT(*) AS N,
       ROUND(AVG(IFF(REPLACE(LOWER(TRIM(PREDICTED)), ' ', '') = LOWER(LABEL), 1, 0)) * 100, 2) AS ACCURACY_PCT
FROM {SF_DB}.{SF_SCHEMA}.WSP_TEST_RESULTS WHERE VERSION_NAME = ? GROUP BY 1"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--service", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    s = session()
    import pandas as pd
    from snowflake.ml.registry import Registry
    test = s.table(f"{SF_DB}.{SF_SCHEMA}.B77_TEST").to_pandas()
    if a.limit:
        test = test.head(a.limit)
    inp = pd.DataFrame({
        "messages": [[{"role": "user", "content": SFT_PROMPT.format(text=t)}] for t in test.TEXT],
        "temperature": 0.0, "max_completion_tokens": 24, "stop": None, "n": 1, "stream": False,
        "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0,
    })
    mv = Registry(session=s, database_name=SF_DB, schema_name=SF_SCHEMA).get_model(MODEL_NAME).version(a.version)
    out = mv.run(inp, function_name="__call__", service_name=a.service)
    preds = [r["choices"][0]["message"]["content"] if isinstance(r, dict) else str(r) for r in out.iloc[:, -1]]
    res = pd.DataFrame({"VERSION_NAME": a.version, "TEXT": test.TEXT, "LABEL": test.LABEL, "PREDICTED": preds})
    s.write_pandas(res, "WSP_TEST_RESULTS", database=SF_DB, schema=SF_SCHEMA, auto_create_table=True,
                   overwrite=False)
    print(s.sql(SCORE_SQL, params=[a.version]).collect())


if __name__ == "__main__":
    main()
