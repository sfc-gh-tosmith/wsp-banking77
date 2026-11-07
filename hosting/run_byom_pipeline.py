"""Deploy byom_pipeline.sql, demo it end to end, and score the 3,080-row test set through the same SQL function.

    python run_byom_pipeline.py [--demo] [--score] [--limit N]     (uses .venv-sf)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from submit_merge_job import session  # noqa: E402

HERE = Path(__file__).resolve().parent
DEMO_MESSAGES = [
    "I still haven't received my new card, it's been two weeks",
    "Why was I charged a fee for withdrawing cash?",
    "My top-up didn't go through, what happened?",
    "Can I change my PIN at an ATM?",
    "I don't recognise this payment on my statement",
]
# SFT benchmark.sql scoring: exact match after REPLACE(LOWER(TRIM(x)), ' ', '').
SCORE_SQL = """
SELECT COUNT(*) AS N,
       SUM(IFF(REPLACE(LOWER(TRIM(PREDICTED)), ' ', '') = LOWER(LABEL), 1, 0)) AS CORRECT,
       ROUND(100 * CORRECT / N, 2) AS ACCURACY_PCT,
       SUM(IFF(PREDICTED IS NULL, 1, 0)) AS NULL_REPLIES
FROM ML_DEMO.BANKING77_WSP.WSP_TEST_RESULTS"""


def statements(sql: str):
    """Split the pipeline file on ';' line endings, keeping $$-quoted bodies intact."""
    out, buf, in_body = [], [], False
    for line in sql.splitlines():
        if line.strip().startswith("--") and not in_body:
            continue
        buf.append(line)
        in_body ^= line.count("$$") % 2 == 1
        if not in_body and line.rstrip().endswith(";"):
            stmt = "\n".join(buf).strip().rstrip(";")
            if stmt:
                out.append(stmt)
            buf = []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    s = session()
    for q in statements((HERE / "byom_pipeline.sql").read_text()):
        print(s.sql(q).collect()[0][0])

    if a.demo:
        s.sql("INSERT INTO INCOMING_MESSAGES (MESSAGE_TEXT) SELECT column1 FROM VALUES "
              + ", ".join("(?)" for _ in DEMO_MESSAGES), params=DEMO_MESSAGES).collect()
        s.sql("EXECUTE TASK CLASSIFY_INCOMING_TASK").collect()
        for _ in range(60):  # wait for the one-off task run
            time.sleep(5)
            st = s.sql("""SELECT STATE, ERROR_MESSAGE FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
                TASK_NAME => 'CLASSIFY_INCOMING_TASK')) ORDER BY SCHEDULED_TIME DESC LIMIT 1""").collect()
            if st and st[0]["STATE"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                print("task run:", st[0]["STATE"], st[0]["ERROR_MESSAGE"] or "")
                break
        for r in s.sql("SELECT MESSAGE_TEXT, INTENT FROM CLASSIFIED_MESSAGES ORDER BY MESSAGE_ID").collect():
            print(f"  {r['INTENT']:<38} <- {r['MESSAGE_TEXT']}")

    if a.score:
        lim = f"LIMIT {a.limit}" if a.limit else ""
        t0 = time.time()
        s.sql(f"""CREATE OR REPLACE TABLE WSP_TEST_RESULTS AS
            SELECT TEXT, LABEL, CLASSIFY_BANKING_INTENT(TEXT) AS PREDICTED
            FROM (SELECT * FROM ML_DEMO.BANKING77_WSP.B77_TEST ORDER BY LABEL, TEXT {lim})""").collect()
        print(f"scored in {(time.time() - t0) / 60:.1f} min")
        print(s.sql(SCORE_SQL).collect()[0].as_dict())


if __name__ == "__main__":
    main()
