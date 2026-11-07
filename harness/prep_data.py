"""Build Banking77 splits and load them into ML_DEMO.BANKING77_WSP.

B77_TRAIN (10,003) / B77_TEST (3,080): PolyAI standard split, same as the SFT benchmark.
B77_DEV: 10 per intent carved from train (editor's self-check set).
B77_TRAIN_PRACTICE: train minus dev (editor's practice set).
B77_PROBE_V2: 2 per intent from test (stand-in for the SFT benchmark's B77_PROBE, whose exact rows are not public).
"""
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

from common import DATA, LABELS, sf_connect

SEED = 42


def load(split):
    df = pd.read_parquet(DATA / f"{split}.parquet")
    df["LABEL"] = df["label"].map(lambda i: LABELS[i])
    return df.rename(columns={"text": "TEXT"})[["TEXT", "LABEL"]].reset_index(drop=True)


def main():
    train, test = load("train"), load("test")
    dev = train.groupby("LABEL", group_keys=False).sample(n=10, random_state=SEED)
    practice = train.drop(index=dev.index)
    probe = test.groupby("LABEL", group_keys=False).sample(n=2, random_state=SEED)
    splits = {"B77_TRAIN": train, "B77_TEST": test, "B77_DEV": dev,
              "B77_TRAIN_PRACTICE": practice, "B77_PROBE_V2": probe}
    for name, df in splits.items():
        print(f"{name:20s} rows={len(df):6d} intents={df.LABEL.nunique()}")
    assert len(train) == 10003 and len(test) == 3080 and len(dev) == 770 and len(probe) == 154

    # Local copies for the editor workspace (practice/dev only; test stays in Snowflake for editor runs).
    practice.to_parquet(DATA / "practice.parquet", index=False)
    dev.to_parquet(DATA / "dev.parquet", index=False)

    con = sf_connect()
    try:
        for name, df in splits.items():
            ok, _, n, _ = write_pandas(con, df.reset_index(drop=True), name, auto_create_table=True, overwrite=True)
            print(f"loaded {name}: {ok} rows={n}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
