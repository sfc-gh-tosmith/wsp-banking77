"""Phase 1 calibration: prefix-cache exactness, label tokenization, M4 throughput."""
import json
import time

import mlx.core as mx
import numpy as np
import pandas as pd

from common import DATA, LABELS, ROOT
from runner import Runner


def main():
    t0 = time.time()
    r = Runner()
    print(f"load {time.time()-t0:.1f}s  prefix_tokens={r.P}  peak_mem={mx.get_peak_memory()/1e9:.2f}GB")
    practice = pd.read_parquet(DATA / "practice.parquet")
    texts = practice.TEXT.tolist()

    # 1) every prompt starts with the shared prefix
    sufs = [r.suffix_ids(t) for t in texts]
    slen = [len(s) for s in sufs]
    print(f"suffix tokens min/mean/max: {min(slen)}/{np.mean(slen):.1f}/{max(slen)}")

    # 2) exactness: prefix-cached, right-padded capture vs stock full-sequence forward
    idx = [0, 1, 2, int(np.argmax(slen))]
    seqs = [sufs[i] for i in idx]
    cap = r.capture(seqs, [[len(s) - 1] for s in seqs], {31: {"resid"}}, topk=1)
    for j, i in enumerate(idx):
        stock = r.model(mx.array([r.prefix + sufs[i]]))[0, -1].astype(mx.float32)
        stock_top = int(mx.argmax(stock))
        lsm = stock - mx.logsumexp(stock)
        print(f"  seq {i}: stock top={stock_top} cached top={int(cap['topk_ids'][j,0])} "
              f"|dlogp|={abs(float(lsm[stock_top]) - float(cap['topk_logprob'][j,0])):.4f}")

    # 3) generation throughput, 512 prompts
    t = time.time()
    outs = r.generate(texts[:512], max_new_tokens=24)
    gt = time.time() - t
    print(f"generate 512: {gt:.1f}s ({512/gt:.2f}/s)  sample: {outs[:3]}")

    # 4) capture throughput: full practice set, last-suffix-token states at 4 layers
    want = {l: {"resid", "mlp_act"} for l in (15, 23, 28, 31)}
    t = time.time()
    cap = r.capture(sufs, [[len(s) - 1] for s in sufs], want)
    ct = time.time() - t
    ntok = sum(slen)
    print(f"capture ALL practice ({len(sufs)} prompts, {ntok} suffix tok): {ct/60:.1f} min ({ntok/ct:.0f} tok/s) "
          f"peak_mem={mx.get_peak_memory()/1e9:.2f}GB")
    np.savez_compressed(ROOT / "runs" / "calib_capture_L31.npz", resid=cap["L31.resid"], seq=cap["seq_index"])
    (ROOT / "runs" / "calibration.json").write_text(json.dumps({
        "prefix_tokens": r.P, "suffix_tokens_mean": float(np.mean(slen)),
        "gen_prompts_per_s_512": 512 / gt, "capture_full_practice_min": ct / 60, "capture_tok_per_s": ntok / ct,
        "peak_mem_gb": mx.get_peak_memory() / 1e9}, indent=1))


if __name__ == "__main__":
    main()
