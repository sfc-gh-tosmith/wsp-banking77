# Hypothesis: the in-task logit gap (best non-label token minus gold label token, unnormalised E.r units) is bounded (~tens of units), so a
# write of alpha * act (act ~0.58 typical at step 0, gate*readout at later steps) can dominate; derive alpha for 99.9% coverage.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
seq_fold = np.load('work/p03_seqfold.npy'); val = (seq_fold == 0)[si]
idx = np.where(val)[0]
R = z['L31.resid'][idx].astype(np.float32)
E = E_full(); tokarr = np.array(TOK)
gaps = np.zeros(len(idx)); lab_gap = np.zeros(len(idx)); rms = np.sqrt((R ** 2).mean(1))
for s in range(0, len(idx), 1000):
    L = R[s:s+1000] @ E.T
    lg = L[np.arange(len(L)), tgt[idx[s:s+1000]]]
    Ll = L[:, tokarr].copy(); L[:, tokarr] = -np.inf
    gaps[s:s+1000] = L.max(1) - lg
    Ll[np.arange(len(L)), [T2J[t] for t in tgt[idx[s:s+1000]]]] = -np.inf
    lab_gap[s:s+1000] = Ll.max(1) - lg
st = step[idx]
for k, nm in [(0, 'step0'), (1, 'step1'), (2, 'step2+')]:
    sel = st == k if k < 2 else st >= 2
    print(f'{nm}: gap to best non-label token (E.r units) median {np.median(gaps[sel]):.1f} p99 {np.percentile(gaps[sel],99):.1f} max {gaps[sel].max():.1f} '
          f'| gap to best other label token median {np.median(lab_gap[sel]):.1f} p99 {np.percentile(lab_gap[sel],99):.1f} | rms {np.median(rms[sel]):.3f} '
          f'(1 E.r unit ~ {1/np.median(rms[sel]):.2f} nats)')
np.save('work/val_gaps.npy', np.c_[gaps, lab_gap, st])
