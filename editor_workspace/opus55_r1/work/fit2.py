# Hypothesis: the read-out's intrinsic accuracy (argmax of label-token scores, ignoring original logits) is limited by ridge strength;
# sweep smaller mu and up-weighting of step-0 to find the classifier ceiling at layer 31.
import sys, gc, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]
A = z['L31.mlp_act']; Av = A[val].astype(np.float32); del A; gc.collect()
evals = np.load('work/G_tr_evals.npy'); V = np.load('work/G_tr_V.npy'); B = np.load('work/B_tr.npy')
VB = V.T @ B; AvV = Av @ V; del Av, V; gc.collect()
gtj = np.array([T2J[t] for t in tgt[val]]); stepv = step[val]; siv = si[val]
print('eval percentiles', np.percentile(evals, [1, 10, 50, 90, 99, 99.9]).round(3))
for mu in [0.03, 0.1, 0.3, 1, 3, 10]:
    S = AvV @ (VB / (evals + mu)[:, None])
    ok = S.argmax(1) == gtj
    seq_ok = np.ones(nseq, bool); seq_ok[siv[~ok]] = False
    print(f'mu={mu}: intrinsic ' + ' '.join(f's{k}:{ok[stepv==k].mean():.3f}' for k in range(4)) + f' | SEQ {seq_ok[val_seq].mean():.4f}'
          + f' | gold score s0 median {np.median(S[stepv==0, :][np.arange((stepv==0).sum()), gtj[stepv==0]]):.2f}', flush=True)
