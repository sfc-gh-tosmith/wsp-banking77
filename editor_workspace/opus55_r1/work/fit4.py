# Hypothesis: the step-0 decision (the hard one) improves if the read-out is fit with step-0 records up-weighted or step-0-only;
# measure intrinsic whole-label accuracy on held-out practice for layer-31 mlp_act features.
import sys, gc, numpy as np, scipy.linalg as sl
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~val_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls].astype(np.float32)
gtj = np.array([T2J[t] for t in tgt])
A = z['L31.mlp_act']; d = A.shape[1]
G0 = np.zeros((d, d), np.float32); B0 = np.zeros((d, len(TOK)), np.float32)
idx = np.where(tr & (step == 0))[0]
for s in range(0, len(idx), 2048):
    j = idx[s:s + 2048]; a = A[j].astype(np.float32); aw = a * w[j][:, None]; G0 += aw.T @ a; np.add.at(B0.T, gtj[j], aw)
Av = A[val].astype(np.float32); del A; gc.collect()
G = np.load('work/G_tr.npy'); B = np.load('work/B_tr.npy')
stepv = step[val]; siv = si[val]; gv = gtj[val]
def report(X, tag):
    S = Av @ X; ok = S.argmax(1) == gv; sq = np.ones(nseq, bool); sq[siv[~ok]] = False
    print(f'{tag}: s0 {ok[stepv==0].mean():.4f} s1 {ok[stepv==1].mean():.4f} s2+ {ok[stepv>=2].mean():.4f} SEQ {sq[val_seq].mean():.4f}', flush=True)
for up in [1, 4]:
    M = G + (up - 1) * G0; M[np.diag_indices(d)] += 0.3 * up
    X = sl.cho_solve(sl.cho_factor(M, overwrite_a=True), B + (up - 1) * B0); del M; gc.collect()
    report(X, f'step0 weight x{up}')
for mu in [0.1, 0.3, 1.0]:
    M = G0.copy(); M[np.diag_indices(d)] += mu
    X0 = sl.cho_solve(sl.cho_factor(M, overwrite_a=True), B0); del M; gc.collect()
    S = Av[stepv == 0] @ X0; ok0 = S.argmax(1) == gv[stepv == 0]
    print(f'step0-only read-out mu={mu}: s0 {ok0.mean():.4f}', flush=True)
