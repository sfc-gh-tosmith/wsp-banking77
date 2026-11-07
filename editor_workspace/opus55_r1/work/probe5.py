# Hypothesis: (a) in-task reply positions vs general-chat positions are linearly separable at layer 31 with a wide margin (task gate);
# (b) a ridge read-out on the 4096-d residual classifies nearly as well as the 14336-d mlp_act read-out (proxy using final resid).
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]; tr = ~val
X = z['L31.resid'].astype(np.float32); X /= np.sqrt((X ** 2).mean(1, keepdims=True))
gtj = np.array([T2J[t] for t in tgt])
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~val_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls]
# (b) token read-out on normalised resid
Xa = np.c_[X, np.ones(len(X))]
G = (Xa[tr] * w[tr, None]).T @ Xa[tr]; B = np.zeros((Xa.shape[1], len(TOK)))
np.add.at(B.T, gtj[tr], Xa[tr] * w[tr, None])
for mu in [1e-1, 1, 10, 100]:
    Wt = np.linalg.solve(G + mu * np.eye(len(G)), B)
    ok = (Xa[val] @ Wt).argmax(1) == gtj[val]
    sq = np.ones(nseq, bool); sq[si[val][~ok]] = False
    print(f'resid read-out mu={mu}: s0 {ok[step[val]==0].mean():.3f} s1 {ok[step[val]==1].mean():.3f} SEQ {sq[val_seq].mean():.4f}', flush=True)
# (a) task gate
zg = np.load('acts/p04_general_L31.npz'); sig = zg['seq_index']
rows = [json.loads(l) for l in open('work/general.jsonl')]
fitg = np.array([rows[s]['split'] == 'fit' for s in sig]); kind = np.array([rows[s]['kind'] for s in sig])
Xg = zg['L31.resid'].astype(np.float32); Xg /= np.sqrt((Xg ** 2).mean(1, keepdims=True))
P = np.r_[X[tr], Xg[fitg]]; t = np.r_[np.ones(tr.sum()), -np.ones(fitg.sum())]
sw = np.r_[np.full(tr.sum(), 0.5 / tr.sum()), np.full(fitg.sum(), 0.5 / fitg.sum())]
Pa = np.c_[P, np.ones(len(P))]
for mu in [1e-4, 1e-3, 1e-2]:
    g = np.linalg.solve((Pa * sw[:, None]).T @ Pa + mu * np.eye(Pa.shape[1]), (Pa * sw[:, None]).T @ t)
    zi = np.c_[X[val], np.ones(val.sum())] @ g; zc = np.c_[Xg[~fitg], np.ones((~fitg).sum())] @ g
    msg = ' '.join(f'{k}:max {zc[kind[~fitg]==k].max():.2f}' for k in np.unique(kind))
    print(f'gate mu={mu}: in-task val min {zi.min():.2f} p1 {np.percentile(zi,1):.2f} med {np.median(zi):.2f} | general check max {zc.max():.2f} p99 {np.percentile(zc,99):.2f} | {msg}', flush=True)
