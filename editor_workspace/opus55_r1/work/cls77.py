# Hypothesis: at step 0, a 77-way class read-out followed by max-aggregation to first tokens beats the direct 43-way first-token read-out
# (the 'card' target is a union of 10 class clusters) -- for linear L31 mlp_in features and for L30 mlp_act features.
import sys, gc, numpy as np, scipy.linalg as sl
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
seq_fold = np.load('work/p03_seqfold.npy'); val = (seq_fold == 0)[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~(seq_fold == 0)], minlength=77)
w = (ncls.mean() / ncls)[ycls]
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)])
a = tr & (step == 0); b = val & (step == 0)
X = np.load('work/p03_mlpin31_rec.npy').astype(np.float32)
G = (X[a] * w[a, None]).T @ X[a]; B = np.zeros((4096, 77)); np.add.at(B.T, ycls[a], X[a] * w[a, None])
for mu_rel in [1e-4, 1e-3]:
    S = X[b] @ np.linalg.solve(G + mu_rel * np.trace(G) / 4096 * np.eye(4096), B)
    pc = S.argmax(1)
    print(f'L31 mlp_in linear 77-way mu_rel={mu_rel}: class acc {(pc == ycls[b]).mean():.4f} | first-token via max {(first_of[pc] == first_of[ycls[b]]).mean():.4f}', flush=True)
del X, G; gc.collect()
z5 = np.load('acts/p05_practice_L28_30.npz'); A = z5['L30.mlp_act']
ia = np.where(a)[0]; Bc = np.zeros((14336, 77), np.float32)
for s in range(0, len(ia), 2048):
    j = ia[s:s+2048]; np.add.at(Bc.T, ycls[j], A[j].astype(np.float32) * w[j, None].astype(np.float32))
Ab = A[b].astype(np.float32); del A, z5; gc.collect()
G0 = np.load('work/G30_tr_s0.npy'); sc = np.trace(G0) / len(G0)
for mu_rel in [1e-3, 3e-3]:
    M = G0.copy(); M[np.diag_indices(len(M))] += mu_rel * sc
    Xc = sl.cho_solve(sl.cho_factor(M, overwrite_a=True), Bc); del M; gc.collect()
    pc = (Ab @ Xc).argmax(1)
    print(f'L30 mlp_act 77-way mu_rel={mu_rel}: class acc {(pc == ycls[b]).mean():.4f} | first-token via max {(first_of[pc] == first_of[ycls[b]]).mean():.4f}', flush=True)
