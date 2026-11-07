# Hypothesis: with step-specialised fits, a LINEAR read-out of layer-31 mlp_in (4096-d, what a gated neuron's up_proj sees)
# is near-perfect for steps>=1 (next label token) but weaker than mlp_act read-outs for the step-0 decision.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
X = np.load('work/p03_mlpin31_rec.npy').astype(np.float32)
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~val_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls]
gtj = np.array([T2J[t] for t in tgt])
for name, sel in [('step0', step == 0), ('step>=1', step >= 1)]:
    a, b = tr & sel, val & sel
    G = (X[a] * w[a, None]).T @ X[a]; B = np.zeros((X.shape[1], len(TOK))); np.add.at(B.T, gtj[a], X[a] * w[a, None])
    sc = np.trace(G) / len(G)
    for mu_rel in [1e-4, 1e-3, 1e-2]:
        Wt = np.linalg.solve(G + mu_rel * sc * np.eye(len(G)), B)
        ok = (X[b] @ Wt).argmax(1) == gtj[b]
        if name == 'step>=1':
            sq = np.ones(nseq, bool); sq[si[b][~ok]] = False
            print(f'{name} mu_rel={mu_rel}: per-step acc {ok.mean():.4f} | labels with all later steps right {sq[val_seq].mean():.4f}', flush=True)
        else:
            print(f'{name} mu_rel={mu_rel}: acc {ok.mean():.4f}', flush=True)
