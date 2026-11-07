# Hypothesis: summing step-0 scores from the linear L31 mlp_in read-out and the L28/L30 mlp_act read-outs (all fit on folds 1-4)
# beats each alone on held-out fold 0; find the mixing weight.
import sys, pickle, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
X = np.load('work/p03_mlpin31_rec.npy').astype(np.float32)
seq_fold = np.load('work/p03_seqfold.npy'); val = (seq_fold == 0)[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~(seq_fold == 0)], minlength=77)
w = (ncls.mean() / ncls)[ycls]
FIRST = sorted({l['token_ids'][0] for l in LABS}); F2J = {t: j for j, t in enumerate(FIRST)}; FA = np.array(FIRST)
fj = np.array([F2J.get(t, -1) for t in tgt])
a = tr & (step == 0); b = val & (step == 0)
G = (X[a] * w[a, None]).T @ X[a]; B = np.zeros((X.shape[1], len(FIRST))); np.add.at(B.T, fj[a], X[a] * w[a, None])
Slin = X[b] @ np.linalg.solve(G + 1e-4 * np.trace(G) / len(G) * np.eye(len(G)), B)
sc = pickle.load(open('work/val_scores_sweep.pkl', 'rb'))
S28 = sc[(28, 's0', 3e-3)]; S30 = sc[(30, 's0', 1e-3)]
y = fj[b]
acc = lambda S: (S.argmax(1) == y).mean()
print('lin %.4f  L28 %.4f  L30 %.4f  L28+L30 %.4f' % (acc(Slin), acc(S28), acc(S30), acc(S28 + S30)))
for wl in [0.25, 0.5, 1.0]:
    print(f'L28+L30 + {wl}*lin: {acc(S28 + S30 + wl * Slin):.4f} | L28 + {wl}*lin: {acc(S28 + wl * Slin):.4f}')
np.save('work/val_s0_scores_lin.npy', Slin)
