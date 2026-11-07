# Hypothesis: a shrinkage-LDA class read-out (equal priors) on true L31 mlp_in at step 0 classifies better than ridge-to-one-hot (.910);
# check class acc and relu-sum first-token acc on held-out fold 0.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy'); ev_ = (seq_fold == 0)[si]; tr = ~ev_
a = np.where(tr & (step == 0))[0]; b = np.where(ev_ & (step == 0))[0]
Xa = X[a].astype(np.float64); Xb = X[b].astype(np.float32); ya = ycls[a]; yb = ycls[b]
mu = Xa.mean(0); C = np.stack([Xa[ya == c].mean(0) for c in range(77)])
Rw = Xa - C[ya]; Sw = Rw.T @ Rw / len(Rw)
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = sorted(set(first_of.tolist()))
M = np.zeros((77, len(FIRST))); M[np.arange(77), [FIRST.index(t) for t in first_of]] = 1
w = 1 / np.bincount(ya, minlength=77)[ya]
G = (Xa * w[:, None]).T @ Xa; Y = np.zeros((len(a), 77)); Y[np.arange(len(a)), ya] = 1
Wr = np.linalg.solve(G + 1e-4 * np.trace(G) / 4096 * np.eye(4096), (Xa * w[:, None]).T @ Y).astype(np.float32)
Sr = Xb @ Wr
print(f'ridge: class acc {(Sr.argmax(1)==yb).mean():.4f} | relu-sum first {(np.array(FIRST)[(np.maximum(Sr-0.15,0)@M).argmax(1)]==tgt[b]).mean():.4f}')
for shr in [0.02, 0.05, 0.1, 0.2]:
    S = (1 - shr) * Sw + shr * np.trace(Sw) / 4096 * np.eye(4096)
    W = np.linalg.solve(S, C.T); bb = -0.5 * (C * W.T).sum(1)
    Sl = (Xb @ W.astype(np.float32)) + bb
    p = Sl.argmax(1)
    # softmax posterior -> first-token mass
    Pz = np.exp(Sl - Sl.max(1, keepdims=True)); Pz /= Pz.sum(1, keepdims=True)
    print(f'LDA shr={shr}: class acc {(p==yb).mean():.4f} | posterior-sum first {(np.array(FIRST)[(Pz@M).argmax(1)]==tgt[b]).mean():.4f} '
          f'| ensemble ridge+LDA-posterior class {((Sr + Pz).argmax(1)==yb).mean():.4f}', flush=True)
