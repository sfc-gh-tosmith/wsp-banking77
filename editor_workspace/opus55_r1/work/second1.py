# Hypothesis: a closed-form learned second layer V (ridge) over gated nonlinear features relu(s_c - t) (s_c = cross-fitted class scores,
# several thresholds t) maps to first tokens better than the fixed relu-sum aggregation (.925). Held-out fold 0, true L31 mlp_in.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy'); ev_ = (seq_fold == 0)[si]; tr = ~ev_
a = np.where(tr & (step == 0))[0]; b = np.where(ev_ & (step == 0))[0]
Xa = X[a].astype(np.float64); Xb = X[b].astype(np.float32); ya = ycls[a]; yb = ycls[b]
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = sorted(set(first_of.tolist()))
fi = np.array([FIRST.index(t) for t in first_of]); M = np.zeros((77, len(FIRST))); M[np.arange(77), fi] = 1
w = 1 / np.bincount(ya, minlength=77)[ya]
Y = np.zeros((len(a), 77)); Y[np.arange(len(a)), ya] = 1
def fitW(rows):
    G = (Xa[rows] * w[rows, None]).T @ Xa[rows]; G += 1e-4 * np.trace(G) / 4096 * np.eye(4096)
    return np.linalg.solve(G, (Xa[rows] * w[rows, None]).T @ Y[rows]).astype(np.float32)
W = fitW(np.arange(len(a))); Sb = Xb @ W
fa = seq_fold[si[a]]                                   # folds 1..4
Sa = np.zeros((len(a), 77), np.float32)
for f in np.unique(fa):
    Sa[fa == f] = Xa[fa == f].astype(np.float32) @ fitW(np.where(fa != f)[0])
print(f'cross-fitted train class acc {(Sa.argmax(1)==ya).mean():.4f} | held-out class acc {(Sb.argmax(1)==yb).mean():.4f}')
gold_b = fi[yb]; gold_a = fi[ya]
print(f'fixed relu-sum (t=.15) held-out first acc {((np.maximum(Sb-.15,0)@M).argmax(1)==gold_b).mean():.4f}')
def feats(S, ts): return np.concatenate([np.maximum(S - t, 0) for t in ts], 1)
for ts in [(0.15,), (0.0, 0.15, 0.3), (0.0, 0.15, 0.3, 0.5)]:
    Fa = feats(Sa, ts); Fb = feats(Sb, ts)
    for tgt_kind in ['first', 'class']:
        T = np.zeros((len(a), len(FIRST) if tgt_kind == 'first' else 77)); T[np.arange(len(a)), gold_a if tgt_kind == 'first' else ya] = 1
        for lam in [1e-2, 1e-1, 1.0]:
            G = (Fa * w[:, None]).T @ Fa; G += lam * np.trace(G) / len(G) * np.eye(len(G))
            V = np.linalg.solve(G, (Fa * w[:, None]).T @ T)
            O = Fb @ V
            if tgt_kind == 'class': O = np.maximum(O - 0.15, 0) @ M
            print(f'ts={ts} target={tgt_kind} lam={lam}: held-out first acc {(O.argmax(1)==gold_b).mean():.4f}', flush=True)
