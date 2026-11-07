# Hypothesis: adding a direct 43-way first-token-group read-out g_f (ridge) to the thresholded class mass improves step-0 first-token
# accuracy over class mass alone (.925). Held-out fold 0, true L31 mlp_in.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy'); ev_ = (seq_fold == 0)[si]; tr = ~ev_
a = np.where(tr & (step == 0))[0]; b = np.where(ev_ & (step == 0))[0]
Xa = X[a].astype(np.float64); Xb = X[b].astype(np.float32); ya = ycls[a]
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = sorted(set(first_of.tolist()))
fi = np.array([FIRST.index(t) for t in first_of]); M = np.zeros((77, len(FIRST))); M[np.arange(77), fi] = 1
w = 1 / np.bincount(ya, minlength=77)[ya]
G = (Xa * w[:, None]).T @ Xa; G += 1e-4 * np.trace(G) / 4096 * np.eye(4096)
Yc = np.zeros((len(a), 77)); Yc[np.arange(len(a)), ya] = 1
Yg = np.zeros((len(a), len(FIRST))); Yg[np.arange(len(a)), fi[ya]] = 1
Wc = np.linalg.solve(G, (Xa * w[:, None]).T @ Yc).astype(np.float32)
Wgp = np.linalg.solve(G, (Xa * w[:, None]).T @ Yg).astype(np.float32)
S = Xb @ Wc; Gs = Xb @ Wgp; gold = tgt[b]
for lam in [0.0, 0.25, 0.5, 1.0]:
    for th in [0.15, 0.25]:
        sc = np.maximum(S - th, 0) @ M + lam * np.maximum(Gs - th, 0)
        print(f'lam={lam} th={th}: first-token acc {(np.array(FIRST)[sc.argmax(1)]==gold).mean():.4f}')
print(f'group read-out alone: {(np.array(FIRST)[Gs.argmax(1)]==gold).mean():.4f}')
