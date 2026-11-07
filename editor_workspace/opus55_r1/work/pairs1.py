# Hypothesis: re-ranking the coarse class read-out's top-2 with pair-specific ridge discriminators (fit only on the two classes)
# fixes a share of step-0 confusions (as branch-specialised read-outs did at branch points). Held-out fold 0, true L31 mlp_in.
import sys, collections, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy'); ev_ = (seq_fold == 0)[si]; tr = ~ev_
a = np.where(tr & (step == 0))[0]; b = np.where(ev_ & (step == 0))[0]
Xa = X[a].astype(np.float64); Xb = X[b].astype(np.float32); ya = ycls[a]; yb = ycls[b]
w = 1 / np.bincount(ya, minlength=77)[ya]
G = (Xa * w[:, None]).T @ Xa; Y = np.zeros((len(a), 77)); Y[np.arange(len(a)), ya] = 1
Gr = G + 1e-4 * np.trace(G) / 4096 * np.eye(4096)
W = np.linalg.solve(Gr, (Xa * w[:, None]).T @ Y).astype(np.float32)
# cross-fitted coarse scores on the training part (2 halves) to find which pairs get confused
fa = seq_fold[si[a]] % 2
Sa = np.zeros((len(a), 77), np.float32)
for h in [0, 1]:
    t_ = fa != h; Gh = (Xa[t_] * w[t_, None]).T @ Xa[t_]; Gh += 1e-4 * np.trace(Gh) / 4096 * np.eye(4096)
    Sa[~t_] = Xa[~t_].astype(np.float32) @ np.linalg.solve(Gh, (Xa[t_] * w[t_, None]).T @ Y[t_]).astype(np.float32)
top2 = np.argsort(-Sa, 1)[:, :2]
pc = collections.Counter(tuple(sorted(p)) for p, y in zip(top2, ya) if y in p)
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)])
Sb = Xb @ W; tb = np.argsort(-Sb, 1)[:, :2]
print(f'coarse class acc {(tb[:,0]==yb).mean():.4f} | top-2 contains gold {np.mean([y in t for t, y in zip(tb, yb)]):.4f}')
for minc, mu_rel in [(10, 1e-2), (10, 1e-1), (20, 1e-1), (5, 1e-1)]:
    pairs = [p for p, c in pc.items() if c >= minc]
    experts = {}
    for (A, B) in pairs:
        idx = np.where((ya == A) | (ya == B))[0]; t = np.where(ya[idx] == A, 1.0, -1.0); ww = w[idx]
        Gp = (Xa[idx] * ww[:, None]).T @ Xa[idx]; Gp += mu_rel * np.trace(Gp) / 4096 * np.eye(4096)
        experts[(A, B)] = np.linalg.solve(Gp, (Xa[idx] * ww[:, None]).T @ t).astype(np.float32)
    pred = tb[:, 0].copy(); used = 0
    for i, (c1, c2) in enumerate(tb):
        key = tuple(sorted((c1, c2)))
        if key in experts:
            used += 1; A, B = key; pred[i] = A if Xb[i] @ experts[key] > 0 else B
    fo = (first_of[pred] == first_of[yb]).mean()
    print(f'pairs>= {minc} co-occurrences ({len(pairs)} experts, mu_rel {mu_rel}): class acc {(pred==yb).mean():.4f} first-token {fo:.4f} | expert used on {used/len(yb):.3f}', flush=True)
