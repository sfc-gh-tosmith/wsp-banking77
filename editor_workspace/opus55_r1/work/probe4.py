# Hypothesis: within groups of labels sharing the first token, the class is better decodable at step 1 (one more position of
# computation, emitted token identical within group) than at step 0 -> deferring the decision (e.g. a leading space token) could help.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p01_layers')
first_tok = np.array([LABS[c]['token_ids'][0] for c in range(77)])
nseq = si.max() + 1
rng = np.random.default_rng(3); fold = rng.permutation(nseq) % 5
def lda_cv(X, y, seqs, shr=0.1):
    accs = []
    for f in range(5):
        tr = fold[seqs] != f; te = ~tr
        cl = np.unique(y[tr]); mu = X[tr].mean(0); Xc = X[tr] - mu
        C = np.stack([Xc[y[tr] == c].mean(0) for c in cl]); yi = np.searchsorted(cl, y[tr])
        Rm = Xc - C[yi]; S = Rm.T @ Rm / len(Rm); S = (1 - shr) * S + shr * np.trace(S) / len(S) * np.eye(len(S))
        W = np.linalg.solve(S, C.T); b = -0.5 * (C * W.T).sum(1)
        p = cl[((X[te] - mu) @ W + b).argmax(1)]; accs.append((p == y[te]).mean())
    return np.mean(accs)
for L in [26, 31]:
    X = z[f'L{L}.resid'].astype(np.float32); X /= np.sqrt((X ** 2).mean(1, keepdims=True))
    for tok, name in [(first_tok[11], 'card'), (first_tok[56], 'top'), (first_tok[45], 'pending')]:
        grp = np.isin(ycls, np.where(first_tok == tok)[0])
        r0 = grp & (step == 0); r1 = grp & (step == 1)
        print(f'L{L} group {name} ({len(np.unique(ycls[r0]))} cls, n={r0.sum()}): step0 {lda_cv(X[r0], ycls[r0], si[r0]):.3f}  step1 {lda_cv(X[r1], ycls[r1], si[r1]):.3f}', flush=True)
