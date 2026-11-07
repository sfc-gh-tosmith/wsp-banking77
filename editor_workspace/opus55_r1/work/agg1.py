# Hypothesis: aggregating 77 class scores to first tokens with a thresholded sum (what gated neurons silu(k(s_c-theta)) implement)
# beats the linear sum and approaches max; class-balanced accuracy (test set is balanced) is higher than plain held-out accuracy.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
seq_fold = np.load('work/p03_seqfold.npy'); val = (seq_fold == 0)[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~(seq_fold == 0)], minlength=77)
w = (ncls.mean() / ncls)[ycls]
FIRST = sorted({l['token_ids'][0] for l in LABS}); F2J = {t: j for j, t in enumerate(FIRST)}
c2f = np.array([F2J[LABS[c]['token_ids'][0]] for c in range(77)]); Mcf = np.zeros((77, len(FIRST))); Mcf[np.arange(77), c2f] = 1
a = tr & (step == 0); b = val & (step == 0)
X = np.load('work/p03_mlpin31_rec.npy').astype(np.float32)
G = (X[a] * w[a, None]).T @ X[a]; B = np.zeros((4096, 77)); np.add.at(B.T, ycls[a], X[a] * w[a, None])
yb = ycls[b]; fb = c2f[yb]
def bal(ok, y): return np.mean([ok[y == c].mean() for c in range(77)])
for mu_rel in [3e-5, 1e-4, 3e-4]:
    S = X[b] @ np.linalg.solve(G + mu_rel * np.trace(G) / 4096 * np.eye(4096), B)
    r = [f'mu_rel={mu_rel}: class acc {(S.argmax(1)==yb).mean():.4f} (balanced {bal(S.argmax(1)==yb, yb):.4f})']
    r.append(f'sum {((S @ Mcf).argmax(1)==fb).mean():.4f}')
    r.append(f'max {(c2f[S.argmax(1)]==fb).mean():.4f}')
    for th in [0.1, 0.2, 0.3, 0.4]:
        r.append(f'relu-sum th{th} {((np.maximum(S - th, 0) @ Mcf + 1e-6 * (S @ Mcf)).argmax(1)==fb).mean():.4f}')
    print(' | '.join(r), flush=True)
    print('   no class above 0.3:', (S.max(1) < 0.3).mean().round(4), ' gold score median', np.median(S[np.arange(len(yb)), yb]).round(3))
