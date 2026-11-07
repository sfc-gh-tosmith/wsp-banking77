# Hypothesis: concatenating L31 mlp_in with L30/L31 mlp_act (14336-d nonlinear features) at step 0 improves held-out class accuracy over
# mlp_in alone (.911); dual-form (kernel) ridge on ~7.4k training rows keeps it cheap. Each block is scaled to unit mean row norm.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
seq_fold = np.load('work/p03_seqfold.npy'); ev_ = (seq_fold == 0)[si]; tr = ~ev_
a = np.where(tr & (step == 0))[0]; b = np.where(ev_ & (step == 0))[0]
ya = ycls[a]; yb = ycls[b]
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = sorted(set(first_of.tolist()))
M = np.zeros((77, len(FIRST))); M[np.arange(77), [FIRST.index(t) for t in first_of]] = 1
srcs = {'in31': ('acts/p10_practice_L31_mlpin.npz', 'L31.mlp_in'), 'act30': ('acts/p05_practice_L28_30.npz', 'L30.mlp_act'),
        'act28': ('acts/p05_practice_L28_30.npz', 'L28.mlp_act'), 'act31': ('acts/p03_practice_L31.npz', 'L31.mlp_act')}
K_tr = {}; K_te = {}
import gc
for k, (f, key) in srcs.items():
    B = np.load(f)[key]
    A = B[a].astype(np.float32); Bt = B[b].astype(np.float32); del B; gc.collect()
    s = np.mean(np.linalg.norm(A, axis=1)); A /= s; Bt /= s
    K_tr[k] = A @ A.T; K_te[k] = Bt @ A.T; del A, Bt; gc.collect()
    print('kernel', k, 'done', flush=True)
w = 1 / np.bincount(ya, minlength=77)[ya]; sw = np.sqrt(w / w.mean())
Y = np.zeros((len(a), 77)); Y[np.arange(len(a)), ya] = 1
def evalK(names, lam):
    Kt = sum(K_tr[n] for n in names); Ke = sum(K_te[n] for n in names)
    Kw = sw[:, None] * Kt * sw[None, :]
    alpha = np.linalg.solve(Kw + lam * np.trace(Kw) / len(Kw) * np.eye(len(Kw)), sw[:, None] * Y)
    S = Ke @ (sw[:, None] * alpha)
    f = (np.array(FIRST)[(np.maximum(S - 0.15, 0) @ M).argmax(1)] == tgt[b]).mean()
    return (S.argmax(1) == yb).mean(), f
for names in [['in31'], ['act31'], ['act30'], ['in31', 'act30'], ['in31', 'act31'], ['in31', 'act30', 'act31'], ['in31', 'act28', 'act30', 'act31']]:
    for lam in [1e-3, 1e-2, 1e-1]:
        r = evalK(names, lam); print(f'{"+".join(names)} lam={lam}: class acc {r[0]:.4f} relu-sum first {r[1]:.4f}', flush=True)
