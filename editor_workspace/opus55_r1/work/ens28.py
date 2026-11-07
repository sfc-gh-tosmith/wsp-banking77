# Hypothesis: adding layer-30 (or 28) MLP activations to L31 mlp_in raises step-0 first-token accuracy by >=0.5% consistently across 5 folds
# (fold 0 showed +0.6%); if not consistent, a two-stage edit is not worth its risk. Dual-form ridge, step-0 rows only.
import sys, gc, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
seq_fold = np.load('work/p03_seqfold.npy')
i0 = np.where(step == 0)[0]; y = ycls[i0]; f = seq_fold[si[i0]]; gold = tgt[i0]
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = np.array(sorted(set(first_of.tolist())))
M = np.zeros((77, len(FIRST))); M[np.arange(77), [list(FIRST).index(t) for t in first_of]] = 1
srcs = {'in31': ('acts/p10_practice_L31_mlpin.npz', 'L31.mlp_in'),         'act28': ('acts/p05_practice_L28_30.npz', 'L28.mlp_act')}
K = {}
for k, (fn, key) in srcs.items():
    A = np.load(fn)[key][i0].astype(np.float32); A /= np.mean(np.linalg.norm(A, axis=1)); K[k] = A @ A.T; del A; gc.collect()
    print('kernel', k, flush=True)
Y = np.zeros((len(i0), 77)); Y[np.arange(len(i0)), y] = 1
def run(names, lam):
    Kall = sum(K[n] for n in names); accs = []
    for fold in range(5):
        a = np.where(f != fold)[0]; b = np.where(f == fold)[0]
        w = 1 / np.bincount(y[a], minlength=77)[y[a]]; sw = np.sqrt(w / w.mean())
        Kw = sw[:, None] * Kall[np.ix_(a, a)] * sw[None, :]
        al = np.linalg.solve(Kw + lam * np.trace(Kw) / len(a) * np.eye(len(a)), sw[:, None] * Y[a])
        S = Kall[np.ix_(b, a)] @ (sw[:, None] * al)
        accs.append((FIRST[(np.maximum(S - 0.15, 0) @ M).argmax(1)] == gold[b]).mean())
    return np.array(accs)
def scores(name, lam):
    out = np.zeros((len(i0), 77))
    for fold in range(5):
        a = np.where(f != fold)[0]; b = np.where(f == fold)[0]
        w = 1 / np.bincount(y[a], minlength=77)[y[a]]; sw = np.sqrt(w / w.mean())
        Kw = sw[:, None] * K[name][np.ix_(a, a)] * sw[None, :]
        al = np.linalg.solve(Kw + lam * np.trace(Kw) / len(a) * np.eye(len(a)), sw[:, None] * Y[a])
        out[b] = K[name][np.ix_(b, a)] @ (sw[:, None] * al)
    return out
S31 = scores('in31', 3e-4); S28 = scores('act28', 1e-3)
def acc(S): return np.array([(FIRST[(np.maximum(S[f==k] - 0.15, 0) @ M).argmax(1)] == gold[f==k]).mean() for k in range(5)])
base = acc(S31); print('in31 alone', base.round(4), base.mean().round(4)); print('act28 alone', acc(S28).mean().round(4))
for lam in [0.25, 0.5, 0.75, 1.0]:
    r = acc((S31 + lam * S28) / (1 + lam)); print(f'ensemble s31 + {lam}*s28: {r.round(4)} mean {r.mean():.4f} diff {np.mean(r-base):+.4f} sd {np.std(r-base):.4f}', flush=True)
