# Hypothesis: with D = pinv(E_label), some non-label tokens (case/space variants) receive boost close to the gold token's (cross-talk ~1),
# so the write must also null the worst competitors; measure max cross-talk per label token and fix by adding competitors to the constraint set.
import sys, gc, numpy as np
sys.path.insert(0, 'work')
from common import *
E = E_full(); ET = E[TOK]
def crosstalk(D):
    ED = E @ D                                # vocab x 151
    ED[TOK, :] = -np.inf                     # exclude label tokens themselves
    mx = ED.max(0); arg = ED.argmax(0)
    return mx, arg
D = np.linalg.pinv(ET)
mx, arg = crosstalk(D)
o = np.argsort(-mx)
print('pinv(E_label): cross-talk max %.3f, >0.8: %d, >0.6: %d tokens; worst:' % (mx.max(), (mx > 0.8).sum(), (mx > 0.6).sum()),
      [(TOK[j], int(arg[j]), round(float(mx[j]), 2)) for j in o[:8]])
# add competitors: for each label token, its top-k non-label boosted tokens, iterate
comp = set()
for it in range(4):
    ED = E @ D; ED[TOK, :] = -np.inf
    top = np.argsort(-ED, axis=0)[:20]          # top-20 competitors per label token
    for j in range(len(TOK)):
        for s in top[:, j]:
            if ED[s, j] > 0.3: comp.add(int(s))
    C = sorted(comp); EA = np.r_[ET, E[C]]
    Dfull = np.linalg.pinv(EA); D = Dfull[:, :len(TOK)]
    mx, arg = crosstalk(D)
    print(f'iter {it}: {len(C)} nulled competitors -> cross-talk max {mx.max():.3f}, >0.5: {(mx>0.5).sum()}, column norm median {np.median(np.linalg.norm(D,axis=0)):.3f} max {np.linalg.norm(D,axis=0).max():.3f}', flush=True)
np.save('work/D_write.npy', D.astype(np.float32)); np.save('work/D_competitors.npy', np.array(C))
