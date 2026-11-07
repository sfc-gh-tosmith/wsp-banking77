# Hypothesis: single-shot competitor rule "top-k non-label tokens boosted by plain pinv(E_label), per label token" + ONE pinv gives relative
# cross-talk close to the multi-round D_write (.37) while staying exactly determined (self-boost 1).
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
E = E_full(); ET = E[TOK]
D0 = np.linalg.pinv(ET); ED0 = E @ D0; ED0[TOK, :] = -np.inf
for k in [10, 15, 20, 25]:
    comp = np.unique(np.argsort(-ED0, axis=0)[:k].ravel())
    D = np.linalg.pinv(np.r_[ET, E[comp]])[:, :len(TOK)]
    S = ET @ D; self_ = np.diag(S)
    ED = E @ D; ED[TOK, :] = -np.inf; mx = ED.max(0) / self_
    print(f'top-{k}: {len(comp)} competitors | self-boost min {self_.min():.3f} | cross-talk/self max {mx.max():.3f} median {np.median(mx):.3f} >.5: {(mx>.5).sum()} | col norm median {np.median(np.linalg.norm(D,axis=0)):.3f}', flush=True)
    np.save(f'work/D_write_top{k}.npy', D.astype(np.float32))
