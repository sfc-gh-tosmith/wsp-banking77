# Hypothesis: the relevant quality of a write matrix is cross-talk / self-boost (and label-vs-label off-diagonal); compare previous D vs single-shot sets.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
E = E_full(); ET = E[TOK]
for nm in ['D_write', 'D_write_ss0.3', 'D_write_ss0.2']:
    D = np.load(f'work/{nm}.npy').astype(np.float32)
    S = ET @ D; self_ = np.diag(S); off = S - np.diag(self_)
    ED = E @ D; ED[TOK, :] = -np.inf; mx = ED.max(0)
    r = mx / self_
    print(f'{nm}: self-boost median {np.median(self_):.3f} min {self_.min():.3f} | label off-diag max/self {np.max(np.abs(off)/self_[None,:]):.3f} | '
          f'non-label cross-talk/self max {r.max():.3f} median {np.median(r):.3f} | gain per unit norm {np.median(self_/np.linalg.norm(D,axis=0)):.3f}')
