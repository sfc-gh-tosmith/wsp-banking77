# Hypothesis: a one-shot competitor set (for each label token, the 40 vocab tokens whose unembedding has the highest cosine with it) gives a
# single closed-form pinv write matrix with cross-talk as low as the earlier constraint-expansion version (max ~.37).
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
E = E_full(); ET = E[TOK]
En = E / np.linalg.norm(E, axis=1, keepdims=True)
cos = En @ En[TOK].T                           # vocab x 151
cos[TOK, :] = -np.inf
for k in [20, 40]:
    comp = np.unique(np.argsort(-cos, axis=0)[:k].ravel())
    D = np.linalg.pinv(np.r_[ET, E[comp]])[:, :len(TOK)]
    ED = E @ D; ED[TOK, :] = -np.inf; mx = ED.max(0)
    print(f'top-{k} cosine competitors: {len(comp)} tokens -> cross-talk max {mx.max():.3f} median {np.median(mx):.3f}, >.5: {(mx>.5).sum()} | '
          f'column norm median {np.median(np.linalg.norm(D,axis=0)):.3f} max {np.linalg.norm(D,axis=0).max():.3f}', flush=True)
    np.save(f'work/D_write_cos{k}.npy', D.astype(np.float32))
old = np.load('work/D_write.npy'); ED = E @ old; ED[TOK, :] = -np.inf
print(f'previous D: cross-talk max {ED.max(0).max():.3f} median {np.median(ED.max(0)):.3f}')
