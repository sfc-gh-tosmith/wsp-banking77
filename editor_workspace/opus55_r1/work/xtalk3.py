# Hypothesis: a single-shot rule (competitors = non-label tokens that the plain pinv(E_label) write boosts above t) followed by ONE closed-form
# pinv gives cross-talk close to the multi-round version (.37), avoiding any repeated refinement.
import sys, numpy as np
sys.path.insert(0, 'work')
from common import *
E = E_full(); ET = E[TOK]
D0 = np.linalg.pinv(ET); ED0 = E @ D0; ED0[TOK, :] = -np.inf
for t in [0.3, 0.2, 0.15]:
    comp = np.where((ED0 > t).any(1))[0]
    D = np.linalg.pinv(np.r_[ET, E[comp]])[:, :len(TOK)]
    ED = E @ D; ED[TOK, :] = -np.inf; mx = ED.max(0)
    print(f'single-shot t={t}: {len(comp)} competitors -> cross-talk max {mx.max():.3f} median {np.median(mx):.3f} >.5: {(mx>.5).sum()} | col norm median {np.median(np.linalg.norm(D,axis=0)):.3f}', flush=True)
    np.save(f'work/D_write_ss{t}.npy', D.astype(np.float32))
