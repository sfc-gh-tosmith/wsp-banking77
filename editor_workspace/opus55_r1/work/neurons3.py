# Hypothesis: the greedy-flip cost of freeing k layer-31 neurons grows with k; freeing ~64 (ranked by mean-square output) costs
# far less than 300, and least-squares compensation (re-fit remaining down columns to absorb removed outputs) cuts it further.
import sys, json, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
Wd = K.load('model.layers.31.mlp.down_proj.weight')
order = np.load('work/L31_neuron_order.npy')       # mean-square-importance ranking
z = np.load('acts/p03_practice_L31.npz'); zg = np.load('acts/p04_general_L31.npz')
R = zg['L31.resid'][1::3].astype(np.float32); Ag = zg['L31.mlp_act'][1::3]
Ri = z['L31.resid'][3::6].astype(np.float32); Ai = z['L31.mlp_act'][3::6]
b0, _ = argmax_full(R); c0, _ = argmax_full(Ri)
for k in [32, 64, 100, 151]:
    sel = order[:k]
    dR = -(Ag[:, sel].astype(np.float32) @ Wd[:, sel].T); dRi = -(Ai[:, sel].astype(np.float32) @ Wd[:, sel].T)
    b1, _ = argmax_full(R + dR); c1, _ = argmax_full(Ri + dRi)
    print(f'free {k:3d} neurons: general flips {int((b0!=b1).sum())}/{len(R)}  in-task flips {int((c0!=c1).sum())}/{len(Ri)}  | median |dr| {np.median(np.linalg.norm(dR,axis=1)):.4f} vs |r| {np.median(np.linalg.norm(R,axis=1)):.2f}', flush=True)
