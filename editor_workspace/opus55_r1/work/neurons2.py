# Hypothesis: ranking by worst-case contribution (max|act| * |down col|) finds neurons whose removal flips ~no greedy token.
import sys, json, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
Wd = K.load('model.layers.31.mlp.down_proj.weight'); dn = np.sqrt((Wd ** 2).sum(0))
z = np.load('acts/p03_practice_L31.npz'); A = z['L31.mlp_act']; zg = np.load('acts/p04_general_L31.npz'); Ag = zg['L31.mlp_act']
mx = np.zeros(A.shape[1])
for X in (A, Ag):
    for s in range(0, len(X), 4096): mx = np.maximum(mx, np.abs(X[s:s+4096].astype(np.float32)).max(0))
wc = mx * dn; order = np.argsort(wc); np.save('work/L31_neuron_order_wc.npy', order)
print('worst-case |contrib| of 151st/300th neuron: %.4f %.4f ; median neuron %.3f' % (wc[order[150]], wc[order[299]], np.median(wc)))
sel = order[:300]
R = zg['L31.resid'][::3].astype(np.float32); a = Ag[::3][:, sel].astype(np.float32)
b0, _ = argmax_full(R); b1, _ = argmax_full(R - a @ Wd[:, sel].T)
Ri = z['L31.resid'][::6].astype(np.float32); ai = A[::6][:, sel].astype(np.float32)
c0, _ = argmax_full(Ri); c1, _ = argmax_full(Ri - ai @ Wd[:, sel].T)
print('zero 300 lowest worst-case: general flips', int((b0 != b1).sum()), '/', len(R), '| in-task flips', int((c0 != c1).sum()), '/', len(Ri))
