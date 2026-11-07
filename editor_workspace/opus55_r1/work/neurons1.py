# Hypothesis: many layer-31 MLP neurons contribute negligibly on both in-task and general text; zeroing the ~300 least important
# changes no greedy token (exact linear check: delta = -act_n * down_n on recorded states).
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
Wd = K.load('model.layers.31.mlp.down_proj.weight'); dn2 = (Wd ** 2).sum(0)
z = np.load('acts/p03_practice_L31.npz'); A = z['L31.mlp_act']
ms_in = np.zeros(A.shape[1])
for s in range(0, len(A), 4096): ms_in += (A[s:s+4096].astype(np.float32) ** 2).sum(0)
ms_in /= len(A)
zg = np.load('acts/p04_general_L31.npz'); Ag = zg['L31.mlp_act']
ms_g = np.zeros(Ag.shape[1]); mx_g = np.zeros(Ag.shape[1])
for s in range(0, len(Ag), 4096):
    a = Ag[s:s+4096].astype(np.float32); ms_g += (a ** 2).sum(0); mx_g = np.maximum(mx_g, np.abs(a).max(0))
ms_g /= len(Ag)
imp = (ms_in + ms_g) * dn2
order = np.argsort(imp)
tot = imp.sum()
print('importance share of least-important 150/300/600 neurons: %.2e %.2e %.2e' % (imp[order[:150]].sum()/tot, imp[order[:300]].sum()/tot, imp[order[:600]].sum()/tot))
print('their max |act| on general:', mx_g[order[:300]].max().round(3), ' median general rms', np.sqrt(ms_g[order[:300]]).max().round(4))
np.save('work/L31_neuron_order.npy', order)
# exact flip check for removing the 300 least important (general reply positions, all; in-task: gold path unchanged?)
sel = order[:300]
rows = [json.loads(l) for l in open('work/general.jsonl')]
R = zg['L31.resid'][::3].astype(np.float32); a = Ag[::3][:, sel].astype(np.float32)
dR = -(a @ Wd[:, sel].T)
b0, _ = argmax_full(R); b1, _ = argmax_full(R + dR)
print('general (every 3rd position): flips after zeroing 300 neurons:', int((b0 != b1).sum()), 'of', len(R), '| max |delta r|', np.abs(dR).max().round(4))
Ri = z['L31.resid'][::6].astype(np.float32); ai = A[::6][:, sel].astype(np.float32)
c0, _ = argmax_full(Ri); c1, _ = argmax_full(Ri - ai @ Wd[:, sel].T)
print('in-task (every 6th): flips:', int((c0 != c1).sum()), 'of', len(Ri))
