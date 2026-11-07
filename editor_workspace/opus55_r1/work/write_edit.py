# Hypothesis: writing the 188 designed neurons into the 188 layer-31 MLP neurons with the least output energy on general text
# (mean act^2 * |down col|^2 over genall positions) implements the classifier with minimal collateral change. Saves the 3 edits.
import sys, json, numpy as np
sys.path.insert(0, 'work')
import wsp_kernel as K
F = np.load('work/neurons_final.npz')
Wg, Wu, Wdn = F['Wg'], F['Wu'], F['Wdn']; N = Wg.shape[0]
zg = np.load('acts/p09_genall_L31.npz'); A = zg['L31.mlp_act']
ms = np.zeros(A.shape[1])
for s in range(0, len(A), 4096): ms += (A[s:s+4096].astype(np.float32) ** 2).sum(0)
ms /= len(A)
base = 'model.layers.31.mlp.'
G = K.load(base + 'gate_proj.weight'); U = K.load(base + 'up_proj.weight'); Dn = K.load(base + 'down_proj.weight')
imp = ms * (Dn ** 2).sum(0)
sel = np.sort(np.argsort(imp)[:N])
print('repurposed neurons: %d, share of general output energy %.2e' % (N, imp[sel].sum() / imp.sum()))
G[sel] = Wg; U[sel] = Wu; Dn[:, sel] = Wdn
K.save_edit(base + 'gate_proj.weight', G); K.save_edit(base + 'up_proj.weight', U); K.save_edit(base + 'down_proj.weight', Dn)
np.save('work/repurposed_neurons.npy', sel)
print(K.list_edits()); print(K.changed_fraction()['fraction'])
