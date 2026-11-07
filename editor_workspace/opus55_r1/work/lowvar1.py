# Hypothesis: the late residual stream has >=151 directions with tiny natural variance (in-task and general) that layer 31's readers
# and the unembedding also read weakly -> a small code written there is both clean (high SNR) and nearly invisible to the model.
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
import wsp_kernel as K
z = np.load('acts/p01_layers.npz'); zg = np.load('acts/p04_general_L31.npz')
X29 = z['L29.resid'].astype(np.float32); X31 = z['L31.resid'].astype(np.float32); Xg = zg['L31.resid'][::2].astype(np.float32)
def cov(X): return (X.T @ X) / len(X)
C = (cov(X29) + cov(X31) + cov(Xg)) / 3          # uncentred second moment (what a read-out sees)
ev, U = np.linalg.eigh(C.astype(np.float64))
avg = np.trace(C) / 4096
print('second-moment eigs / avg-dir: min %.2e  151st %.2e  400th %.2e  median %.2e  top %.1f' % (ev[0]/avg, ev[150]/avg, ev[399]/avg, np.median(ev)/avg, ev[-1]/avg))
Ub = U[:, :151].astype(np.float32)
for nm, X in [('L29 in-task', X29), ('L31 in-task', X31), ('L31 general', Xg)]:
    pr = X @ Ub; print(f'  {nm}: rms along bottom-151 dirs {np.sqrt((pr**2).mean()):.4f} vs rms per coordinate {np.sqrt((X**2).mean()):.4f}')
g1 = K.load('model.layers.31.input_layernorm.weight'); g2 = K.load('model.layers.31.post_attention_layernorm.weight'); gf = K.load('model.norm.weight')
def gain(W, g):
    Wg = W * g[None, :]; num = np.linalg.norm(Wg @ Ub, axis=0) ** 2; den = (Wg ** 2).sum() / 4096
    return num / den
for nm, W, g in [('q', K.load('model.layers.31.self_attn.q_proj.weight'), g1), ('k', K.load('model.layers.31.self_attn.k_proj.weight'), g1),
                 ('v', K.load('model.layers.31.self_attn.v_proj.weight'), g1), ('gate', K.load('model.layers.31.mlp.gate_proj.weight'), g2),
                 ('up', K.load('model.layers.31.mlp.up_proj.weight'), g2)]:
    gn = gain(W, g); print(f'  layer31 {nm:4s} gain on bottom-151: mean {gn.mean():.3f} max {gn.max():.3f}'); del W; gc.collect()
E = K.load('lm_head.weight'); num = np.zeros(151); den = 0.0
for s in range(0, E.shape[0], 16384):
    c = E[s:s+16384] * gf[None, :]; num += (np.asarray(c @ Ub) ** 2).sum(0); den += (c ** 2).sum() / 4096
print(f'  unembed gain on bottom-151: mean {(num/den).mean():.3f} max {(num/den).max():.3f}')
np.save('work/lowvar_U.npy', U[:, :600].astype(np.float32)); np.save('work/lowvar_ev.npy', ev)
