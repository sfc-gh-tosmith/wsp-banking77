# Hypothesis: the residual stream after layer 30 has a subspace of >=151 directions that layer 31 (q/k/v, gate/up) and the
# unembedding barely read ("dark" directions), usable as a private code channel; quantify reader gain vs typical directions.
import sys, gc, numpy as np
sys.path.insert(0, 'work')
import wsp_kernel as K
g1 = K.load('model.layers.31.input_layernorm.weight'); g2 = K.load('model.layers.31.post_attention_layernorm.weight')
gf = K.load('model.norm.weight')
d = 4096
blocks = {}
def add(name, W, g):
    Wg = W * g[None, :]
    Gm = (Wg.T @ Wg).astype(np.float64)
    blocks[name] = Gm / np.trace(Gm) * d      # normalised so a random unit direction has gain 1
for nm in ['q_proj', 'k_proj', 'v_proj']:
    add(nm, K.load(f'model.layers.31.self_attn.{nm}.weight'), g1); gc.collect()
for nm in ['gate_proj', 'up_proj']:
    add(nm, K.load(f'model.layers.31.mlp.{nm}.weight'), g2); gc.collect()
E = K.load('lm_head.weight'); Ge = np.zeros((d, d))
for s in range(0, E.shape[0], 16384):
    c = E[s:s + 16384] * gf[None, :]; Ge += c.T @ c
del E; gc.collect()
blocks['unembed'] = Ge / np.trace(Ge) * d
M = sum(blocks.values())
ev, U = np.linalg.eigh(M)
print('combined gain eigenvalues: min %.4f  p1(41st) %.4f  151st %.4f  p10 %.4f  median %.3f  max %.1f' % (ev[0], ev[40], ev[150], ev[409], np.median(ev), ev[-1]))
Ud = U[:, :151]
for k, Gm in blocks.items():
    gains = np.einsum('di,de,ei->i', Ud, Gm, Ud)
    print(f'  {k:10s} gain on 151 darkest dirs: max {gains.max():.4f} mean {gains.mean():.4f}  (random dir = 1)')
np.save('work/dark_U.npy', U[:, :400].astype(np.float32)); np.save('work/dark_ev.npy', ev)
# natural variance of residual (layer 29 / 31 proxies) along dark dirs vs average direction
z = np.load('acts/p01_layers.npz')
for L in [29, 31]:
    X = z[f'L{L}.resid'][:6000].astype(np.float32)
    tot = (X ** 2).mean(0).sum() / d
    pr = (X @ Ud)
    print(f'L{L}: mean-square per dark dir {np.mean(pr**2):.4g} (max dir {np.max(np.mean(pr**2,0)):.4g}) vs avg dir {tot:.4g}; std per dark dir {np.mean(pr.std(0)):.4g}')
