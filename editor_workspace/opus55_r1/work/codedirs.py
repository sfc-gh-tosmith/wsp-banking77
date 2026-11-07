# Hypothesis: there are >=194 residual directions with BOTH low downstream read gain (layer-31 q/k/v/gate/up + unembed) and low
# centred natural variance; writing the stage-1 code there (scale kappa) leaves general greedy tokens unchanged via the direct path.
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
d = 4096
g1 = K.load('model.layers.31.input_layernorm.weight'); g2 = K.load('model.layers.31.post_attention_layernorm.weight'); gf = K.load('model.norm.weight')
Gain = np.zeros((d, d))
def addg(W, g):
    global Gain
    Wg = W * g[None, :]; Gm = Wg.T @ Wg; Gain += Gm / np.trace(Gm) * d
for nm in ['q_proj', 'k_proj', 'v_proj']: addg(K.load(f'model.layers.31.self_attn.{nm}.weight'), g1); gc.collect()
for nm in ['gate_proj', 'up_proj']: addg(K.load(f'model.layers.31.mlp.{nm}.weight'), g2); gc.collect()
Ge = np.zeros((d, d)); E = K.load('lm_head.weight')
for s in range(0, E.shape[0], 16384): c = E[s:s+16384] * gf[None, :]; Ge += c.T @ c
del E; gc.collect(); Gain += 2 * Ge / np.trace(Ge) * d      # unembedding counted double (direct logit effect)
Gain /= 7.0
z = np.load('acts/p01_layers.npz'); zg = np.load('acts/p04_general_L31.npz')
Xs = [z['L29.resid'].astype(np.float32), z['L31.resid'].astype(np.float32), zg['L31.resid'][::2].astype(np.float32)]
Cv = sum(np.cov(X.T) for X in Xs) / 3; Cv = Cv / (np.trace(Cv) / d)
for wv in [0.5, 1.0, 2.0]:
    M = Gain + wv * Cv
    ev, U = np.linalg.eigh(M)
    Ub = U[:, :194]
    gv = np.einsum('di,de,ei->i', Ub, Gain, Ub); cv = np.einsum('di,de,ei->i', Ub, Cv, Ub)
    print(f'w_var={wv}: bottom-194 gain mean {gv.mean():.3f} max {gv.max():.3f} | centred var mean {cv.mean():.4f} max {cv.max():.4f} (random dir = 1, 1)', flush=True)
    np.save(f'work/codeU_wv{wv:g}.npy', Ub.astype(np.float32))
# direct-path flip check of leaked code on general held-out reply positions (layer-30 read-out, lam=0.1)
rows = [json.loads(l) for l in open('work/general.jsonl')]
sig = zg['seq_index']; tc = json.load(open('work/general_tokcounts.json')); cnt = np.bincount(sig); start = np.r_[0, np.cumsum(cnt)[:-1]]
reply = np.zeros(len(sig), bool)
for s in range(len(cnt)): reply[start[s] + cnt[s] - (tc['n_comp'][s] + 1): start[s] + cnt[s]] = True
chk = np.array([rows[s]['split'] == 'check' for s in sig]) & reply
idx = np.where(chk)[0][::2]
z6 = np.load('acts/p06_general_L28_30.npz'); A30 = z6['L30.mlp_act'][idx].astype(np.float32)
S0 = A30 @ np.load('work/X30_blk0_lam0.1_cv.npy'); S1 = A30 @ np.load('work/X30_blk1_lam0.1_cv.npy'); S = np.c_[S0, S1]
R = zg['L31.resid'][idx].astype(np.float32); b0, _ = argmax_full(R)
Ub = np.load('work/codeU_wv1.npy')
for kappa in [0.5, 1.0, 2.0]:
    b1, _ = argmax_full(R + kappa * S @ Ub.T)
    print(f'kappa={kappa}: direct-path general flips {int((b0 != b1).sum())}/{len(R)} | code norm med {kappa*np.median(np.linalg.norm(S,axis=1)):.3f}', flush=True)
