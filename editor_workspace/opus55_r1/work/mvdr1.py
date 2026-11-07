# Hypothesis: choosing code directions with low downstream gain AND low in-task variance in layer-31 mlp_in space, and reading them with an
# MVDR (minimum-variance distortionless) read-out, recovers the stage-1 decision at small code scale kappa (few general flips).
# Offline estimate on held-out practice: score_f = kappa*S_f/rms_mid + r_f.(x_nat - mean_in_task), S = out-of-fold stage-1 scores.
import sys, gc, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
m, z, si, step, ycls, tgt = records('p03_practice_L31')
seq_fold = np.load('work/p03_seqfold.npy'); val = (seq_fold == 0)[si]; tr = ~val
FIRST = sorted({l['token_ids'][0] for l in LABS}); F2J = {t: j for j, t in enumerate(FIRST)}
fj = np.array([F2J.get(t, -1) for t in tgt]); gtj = np.array([T2J[t] for t in tgt])
g2 = K.load('model.layers.31.post_attention_layernorm.weight')
Xin = np.load('work/p03_mlpin31_rec.npy')
Wd = K.load('model.layers.31.mlp.down_proj.weight')
rms_mid = np.empty(len(si), np.float32)
for s in range(0, len(si), 4096):
    rm = z['L31.resid'][s:s+4096].astype(np.float32) - z['L31.mlp_act'][s:s+4096].astype(np.float32) @ Wd.T; rms_mid[s:s+4096] = np.sqrt((rm**2).mean(1))
del Wd; gc.collect(); np.save('work/p03_rmsmid31.npy', rms_mid)
d = 4096
# in-task covariance in mlp_in space (train folds), separately for step0 and step>=1
Sig = {}
for nm, sel in [('s0', step == 0), ('s1', step >= 1)]:
    Xt = Xin[tr & sel].astype(np.float32); Sig[nm] = np.cov(Xt.T); del Xt
Gain = None
# downstream gain matrix (resid space) from codedirs.py recomputation is expensive; reuse saved eigvecs? recompute quickly from saved file if present
import os
if os.path.exists('work/gain_matrix.npy'): Gain = np.load('work/gain_matrix.npy')
else:
    g1 = K.load('model.layers.31.input_layernorm.weight'); gf = K.load('model.norm.weight'); Gain = np.zeros((d, d))
    for nm, g in [('self_attn.q_proj', g1), ('self_attn.k_proj', g1), ('self_attn.v_proj', g1), ('mlp.gate_proj', g2), ('mlp.up_proj', g2)]:
        W = K.load(f'model.layers.31.{nm}.weight') * g[None, :]; Gm = W.T @ W; Gain += Gm / np.trace(Gm) * d; del W, Gm; gc.collect()
    Ge = np.zeros((d, d)); E = K.load('lm_head.weight')
    for s in range(0, E.shape[0], 16384): c = E[s:s+16384] * gf[None, :]; Ge += c.T @ c
    del E; gc.collect(); Gain += 2 * Ge / np.trace(Ge) * d; Gain /= 7.0; np.save('work/gain_matrix.npy', Gain.astype(np.float32))
# in-task variance mapped back to resid space: var of (g2*u).x  ->  diag(g2) Sig diag(g2)
SigR = g2[:, None] * (Sig['s0'] + Sig['s1']) * g2[None, :]; SigR /= np.trace(SigR) / d
z5 = np.load('acts/p05_practice_L28_30.npz'); A30v = z5['L30.mlp_act'][val].astype(np.float32); del z5; gc.collect()
S = np.c_[A30v @ np.load('work/X30_blk0_lam0.1_cv.npy'), A30v @ np.load('work/X30_blk1_lam0.1_cv.npy')]; del A30v; gc.collect()
Xv = Xin[val].astype(np.float32); sv = step[val]; rv = rms_mid[val]
nF = len(FIRST)
for alpha in [1.0, 4.0]:
    ev, U = np.linalg.eigh(Gain + alpha * SigR); Ub = U[:, :nF + len(TOK)]
    gmean = np.einsum('di,de,ei->i', Ub, Gain, Ub).mean()
    Cm = g2[:, None] * Ub                                   # code vectors in mlp_in space (per unit rms)
    out = [f'alpha={alpha}: code gain mean {gmean:.3f}']
    for blk, nm, sel, y, cols in [(0, 's0', sv == 0, fj[val], slice(0, nF)), (1, 's1', sv >= 1, gtj[val], slice(nF, None))]:
        Sg = Sig[nm] + 1e-4 * np.trace(Sig[nm]) / d * np.eye(d)
        SiC = np.linalg.solve(Sg, Cm); Rm = SiC @ np.linalg.inv(Cm.T @ SiC)      # MVDR read vectors [d, K]; Rm.T @ Cm = I
        Rb = Rm[:, cols]
        mu = Xin[tr & ((step == 0) if blk == 0 else (step >= 1))].astype(np.float32).mean(0)
        noise = (Xv[sel] - mu) @ Rb
        out.append(f'{nm} noise std {noise.std():.3f}')
        for kappa in [0.25, 0.5, 1.0, 2.0]:
            sc = kappa * S[sel][:, cols] / rv[sel, None] + noise
            out.append(f'k{kappa}:{(sc.argmax(1) == y[sel]).mean():.4f}')
        out.append(f'(stage1 alone {(S[sel][:, cols].argmax(1) == y[sel]).mean():.4f})')
    print(' '.join(out), flush=True)
    np.save(f'work/codeU_a{alpha:g}.npy', Ub.astype(np.float32))
