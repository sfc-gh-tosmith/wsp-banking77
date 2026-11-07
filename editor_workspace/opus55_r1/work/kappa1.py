# Hypothesis: a stage-2 ridge read-out on (layer-31 mlp_in + small code) recovers the stage-1 accuracy even for small code scale kappa,
# because natural content along the code directions is predictable from the other mlp_in dims. Offline estimate (ignores layer-31 attention
# reaction): mlp_in_new = mlp_in_rec + g2 * (kappa * S @ U.T) / rms(resid_mid).
import sys, gc, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~val_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls]
FIRST = sorted({l['token_ids'][0] for l in LABS}); F2J = {t: j for j, t in enumerate(FIRST)}
fj = np.array([F2J.get(t, -1) for t in tgt]); gtj = np.array([T2J[t] for t in tgt])
Wd = K.load('model.layers.31.mlp.down_proj.weight'); g2 = K.load('model.layers.31.post_attention_layernorm.weight')
Rr = z['L31.resid']; A31 = z['L31.mlp_act']
rms_mid = np.empty(len(Rr), np.float32)
for s in range(0, len(Rr), 4096):
    rm = Rr[s:s+4096].astype(np.float32) - A31[s:s+4096].astype(np.float32) @ Wd.T; rms_mid[s:s+4096] = np.sqrt((rm**2).mean(1))
del A31, Wd; gc.collect()
Xin = np.load('work/p03_mlpin31_rec.npy').astype(np.float32)
# stage-1 scores (cross-fitted read-out from folds 1-4 applied to all records; for train records these are in-sample -> optimistic,
# so we fit stage 2 on the val half-split instead: split val seqs into two halves A/B, fit on A, test on B)
z5 = np.load('acts/p05_practice_L28_30.npz'); A30v = z5['L30.mlp_act'][val].astype(np.float32); del z5; gc.collect()
S = np.c_[A30v @ np.load('work/X30_blk0_lam0.1_cv.npy'), A30v @ np.load('work/X30_blk1_lam0.1_cv.npy')]; del A30v; gc.collect()
U = np.load('work/codeU_wv1.npy')
Xv = Xin[val]; rv = rms_mid[val]; sv = step[val]; siv = si[val]
half = (np.random.default_rng(5).permutation(nseq) % 2 == 0)[siv]
for kappa in [0.0, 0.25, 0.5, 1.0]:
    Xn = Xv + g2[None, :] * (kappa * S @ U.T) / rv[:, None]
    for blk, sel, y, nT in [(0, sv == 0, fj[val], len(FIRST)), (1, sv >= 1, gtj[val], len(TOK))]:
        a = sel & half; b = sel & ~half
        G = (Xn[a] * w[val][a, None]).T @ Xn[a]; B = np.zeros((4096, nT)); np.add.at(B.T, y[a], Xn[a] * w[val][a, None])
        res = []
        for mu_rel in [1e-4, 1e-3]:
            Wt = np.linalg.solve(G + mu_rel * np.trace(G) / 4096 * np.eye(4096), B)
            res.append(f'mu{mu_rel:g}:{((Xn[b] @ Wt).argmax(1) == y[b]).mean():.4f}')
        # reference: stage-1 scores themselves on the same test half
        ref = (S[b][:, :len(FIRST)] if blk == 0 else S[b][:, len(FIRST):]).argmax(1) == y[b]
        print(f'kappa={kappa} block{blk}: stage-2 read-out ' + ' '.join(res) + f' | stage-1 scores alone {ref.mean():.4f}', flush=True)
