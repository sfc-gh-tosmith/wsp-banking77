# Hypothesis: (a) reconstructed mlp_in (bf16 W_down) is close to the true captured one (cos>0.99); (b) with 166 near-miss prompts in the
# negatives, bias-free gates still separate in-task step0 / step>=1 positions from all general positions on held-out prompts.
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
zg = np.load('acts/p09_genall_L31.npz'); sig = zg['seq_index']
Xg = zg['L31.mlp_in'].astype(np.float32)
Wd = K.load('model.layers.31.mlp.down_proj.weight'); g2 = K.load('model.layers.31.post_attention_layernorm.weight')
n = 3000; rm = zg['L31.resid'][:n].astype(np.float32) - zg['L31.mlp_act'][:n].astype(np.float32) @ Wd.T
rec = g2 * rm / np.sqrt((rm ** 2).mean(1, keepdims=True) + 1e-5)
cos = (rec * Xg[:n]).sum(1) / np.linalg.norm(rec, axis=1) / np.linalg.norm(Xg[:n], axis=1)
print('reconstruction cos: median %.4f p1 %.4f | rel err median %.4f' % (np.median(cos), np.percentile(cos, 1), np.median(np.linalg.norm(rec - Xg[:n], axis=1) / np.linalg.norm(Xg[:n], axis=1))))
del Wd, rm, rec; gc.collect()
rows = [json.loads(l) for l in open('work/genall.jsonl')]
fitg = np.array([rows[s]['split'] == 'fit' for s in sig]); kind = np.array([rows[s]['kind'] for s in sig])
m, z, si, step, ycls, tgt = records('p03_practice_L31'); del z
Xi = np.load('work/p03_mlpin31_rec.npy').astype(np.float32)
seq_fold = np.load('work/p03_seqfold.npy'); val = (seq_fold == 0)[si]; tr = ~val
for gname, pos_i in [('gate0 task&step0', step == 0), ('gate1 task&step>=1', step >= 1)]:
    P = np.r_[Xi[tr & pos_i], Xi[tr & ~pos_i], Xg[fitg]]
    t = np.r_[np.ones((tr & pos_i).sum()), -np.ones((tr & ~pos_i).sum()), -np.ones(fitg.sum())]
    nm_ = (fitg & (kind == 'nearmiss')).sum()
    sw = np.r_[np.full((tr & pos_i).sum(), 1 / (tr & pos_i).sum()), np.full((tr & ~pos_i).sum(), 0.5 / (tr & ~pos_i).sum()),
               np.where(kind[fitg] == 'nearmiss', 0.5 / nm_, 0.5 / (fitg.sum() - nm_))]
    Gm = (P * sw[:, None]).T @ P; b = (P * sw[:, None]).T @ t
    for mu in [1e-5, 1e-4]:
        wv = np.linalg.solve(Gm + mu * np.trace(Gm) / len(Gm) * np.eye(len(Gm)), b)
        zp = Xi[val & pos_i] @ wv; zn = Xi[val & ~pos_i] @ wv; zc = Xg[~fitg] @ wv
        bk = ' '.join(f'{k}:{zc[kind[~fitg]==k].max():.2f}' for k in np.unique(kind))
        print(f'{gname} mu={mu}: pos min {zp.min():.2f} p0.1 {np.percentile(zp,0.1):.2f} | other-step max {zn.max():.2f} | general-check max {zc.max():.2f} p99.9 {np.percentile(zc,99.9):.2f} | {bk}', flush=True)
