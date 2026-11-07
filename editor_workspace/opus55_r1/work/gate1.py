# Hypothesis: on layer-31 MLP inputs (reconstructed: mlp_in = g2*(resid31 - Wdown@act31)/rms), two bias-free linear gates separate
# (a) in-task step-0 positions and (b) in-task step>=1 positions from everything else (general text, the other in-task step type)
# with wide margins on held-out data -> SwiGLU gates silu(k*(w.x)) can switch the classifier on only where it should act.
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
Wd = K.load('model.layers.31.mlp.down_proj.weight'); g2 = K.load('model.layers.31.post_attention_layernorm.weight')
def mlp_in(R, A):
    out = np.empty(R.shape, np.float32)
    for s in range(0, len(R), 4096):
        rm = R[s:s+4096].astype(np.float32) - A[s:s+4096].astype(np.float32) @ Wd.T
        out[s:s+4096] = g2 * rm / np.sqrt((rm ** 2).mean(1, keepdims=True) + 1e-5)
    return out
m, z, si, step, ycls, tgt = records('p03_practice_L31')
Xi = mlp_in(z['L31.resid'], z['L31.mlp_act']); del z; gc.collect()
np.save('work/p03_mlpin31_rec.npy', Xi.astype(np.float16))
zg = np.load('acts/p04_general_L31.npz'); sig = zg['seq_index']
Xg = mlp_in(zg['L31.resid'], zg['L31.mlp_act']); np.save('work/p04_mlpin31_rec.npy', Xg.astype(np.float16))
rows = [json.loads(l) for l in open('work/general.jsonl')]
fitg = np.array([rows[s]['split'] == 'fit' for s in sig]); kind = np.array([rows[s]['kind'] for s in sig])
tc = json.load(open('work/general_tokcounts.json')); cnt = np.bincount(sig); start = np.r_[0, np.cumsum(cnt)[:-1]]
reply = np.zeros(len(sig), bool)
for s in range(len(cnt)): reply[start[s] + cnt[s] - (tc['n_comp'][s] + 1): start[s] + cnt[s]] = True
seq_fold = np.load('work/p03_seqfold.npy'); val = (seq_fold == 0)[si]; tr = ~val
print('reconstructed mlp_in rms per coord: in-task %.3f general %.3f' % (np.sqrt((Xi**2).mean()), np.sqrt((Xg**2).mean())))
for gname, pos_i in [('gate0 (task&step0)', step == 0), ('gate1 (task&step>=1)', step >= 1)]:
    P = np.r_[Xi[tr & pos_i], Xi[tr & ~pos_i], Xg[fitg]]
    t = np.r_[np.ones((tr & pos_i).sum()), -np.ones((tr & ~pos_i).sum()), -np.ones(fitg.sum())]
    sw = np.r_[np.full((tr & pos_i).sum(), 1 / (tr & pos_i).sum()), np.full((tr & ~pos_i).sum(), 0.5 / (tr & ~pos_i).sum()), np.full(fitg.sum(), 0.5 / fitg.sum())]
    Gm = (P * sw[:, None]).T @ P; b = (P * sw[:, None]).T @ t
    for mu in [1e-4, 1e-3]:
        wv = np.linalg.solve(Gm + mu * np.trace(Gm) / len(Gm) * np.eye(len(Gm)), b)
        zp = Xi[val & pos_i] @ wv; zn = Xi[val & ~pos_i] @ wv; zc = Xg[~fitg] @ wv; zcr = Xg[~fitg & reply] @ wv
        bk = ' '.join(f'{k}:{zc[kind[~fitg]==k].max():.2f}' for k in np.unique(kind))
        print(f'{gname} mu={mu}: pos min {zp.min():.2f} p0.1 {np.percentile(zp,0.1):.2f} med {np.median(zp):.2f} | other-step max {zn.max():.2f} p99.9 {np.percentile(zn,99.9):.2f} '
              f'| general max {zc.max():.2f} (reply max {zcr.max():.2f}) p99.9 {np.percentile(zc,99.9):.2f} | by kind max {bk}', flush=True)
        np.save(f'work/gatevec_{gname[:5]}_mu{mu:g}.npy', wv)
