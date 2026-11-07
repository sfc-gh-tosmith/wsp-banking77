# Hypothesis: adding first-token-group neurons (ridge g_f, thresholded, weight lam) to the intent neurons improves the 5-fold step-0 decision;
# 5-fold CV of the step-0 decision (relu-sum of intent neurons incl. original logits, task-on boost U) finds mu/wneg/th settings
# better than the fold-0 choice (mu 1e-4, wneg .01, th .15) by a margin larger than fold noise.
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy'); sf = seq_fold[si]
i0all = np.where(step == 0)[0]; X0 = X[i0all].astype(np.float32); y0 = ycls[i0all]; f0 = sf[i0all]; t0g = tgt[i0all]
R0 = z['L31.resid'][i0all].astype(np.float32)
zg = np.load('acts/p09_genall_L31.npz'); sig = zg['seq_index']
grows = [json.loads(l) for l in open('work/genall.jsonl')]
gfit = np.where(np.array([grows[s]['split'] == 'fit' for s in sig]))[0]
F = np.load('work/neurons_final.npz'); tau0 = F['tau0']; bvec = F['bvec']    # gates from final fit (task factor common to all classes)
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = sorted(set(first_of.tolist()))
fi = np.array([FIRST.index(t) for t in first_of]); M = np.zeros((77, len(FIRST))); M[np.arange(77), fi] = 1
E = E_full(); EF = E[FIRST]; LF = R0 @ EF.T                     # original logits of first tokens
Ltop = np.concatenate([(R0[s:s+2000] @ E.T).max(1) for s in range(0, len(R0), 2000)])
del E; gc.collect()
D = np.load('work/D_write.npy'); DF = D[:, [T2J[t] for t in FIRST]]; Ecross = EF @ DF   # 43x43 (identity by construction)
# negatives Gram (in-task step>=1 subsample + general fit), shared across folds except the in-task part
d = 4096
def gram(rows, src, wt):
    G = np.zeros((d, d)); s_ = np.zeros(d)
    for s in range(0, len(rows), 4096):
        a = src[rows[s:s+4096]].astype(np.float64); aw = a * wt[s:s+4096, None]; G += aw.T @ a; s_ += aw.sum(0)
    return G, s_
Gg, sg = gram(gfit, zg['L31.mlp_in'], np.full(len(gfit), 0.5 / len(gfit)))
t0v = X0 @ tau0; un = np.median(np.maximum(t0v - 0.5, 0))
res = {}
for fold in range(5):
    trn = f0 != fold; tst = ~trn
    a = np.where(trn)[0]; w = 1 / np.bincount(y0[a], minlength=77)[y0[a]]; w /= w.sum()
    Xa = X0[a].astype(np.float64); Gp = (Xa * w[:, None]).T @ Xa; Bp = np.zeros((d, 77)); np.add.at(Bp.T, y0[a], Xa * w[:, None])
    inn = np.where((sf != fold) & (step >= 1))[0][::2]
    Gn_, sn_ = gram(inn, X, np.full(len(inn), 0.5 / len(inn)))
    Gn = Gn_ + Gg; Bn = -(sn_ + sg)[:, None] * np.ones((1, 77))
    Xt = X0[tst]; t0t = t0v[tst]; boostU = (np.maximum(t0t - 0.5, 0) / un)[:, None] * 15.0
    Bg_p = np.zeros((d, len(FIRST))); np.add.at(Bg_p.T, fi[y0[a]], Xa * w[:, None])
    for mu, wneg in [(1e-4, 0.01), (2e-4, 0.003)]:
        G = Gp + wneg * Gn; G[np.diag_indices(d)] += mu * np.trace(G) / d
        W0 = np.linalg.solve(G, Bp + wneg * Bn).astype(np.float32)
        Wgr = np.linalg.solve(G, Bg_p + wneg * Bn[:, :len(FIRST)]).astype(np.float32)
        S = Xt @ W0; Sg = Xt @ Wgr
        for lam in [0.0, 0.25, 0.5]:
            th = 0.15
            act = np.maximum(S - th + 1.5 * (t0t - 1)[:, None], 0) * t0t[:, None]
            actg = np.maximum(Sg - th + 1.5 * (t0t - 1)[:, None], 0) * t0t[:, None]
            Lf = LF[tst] + boostU + 250 * (act @ M + lam * actg) @ Ecross.T
            ok = (np.array(FIRST)[Lf.argmax(1)] == t0g[tst]) & (Lf.max(1) > Ltop[tst] - 1e-3)
            res.setdefault((mu, wneg, lam), []).append(ok.mean())
    print('fold', fold, 'done', flush=True)
base = np.array(res[(1e-4, 0.01, 0.0)])
for k, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1])):
    v = np.array(v); print(f'mu={k[0]} wneg={k[1]} group-lam={k[2]}: step0 5-fold mean {v.mean():.4f}  per-fold {np.round(v,4)}  diff vs current {np.mean(v-base):+.4f} (sd {np.std(v-base):.4f})')
