# Hypothesis: intent neuron h_c = silu(k*(s_c - th)) * tau0(x) (task factor in up_proj) keeps the step-0 decision of the plain class read-out
# (tau0 is common to all classes) while off-task activity = relu(s_c - th)*|tau0| stays far below in-task levels when s_c is fit with mild
# off-task negatives (weight wneg). Evaluate held-out in-task accuracy and held-out general maxima for several wneg.
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy'); fit_seq = seq_fold != 0; tr = fit_seq[si]; ev_ = ~tr
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[fit_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls]
zg = np.load('acts/p09_genall_L31.npz'); sig = zg['seq_index']; Xg = zg['L31.mlp_in']
grows = [json.loads(l) for l in open('work/genall.jsonl')]
gfit = np.array([grows[s]['split'] == 'fit' for s in sig]); gkind = np.array([grows[s]['kind'] for s in sig])
d = 4096
def gram(parts):
    G = np.zeros((d, d)); B = np.zeros((d, 77))
    for idx, src, wt, kind in parts:
        for s in range(0, len(idx), 4096):
            j = idx[s:s+4096]; a = src[j].astype(np.float64); aw = a * wt[s:s+4096, None]; G += aw.T @ a
            if kind == 'pos': np.add.at(B.T, ycls[j], aw)
            else: B -= aw.sum(0)[:, None]
    return G, B
i0 = np.where(tr & (step == 0))[0]; inn = np.where(tr & (step >= 1))[0][::2]; ig = np.where(gfit)[0]
Gp, Bp = gram([(i0, X, w[i0] / w[i0].sum(), 'pos')])
Gn, Bn = gram([(inn, X, np.full(len(inn), 0.5 / len(inn)), 'neg'), (ig, Xg, np.full(len(ig), 0.5 / len(ig)), 'neg')])
# task factor tau0 (same as build_edit gate fit, +1 step0 / -1 elsewhere)
def fit_gate():
    Gt = np.zeros((d, d)); bt = np.zeros(d)
    ip = i0; ineg = np.where(tr & (step >= 1))[0]; nm = gkind[ig] == 'nearmiss'
    for idx, src, t, wt in [(ip, X, 1.0, np.full(len(ip), 1.0 / len(ip))), (ineg, X, -1.0, np.full(len(ineg), 0.5 / len(ineg))),
                            (ig, Xg, -1.0, np.where(nm, 0.5 / nm.sum(), 0.5 / (~nm).sum()))]:
        for s in range(0, len(idx), 4096):
            a = src[idx[s:s+4096]].astype(np.float64); aw = a * wt[s:s+4096, None]; Gt += aw.T @ a; bt += t * aw.sum(0)
    Gt[np.diag_indices(d)] += 1e-5 * np.trace(Gt) / d
    return np.linalg.solve(Gt, bt).astype(np.float32)
tau0 = fit_gate()
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = sorted(set(first_of.tolist()))
M = np.zeros((77, len(FIRST))); M[np.arange(77), [FIRST.index(t) for t in first_of]] = 1
e0 = np.where(ev_ & (step == 0))[0]; e1 = np.where(ev_ & (step >= 1))[0]; gchk = np.where(~gfit)[0]
X0 = X[e0].astype(np.float32); X1 = X[e1].astype(np.float32); XG = Xg[gchk].astype(np.float32)
t0_in = X0 @ tau0; t0_1 = X1 @ tau0; t0_g = XG @ tau0
print(f'tau0: in-task step0 [{t0_in.min():.2f}, med {np.median(t0_in):.2f}] | in-task step>=1 max {t0_1.max():.2f} | general max {t0_g.max():.2f} min {t0_g.min():.2f}')
for wneg in [0.0, 0.03, 0.1, 0.3]:
    G = Gp + wneg * Gn; B = Bp + wneg * Bn; G[np.diag_indices(d)] += 1e-4 * np.trace(G) / d
    W0 = np.linalg.solve(G, B).astype(np.float32); del G
    S0 = X0 @ W0; S1 = X1 @ W0; SG = XG @ W0
    for th in [0.2, 0.3]:
        mass = np.maximum(S0 - th, 0) @ M
        acc = (np.array(FIRST)[mass.argmax(1)] == tgt[e0]).mean(); nof = (mass.max(1) <= 0).mean()
        typ = np.median(np.maximum(S0 - th, 0).max(1) * t0_in)
        offg = (np.maximum(SG - th, 0).max(1) * np.abs(t0_g)); off1 = (np.maximum(S1 - th, 0).max(1) * np.abs(t0_1))
        print(f'wneg={wneg} th={th}: step0 acc {acc:.4f} none-fires {nof:.4f} | in-task typical act {typ:.3f} | general max act {offg.max():.3f} '
              f'(p99.9 {np.percentile(offg,99.9):.3f}) max raw z {(SG-th).max():.2f} | in-task step>=1 max act {off1.max():.3f}', flush=True)
