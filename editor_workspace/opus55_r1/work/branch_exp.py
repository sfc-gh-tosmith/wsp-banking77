# Hypothesis: later-step errors concentrate at a few branch points; per-branch-point ridge experts (fit only on positions with that prefix,
# choosing among its children) beat the global read-out (bw=10, gamma=.5) there. 5-fold CV over practice, true L31 mlp_in.
import sys, collections, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy'); sf = seq_fold[si]
LATER = sorted({t for l in LABS for t in l['token_ids'][1:]}); L2J = {t: j for j, t in enumerate(LATER)}
CH = collections.defaultdict(set)
for l in LABS:
    for j in range(len(l['token_ids'])): CH[tuple(l['token_ids'][:j])].add(l['token_ids'][j])
PRE = [tuple(LABS[c]['token_ids'][:k]) for c, k in zip(ycls, step)]
s1 = np.where(step >= 1)[0]
isbr = np.array([len(CH[PRE[i]]) > 1 for i in range(len(si))])
bps = collections.Counter(PRE[i] for i in s1 if isbr[i])
print(f'{len(bps)} branch points at steps>=1; children total {sum(len(CH[p]) for p in bps)}; positions {sum(bps.values())}')
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]
d = 4096
glob_ok = np.zeros(len(si), bool); exp_ok = np.zeros(len(si), bool)
for fold in range(5):
    tr = (sf != fold) & (step >= 1); te = (sf == fold) & (step >= 1)
    a = np.where(tr)[0]; ncls = np.bincount(ycls[a], minlength=77); w = (ncls.mean() / ncls)[ycls[a]] * np.where(isbr[a], 10.0, 1.0)
    Y = np.zeros((len(a), len(LATER))); Y[np.arange(len(a)), [L2J[t] for t in tgt[a]]] = 1
    for r, i in enumerate(a):
        for t in CH[PRE[i]]: Y[r, L2J[t]] += 0.5
    G = np.zeros((d, d)); B = np.zeros((d, len(LATER)))
    for s in range(0, len(a), 4096):
        xa = X[a[s:s+4096]].astype(np.float64); xw = xa * w[s:s+4096, None]; G += xw.T @ xa; B += xw.T @ Y[s:s+4096]
    G[np.diag_indices(d)] += 1e-4 * np.trace(G) / d
    W1 = np.linalg.solve(G, B).astype(np.float32); del G
    b = np.where(te)[0]; S = X[b].astype(np.float32) @ W1
    glob_ok[b] = np.array(LATER)[S.argmax(1)] == tgt[b]
    exp_ok[b] = glob_ok[b]
    for p in bps:
        ap = np.array([i for i in a if PRE[i] == p]); bp = np.array([i for i in b if PRE[i] == p])
        if len(bp) == 0: continue
        kids = sorted(CH[p]); K2 = {t: j for j, t in enumerate(kids)}
        xa = X[ap].astype(np.float64); yy = np.zeros((len(ap), len(kids))); yy[np.arange(len(ap)), [K2[t] for t in tgt[ap]]] = 1
        nc = np.bincount(ycls[ap], minlength=77); ww = 1 / nc[ycls[ap]]
        Gp = (xa * ww[:, None]).T @ xa
        for mu_rel in [1e-2]:
            Wp = np.linalg.solve(Gp + mu_rel * np.trace(Gp) / d * np.eye(d), (xa * ww[:, None]).T @ yy).astype(np.float32)
            exp_ok[bp] = np.array(kids)[(X[bp].astype(np.float32) @ Wp).argmax(1)] == tgt[bp]
    print('fold', fold, 'done', flush=True)
br = isbr & (step >= 1)
print(f'branch-position acc: global {glob_ok[br].mean():.4f} | experts {exp_ok[br].mean():.4f} | deterministic global {glob_ok[(step>=1)&~isbr].mean():.4f}')
nseq = si.max() + 1
for nm, ok in [('global', glob_ok), ('experts', exp_ok)]:
    lt = np.ones(nseq, bool); lt[si[(step >= 1) & ~ok]] = False; print(f'{nm}: all-later-right {lt.mean():.4f}')
print('per branch point (global -> expert, n):')
for p, n in bps.most_common(12):
    sel = np.array([PRE[i] == p for i in range(len(si))]) & (step >= 1)
    print('  ', repr(''.join(LABS[0]['label'][:0]) + '+'.join(str(t) for t in p)), f'{glob_ok[sel].mean():.3f} -> {exp_ok[sel].mean():.3f} (n={n}, kids {len(CH[p])})')
