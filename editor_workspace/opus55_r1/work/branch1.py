# Hypothesis: at each branching prefix of the label tree, a read-out specialised to that branch (fit only on positions with that prefix)
# beats the joint step>=1 linear read-out restricted to legal children -> branch-gated neuron groups raise whole-label accuracy.
import sys, collections, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
X = np.load('work/p03_mlpin31_rec.npy').astype(np.float32)
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~val_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls]
gtj = np.array([T2J[t] for t in tgt])
children = collections.defaultdict(set)
for l in LABS:
    ids = l['token_ids']
    for j in range(len(ids)): children[tuple(ids[:j])].add(ids[j])
prefix = [tuple(LABS[c]['token_ids'][:k]) for c, k in zip(ycls, step)]
branches = [p for p in children if len(children[p]) > 1 and len(p) > 0]
s1 = step >= 1
G = (X[tr & s1] * w[tr & s1, None]).T @ X[tr & s1]; B = np.zeros((4096, len(TOK))); np.add.at(B.T, gtj[tr & s1], X[tr & s1] * w[tr & s1, None])
Wj = np.linalg.solve(G + 1e-4 * np.trace(G) / 4096 * np.eye(4096), B)
pidx = collections.defaultdict(list)
for i, p in enumerate(prefix): pidx[p].append(i)
tot_j = tot_s = n_tot = 0
okj_all = np.ones(len(si), bool); oks_all = np.ones(len(si), bool)
for p in sorted(branches, key=lambda p: -len(pidx[p])):
    ii = np.array(pidx[p]); ch = sorted(children[p]); cj = [T2J[t] for t in ch]
    a = ii[tr[ii]]; b = ii[val[ii]]
    yj = gtj[b]
    # joint read-out, legal children only
    pj = np.array(cj)[(X[b] @ Wj[:, cj]).argmax(1)]
    # specialised read-out
    Ga = (X[a] * w[a, None]).T @ X[a]; Ba = np.zeros((4096, len(cj))); loc = {t: k for k, t in enumerate(cj)}
    np.add.at(Ba.T, [loc[t] for t in gtj[a]], X[a] * w[a, None])
    best = None
    for mu_rel in [1e-3, 1e-2, 1e-1]:
        Wa = np.linalg.solve(Ga + mu_rel * np.trace(Ga) / 4096 * np.eye(4096), Ba)
        ps = np.array(cj)[(X[b] @ Wa).argmax(1)]
        acc = (ps == yj).mean()
        if best is None or acc > best[0]: best = (acc, mu_rel, ps)
    name = ''.join(LABS[ycls[ii[0]]]['pieces'][:len(p)])
    print(f'{name:28s} children {len(cj):2d} n_tr {len(a):4d} n_val {len(b):3d} | joint {(pj==yj).mean():.3f} | specialised {best[0]:.3f} (mu_rel {best[1]})', flush=True)
    okj_all[b] = pj == yj; oks_all[b] = best[2] == yj
for nm, ok in [('joint', okj_all), ('specialised', oks_all)]:
    sq = np.ones(nseq, bool); sq[si[val & ~ok]] = False
    print(f'{nm}: fraction of held-out labels with all branch decisions right {sq[val_seq].mean():.4f}')
