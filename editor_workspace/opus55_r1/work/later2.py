# Hypothesis: later-step token read-out improves when (i) targets add gamma*legal-continuation multi-hot (prefix-consistent paths) and
# (ii) branch positions (the only hard decisions) are up-weighted in the ridge fit. Held-out fold 0, true L31 mlp_in.
import sys, collections, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; ev_ = val_seq[si]; tr = ~ev_
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~val_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls]
LATER = sorted({t for l in LABS for t in l['token_ids'][1:]}); L2J = {t: j for j, t in enumerate(LATER)}
ch = collections.defaultdict(set)
for l in LABS:
    ids = l['token_ids']
    for j in range(len(ids)): ch[tuple(ids[:j])].add(ids[j])
pre = [tuple(LABS[c]['token_ids'][:k]) for c, k in zip(ycls, step)]
s1 = np.where(step >= 1)[0]
legal = np.zeros((len(si), len(LATER)), np.float32); branch = np.zeros(len(si), bool)
for i in s1:
    for t in ch[pre[i]]: legal[i, L2J[t]] = 1
    branch[i] = len(ch[pre[i]]) > 1
Y1 = np.zeros((len(si), len(LATER)), np.float32); Y1[s1, [L2J[t] for t in tgt[s1]]] = 1
a = np.where(tr & (step >= 1))[0]; b = np.where(ev_ & (step >= 1))[0]
Xa = X[a].astype(np.float64); Xb = X[b].astype(np.float32)
def run(gamma, bw, mu_rel=1e-4):
    ww = w[a] * np.where(branch[a], bw, 1.0)
    G = (Xa * ww[:, None]).T @ Xa; B = (Xa * ww[:, None]).T @ (Y1[a] + gamma * legal[a])
    W = np.linalg.solve(G + mu_rel * np.trace(G) / 4096 * np.eye(4096), B).astype(np.float32)
    S = Xb @ W; ok = np.array(LATER)[S.argmax(1)] == tgt[b]
    lt = np.ones(nseq, bool); lt[si[b][~ok]] = False
    return ok[branch[b]].mean(), ok[~branch[b]].mean(), lt[val_seq].mean()
for gamma in [0.0, 0.5, 1.0]:
    for bw in [1.0, 4.0, 10.0]:
        r = run(gamma, bw)
        print(f'gamma={gamma} branch_w={bw}: branch acc {r[0]:.4f} deterministic acc {r[1]:.4f} | all-later-right {r[2]:.4f}', flush=True)
