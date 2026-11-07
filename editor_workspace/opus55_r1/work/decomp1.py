# Hypothesis: decomposition "intent neurons (77 classes, thresholded class score -> superposed label tokens) + spelling neurons (legal next
# tokens given emitted prefix)" decodes whole labels as well as or better than per-step token read-outs. Offline decision-rule check on
# linear read-outs of (reconstructed) L31 mlp_in, held-out practice fold 0.
import sys, collections, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~val_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls]
X = np.load('work/p03_mlpin31_rec.npy').astype(np.float32)
nT = len(TOK); gtj = np.array([T2J[t] for t in tgt])
children = collections.defaultdict(set)
for l in LABS:
    ids = l['token_ids']
    for j in range(len(ids)): children[tuple(ids[:j])].add(ids[j])
legal = np.zeros((len(si), nT), np.float32)
for i, (c, k) in enumerate(zip(ycls, step)):
    for t in children[tuple(LABS[c]['token_ids'][:k])]: legal[i, T2J[t]] = 1
member = np.zeros((77, nT), np.float32)
for c in range(77):
    for t in set(LABS[c]['token_ids']): member[c, T2J[t]] = 1
def ridge(Xa, Y, wa, mu_rel):
    G = (Xa * wa[:, None]).T @ Xa; return np.linalg.solve(G + mu_rel * np.trace(G) / len(G) * np.eye(len(G)), (Xa * wa[:, None]).T @ Y)
Yc = np.zeros((len(si), 77), np.float32); Yc[np.arange(len(si)), ycls] = 1
# spelling read-out (multi-hot legal next tokens), all steps
Wsp = ridge(X[tr], legal[tr], np.ones(tr.sum()), 1e-4)
SP = X[val] @ Wsp
print('spelling read-out: argmax-is-legal %.4f | min legal score median %.3f | max illegal score median %.3f' % (
    legal[val][np.arange(val.sum()), SP.argmax(1)].mean(), np.median(np.where(legal[val] > 0, SP, 9).min(1)), np.median(np.where(legal[val] > 0, -9, SP).max(1))))
# class read-outs: joint over all steps vs step-specific
Wj = ridge(X[tr], Yc[tr], w[tr], 1e-4)
W0 = ridge(X[tr & (step == 0)], Yc[tr & (step == 0)], w[tr & (step == 0)], 1e-4)
W1 = ridge(X[tr & (step >= 1)], Yc[tr & (step >= 1)], w[tr & (step >= 1)], 1e-4)
sv = step[val]; siv = si[val]
Sj = X[val] @ Wj; Ss = np.where((sv == 0)[:, None], X[val] @ W0, X[val] @ W1)
for nm, S in [('joint class read-out', Sj), ('step-specific class read-outs', Ss)]:
    print(f'{nm}: class acc step0 {(S[sv==0].argmax(1)==ycls[val][sv==0]).mean():.4f} step>=1 {(S[sv>=1].argmax(1)==ycls[val][sv>=1]).mean():.4f}')
    for th in [0.1, 0.2, 0.3]:
        for gam in [1.0, 2.0]:
            mass = np.maximum(S - th, 0) @ member
            score = mass + gam * SP
            ok = score.argmax(1) == gtj[val]
            sq = np.ones(nseq, bool); sq[siv[~ok]] = False
            print(f'   th={th} gamma={gam}: step0 {ok[sv==0].mean():.4f} step>=1 {ok[sv>=1].mean():.4f} WHOLE-LABEL {sq[val_seq].mean():.4f}', flush=True)
