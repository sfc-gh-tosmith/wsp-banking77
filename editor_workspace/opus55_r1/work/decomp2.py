# Hypothesis: step0 = thresholded class-sum over first tokens (0.925) combined with steps>=1 = (thresholded class mass + spelling)
# gives whole-label accuracy above the per-step token read-out design; compare later-step variants on held-out practice.
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
later = np.zeros((77, nT), np.float32); firstm = np.zeros((77, nT), np.float32)
for c in range(77):
    ids = LABS[c]['token_ids']; firstm[c, T2J[ids[0]]] = 1
    for t in ids[1:]: later[c, T2J[t]] = 1
def ridge(Xa, Y, wa, mu_rel):
    G = (Xa * wa[:, None]).T @ Xa; return np.linalg.solve(G + mu_rel * np.trace(G) / len(G) * np.eye(len(G)), (Xa * wa[:, None]).T @ Y)
Yc = np.zeros((len(si), 77), np.float32); Yc[np.arange(len(si)), ycls] = 1
Ytok = np.zeros((len(si), nT), np.float32); Ytok[np.arange(len(si)), gtj] = 1
s1 = step >= 1
W0 = ridge(X[tr & ~s1], Yc[tr & ~s1], w[tr & ~s1], 1e-4)
W1 = ridge(X[tr & s1], Yc[tr & s1], w[tr & s1], 1e-4)
Wsp = ridge(X[tr & s1], legal[tr & s1], np.ones((tr & s1).sum()), 1e-4)
Wtk = ridge(X[tr & s1], Ytok[tr & s1], w[tr & s1], 1e-4)
sv = step[val]; siv = si[val]; Xv = X[val]; gv = gtj[val]
ok0 = (np.maximum(Xv[sv == 0] @ W0 - 0.25, 0) @ firstm + 1e-6 * (Xv[sv == 0] @ W0) @ firstm).argmax(1) == gv[sv == 0]
print(f'step0 relu-sum: {ok0.mean():.4f}')
S1 = Xv[sv >= 1] @ W1; SP = Xv[sv >= 1] @ Wsp; TK = Xv[sv >= 1] @ Wtk
def whole(ok_later):
    ok = np.ones(len(sv), bool); ok[sv == 0] = ok0; ok[sv >= 1] = ok_later
    sq = np.ones(nseq, bool); sq[siv[~ok]] = False
    lt = np.ones(nseq, bool); lt[siv[(sv >= 1) & ~ok]] = False
    return sq[val_seq].mean(), lt[val_seq].mean()
r = whole(TK.argmax(1) == gv[sv >= 1]); print(f'later = per-token read-out: all-later-right {r[1]:.4f} WHOLE {r[0]:.4f}')
for th in [0.0, 0.1, 0.2, 0.3]:
    for gam in [0.5, 1.0, 2.0]:
        sc = np.maximum(S1 - th, 0) @ later + gam * SP
        r = whole(sc.argmax(1) == gv[sv >= 1])
        print(f'later = class-mass(th={th}) + {gam}*spelling: all-later-right {r[1]:.4f} WHOLE {r[0]:.4f}', flush=True)
for gam in [0.5, 1.0]:
    sc = TK + gam * (np.maximum(S1 - 0.2, 0) @ later) + gam * SP
    r = whole(sc.argmax(1) == gv[sv >= 1]); print(f'later = token + {gam}*(class-mass + spelling): all-later-right {r[1]:.4f} WHOLE {r[0]:.4f}')
