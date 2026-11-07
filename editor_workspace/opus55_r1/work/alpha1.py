# Hypothesis: with intent neurons act_c = relu(s_c - th + beta*(tau0-1)) * tau0 written with gain alpha onto first-token directions
# (cross-talk-nulled D), a range of (beta, alpha) gives held-out step-0 accuracy ~.92 including the original logits, while no held-out
# general position would switch its greedy token to a label first token (bleed risk = alpha*act - (orig top logit - orig label logit) > 0).
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy'); ev_ = (seq_fold == 0)[si]
zg = np.load('acts/p09_genall_L31.npz'); sig = zg['seq_index']
grows = [json.loads(l) for l in open('work/genall.jsonl')]
gchk = np.where(np.array([grows[s]['split'] == 'check' for s in sig]))[0]
gkind = np.array([grows[s]['kind'] for s in sig])[gchk]
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = sorted(set(first_of.tolist()))
fidx = np.array([FIRST.index(t) for t in first_of]); M = np.zeros((77, len(FIRST))); M[np.arange(77), fidx] = 1
W0 = np.load('work/W0_cv_wneg0.01.npy')
exec(open('work/intent3.py').read().split('tau0 = fit_gate()')[0].split('m, _, si')[0])  # imports only
# recompute tau0 exactly as in intent3 (fit on folds 1-4)
src = open('work/intent3.py').read()
ns = {}
code = src.split('first_of = np.array')[0]
exec(code, ns)
tau0 = ns['tau0']
D = np.load('work/D_write.npy')                      # 4096 x 151 (label tokens), cross-talk nulled
E = E_full()
FJ = [T2J[t] for t in FIRST]
DF = D[:, FJ]                                        # write directions for first tokens
# in-task held-out step 0
e0 = np.where(ev_ & (step == 0))[0]
X0 = X[e0].astype(np.float32); R0 = z['L31.resid'][e0].astype(np.float32)
S0 = X0 @ W0; t0 = X0 @ tau0
Lorig0 = R0 @ E.T                                    # [n, vocab]
# general held-out
XG = zg['L31.mlp_in'][gchk].astype(np.float32); RG = zg['L31.resid'][gchk].astype(np.float32)
SG = XG @ W0; tg = XG @ tau0
EF = E[FIRST]                                        # [43, 4096]
LGf = RG @ EF.T                                      # orig logits of first tokens at general positions
LGtop = np.concatenate([(RG[s:s+2000] @ E.T).max(1) for s in range(0, len(RG), 2000)])
ED = E @ DF                                          # [vocab, 43] boost per unit act on each vocab token
for beta in [1.0, 1.5, 2.0]:
    for th in [0.15]:
        a0 = np.maximum(S0 - th + beta * (t0 - 1)[:, None], 0) * t0[:, None]; mass0 = a0 @ M
        ag = np.maximum(SG - th + beta * (tg - 1)[:, None], 0) * np.maximum(tg, 0)[:, None]; massg = ag @ M
        for alpha in [30, 60, 100, 200]:
            L = Lorig0 + alpha * (mass0 @ ED.T)
            acc = (L.argmax(1) == tgt[e0]).mean()
            # bleed risk at general positions: boost on first token f minus gap to orig top (cross-talk ignored for non-label)
            risk = (alpha * massg + LGf - LGtop[:, None]).max(1)
            nrisk = (risk > 0).sum()
            print(f'beta={beta} th={th} alpha={alpha}: step0 acc incl. orig logits {acc:.4f} | general held-out positions where a label first token '
                  f'would win: {nrisk}/{len(RG)} (max margin {risk.max():.1f}) kinds {dict(zip(*np.unique(gkind[risk>0], return_counts=True)))}', flush=True)
