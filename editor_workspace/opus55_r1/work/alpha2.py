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
DU = DF.sum(1)                                        # uniform write onto all first tokens
EDU = E @ DU
print('tau0 held-out in-task step0 min %.3f p1 %.3f | general check max %.3f' % (t0.min(), np.percentile(t0, 1), tg.max()))
for th0 in [0.5]:
    u0 = np.maximum(t0 - th0, 0); ug = np.maximum(tg - th0, 0)
    print('uniform unit act: in-task median %.3f min %.3f | general max %.3f' % (np.median(u0), u0.min(), ug.max()))
    for U in [0.0, 20.0, 40.0]:
        for beta in [1.0, 1.5]:
            th = 0.15
            a0 = np.maximum(S0 - th + beta * (t0 - 1)[:, None], 0) * t0[:, None]; mass0 = a0 @ M
            ag = np.maximum(SG - th + beta * (tg - 1)[:, None], 0) * np.maximum(tg, 0)[:, None]; massg = ag @ M
            for alpha in [100, 200]:
                L = Lorig0 + alpha * (mass0 @ ED.T) + (U / np.median(u0)) * u0[:, None] * EDU[None, :]
                acc = (L.argmax(1) == tgt[e0]).mean()
                risk = (alpha * massg + (U / np.median(u0)) * ug[:, None] + LGf - LGtop[:, None]).max(1)
                print(f'U={U} beta={beta} alpha={alpha}: step0 acc {acc:.4f} | general risk positions {(risk>0).sum()}/{len(RG)} max {risk.max():.1f}', flush=True)
