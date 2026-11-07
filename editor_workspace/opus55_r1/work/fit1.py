# Hypothesis: a ridge read-out from layer-31 mlp_act to label-token scores (balanced classes), written through D = pinv(E_label),
# makes greedy decoding emit the gold label on held-out practice; measure per-step and whole-label accuracy vs ridge strength.
import sys, json, gc, numpy as np
sys.path.insert(0, 'work')
from common import *

m, z, si, step, ycls, tgt = records('p03_practice_L31')
N = len(si); nseq = si.max() + 1
print('records', N, 'seqs', nseq, flush=True)
Rr = z['L31.resid']
topk = z['topk_ids']
rms = np.sqrt((Rr.astype(np.float32) ** 2).mean(1))
for k in range(4):
    print(f'step{k} rms median {np.median(rms[step==k]):.3f}  frac orig top1==gold {(topk[step==k,0]==tgt[step==k]).mean():.3f}')
np.save('work/p03_rms.npy', rms)

rng = np.random.default_rng(0)
seq_fold = rng.permutation(nseq) % 5
np.save('work/p03_seqfold.npy', seq_fold)
val_seq = seq_fold == 0
val = val_seq[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]
ncls = np.bincount(cls_seq[~val_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls].astype(np.float32)
gtj_all = np.array([T2J[t] for t in tgt])

A = z['L31.mlp_act']
d = A.shape[1]
G = np.zeros((d, d), np.float32); B = np.zeros((d, len(TOK)), np.float32)
idx = np.where(tr)[0]
for s in range(0, len(idx), 2048):
    j = idx[s:s + 2048]; a = A[j].astype(np.float32); aw = a * w[j][:, None]
    G += aw.T @ a
    np.add.at(B.T, gtj_all[j], aw)
print('gram done', flush=True)
Av = A[val].astype(np.float32)
del A; gc.collect()
np.save('work/G_tr.npy', G); np.save('work/B_tr.npy', B)
evals, V = np.linalg.eigh(G); del G; gc.collect()
print('eig done; top eig %.3g median %.3g' % (evals[-1], np.median(evals)), flush=True)
np.save('work/G_tr_evals.npy', evals); np.save('work/G_tr_V.npy', V)
VB = V.T @ B
AvV = Av @ V; del Av; gc.collect()

stepv = step[val]; gtj = gtj_all[val]; siv = si[val]; Rv = Rr[val].astype(np.float32)
ET = E_rows(TOK)
D = np.linalg.pinv(ET)                    # 4096 x 151, ET @ D = I
cand = topk[val]
islab = np.isin(cand, TOK)
Ecand = [E_rows(cand[:, k]) for k in range(cand.shape[1])]
for mu in [c * evals.mean() for c in (1e-3, 1e-2, 3e-2, 1e-1)]:
    S = AvV @ (VB / (evals + mu)[:, None])  # val scores [n,151]
    for beta in [10, 20, 40]:
        alpha = beta * np.median(rms[tr])
        R2 = Rv + alpha * (S @ D.T)
        Lt = R2 @ ET.T
        ok_lab = Lt.argmax(1) == gtj
        Lc = np.stack([np.einsum('nd,nd->n', R2, Ecand[k]) for k in range(cand.shape[1])], 1)
        Lc[islab] = -np.inf
        ok = ok_lab & (Lt[np.arange(len(Rv)), gtj] > Lc.max(1))
        seq_ok = np.ones(nseq, bool); seq_ok[siv[~ok]] = False
        per = ' '.join(f's{k}:{ok[stepv==k].mean():.3f}' for k in range(3))
        print(f'mu={mu:g} beta={beta}: lab-argmax s0 {ok_lab[stepv==0].mean():.3f} | ok {per} | SEQ {seq_ok[val_seq].mean():.4f}', flush=True)
