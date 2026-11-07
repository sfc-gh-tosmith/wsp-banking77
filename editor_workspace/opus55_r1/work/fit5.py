# Hypothesis: layer-30 (and 28) mlp_act read-outs classify about as well as layer 31's; a step-0-specialised read-out beats the joint
# one at step 0; averaging layer 28 and 30 scores adds accuracy. Intrinsic (argmax of scores) held-out practice accuracy.
import sys, gc, numpy as np, scipy.linalg as sl
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]; tr = ~val
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[~val_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls].astype(np.float32)
gtj = np.array([T2J[t] for t in tgt]); stepv = step[val]; siv = si[val]; gv = gtj[val]
z5 = np.load('acts/p05_practice_L28_30.npz')
assert (z5['seq_index'] == si).all()
FIRST = sorted({l['token_ids'][0] for l in LABS}); F2J = {t: j for j, t in enumerate(FIRST)}
first_j = np.array([F2J.get(t, -1) for t in tgt])
scores = {}
for L in [30, 28]:
    A = z5[f'L{L}.mlp_act']; d = A.shape[1]
    G = np.zeros((d, d), np.float32); B = np.zeros((d, len(TOK)), np.float32)
    G0 = np.zeros((d, d), np.float32); B0 = np.zeros((d, len(FIRST)), np.float32)
    idx = np.where(tr)[0]
    for s in range(0, len(idx), 2048):
        j = idx[s:s + 2048]; a = A[j].astype(np.float32)
        ww = w[j] * np.where(step[j] == 0, 4.0, 1.0); aw = a * ww[:, None]
        G += aw.T @ a; np.add.at(B.T, gtj[j], aw)
        k0 = step[j] == 0; a0 = a[k0] * w[j][k0][:, None]
        G0 += a0.T @ a[k0]; np.add.at(B0.T, first_j[j][k0], a0)
    Av = A[val].astype(np.float32); del A; gc.collect()
    np.save(f'work/G{L}_tr_w4.npy', G); np.save(f'work/B{L}_tr_w4.npy', B); np.save(f'work/G{L}_tr_s0.npy', G0); np.save(f'work/B{L}_tr_s0.npy', B0)
    sc = np.trace(G) / d
    for mu_rel in [3e-3, 1e-2, 3e-2]:
        M = G.copy(); M[np.diag_indices(d)] += mu_rel * sc
        X = sl.cho_solve(sl.cho_factor(M, overwrite_a=True), B); del M; gc.collect()
        S = Av @ X; ok = S.argmax(1) == gv; sq = np.ones(nseq, bool); sq[siv[~ok]] = False
        print(f'L{L} joint(w4) mu_rel={mu_rel}: s0 {ok[stepv==0].mean():.4f} s1+ {ok[stepv>=1].mean():.4f} SEQ {sq[val_seq].mean():.4f}', flush=True)
        scores[(L, 'joint', mu_rel)] = S
    sc0 = np.trace(G0) / d
    for mu_rel in [1e-2, 3e-2, 1e-1]:
        M = G0.copy(); M[np.diag_indices(d)] += mu_rel * sc0
        X0 = sl.cho_solve(sl.cho_factor(M, overwrite_a=True), B0); del M; gc.collect()
        S0 = Av[stepv == 0] @ X0
        print(f'L{L} step0-only mu_rel={mu_rel}: s0 {(np.array(FIRST)[S0.argmax(1)] == tgt[val][stepv==0]).mean():.4f}', flush=True)
        scores[(L, 's0', mu_rel)] = S0
    del G, G0, Av; gc.collect()
np.save('work/val_scores_L28_30.npy', np.array([0]))
import pickle; pickle.dump(scores, open('work/val_scores_L28_30.pkl', 'wb'))
# ensembles
for k in [3e-3, 1e-2, 3e-2]:
    S = scores[(30, 'joint', k)] + scores[(28, 'joint', k)]
    ok = S.argmax(1) == gv; sq = np.ones(nseq, bool); sq[siv[~ok]] = False
    print(f'ensemble L28+L30 joint mu_rel={k}: s0 {ok[stepv==0].mean():.4f} SEQ {sq[val_seq].mean():.4f}')
for k in [1e-2, 3e-2, 1e-1]:
    S0 = scores[(30, 's0', k)] + scores[(28, 's0', k)]
    print(f'ensemble L28+L30 step0-only mu_rel={k}: s0 {(np.array(FIRST)[S0.argmax(1)] == tgt[val][stepv==0]).mean():.4f}')
