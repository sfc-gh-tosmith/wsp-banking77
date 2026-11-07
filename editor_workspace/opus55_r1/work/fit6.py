# Hypothesis: accuracy keeps rising at smaller ridge strengths; sweep mu via one eigendecomposition per Gram (L28, L30; joint-w4 and step0-only).
import sys, gc, pickle, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]
gtj = np.array([T2J[t] for t in tgt]); stepv = step[val]; siv = si[val]; gv = gtj[val]
FIRST = sorted({l['token_ids'][0] for l in LABS}); FIRST_a = np.array(FIRST)
z5 = np.load('acts/p05_practice_L28_30.npz')
best = {}
for L in [28, 30]:
    Av = z5[f'L{L}.mlp_act'][val].astype(np.float32)
    for kind in ['w4', 's0']:
        G = np.load(f'work/G{L}_tr_{kind}.npy'); B = np.load(f'work/B{L}_tr_{kind}.npy'); sc = np.trace(G) / len(G)
        ev, V = np.linalg.eigh(G); del G; gc.collect()
        VB = V.T @ B
        AV = (Av if kind == 'w4' else Av[stepv == 0]) @ V
        for mu_rel in [1e-4, 3e-4, 1e-3, 3e-3]:
            S = AV @ (VB / (ev + mu_rel * sc)[:, None])
            if kind == 'w4':
                ok = S.argmax(1) == gv; sq = np.ones(nseq, bool); sq[siv[~ok]] = False
                print(f'L{L} joint-w4 mu_rel={mu_rel:g}: s0 {ok[stepv==0].mean():.4f} s1+ {ok[stepv>=1].mean():.4f} SEQ {sq[val_seq].mean():.4f}', flush=True)
            else:
                print(f'L{L} step0-only mu_rel={mu_rel:g}: s0 {(FIRST_a[S.argmax(1)] == tgt[val][stepv==0]).mean():.4f}', flush=True)
            best[(L, kind, mu_rel)] = S
        del V, AV; gc.collect()
    del Av; gc.collect()
pickle.dump(best, open('work/val_scores_sweep.pkl', 'wb'))
for mu_rel in [3e-4, 1e-3, 3e-3]:
    S0 = best[(28, 's0', mu_rel)] + best[(30, 's0', mu_rel)]
    S = best[(28, 'w4', mu_rel)] + best[(30, 'w4', mu_rel)]
    ok = S.argmax(1) == gv
    print(f'ensemble mu_rel={mu_rel:g}: step0-only s0 {(FIRST_a[S0.argmax(1)] == tgt[val][stepv==0]).mean():.4f} | joint s1+ {ok[stepv>=1].mean():.4f}')
