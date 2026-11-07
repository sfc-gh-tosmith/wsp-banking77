# Hypothesis: a ridge code V (14336 -> 77) from L28 mlp_act, fit with one-hot targets at in-task step 0 and zero targets on general text,
# keeps held-out step-0 first-token accuracy ~.93 while staying small (|code| << in-task) on held-out general positions -> usable as an
# ungated stage-1 write in L28 down_proj. Primal ridge in float32 Gram (memory).
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
m, _, si, step, ycls, tgt = records('p03_practice_L31')
seq_fold = np.load('work/p03_seqfold.npy'); sf = seq_fold[si]
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = np.array(sorted(set(first_of.tolist())))
M = np.zeros((77, len(FIRST))); M[np.arange(77), [list(FIRST).index(t) for t in first_of]] = 1
i0 = np.where(step == 0)[0]; tr0 = i0[sf[i0] != 0]; te0 = i0[sf[i0] == 0]
iN = np.where(step >= 1)[0]; trN = iN[sf[iN] != 0][::3]; teN = iN[sf[iN] == 0][::3]
A = np.load('acts/p05_practice_L28_30.npz')['L28.mlp_act']
d = A.shape[1]
w = 1 / np.bincount(ycls[tr0], minlength=77)[ycls[tr0]]; w = w / w.sum()
Gp = np.zeros((d, d), np.float32); Bp = np.zeros((d, 77), np.float32)
for s in range(0, len(tr0), 2048):
    j = tr0[s:s+2048]; a = A[j].astype(np.float32); aw = a * w[s:s+2048, None]; Gp += aw.T @ a; np.add.at(Bp.T, ycls[j], aw)
Gn = np.zeros((d, d), np.float32)
for s in range(0, len(trN), 2048):
    a = A[trN[s:s+2048]].astype(np.float32); Gn += 0.5 * (a.T @ a) / len(trN)
Ate0 = A[te0].astype(np.float32); AteN = A[teN].astype(np.float32); del A; gc.collect()
zg = np.load('acts/p06_general_L28_30.npz'); sig = zg['seq_index']; Ag = zg['L28.mlp_act']
grows = [json.loads(l) for l in open('work/general.jsonl')]
gfit = np.array([grows[s]['split'] == 'fit' for s in sig]); gk = np.array([grows[s]['kind'] for s in sig])
ig = np.where(gfit)[0]
for s in range(0, len(ig), 2048):
    a = Ag[ig[s:s+2048]].astype(np.float32); Gn += 0.5 * (a.T @ a) / len(ig)
Agc = Ag[np.where(~gfit)[0][::2]].astype(np.float32); del Ag, zg; gc.collect()
tr_ = np.trace(Gp)
for wz in [0.0, 0.1, 0.3, 1.0]:
    for mu in [1e-3, 3e-3]:
        G = Gp + wz * Gn; G[np.diag_indices(d)] += mu * np.trace(G) / d
        V = np.linalg.solve(G, Bp); del G; gc.collect()
        S = Ate0 @ V; fa = (FIRST[(np.maximum(S - 0.15, 0) @ M).argmax(1)] == tgt[te0]).mean()
        gold = S[np.arange(len(te0)), ycls[te0]]
        nin = np.linalg.norm(S, axis=1); nN = np.linalg.norm(AteN @ V, axis=1); ng = np.linalg.norm(Agc @ V, axis=1)
        print(f'wz={wz} mu={mu}: step0 class acc {(S.argmax(1)==ycls[te0]).mean():.4f} first {fa:.4f} | gold score median {np.median(gold):.3f} | '
              f'|code| in-task step0 median {np.median(nin):.3f}, steps>=1 median {np.median(nN):.3f}, general-check median {np.median(ng):.3f} p99 {np.percentile(ng,99):.3f} max {ng.max():.3f}', flush=True)
        if wz == 0.3 and mu == 1e-3: np.save('work/V28_cv.npy', V)
