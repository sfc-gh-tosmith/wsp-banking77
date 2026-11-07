# Hypothesis: a general-text preserve term in the stage-1 (layer-28 mlp_act) read-out shrinks its off-task output (code norm on general
# text) by a large factor while keeping held-out in-task accuracy (step-0 first-token block; step>=1 next-token block).
import sys, gc, json, numpy as np, scipy.linalg as sl
sys.path.insert(0, 'work')
from common import *
L = int(sys.argv[1]) if len(sys.argv) > 1 else 28
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]
FIRST = sorted({l['token_ids'][0] for l in LABS}); F2J = {t: j for j, t in enumerate(FIRST)}
fj = np.array([F2J.get(t, -1) for t in tgt]); gtj = np.array([T2J[t] for t in tgt])
z5 = np.load('acts/p05_practice_L28_30.npz'); Av = z5[f'L{L}.mlp_act'][val].astype(np.float32); del z5; gc.collect()
G0 = np.load(f'work/G{L}_tr_s0.npy'); B0 = np.load(f'work/B{L}_tr_s0.npy')
Gw = np.load(f'work/G{L}_tr_w4.npy'); Bw = np.load(f'work/B{L}_tr_w4.npy')
G1 = Gw - 4 * G0; del Gw; gc.collect()
B1 = Bw.copy()
FT = np.array([T2J[t] for t in FIRST]); B1[:, FT] -= 4 * B0     # remove step-0 contributions
zg = np.load(f'acts/p06_general_L28_30.npz'); sig = zg['seq_index']
rows = [json.loads(l) for l in open('work/general.jsonl')]
fitg = np.array([rows[s]['split'] == 'fit' for s in sig])
Ag = zg[f'L{L}.mlp_act']; d = Ag.shape[1]
Gg = np.zeros((d, d), np.float32); idx = np.where(fitg)[0]
for s in range(0, len(idx), 2048): a = Ag[idx[s:s+2048]].astype(np.float32); Gg += a.T @ a
Agc = Ag[~fitg][::2].astype(np.float32); del Ag, zg; gc.collect()
np.save(f'work/G{L}_gen_fit.npy', Gg)
stepv = step[val]; siv = si[val]
FA = np.array(FIRST)
for blk, G, B, sel, nm in [(0, G0, B0, stepv == 0, 'step0 first-token'), (1, G1, B1, stepv >= 1, 'step>=1 next-token')]:
    sc = np.trace(G) / d; scg = np.trace(Gg) / d
    mu_rel = 3e-3 if blk == 0 else 1e-4
    for lam in [0.0, 0.1, 1.0]:
        M = G + (lam * sc / scg) * Gg; M[np.diag_indices(d)] += mu_rel * sc
        X = sl.cho_solve(sl.cho_factor(M, overwrite_a=True), B); del M; gc.collect()
        S = Av[sel] @ X; Sg = Agc @ X
        y = fj[val][sel] if blk == 0 else gtj[val][sel]
        acc = (S.argmax(1) == y).mean()
        gold = S[np.arange(len(y)), y]
        print(f'L{L} {nm} lam={lam}: held-out acc {acc:.4f} | in-task gold score med {np.median(gold):.3f} | in-task code norm med {np.median(np.linalg.norm(S,axis=1)):.3f} '
              f'| general code norm med {np.median(np.linalg.norm(Sg,axis=1)):.3f} p99 {np.percentile(np.linalg.norm(Sg,axis=1),99):.3f}', flush=True)
        np.save(f'work/X{L}_blk{blk}_lam{lam:g}_cv.npy', X)
