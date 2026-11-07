# Hypothesis: a single-stage layer-31 MLP edit -- 77 intent neurons (gate = kA*(class score - th + beta*(task0 - 1)), up = bias dir,
# down = alphaA * write(first token)) + token neurons for steps>=1 (gate = kB*(task1 - th1), up = next-token read-out, down = alphaB *
# write(token)) -- classifies held-out practice like the read-outs, is OFF (gate<0) on every held-out general position, and dominates
# the original logits in-task. Usage: kpython build_edit.py cv|final
import sys, gc, json, collections, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
MODE = sys.argv[1]
P = dict(mu=1e-4, th=0.2, beta=2.0, th1=0.2, kA=12.0, kB=40.0, alphaA=None, alphaB=None, mu_gate=1e-5)
for kv in sys.argv[2:]:
    k, v = kv.split('='); P[k] = float(v)
m, _, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
assert len(X) == len(si)
seq_fold = np.load('work/p03_seqfold.npy')
fit_seq = (seq_fold != 0) if MODE == 'cv' else np.ones(nseq, bool)
tr = fit_seq[si]; ev_ = (~fit_seq)[si] if MODE == 'cv' else np.zeros(len(si), bool)
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[fit_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls]
zg = np.load('acts/p09_genall_L31.npz'); sig = zg['seq_index']; Xg = zg['L31.mlp_in']
grows = [json.loads(l) for l in open('work/genall.jsonl')]
gfit = np.array([grows[s]['split'] == 'fit' for s in sig]); gkind = np.array([grows[s]['kind'] for s in sig])
d = 4096
def ridge(rows_idx, Y, wts, mu_rel, Xs=X):
    G = np.zeros((d, d)); B = np.zeros((d, Y.shape[1]))
    for s in range(0, len(rows_idx), 4096):
        j = rows_idx[s:s+4096]; a = Xs[j].astype(np.float64); aw = a * wts[s:s+4096, None]
        G += aw.T @ a; B += aw.T @ Y[s:s+4096]
    G[np.diag_indices(d)] += mu_rel * np.trace(G) / d
    return np.linalg.solve(G, B).astype(np.float32)
# ---- gates (bias-free ridge, +1 on target in-task positions, -1 on other in-task step type and general fit positions)
def fit_gate(pos_mask):
    ip = np.where(tr & pos_mask)[0]; ineg = np.where(tr & ~pos_mask & (step >= 0))[0]; ig = np.where(gfit)[0]
    nm = (gkind[ig] == 'nearmiss')
    G = np.zeros((d, d)); B = np.zeros(d)
    for idx, src, t, wt in [(ip, X, 1.0, np.full(len(ip), 1.0 / len(ip))), (ineg, X, -1.0, np.full(len(ineg), 0.5 / len(ineg))),
                            (ig, Xg, -1.0, np.where(nm, 0.5 / nm.sum(), 0.5 / (~nm).sum()))]:
        for s in range(0, len(idx), 4096):
            a = src[idx[s:s+4096]].astype(np.float64); aw = a * wt[s:s+4096, None]; G += aw.T @ a; B += t * aw.sum(0)
    G[np.diag_indices(d)] += P['mu_gate'] * np.trace(G) / d
    return np.linalg.solve(G, B).astype(np.float32)
tau0 = fit_gate(step == 0); tau1 = fit_gate(step >= 1)
# bias direction: b.x ~= 1 on in-task reply positions and general positions
ib = np.where(tr)[0][::3]; igb = np.where(gfit)[0]
Gb = np.zeros((d, d)); Bb = np.zeros(d)
for idx, src in [(ib, X), (igb, Xg)]:
    for s in range(0, len(idx), 4096):
        a = src[idx[s:s+4096]].astype(np.float64); Gb += a.T @ a; Bb += a.sum(0)
Gb[np.diag_indices(d)] += 1e-6 * np.trace(Gb) / d
bvec = np.linalg.solve(Gb, Bb).astype(np.float32)
# ---- read-outs
i0 = np.where(tr & (step == 0))[0]; Yc = np.zeros((len(i0), 77)); Yc[np.arange(len(i0)), ycls[i0]] = 1
W0 = ridge(i0, Yc, w[i0], P['mu'])
LATER = sorted({t for l in LABS for t in l['token_ids'][1:]}); L2J = {t: j for j, t in enumerate(LATER)}
i1 = np.where(tr & (step >= 1))[0]; Yt = np.zeros((len(i1), len(LATER))); Yt[np.arange(len(i1)), [L2J[t] for t in tgt[i1]]] = 1
W1 = ridge(i1, Yt, w[i1], P['mu'])
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)])
FIRST = sorted(set(first_of.tolist()))
print(f'{MODE}: {len(LATER)} later tokens, {len(FIRST)} first tokens -> neurons {77 + len(LATER)}', flush=True)
# ---- diagnostics: gate pre-activations (linear read-outs) on held-out in-task and held-out general
def gA(Xs):   # [n,77] intent gate pre-activation / kA
    Xs = Xs.astype(np.float32); return Xs @ W0 - P['th'] * (Xs @ bvec)[:, None] + P['beta'] * ((Xs @ tau0) - (Xs @ bvec))[:, None]
def gB(Xs):
    Xs = Xs.astype(np.float32); return Xs @ tau1 - P['th1'] * (Xs @ bvec)
gchk = np.where(~gfit)[0]
zA_g = np.concatenate([gA(Xg[gchk[s:s+4096]]).max(1) for s in range(0, len(gchk), 4096)])
zB_g = np.concatenate([gB(Xg[gchk[s:s+4096]]) for s in range(0, len(gchk), 4096)])
bg = Xg[gchk].astype(np.float32) @ bvec
print(f'bias dir b.x: general-check median {np.median(bg):.3f} [p1 {np.percentile(bg,1):.3f}, p99 {np.percentile(bg,99):.3f}]')
print(f'general-check: max intent gate z {zA_g.max():.3f} (p99.9 {np.percentile(zA_g,99.9):.3f}) | max token gate z {zB_g.max():.3f} (p99.9 {np.percentile(zB_g,99.9):.3f})',
      ' by kind A:', {k: round(float(zA_g[gkind[gchk]==k].max()), 2) for k in np.unique(gkind)}, ' B:', {k: round(float(zB_g[gkind[gchk]==k].max()), 2) for k in np.unique(gkind)})
if MODE == 'cv':
    e0 = np.where(ev_ & (step == 0))[0]; e1 = np.where(ev_ & (step >= 1))[0]
    Z0 = gA(X[e0]); Z1A = gA(X[e1]).max(1); ZB0 = gB(X[e0]); ZB1 = gB(X[e1])
    print(f'held-out in-task: intent gates at step>=1 max {Z1A.max():.3f} | token gate at step0 max {ZB0.max():.3f} | token gate at step>=1 min {ZB1.min():.3f} p0.1 {np.percentile(ZB1,0.1):.3f}')
    fidx = np.array([FIRST.index(t) for t in first_of]); M = np.zeros((77, len(FIRST))); M[np.arange(77), fidx] = 1
    mass = np.maximum(Z0, 0) @ M
    ok0 = np.array(FIRST)[mass.argmax(1)] == tgt[e0]
    nofire = mass.max(1) <= 0
    gold_z = Z0[np.arange(len(e0)), ycls[e0]]
    print(f'step0 first-token acc {ok0.mean():.4f} | no intent neuron fires {nofire.mean():.4f} | gold z median {np.median(gold_z):.3f} p10 {np.percentile(gold_z,10):.3f}')
    S1 = X[e1].astype(np.float32) @ W1; ok1 = np.array(LATER)[S1.argmax(1)] == tgt[e1]
    ok = np.ones(len(si), bool); ok[e0] = ok0; ok[e1] = ok1
    sq = np.ones(nseq, bool); sq[si[ev_ & ~ok]] = False
    accs = sq[~fit_seq]; ycl = cls_seq[~fit_seq]
    print(f'HELD-OUT WHOLE-LABEL acc {accs.mean():.4f} | class-balanced {np.mean([accs[ycl==c].mean() for c in range(77)]):.4f}', flush=True)
    np.save('work/cv_pred_ok.npy', accs)
if MODE == 'final':
    np.savez('work/final_readouts.npz', tau0=tau0, tau1=tau1, bvec=bvec, W0=W0, W1=W1, LATER=np.array(LATER), params=json.dumps(P))
    print('saved work/final_readouts.npz')
