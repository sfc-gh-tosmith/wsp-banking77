# Hypothesis: the full single-stage layer-31 edit (77 intent neurons + 1 task-on neuron at step 0, 110 next-token neurons at steps>=1,
# all task-gated; written onto cross-talk-nulled label-token directions) reaches held-out whole-label accuracy ~.90 including the
# original logits, with no held-out general position at risk. Usage: kpython build_final.py cv|final [key=val ...]
import sys, gc, json, collections, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
MODE = sys.argv[1]
P = dict(mu=1e-4, wneg=0.01, th=0.15, beta=1.5, alphaA=250.0, U=10.0, thU=0.5, th1=0.2, alphaB=250.0, kA=20.0, kB=50.0, mu_gate=1e-5, bw=1.0, gamma=0.0, fold=0)
for kv in sys.argv[2:]:
    k, v = kv.split('='); P[k] = float(v)
print(MODE, P, flush=True)
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
X = np.load('acts/p10_practice_L31_mlpin.npz')['L31.mlp_in']
seq_fold = np.load('work/p03_seqfold.npy')
fit_seq = (seq_fold != int(P['fold'])) if MODE == 'cv' else np.ones(nseq, bool)
tr = fit_seq[si]; ev_ = ~tr
cls_seq = ycls[np.r_[True, si[1:] != si[:-1]]]; ncls = np.bincount(cls_seq[fit_seq], minlength=77)
w = (ncls.mean() / ncls)[ycls]
zg = np.load('acts/p09_genall_L31.npz'); sig = zg['seq_index']; Xg = zg['L31.mlp_in']
grows = [json.loads(l) for l in open('work/genall.jsonl')]
gfit = np.array([grows[s]['split'] == 'fit' for s in sig]) if MODE == 'cv' else np.ones(len(sig), bool)
gkind = np.array([grows[s]['kind'] for s in sig])
d = 4096
def acc_gram(parts, nout):
    G = np.zeros((d, d)); B = np.zeros((d, nout))
    for idx, src, wt, Yfn in parts:
        for s in range(0, len(idx), 4096):
            j = idx[s:s+4096]; a = src[j].astype(np.float64); aw = a * wt[s:s+4096, None]; G += aw.T @ a; B += aw.T @ Yfn(j)
    return G, B
def solve(G, B, mu_rel):
    G = G.copy(); G[np.diag_indices(d)] += mu_rel * np.trace(G) / d; return np.linalg.solve(G, B).astype(np.float32)
ig = np.where(gfit)[0]; nm = gkind[ig] == 'nearmiss'
wg_gate = np.where(nm, 0.5 / nm.sum(), 0.5 / (~nm).sum())
def gate_vec(pos):
    ip = np.where(tr & pos)[0]; ineg = np.where(tr & ~pos)[0]
    G, B = acc_gram([(ip, X, np.full(len(ip), 1 / len(ip)), lambda j: np.ones((len(j), 1))),
                     (ineg, X, np.full(len(ineg), 0.5 / len(ineg)), lambda j: -np.ones((len(j), 1))),
                     (ig, Xg, wg_gate, lambda j: -np.ones((len(j), 1)))], 1)
    return solve(G, B, P['mu_gate'])[:, 0]
tau0 = gate_vec(step == 0); tau1 = gate_vec(step >= 1)
ib = np.where(tr)[0][::3]
G, B = acc_gram([(ib, X, np.ones(len(ib)), lambda j: np.ones((len(j), 1))), (ig, Xg, np.ones(len(ig)), lambda j: np.ones((len(j), 1)))], 1)
bvec = solve(G, B, 1e-6)[:, 0]
i0 = np.where(tr & (step == 0))[0]; inn = np.where(tr & (step >= 1))[0][::2]
def onehot_cls(j): Y = np.zeros((len(j), 77)); Y[np.arange(len(j)), ycls[j]] = 1; return Y
Gp, Bp = acc_gram([(i0, X, w[i0] / w[i0].sum(), onehot_cls)], 77)
Gn, Bn = acc_gram([(inn, X, np.full(len(inn), 0.5 / len(inn)), lambda j: -np.ones((len(j), 77))),
                   (ig, Xg, np.full(len(ig), 0.5 / len(ig)), lambda j: -np.ones((len(j), 77)))], 77)
W0 = solve(Gp + P['wneg'] * Gn, Bp + P['wneg'] * Bn, P['mu']); del Gp, Gn; gc.collect()
LATER = sorted({t for l in LABS for t in l['token_ids'][1:]}); L2J = {t: j for j, t in enumerate(LATER)}
i1 = np.where(tr & (step >= 1))[0]
CH = collections.defaultdict(set)
for l in LABS:
    for j in range(len(l['token_ids'])): CH[tuple(l['token_ids'][:j])].add(l['token_ids'][j])
PRE = [tuple(LABS[c]['token_ids'][:k]) for c, k in zip(ycls, step)]
isbr = np.array([len(CH[p]) > 1 for p in PRE])
def onehot_later(j):
    Y = np.zeros((len(j), len(LATER))); Y[np.arange(len(j)), [L2J[t] for t in tgt[j]]] = 1
    for r, i in enumerate(j):
        for t in CH[PRE[i]]: Y[r, L2J[t]] += P['gamma']
    return Y
G, B = acc_gram([(i1, X, w[i1] * np.where(isbr[i1], P['bw'], 1.0), onehot_later)], len(LATER)); W1 = solve(G, B, P['mu']); del G; gc.collect()
first_of = np.array([LABS[c]['token_ids'][0] for c in range(77)]); FIRST = sorted(set(first_of.tolist()))
D = np.load('work/D_write.npy'); DF = D[:, [T2J[t] for t in FIRST]]; DL = D[:, [T2J[t] for t in LATER]]
M = np.zeros((77, len(FIRST))); M[np.arange(77), [FIRST.index(t) for t in first_of]] = 1
t0_tr = X[i0].astype(np.float32) @ tau0; un = np.median(np.maximum(t0_tr - P['thU'], 0))
# ---- neuron weights (gate rows, up rows in mlp_in space; down columns in residual space)
nA = 77; nT = len(LATER); N = nA + 1 + nT
Wg = np.zeros((N, d), np.float32); Wu = np.zeros((N, d), np.float32); Wdn = np.zeros((d, N), np.float32)
for c in range(77):
    Wg[c] = P['kA'] * (W0[:, c] - (P['th'] + P['beta']) * bvec + P['beta'] * tau0)
    Wu[c] = tau0 / P['kA']
    Wdn[:, c] = P['alphaA'] * DF[:, FIRST.index(first_of[c])]
Wg[nA] = P['kA'] * (tau0 - P['thU'] * bvec); Wu[nA] = bvec / P['kA'] / un; Wdn[:, nA] = P['U'] * DF.sum(1)
for k in range(nT):
    Wg[nA + 1 + k] = P['kB'] * (tau1 - P['th1'] * bvec)
    Wu[nA + 1 + k] = W1[:, k] / (P['kB'] * (1 - P['th1'])) / (1 + P['gamma'])
    Wdn[:, nA + 1 + k] = P['alphaB'] * DL[:, k]
print(f'neurons: {N} | un={un:.3f}', flush=True)
if MODE == 'cv':
    # decision-rule evaluation with relu in place of silu (gate scales kA,kB are large), including original logits
    E = E_full(); Rall = z['L31.resid']; RGall = zg['L31.resid']
    ev_idx = np.where(ev_)[0]
    ok = np.zeros(len(si), bool)
    for s in range(0, len(ev_idx), 800):
        j = ev_idx[s:s+800]; Xs = X[j].astype(np.float32); Rs = Rall[j].astype(np.float32)
        H = np.maximum(Xs @ Wg.T, 0) * (Xs @ Wu.T)
        L = (Rs + H @ Wdn.T) @ E.T
        ok[j] = L.argmax(1) == tgt[j]
    sq = np.ones(nseq, bool); sq[si[ev_ & ~ok]] = False
    accs = sq[~fit_seq]; ycl = cls_seq[~fit_seq]
    print(f'held-out per-step: step0 {ok[ev_&(step==0)].mean():.4f} step>=1 {ok[ev_&(step>=1)].mean():.4f} | WHOLE-LABEL {accs.mean():.4f} '
          f'class-balanced {np.mean([accs[ycl==c].mean() for c in range(77)]):.4f}', flush=True)
    gchk = np.where(~gfit)[0]; risk = 0; worst = -1e9; kinds = []
    for s in range(0, len(gchk), 800):
        j = gchk[s:s+800]; Xs = Xg[j].astype(np.float32); Rs = RGall[j].astype(np.float32)
        H = np.maximum(Xs @ Wg.T, 0) * (Xs @ Wu.T)
        b0 = (Rs @ E.T).argmax(1); b1 = ((Rs + H @ Wdn.T) @ E.T).argmax(1)
        fl = b0 != b1; risk += fl.sum(); kinds += list(gkind[j][fl])
        worst = max(worst, np.abs(H).max())
    print(f'held-out general positions whose greedy token changes (new neurons only): {risk}/{len(gchk)} {dict(zip(*np.unique(kinds, return_counts=True))) if kinds else {}} | max |neuron act| {worst:.4f}')
else:
    np.savez('work/neurons_final.npz', Wg=Wg, Wu=Wu, Wdn=Wdn, tau0=tau0, tau1=tau1, bvec=bvec, W0=W0, W1=W1, LATER=np.array(LATER),
             FIRST=np.array(FIRST), params=json.dumps(P))
    print('saved work/neurons_final.npz')
