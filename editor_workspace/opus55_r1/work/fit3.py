# Hypothesis: adding general-text layer-31 activations as a zero-target (preserve) term keeps the read-out silent outside the task
# (no greedy-token flips on held-out general replies) at little cost to held-out in-task accuracy.
import sys, gc, json, numpy as np, scipy.linalg as sl
sys.path.insert(0, 'work')
from common import *

# ---------- in-task held-out data
m, z, si, step, ycls, tgt = records('p03_practice_L31')
nseq = si.max() + 1
rms_in = np.load('work/p03_rms.npy')
seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]
Av = z['L31.mlp_act'][val].astype(np.float32)
Rv = z['L31.resid'][val].astype(np.float32); cand = z['topk_ids'][val]
gtj = np.array([T2J[t] for t in tgt[val]]); stepv = step[val]; siv = si[val]
del z; gc.collect()

# ---------- general data
zg = np.load('acts/p04_general_L31.npz'); sig = zg['seq_index']
rows = [json.loads(l) for l in open('work/general.jsonl')]
tc = json.load(open('work/general_tokcounts.json'))
cnt = np.bincount(sig); start = np.r_[0, np.cumsum(cnt)[:-1]]
reply = np.zeros(len(sig), bool)
for s in range(len(cnt)):
    reply[start[s] + cnt[s] - (tc['n_comp'][s] + 1): start[s] + cnt[s]] = True
fit_g = np.array([rows[s]['split'] == 'fit' for s in sig])
kind_g = np.array([rows[s]['kind'] for s in sig])
Ag = zg['L31.mlp_act']
d = Ag.shape[1]
Gg = np.zeros((d, d), np.float32)
idx = np.where(fit_g)[0]
for s0 in range(0, len(idx), 2048):
    a = Ag[idx[s0:s0 + 2048]].astype(np.float32); Gg += a.T @ a
np.save('work/G_gen_fit.npy', Gg)
chk = (~fit_g) & reply
Ac = Ag[chk].astype(np.float32); Rc = zg['L31.resid'][chk].astype(np.float32)
top1c = zg['topk_ids'][chk][:, 0]; candc = zg['topk_ids'][chk]; kindc = kind_g[chk]; sigc = sig[chk]
del Ag, zg; gc.collect()
print('in-task val records', len(Av), '| general fit records', fit_g.sum(), '| general check reply records', chk.sum(), flush=True)

ET = E_rows(TOK); D = np.linalg.pinv(ET)
def cand_pre(R, cand):
    u, inv = np.unique(cand, return_inverse=True); inv = inv.reshape(cand.shape)
    Eu = E_rows(u); L0 = np.stack([np.einsum('nd,nd->n', R, Eu[inv[:, k]]) for k in range(cand.shape[1])], 1)
    return L0, inv, Eu @ D
L0v, invv, DEv = cand_pre(Rv, cand); islab = np.isin(cand, TOK)
L0c, invc, DEc = cand_pre(Rc, candc)
LTv = Rv @ ET.T; LTc = Rc @ ET.T
alpha0 = np.median(rms_in)

def in_task_acc(S, beta):
    g = beta * alpha0
    Lt = LTv + g * S
    Lc = L0v + g * np.einsum('nj,nkj->nk', S, DEv[invv]); Lc[islab] = -np.inf
    ok = (Lt.argmax(1) == gtj) & (Lt[np.arange(len(Lt)), gtj] > Lc.max(1))
    sq = np.ones(nseq, bool); sq[siv[~ok]] = False
    return ok[stepv == 0].mean(), sq[val_seq].mean()

def general_flips(S, beta):
    g = beta * alpha0
    Lt = LTc + g * S
    Lc = L0c + g * np.einsum('nj,nkj->nk', S, DEc[invc])
    allL = np.concatenate([Lc, Lt], 1); ids = np.concatenate([candc, np.tile(TOK, (len(Lt), 1))], 1)
    new = ids[np.arange(len(Lt)), allL.argmax(1)]
    flip = new != top1c
    lab_new = np.isin(new, TOK) & ~np.isin(top1c, TOK)
    # first-flip per sequence matters most
    return flip.mean(), lab_new.mean(), len(np.unique(sigc[flip])), {k: flip[kindc == k].mean().round(4) for k in np.unique(kindc)}

G = np.load('work/G_tr.npy'); B = np.load('work/B_tr.npy')
w_in = np.trace(G); w_g = np.trace(Gg)
print('trace in-task %.3g general %.3g' % (w_in, w_g), flush=True)
mu = 0.3
for lam in [0.0, 1.0, 3.0, 10.0]:
    M = G + (lam * w_in / w_g) * Gg; M[np.diag_indices(d)] += mu
    X = sl.cho_solve(sl.cho_factor(M, overwrite_a=True), B); del M; gc.collect()
    np.save(f'work/X_lam{lam:g}.npy', X)
    S = Av @ X; Sc = Ac @ X
    print(f'lam={lam:g}: general score max-abs median {np.median(np.abs(Sc).max(1)):.3f} p99 {np.percentile(np.abs(Sc).max(1), 99):.3f} | in-task gold s0 med {np.median(S[stepv==0][np.arange((stepv==0).sum()), gtj[stepv==0]]):.2f}', flush=True)
    for beta in [20, 30, 40, 60]:
        a0, sq = in_task_acc(S, beta)
        fl, ln, nseqf, byk = general_flips(Sc, beta)
        print(f'   beta={beta}: in-task s0 {a0:.3f} SEQ {sq:.4f} | general flip rate {fl:.4f} label-token flips {ln:.4f} seqs-with-flip {nseqf} {byk}', flush=True)
