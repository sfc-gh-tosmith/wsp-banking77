# Hypothesis: with a correct (full-vocab, offline) baseline argmax, the down_proj read-out's leak on held-out general replies is small;
# exact full-vocab evaluation of in-task held-out accuracy and general flips for saved read-outs X and boost betas.
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *

def load_sets():
    m, z, si, step, ycls, tgt = records('p03_practice_L31')
    nseq = si.max() + 1
    seq_fold = np.load('work/p03_seqfold.npy'); val_seq = seq_fold == 0; val = val_seq[si]
    I = dict(A=z['L31.mlp_act'][val].astype(np.float32), R=z['L31.resid'][val].astype(np.float32),
             gold=tgt[val], step=step[val], si=si[val], nseq=nseq, val_seq=val_seq)
    del z; gc.collect()
    zg = np.load('acts/p04_general_L31.npz'); sig = zg['seq_index']
    rows = [json.loads(l) for l in open('work/general.jsonl')]
    tc = json.load(open('work/general_tokcounts.json'))
    cnt = np.bincount(sig); start = np.r_[0, np.cumsum(cnt)[:-1]]
    reply = np.zeros(len(sig), bool)
    for s in range(len(cnt)):
        reply[start[s] + cnt[s] - (tc['n_comp'][s] + 1): start[s] + cnt[s]] = True
    chk = np.array([rows[s]['split'] == 'check' for s in sig]) & reply
    Gs = dict(A=zg['L31.mlp_act'][chk].astype(np.float32), R=zg['L31.resid'][chk].astype(np.float32),
              si=sig[chk], kind=np.array([rows[s]['kind'] for s in sig[chk]]))
    return I, Gs

def evaluate(I, Gs, Xs, betas, D, alpha0, chunk=700):
    SIs = {nm: I['A'] @ X for nm, X in Xs.items()}; SGs = {nm: Gs['A'] @ X for nm, X in Xs.items()}
    I.pop('A', None); Gs.pop('A', None); gc.collect()
    E = E_full(); ED = E @ D                     # vocab x 151
    tokarr = np.array(TOK)
    out = {}
    for nm in Xs:
        for b in betas:
            out[(nm, b)] = dict(SI=SIs[nm], SG=SGs[nm])
    resI = {k: np.zeros(len(I['R']), bool) for k in out}; resG = {k: np.zeros(len(Gs['R']), int) for k in out}
    base = np.zeros(len(Gs['R']), int)
    for s in range(0, len(I['R']), chunk):
        L0 = I['R'][s:s + chunk] @ E.T
        for k, v in out.items():
            L = L0 + (k[1] * alpha0) * (v['SI'][s:s + chunk] @ ED.T)
            resI[k][s:s + chunk] = L.argmax(1) == I['gold'][s:s + chunk]
        del L0; gc.collect()
    for s in range(0, len(Gs['R']), chunk):
        L0 = Gs['R'][s:s + chunk] @ E.T
        base[s:s + chunk] = L0.argmax(1)
        for k, v in out.items():
            L = L0 + (k[1] * alpha0) * (v['SG'][s:s + chunk] @ ED.T)
            resG[k][s:s + chunk] = L.argmax(1)
        del L0; gc.collect()
    for k in out:
        ok = resI[k]; sq = np.ones(I['nseq'], bool); sq[I['si'][~ok]] = False
        fl = resG[k] != base; lab = np.isin(resG[k], tokarr) & ~np.isin(base, tokarr)
        seqf = len(np.unique(Gs['si'][fl])); nsg = len(np.unique(Gs['si']))
        byk = ' '.join(f"{kk}:{len(np.unique(Gs['si'][fl & (Gs['kind']==kk)]))}/{len(np.unique(Gs['si'][Gs['kind']==kk]))}" for kk in np.unique(Gs['kind']))
        print(f'{k[0]} beta={k[1]}: in-task s0 {ok[I["step"]==0].mean():.3f} SEQ {sq[I["val_seq"]].mean():.4f} | general pos-flip {fl.mean():.4f} '
              f'to-label {lab.mean():.4f} | seqs with any flip {seqf}/{nsg} ({byk})', flush=True)
    return resI, resG, base

if __name__ == '__main__':
    I, Gs = load_sets()
    ET = E_rows(TOK); D = np.linalg.pinv(ET); alpha0 = np.median(np.load('work/p03_rms.npy'))
    Xs = {f'lam{l}': np.load(f'work/X_lam{l}.npy') for l in ['0', '1', '10']}
    evaluate(I, Gs, Xs, [30, 60], D, alpha0)
