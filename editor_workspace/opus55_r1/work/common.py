# Shared helpers: load captures, label/token maps, exact offline logit read-out of a final-layer residual change.
import json, numpy as np
import wsp_kernel as K

LABS = json.load(open('data/labels.json'))
L2I = {l['label']: l['label_id'] for l in LABS}
EOT = 128009
TOK = sorted({t for l in LABS for t in l['token_ids']})          # 151 label tokens incl eot
T2J = {t: j for j, t in enumerate(TOK)}


def records(name):
    """Per-record class id, step within the reply, gold next-token id, seq index."""
    m = json.load(open(f'acts/{name}.meta.json'))
    z = np.load(f'acts/{name}.npz')
    si = z['seq_index']
    step = np.zeros(len(si), int)
    for i in range(1, len(si)):
        step[i] = step[i - 1] + 1 if si[i] == si[i - 1] else 0
    items = m['items']
    ycls = np.array([L2I[items[s]['gold']] if items[s].get('gold') in L2I else -1 for s in si])
    tgt = np.array([LABS[c]['token_ids'][k] if c >= 0 else -1 for c, k in zip(ycls, step)])
    return m, z, si, step, ycls, tgt


_E = None
def E_full():
    """lm_head rows scaled by the final-norm weight: logits = E @ r / rms(r) (argmax is rms-invariant)."""
    global _E
    if _E is None:
        g = K.load('model.norm.weight')
        _E = (K.load('lm_head.weight') * g[None, :]).astype(np.float32)
    return _E


def E_rows(ids):
    g = K.load('model.norm.weight')
    return K.load_rows('lm_head.weight', list(ids)) * g[None, :]


def argmax_full(R, chunk=16384):
    """Full-vocab argmax and max logit (unnormalised, per row of R [n,4096])."""
    E = E_full()
    best = np.full(len(R), -np.inf, np.float32); arg = np.zeros(len(R), int)
    for s in range(0, E.shape[0], chunk):
        L = R @ E[s:s + chunk].T
        a = L.argmax(1); v = L[np.arange(len(R)), a]
        upd = v > best; best[upd] = v[upd]; arg[upd] = a[upd] + s
    return arg, best


def bf16(x):
    x = np.ascontiguousarray(x, np.float32)
    return ((x.view(np.uint32) + 0x7FFF + ((x.view(np.uint32) >> 16) & 1)) >> 16 << 16).view(np.float32)
