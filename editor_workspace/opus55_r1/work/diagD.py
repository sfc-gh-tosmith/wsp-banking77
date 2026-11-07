# Hypothesis: the general-text flips are inflated by an ill-conditioned write matrix D = pinv(E_label); compare its norm and the
# induced residual change vs a regularised / plain-transpose D, using the lam=1 read-out on held-out general reply positions.
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
ET = E_rows(TOK)
sv = np.linalg.svd(ET, compute_uv=False)
print('E_label singular values: max %.3f  min %.4f  cond %.1f' % (sv[0], sv[-1], sv[0] / sv[-1]))
D = np.linalg.pinv(ET)
print('pinv column norms: median %.3f max %.3f | ET row norms median %.3f' % (np.median(np.linalg.norm(D, axis=0)), np.linalg.norm(D, axis=0).max(), np.median(np.linalg.norm(ET, axis=1))))
zg = np.load('acts/p04_general_L31.npz'); sig = zg['seq_index']
rows = [json.loads(l) for l in open('work/general.jsonl')]
chk = np.array([rows[s]['split'] == 'check' for s in sig])
idx = np.where(chk)[0][::5]
A = zg['L31.mlp_act'][idx].astype(np.float32); R = zg['L31.resid'][idx].astype(np.float32); top1 = zg['topk_ids'][idx][:, 0]
X = np.load('work/X_lam1.npy'); S = A @ X
rms_in = np.median(np.load('work/p03_rms.npy'))
rn = np.linalg.norm(R, axis=1)
for name, Dm in [('pinv', D), ('ridge', ET.T @ np.linalg.inv(ET @ ET.T + 0.1 * sv[0] ** 2 * np.eye(len(TOK)))), ('transpose', ET.T / (np.linalg.norm(ET, axis=1) ** 2)[None, :])]:
    dr = 40 * rms_in * (S @ Dm.T)
    arg, _ = argmax_full(R + dr)
    gain = np.diag(ET @ Dm)
    print(f'{name:9s}: self-gain median {np.median(gain):.3f} | |dr|/|r| median {np.median(np.linalg.norm(dr,axis=1)/rn):.3f} | full-vocab flip rate {(arg != top1).mean():.4f}', flush=True)
arg0, _ = argmax_full(R); print('sanity: flip with no edit', (arg0 != top1).mean())
