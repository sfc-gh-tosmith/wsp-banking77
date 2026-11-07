# Hypothesis: merely switching layer-31 down_proj from 4-bit to bf16 (unavoidable for any edit) already flips a noticeable fraction of
# general greedy tokens; estimate with a simulated affine 4-bit, group-64 quantisation (scheme of the runner unknown -> rough floor).
import sys, gc, json, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
W = K.load('model.layers.31.mlp.down_proj.weight')
Wg = W.reshape(W.shape[0], -1, 64); mn = Wg.min(2, keepdims=True); mx = Wg.max(2, keepdims=True)
sc = (mx - mn) / 15; sc[sc == 0] = 1
Wq = (np.round((Wg - mn) / sc) * sc + mn).reshape(W.shape); dW = (W - Wq).astype(np.float32); del Wg, Wq; gc.collect()
print('rel weight error rms %.4f' % (np.sqrt((dW**2).mean()) / np.sqrt((W**2).mean())))
zg = np.load('acts/p04_general_L31.npz'); sig = zg['seq_index']
rows = [json.loads(l) for l in open('work/general.jsonl')]
tc = json.load(open('work/general_tokcounts.json')); cnt = np.bincount(sig); start = np.r_[0, np.cumsum(cnt)[:-1]]
reply = np.zeros(len(sig), bool)
for s in range(len(cnt)): reply[start[s] + cnt[s] - (tc['n_comp'][s] + 1): start[s] + cnt[s]] = True
idx = np.where(reply)[0][::2]
R = zg['L31.resid'][idx].astype(np.float32); A = zg['L31.mlp_act'][idx].astype(np.float32)
b0, _ = argmax_full(R); dR = A @ dW.T
b1, _ = argmax_full(R + dR)
print(f'4bit->bf16 (simulated) general reply flips {int((b0!=b1).sum())}/{len(R)} = {(b0!=b1).mean():.4f} | median |dr| {np.median(np.linalg.norm(dR,axis=1)):.3f}')
