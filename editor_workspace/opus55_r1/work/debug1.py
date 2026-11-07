# Hypothesis: offline logits (final norm + lm_head on captured L31.resid) should reproduce captured top-k; find which assumption breaks.
import sys, json, numpy as np
sys.path.insert(0, 'work')
from common import *
import wsp_kernel as K
for name in ['p03_practice_L31', 'p04_general_L31']:
    z = np.load(f'acts/{name}.npz'); n = 400
    R = z['L31.resid'][:n].astype(np.float32); tk = z['topk_ids'][:n]; tl = z['topk_logprob'][:n]
    print(name, 'resid abs max', np.abs(R).max(), 'rms median', np.median(np.sqrt((R**2).mean(1))))
    gf = K.load('model.norm.weight')
    E = E_full()
    Ln = (R / np.sqrt((R**2).mean(1, keepdims=True) + 1e-5)) @ E.T
    lp = Ln - (np.log(np.exp(Ln - Ln.max(1, keepdims=True)).sum(1, keepdims=True)) + Ln.max(1, keepdims=True))
    a = lp.argmax(1)
    print('  argmax==topk0', (a == tk[:, 0]).mean(), '| my lp at topk0 vs captured (median abs diff)', np.median(np.abs(lp[np.arange(n), tk[:, 0]] - tl[:, 0])))
    # maybe state at record i predicts via next record's resid? test shift
    a2 = np.r_[a[1:], -1]
    print('  argmax(shift+1)==topk0', (a2 == tk[:, 0]).mean(), ' argmax(shift-1)==topk0', (np.r_[-1, a[:-1]] == tk[:, 0]).mean())
    print('  examples my top', a[:8], 'captured', tk[:8, 0])
