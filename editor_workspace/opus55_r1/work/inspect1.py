# Hypothesis: capture records are ordered by seq, with pos counting from shared prefix end; check layout and baseline top-1 of original.
import json, numpy as np
m=json.load(open('acts/p01_layers.meta.json'))
print([k for k in m.keys()])
for k in m:
    if k not in ('items','request'): print(k, str(m[k])[:300])
z=np.load('acts/p01_layers.npz')
si=z['seq_index']; pos=z['pos']
print(si[:15], pos[:15])
print('logprob first', z['logprob'][:12])
print('topk ids', z['topk_ids'][:3,:5])
