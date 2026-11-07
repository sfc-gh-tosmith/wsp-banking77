# Hypothesis: intent is linearly decodable (shrinkage-LDA) at the last prompt token from mid layers, and remains decodable at later reply positions.
import json, numpy as np
labs=json.load(open('data/labels.json')); L2I={l['label']:l['label_id'] for l in labs}
m=json.load(open('acts/p01_layers.meta.json')); z=np.load('acts/p01_layers.npz')
si=z['seq_index']; pos=z['pos']
y_seq=np.array([L2I[it['gold']] for it in m['items']])
step=np.zeros(len(si),int)
for i in range(1,len(si)):
    step[i]=step[i-1]+1 if si[i]==si[i-1] else 0
# check record counts match label token counts
cnt=np.bincount(si); ok=all(cnt[s]==len(labs[y_seq[s]]['token_ids']) for s in range(len(cnt)))
print('records per seq == label tokens incl eot:', ok)
y=y_seq[si]
rng=np.random.default_rng(0); perm=rng.permutation(len(y_seq)); test_seq=set(perm[:500].tolist())
te=np.array([s in test_seq for s in si]); tr=~te
def lda(Xtr,ytr,Xte,shr=0.1):
    mu=Xtr.mean(0); Xc=Xtr-mu; C=np.zeros((77,Xtr.shape[1]))
    for c in range(77):
        idx=ytr==c
        if idx.any(): C[c]=Xc[idx].mean(0)
    R=Xc-C[ytr]; S=R.T@R/len(R); S=(1-shr)*S+shr*np.trace(S)/S.shape[0]*np.eye(S.shape[0])
    W=np.linalg.solve(S,C.T)  # d x 77
    b=-0.5*np.sum(C*W.T,1)
    return ((Xte-mu)@W+b).argmax(1)
for L in [6,10,14,18,22,26,29,31]:
    X=z[f'L{L}.resid'].astype(np.float32)
    X=X/np.sqrt((X**2).mean(1,keepdims=True))  # rms-normalize
    res=[]
    for sel_name,sel in [('step0',step==0),('step1',step==1),('step>=2',step>=2)]:
        a=tr&sel; b=te&sel
        p=lda(X[a],y[a],X[b])
        res.append(f'{sel_name}:{(p==y[b]).mean():.3f}(n={b.sum()})')
    print(f'L{L}', ' '.join(res), flush=True)
