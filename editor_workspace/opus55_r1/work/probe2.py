# Hypothesis: step-0 intent accuracy improves with tuned shrinkage and multi-layer concatenation; quantify headroom before choosing the edit site.
import json, numpy as np
labs=json.load(open('data/labels.json')); L2I={l['label']:l['label_id'] for l in labs}
m=json.load(open('acts/p01_layers.meta.json')); z=np.load('acts/p01_layers.npz')
si=z['seq_index']
first=np.r_[True, si[1:]!=si[:-1]]
y=np.array([L2I[it['gold']] for it in m['items']])
def get(L):
    X=z[f'L{L}.resid'][first].astype(np.float32); return X/np.sqrt((X**2).mean(1,keepdims=True))
def lda_fit(X,y,shr):
    mu=X.mean(0); Xc=X-mu; C=np.stack([Xc[y==c].mean(0) for c in range(77)])
    R=Xc-C[y]; S=R.T@R/len(R); S=(1-shr)*S+shr*np.trace(S)/S.shape[0]*np.eye(S.shape[0])
    W=np.linalg.solve(S,C.T); b=-0.5*np.sum(C*W.T,1); return mu,W,b
rng=np.random.default_rng(1); folds=rng.permutation(len(y))%5
def cv(X,shr):
    acc=[]
    for f in range(5):
        tr=folds!=f; te=~tr
        mu,W,b=lda_fit(X[tr],y[tr],shr); acc.append((((X[te]-mu)@W+b).argmax(1)==y[te]).mean())
    return np.mean(acc)
for L in [22,26,29,31]:
    X=get(L); print(f'L{L}', ' '.join(f'shr{s}:{cv(X,s):.3f}' for s in [0.02,0.05,0.1,0.2,0.4]), flush=True)
X=np.concatenate([get(22),get(26),get(31)],1); print('concat22+26+31', ' '.join(f'shr{s}:{cv(X,s):.3f}' for s in [0.05,0.1,0.2]))
