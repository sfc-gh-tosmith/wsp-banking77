# Hypothesis: step-0 LDA accuracy keeps rising with training size; extrapolate to 9233 practice prompts.
import json, numpy as np
labs=json.load(open('data/labels.json')); L2I={l['label']:l['label_id'] for l in labs}
m=json.load(open('acts/p01_layers.meta.json')); z=np.load('acts/p01_layers.npz')
si=z['seq_index']; first=np.r_[True, si[1:]!=si[:-1]]
y=np.array([L2I[it['gold']] for it in m['items']])
X=z['L29.resid'][first].astype(np.float32); X/=np.sqrt((X**2).mean(1,keepdims=True))
def lda_fit(X,y,shr):
    mu=X.mean(0); Xc=X-mu; C=np.stack([Xc[y==c].mean(0) if (y==c).any() else np.zeros(X.shape[1]) for c in range(77)])
    R=Xc-C[y]; S=R.T@R/len(R); S=(1-shr)*S+shr*np.trace(S)/S.shape[0]*np.eye(S.shape[0])
    W=np.linalg.solve(S,C.T); b=-0.5*np.sum(C*W.T,1); return mu,W,b
rng=np.random.default_rng(2); perm=rng.permutation(len(y)); te=perm[:500]
for n in [250,500,1000,1500,2000]:
    tr=perm[500:500+n]; mu,W,b=lda_fit(X[tr],y[tr],0.1)
    print(n, round(float((((X[te]-mu)@W+b).argmax(1)==y[te]).mean()),3))
