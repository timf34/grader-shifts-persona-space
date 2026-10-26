import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
Q=Path('/Users/timf34/Documents/VSCode/grader-shifts-persona-space/outputs')
O=Path('/Users/timf34/Documents/VSCode/grader-shifts-psm/outputs/_hf_q1')
src={'qwen3.6-27b':Q/'qwen3.6-27b/analysis/vectors','qwen3.8-27b':Q/'qwen3.8-27b/analysis/vectors',
     'olmo3-32b-dpo':O/'olmo3-32b-dpo/analysis/vectors','olmo31-32b-final':O/'olmo31-32b-final/analysis/vectors'}
def load(m,rep,L):
    z=np.load(src[m]/f'original_{rep}_L{L}_condition_centroids.npz',allow_pickle=True)
    roles=[str(r) for r in z['roles']]; b=[str(c) for c in z['conds']].index('baseline')
    d=z['mu_discovery'][:,b].astype(float); v=z['mu_validation'][:,b].astype(float)
    dd=z['mu_default_discovery'][b].astype(float); dv=z['mu_default_validation'][b].astype(float)
    return roles,d,v,dd,dv
def D(X): return np.sqrt(((X[:,None]-X[None])**2).sum(-1))
def pr(X):
    Xc=X-X.mean(0); s=np.linalg.svd(Xc,compute_uv=False)**2; return s.sum()**2/(s**2).sum(), s[0]/s.sum()
for a,b in [('qwen3.6-27b','qwen3.8-27b'),('olmo3-32b-dpo','olmo31-32b-final')]:
    for rep in ['pre','resp']:
        L=36
        ra,da,va,dda,dva=load(a,rep,L); rb,db,vb,ddb,dvb=load(b,rep,L)
        common=[r for r in ra if r in rb]; ia=[ra.index(r) for r in common]; ib=[rb.index(r) for r in common]
        iu=np.triu_indices(len(common),1)
        Xa=(da[ia]+va[ia])/2; Xb=(db[ib]+vb[ib])/2
        rsa=spearmanr(D(Xa)[iu],D(Xb)[iu]).correlation
        # reliability: discovery vs validation question halves within each model
        rel_a=spearmanr(D(da[ia])[iu],D(va[ia])[iu]).correlation; rel_b=spearmanr(D(db[ib])[iu],D(vb[ib])[iu]).correlation
        # cross-half cross-model (noise-free-ish): a discovery vs b validation
        rsa_x=np.mean([spearmanr(D(da[ia])[iu],D(vb[ib])[iu]).correlation, spearmanr(D(va[ia])[iu],D(db[ib])[iu]).correlation])
        def assist(X,dd):
            c=X.mean(0); spread=np.sqrt(((X-c)**2).sum(1)).mean(); return np.linalg.norm(dd-c)/spread
        nn=lambda X,dd,roles: [roles[i] for i in np.argsort(np.linalg.norm(X-dd,axis=1))[:4]]
        pa,p1a=pr(Xa); pb,p1b=pr(Xb)
        print(f'{a}->{b} {rep} L{L}: roles {len(common)} | RSA {rsa:.3f} (cross-half {rsa_x:.3f}; reliability {rel_a:.3f}/{rel_b:.3f}) | particip. ratio {pa:.1f}->{pb:.1f}, PC1 share {p1a:.2f}->{p1b:.2f} | default dist/spread {assist(Xa,(dda+dva)/2):.2f}->{assist(Xb,(ddb+dvb)/2):.2f}')
        print('   default nearest:', nn(Xa,(dda+dva)/2,common), '->', nn(Xb,(ddb+dvb)/2,common))
