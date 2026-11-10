"""Expanded chained-equations imputation, frozen for the revised analysis.
PMM5 for continuous targets and approximate posterior draws for binary/nominal targets.
Observed outcomes, alternative exposures and structural non-applicability are not imputed.
"""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logsumexp
from scipy.linalg import cho_factor,cho_solve,solve_triangular,qr
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from config import P
from model_spec import STRUCT,rcs
threadpool_limits(1)
CONT=['pir','bmi','egfr','log_crp','sleep_h']
BIN=['insured','food_insec','pa_active','diabetes','cvd','cancer','htn','rx']
CAT={'edu':['<High school','High school/GED','Some college','College graduate+'],
     'marital':['Married/partnered','Widowed/divorced/separated','Never married'],
     'smoke':['Never','Former','Current'],
     'alc':['None','Moderate (<=7/wk)','Heavy (>7/wk)']}
TARGETS=CONT+BIN+list(CAT)

def posterior_logit(X,y,k,rng,start=None):
 # Weakly informative N(0,5^2) priors on slopes; diffuse intercept.
 p=X.shape[1];pen=np.full(p,.04);pen[0]=1e-6
 b=np.zeros((k-1,p)) if start is None or start.shape!=(k-1,p) else start.copy()
 Y=np.eye(k)[y.astype(int)][:,:k-1]
 def evaluate(bb,hessian=True):
  eta=X@bb.T;all_eta=np.column_stack([eta,np.zeros(len(X))]);P=np.exp(all_eta-logsumexp(all_eta,axis=1)[:,None])[:,:k-1]
  obj=np.sum(logsumexp(all_eta,axis=1))-np.sum(Y*eta)+.5*np.sum(bb*bb*pen)
  grad=(P-Y).T@X+pen*bb
  if not hessian:return obj,grad,None
  H=np.empty(((k-1)*p,(k-1)*p))
  for j in range(k-1):
   for l in range(j,k-1):
    w=P[:,j]*((1 if j==l else 0)-P[:,l]);block=X.T@(X*w[:,None])
    if j==l:block+=np.diag(pen)
    H[j*p:(j+1)*p,l*p:(l+1)*p]=block
    if j!=l:H[l*p:(l+1)*p,j*p:(j+1)*p]=block.T
  return obj,grad,H
 for it in range(40):
  obj,grad,H=evaluate(b);cf=cho_factor(H,lower=True);step=cho_solve(cf,grad.ravel()).reshape(b.shape)
  t=1.
  for j in range(25):
   bn=b-t*step;on,_,_=evaluate(bn,False)
   if on<=obj+1e-8:break
   t*=.5
  b=bn
  if np.max(np.abs(t*step))<1e-6:break
 obj,grad,H=evaluate(b)
 if np.max(np.abs(grad))>1e-3:raise RuntimeError(('conditional model did not converge',np.max(np.abs(grad)),k))
 L=np.linalg.cholesky(H);bd=b+solve_triangular(L.T,rng.normal(size=b.size),lower=False).reshape(b.shape)
 return bd,b,it+1

def fixed_predictors(base,diet):
 age=(base.age.to_numpy()-50)/10;w=base['WTDRD1' if diet else 'w'].to_numpy();lw=np.log(w);z=base.o3i_z.to_numpy();cols=[np.ones(len(base)),age,age**2,base.female.to_numpy(float),base.usborn.to_numpy(float),z,base.phq9.to_numpy()/5,base.dep.to_numpy(),base.dep_sev.to_numpy(),base.age.lt(20).to_numpy(float),(lw-lw.mean())/lw.std()]
 def add(v,scale=False):
  s=np.asarray(v,float);m=~np.isfinite(s);s=s.copy();s[m]=np.nanmedian(s)
  if scale:s=(s-s.mean())/max(s.std(),1e-8)
  cols.append(s)
  if m.any():cols.append(m.astype(float))
 for lv in ['Mexican American','Other Hispanic','NH Black','NH Asian','Other/Multiracial']:add(base.race.eq(lv))
 psu=pd.Categorical(pd.Series(list(zip(base.SDMVSTRA,base.SDMVPSU)))).codes
 for j in sorted(set(psu))[1:]:add(psu==j)
 for t in [3,5,8,12,20]:add(base.phq9.ge(t))
 for i in range(1,10):
  s=base[f'DPQ{i*10:03d}'];
  if i<9:add(s/3)
  add(s.ge(2))
 for v in ['mel_any','non_any','impair','anx_freq','anx_severe','dep_wg_severe','mh_visit','o3sup']:add(base[v])
 for q in [2,3,4]:add(base.o3i_q.eq(q))
 add(base.o3i.lt(4));add(base.o3i.between(4,8,inclusive='left'))
 for probs in [[.05,.35,.65,.95],[.10,.50,.90]]:
  B=rcs(base.o3i,np.quantile(base.o3i,probs))
  for j in range(B.shape[1]):add(B[:,j],True)
 for v in ['epa_z','dha_z','dpa_z','ala_z','aa_z','la_z','hufa_z','n6n3_z','aaepa_z','sfa_z']:add(base[v],True)
 for s in [np.log(base.kcal.where(base.kcal>0)),np.log1p(base.diet_epa_dha),base.hei]:add(s,True)
 for s in [base.dep.to_numpy()*z,base.dep_sev.to_numpy()*z,base.female.to_numpy()*z,base.age.ge(45).to_numpy()*z,base.age.ge(65).to_numpy()*z]:add(s)
 F=np.column_stack(cols);R=F[:,1:]-F[:,1:].mean(axis=0);Q,T,piv=qr(R,mode='economic',pivoting=True);rank=np.sum(np.abs(np.diag(T))>1e-8*np.max(np.abs(np.diag(T))));keep=[0]+sorted((piv[:rank]+1).tolist());return F[:,keep]

def chain(base,diet,seed,burn=50,thin=5,keep=10):
 rng=np.random.default_rng(seed);n=len(base);u20=base.age.lt(20).to_numpy();fixed=fixed_predictors(base,diet)
 miss={v:base[v].isna().to_numpy()&(~u20 if v in STRUCT else True) for v in TARGETS}
 obs={v:base[v].notna().to_numpy() for v in TARGETS}
 values={};std={};post={};maxit=0
 for v in TARGETS:
  if v in CAT:
   s=pd.Categorical(base[v],categories=CAT[v]).codes.astype(float);s[s<0]=np.nan
  else:s=base[v].to_numpy(float).copy()
  if miss[v].any():s[miss[v]]=rng.choice(s[obs[v]],miss[v].sum())
  s[np.isnan(s)]=0.;values[v]=s
  if v in CONT:std[v]=(np.mean(s[obs[v]]),np.std(s[obs[v]]))
 def encode(v):
  s=values[v]
  if v in CONT:
   C=[(s-std[v][0])/std[v][1]]
   if v=='pir':C.extend([s<=1.3,(s>1.3)&(s<=3.5)])
   A=np.column_stack(C).astype(float)
  elif v in CAT:A=np.column_stack([s==j for j in range(1,len(CAT[v]))]).astype(float)
  else:A=s[:,None]
  if v in ['pir','bmi','smoke','log_crp']:A=np.column_stack([A,A*base.o3i_z.to_numpy()[:,None]])
  return A
 traces=[];completed=[];finger=[]
 for cycle in range(burn+thin*keep):
  for v in TARGETS:
   m=miss[v];o=obs[v]
   if not m.any():continue
   XX=np.column_stack([fixed]+[encode(vv) for vv in TARGETS if vv!=v])
   cols=np.r_[True,np.std(XX[o,1:],axis=0)>1e-10];X=XX[:,cols];xo=X[o];xm=X[m];y=values[v][o]
   if v in CONT:
    A=xo.T@xo;A+=np.diag(np.maximum(np.diag(A),1)*1e-6);cf=cho_factor(A,lower=True);b=cho_solve(cf,xo.T@y)
    sigma2=np.sum((y-xo@b)**2)/rng.chisquare(max(len(y)-xo.shape[1],1));L=np.linalg.cholesky(A);bd=b+np.sqrt(sigma2)*solve_triangular(L.T,rng.normal(size=len(b)),lower=False)
    pred_o=xo@b;pred_m=xm@bd;tree=cKDTree(pred_o[:,None]);dist,nn=tree.query(pred_m[:,None],k=5);donor=nn[np.arange(m.sum()),rng.integers(0,5,m.sum())];values[v][m]=y[donor]
   else:
    k=len(CAT[v]) if v in CAT else 2;bd,b,iterations=posterior_logit(xo,y,k,rng,post.get(v));post[v]=b;maxit=max(maxit,iterations)
    eta=np.column_stack([xm@bd.T,np.zeros(m.sum())]);prob=np.exp(eta-logsumexp(eta,axis=1)[:,None]);pick=(prob.cumsum(axis=1)<rng.random(m.sum())[:,None]).sum(axis=1);values[v][m]=np.minimum(pick,k-1)
  trace={}
  for v in TARGETS:
   if not miss[v].any():continue
   if v in CAT:
    for j in range(len(CAT[v])-1):trace[v+':'+str(j)]=float(np.mean(values[v][miss[v]]==j))
   else:trace[v]=float(np.mean(values[v][miss[v]]))
  traces.append(trace)
  if cycle>=burn and (cycle-burn+1)%thin==0:
   df=base.copy()
   for v in TARGETS:
    if v in CAT:
     df[v]=df[v].astype(object);df.loc[miss[v],v]=[CAT[v][int(j)] for j in values[v][miss[v]]]
    else:df.loc[miss[v],v]=values[v][miss[v]]
   df['pir_cat']=pd.cut(df.pir,[-.01,1.3,3.5,99],labels=['<1.3','1.3-3.5','>3.5'])
   for v in TARGETS:
    assert df.loc[obs[v],v].equals(base.loc[obs[v],v]) or np.array_equal(df.loc[obs[v],v].to_numpy(),base.loc[obs[v],v].to_numpy())
    assert df.loc[miss[v],v].notna().all()
   finger.append(hashlib.sha256(b''.join(values[v][miss[v]].tobytes() for v in TARGETS)).hexdigest());completed.append(df)
 return completed,traces,finger,maxit

def diagnose(traces,burn=50):
 # Split-chain R-hat for the monitored imputation means/proportions.
 arr=np.array([[list(t.values()) for t in chain[burn:]] for chain in traces]);names=list(traces[0][0]);n=arr.shape[1]//2;arr=np.concatenate([arr[:,:n],arr[:,-n:]],axis=0)
 W=arr.var(axis=1,ddof=1).mean(axis=0);B=n*arr.mean(axis=1).var(axis=0,ddof=1);V=(n-1)/n*W+B/n;rh=np.sqrt(np.divide(V,W,out=np.ones_like(W),where=W>0))
 return dict(zip(names,rh.tolist()))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--diet',action='store_true');ap.add_argument('--seed',type=int,default=20261120);args=ap.parse_args()
 full=pd.read_pickle(P/'full_v2.pkl');base=full[full.insample.eq(1)].copy()
 if args.diet:base=base[base.hei.notna()&base.kcal.gt(0)&base.WTDRD1.gt(0)].copy()
 base=base.reset_index(drop=True);label=('diet' if args.diet else 'full')+('' if args.seed==20261120 else '_'+str(args.seed));allimp=[];traces=[];finger=[];mx=[]
 print('IMPUTATION',label,'N',len(base),'fixed predictors',fixed_predictors(base,args.diet).shape[1],flush=True)
 for j in range(4):
  imp,tr,fi,iterations=chain(base,args.diet,args.seed+j*100,burn=50,thin=5,keep=10)
  for df in imp:df['crp']=np.exp(df.log_crp);df['crp_hi']=(df.crp>=3).astype(float);df['obese']=(df.bmi>=30).astype(float)
  allimp+=imp;traces.append(tr);finger+=fi;mx.append(iterations);print('CHAIN',label,j+1,'done',flush=True);(P/('progress_'+label+'.json')).write_text(json.dumps({'chains':j+1,'M':len(allimp)}))
 rh=diagnose(traces);print('DIAGNOSTICS',label,'rhatmax',max(rh.values()),'unique',len(set(finger)),flush=True)
 pd.to_pickle(allimp,P/('imputed_'+label+'.pkl'));(P/('imputation_'+label+'.json')).write_text(json.dumps({'rhat':rh,'traces':traces,'fingerprints':finger,'max_iterations':mx,'fixed_predictors':fixed_predictors(base,args.diet).shape[1],'seed':args.seed,'burn':50,'cycles':100,'thin':5},indent=2))
