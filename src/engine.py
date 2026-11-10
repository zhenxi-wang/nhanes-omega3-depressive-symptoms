import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from scipy.special import expit
from scipy.linalg import qr
from threadpoolctl import threadpool_limits
from config import P,ROOT,AUD
from model_spec import Xcorr,STRUCT,VS,rcs
from survey import _irls_logit
threadpool_limits(1)
META=json.loads((P/'metadata.json').read_text())
def load(diet=False):return pd.read_pickle(P/('imputed_diet.pkl' if diet else 'imputed_full.pkl'))
def raw():
 d=pd.read_pickle(P/'full_v2.pkl');return d[d.insample.eq(1)].reset_index(drop=True)
def model_X(d,model='m3',missing=False):
 X,nm=Xcorr(d,missing)
 if model=='m0':return np.ones((len(d),1)),['Intercept']
 if model=='m1':
  keep=[i for i,n in enumerate(nm) if n in ['Intercept','Age/10','Age squared','Female'] or n.startswith('race:')];return X[:,keep],[nm[i] for i in keep]
 if model=='m2':
  keep=[i for i,n in enumerate(nm) if i<5 or any(n==v or n.startswith(v+':') or n.startswith(v+' missing') for v in ['race','edu','pir_cat','marital','insured','food_insec','usborn'])];return X[:,keep],[nm[i] for i in keep]
 if model=='m4':X=np.column_stack([X,d.log_crp]);nm+=['log hsCRP']
 return X,nm
def vcov_if(infl,d):
 groups=list(zip(d.SDMVSTRA,d.SDMVPSU));codes,labels=pd.factorize(pd.Series(groups),sort=True);S=np.zeros((len(labels),infl.shape[1]));np.add.at(S,codes,infl);V=np.zeros((infl.shape[1],infl.shape[1]))
 for h in sorted(set(t[0] for t in labels)):
  v=S[np.array([t[0]==h for t in labels])];n=len(v)
  if n>1:
   a=v-v.mean(axis=0);V+=n/(n-1)*(a.T@a)
 return V
def fit(d,y,extra=None,enames=None,model='m3',kind='OR',mask=None,weight='w',missing=False,need_if=False,Xoverride=None,noQR=False):
 X,nm=model_X(d,model,missing) if Xoverride is None else Xoverride
 if extra is not None:X=np.column_stack([X]+[np.asarray(x,float) for x in extra]);nm=nm+list(enames)
 y=np.asarray(d[y] if isinstance(y,str) else y,float);w=d[weight].fillna(0).to_numpy(float);valid=np.isfinite(y)&np.isfinite(X).all(axis=1)&(w>0)
 if mask is not None:valid &= np.asarray(mask,bool)
 if valid.sum()==0:raise ValueError('empty model')
 keep=np.r_[True,np.std(X[valid,1:],axis=0)>1e-12];X=X[:,keep];nm=[n for n,k in zip(nm,keep) if k]
 if not noQR:
  R=X[valid,1:]-X[valid,1:].mean(axis=0);_,T,piv=qr(R,mode='economic',pivoting=True);rank=np.sum(np.abs(np.diag(T))>max(np.max(np.abs(np.diag(T))),1)*1e-10);sel=[0]+sorted((piv[:rank]+1).tolist());X=X[:,sel];nm=[nm[i] for i in sel]
 xx=np.nan_to_num(X);yy=np.nan_to_num(y);ww=np.where(valid,w,0.)
 if kind=='OR':
  b,mu=_irls_logit(xx,yy,ww);H=xx.T@(xx*(ww*mu*(1-mu))[:,None])
 else:
  H=xx.T@(xx*ww[:,None]);b=np.linalg.pinv(H)@(xx.T@(ww*yy));mu=xx@b
 inv=np.linalg.pinv(H);score=xx*(ww*(yy-mu))[:,None];influence=score@inv.T;V=vcov_if(influence,d)
 result={'b':b,'V':V,'names':nm,'n':int(valid.sum()),'events':int(yy[valid].sum()) if kind=='OR' else None,'valid':valid,'weight':weight,'kind':kind,'X':xx}
 if need_if:result['IF']=influence;result['Hinv']=inv
 return result
def scalar_pool(Q,U,kind='OR',dfcom=15):
 Q=np.asarray(Q,float);U=np.asarray(U,float);m=len(Q);q=Q.mean();u=U.mean();b=np.var(Q,ddof=1) if m>1 else 0.;t=u+(1+1/m)*b;lam=(1+1/m)*b/t if t>0 else 0
 if m>1:
  old=(m-1)/max(lam,1e-15)**2;obs=(dfcom+1)/(dfcom+3)*dfcom*(1-lam);df=1/(1/old+1/obs)
 else:df=float(dfcom)
 se=np.sqrt(max(t,0));crit=stats.t.ppf(.975,df);lo=q-crit*se;hi=q+crit*se;p=2*stats.t.sf(abs(q/se),df) if se>0 else 1.
 if kind=='OR':est,l,h=np.exp(np.clip([q,lo,hi],-700,700))
 elif kind=='risk':est,l,h=expit([q,lo,hi])
 else:est,l,h=q,lo,hi
 r=(1+1/m)*b/u if u>0 else 0;fmi=(r+2/(df+3))/(1+r) if m>1 else 0
 return dict(estimate=float(est),lo=float(l),hi=float(h),p=float(p),beta=float(q),se=float(se),df=float(df),M=m,U=float(u),B=float(b),T=float(t),lambda_=float(lam),FMI=float(fmi),MCSE=float(np.sqrt(b/m)))
def vector_pool(Q,U,dfcom=14):
 Q=np.asarray(Q,float);U=np.asarray(U,float);m,k=Q.shape;q=Q.mean(axis=0);u=U.mean(axis=0);B=np.cov(Q,rowvar=False) if m>1 else np.zeros((k,k));B=np.atleast_2d(B);inv=np.linalg.pinv(u);r=float((1+1/m)*np.trace(B@inv)/k);r=max(r,0);F=float(q@inv@q/k/(1+r))
 if k==1:
  one=scalar_pool(Q[:,0],U[:,0,0],'beta',dfcom);return dict(F=(one['beta']/one['se'])**2,df1=1,df2=one['df'],p=one['p'],RIV=r,M=m,dfcom=dfcom)
 if m==1:df=float(dfcom)
 else:
  t=k*(m-1);a=r*t/(t-2);vstar=(dfcom+1)/(dfcom+3)*dfcom;c0=1/(t-4);c1=vstar-2*(1+a);c2=vstar-4*(1+a)
  z=1/c2+c0*a*a*c1/((1+a)**2*c2)+c0*(8*a*a*c1/((1+a)*c2*c2)+4*a*a/((1+a)*c2))+c0*(4*a*a/(c2*c1)+16*a*a*c1/c2**3)+c0*8*a*a/c2**2;df=4+1/z
 return dict(F=F,df1=k,df2=float(df),p=float(stats.f.sf(F,k,df)),RIV=r,M=m,dfcom=dfcom)
class Book:
 def __init__(self):self.scalar={};self.vector={}
 def add_est(self,key,Q,U,kind='OR',meta=None):
  if key not in self.scalar:self.scalar[key]={'Q':[],'U':[],'kind':kind,'meta':meta or {}}
  self.scalar[key]['Q'].append(float(Q));self.scalar[key]['U'].append(float(U))
 def add(self,key,r,term='O3I',kind=None,meta=None):
  if term not in r['names']:raise ValueError((key,'term not estimable',term,r['names']))
  j=r['names'].index(term);self.add_est(key,r['b'][j],r['V'][j,j],kind or r['kind'],{'n':r['n'],'events':r['events'],**(meta or {})})
 def add_vector(self,key,Q,U,meta=None):
  if key not in self.vector:self.vector[key]={'Q':[],'U':[],'meta':meta or {}}
  self.vector[key]['Q'].append(np.asarray(Q,float).tolist());self.vector[key]['U'].append(np.asarray(U,float).tolist())
 def contrast(self,key,r,c,kind='OR',meta=None):
  c=np.asarray(c,float);self.add_est(key,c@r['b'],c@r['V']@c,kind,{'n':r['n'],**(meta or {})})
 def test(self,key,r,terms,meta=None):
  idx=[r['names'].index(t) for t in terms];self.add_vector(key,r['b'][idx],r['V'][np.ix_(idx,idx)],{'n':r['n'],**(meta or {})})
 def save(self,label):
  out=[]
  for key,v in self.scalar.items():out.append(dict(key=key,kind=v['kind'],**v['meta'],**scalar_pool(v['Q'],v['U'],v['kind'])))
  tests=[]
  for key,v in self.vector.items():tests.append(dict(key=key,**v['meta'],**vector_pool(v['Q'],v['U'],16-len(v['Q'][0]))))
  pd.DataFrame(out).to_csv(P/(label+'.csv'),index=False);pd.DataFrame(tests).to_csv(P/(label+'_tests.csv'),index=False);(P/(label+'_raw.json')).write_text(json.dumps({'scalar':self.scalar,'vector':self.vector}));return out,tests
def marginal(r,d,assignments,mask=None):
 valid=r['valid'] if mask is None else r['valid']&np.asarray(mask,bool);w=np.where(valid,d[r['weight']].fillna(0).to_numpy(),0);W=w.sum();vals=[];infl=[]
 for mapping in assignments:
  X=r['X'].copy()
  for term,val in mapping.items():X[:,r['names'].index(term)]=val
  pr=expit(X@r['b']);mu=np.dot(w,pr)/W;grad=np.sum(X*(w*pr*(1-pr))[:,None],axis=0)/W
  IF=r['IF']@grad+w/W*(pr-mu);vals.append(mu);infl.append(IF)
 IF=np.column_stack(infl);return np.array(vals),vcov_if(IF,d)
def add_risks(book,key,R,V,meta=None):
 for j,v in enumerate(R):
  lg=np.log(v/(1-v));u=V[j,j]/(v*(1-v))**2;book.add_est(key+f'/risk{j}',lg,u,'risk',meta)
def evalue(est,lo,hi):
 def ev(x):z=1/x if x<1 else x;return z+np.sqrt(z*(z-1))
 limit=hi if est<1 else lo;return ev(est),1. if lo<=1<=hi else ev(limit)
def correct_missing_mask(d):
 M=pd.DataFrame({v:d[v].isna()&(~d.age.lt(20) if v in STRUCT else True) for v in VS});return ~M.any(axis=1).to_numpy()
