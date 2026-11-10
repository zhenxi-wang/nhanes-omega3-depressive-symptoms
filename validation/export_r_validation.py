import sys,json
from pathlib import Path
import numpy as np,pandas as pd
S=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(S));sys.dont_write_bytecode=True
from engine import *
V=P/'r_validation';F=V/'inputs';F.mkdir(parents=True,exist_ok=True)
base=raw();full=load();diet=load(True);models=[];coeff=[];cov=[]
def record(cid,mi,d,yy,extra,names,mask=None,weight='w',kind='OR',special=None):
 r=fit(d,yy,extra,names,mask=mask,weight=weight,kind=kind,Xoverride=special,need_if=False,noQR=(special is not None))
 cols=['x'+str(j) for j in range(r['X'].shape[1])];frame=pd.DataFrame(r['X'],columns=cols);frame['outcome']=d[yy].to_numpy() if isinstance(yy,str) else yy;frame['domain']=r['valid'].astype(int);frame['weight']=d[weight].to_numpy();frame['strata']=d.SDMVSTRA.to_numpy();frame['psu']=d.SDMVPSU.to_numpy()
 path=f'{cid}_{mi:02d}.csv';frame.to_csv(F/path,index=False)
 models.append({'case':cid,'mi':mi,'path':path,'cols':cols,'kind':kind,'n':r['n'],'terms':r['names']})
 for j,col in enumerate(cols):coeff.append({'case':cid,'mi':mi,'term':col,'label':r['names'][j],'beta':r['b'][j],'SE':np.sqrt(r['V'][j,j])})
 for j,col in enumerate(cols):
  for k,col2 in enumerate(cols):cov.append({'case':cid,'mi':mi,'term1':col,'term2':col2,'covariance':r['V'][j,k]})
for m,d in enumerate(full):
 z=d.o3i_z
 for y in ['dep','dep_sev']:record('main_'+y,m+1,d,y,[z],['O3I'])
 if m<10:
  record('continuous_phq',m+1,d,'phq9',[z],['O3I'],kind='beta')
  record('quartiles',m+1,d,'dep_sev',[d.o3i_q.eq(k).to_numpy(float) for k in [2,3,4]],['Q2','Q3','Q4'])
 if m<5:
  hi=base.crp.ge(3).to_numpy(float);record('CRP_interaction',m+1,d,'dep',[z,hi,z*hi],['O3I','CRP high','interaction'],mask=base.crp.notna())
  B=rcs(base.o3i,META['knots_sev']);record('spline',m+1,d,'dep_sev',[base.o3i]+[B[:,j] for j in range(B.shape[1])],['spline_linear']+[f'spline_nonlinear{j+1}' for j in range(B.shape[1])])
  record('no_prescription',m+1,d,'dep_sev',[z],['O3I'],mask=base.RXQ033.eq(2))
  record('no_supplement',m+1,d,'dep_sev',[z],['O3I'],mask=base.o3sup.eq(0)&base.WTDRD1.gt(0),weight='WTDRD1')
 dd=diet[m];record('diet_HEI',m+1,dd,'dep_sev',[dd.o3i_z,dd.hei/10],['O3I','HEI per10'],weight='WTDRD1')
 if m==0:
  s=pd.concat([d]*3,ignore_index=True);g=np.repeat(np.arange(3),len(d));X,nm=model_X(d);X=np.column_stack([X,z]);nm+=['O3I'];block=np.column_stack([np.tile(X,(3,1))*(g==j)[:,None] for j in range(3)]);labels=[f'group{j} {x}' for j in range(3) for x in nm];yy=np.concatenate([d.phq9.between(5,9).to_numpy(float),d.phq9.between(10,14).to_numpy(float),d.phq9.ge(15).to_numpy(float)]);mask=np.concatenate([(d.phq9.lt(5)|d.phq9.between(5,9)),(d.phq9.lt(5)|d.phq9.between(10,14)),(d.phq9.lt(5)|d.phq9.ge(15))]);record('severity_stack',1,s,yy,None,None,mask=mask,special=(block,labels))
  yy=np.concatenate([d.phq9.ge(t).to_numpy(float) for t in [5,10,15]]);zz=np.tile(z,3);record('cumulative_stack',1,s,yy,[(g==1).astype(float),(g==2).astype(float),zz,zz*g],['cut10','cut15','O3I','O3I x cutpoint'])
 if (m+1)%10==0:print('EXPORTED',m+1,flush=True)
pd.DataFrame(coeff).to_csv(V/'python_coefficients.csv',index=False);pd.DataFrame(cov).to_csv(V/'python_covariances.csv',index=False);(V/'models.json').write_text(json.dumps(models,indent=2));pd.DataFrame(models).drop(columns=['cols','terms']).to_csv(V/'models.csv',index=False)
print('VALIDATION INPUTS COMPLETE',len(models),flush=True)
