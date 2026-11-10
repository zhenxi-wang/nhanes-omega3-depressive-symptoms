from engine import *
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm,hashlib
def main():
 imps=load();base=raw();book=Book();checks={};d=imps[0];z=d.o3i_z.to_numpy()
 for y in ['dep','dep_sev']:
  r=fit(d,y,[z],['O3I'],need_if=True);X=r['X'];w=d.w.to_numpy();w=w/w.mean();g=sm.GLM(d[y],X,family=sm.families.Binomial(),freq_weights=w).fit(tol=1e-11,maxiter=100);mu=expit(X@g.params);H=X.T@(X*(w*mu*(1-mu))[:,None]);S=X*(w*(d[y].to_numpy()-mu))[:,None];V=vcov_if(S@np.linalg.inv(H).T,d);checks['GLM_'+y]={'maximum_coefficient_difference':float(np.max(np.abs(g.params-r['b']))),'maximum_covariance_difference':float(np.max(np.abs(V-r['V'])))}
 # Check the separate-logit PSU covariance against an explicitly stacked, fully interacted fit.
 qs=[];IF=[];frames=[];xx=[];yy=[];labels=[]
 for k,(lo,hi) in enumerate([(5,10),(10,15),(15,100)]):
  yes=d.phq9.ge(lo)&d.phq9.lt(hi);mask=yes|d.phq9.lt(5);r=fit(d,yes.to_numpy(float),[z],['O3I'],mask=mask,need_if=True);j=r['names'].index('O3I');qs.append(r['b'][j]);IF.append(r['IF'][:,j]);f=d[mask].copy();frames.append(f);X,nm=model_X(f);A=np.column_stack([X,f.o3i_z]);Xbig=np.zeros((len(f),3*A.shape[1]));Xbig[:,k*A.shape[1]:(k+1)*A.shape[1]]=A;xx.append(Xbig);yy.append(yes[mask].to_numpy(float));labels += [f'{k}:{n}' for n in nm+['O3I']]
 s=pd.concat(frames,ignore_index=True);r=fit(s,np.concatenate(yy),Xoverride=(np.vstack(xx),labels),noQR=True);idx=[r['names'].index(f'{k}:O3I') for k in range(3)];U=vcov_if(np.column_stack(IF),d);checks['stacked_severity']={'coefficient_difference':float(np.max(np.abs(np.array(qs)-r['b'][idx]))),'covariance_difference':float(np.max(np.abs(U-r['V'][np.ix_(idx,idx)])))}
 for imp in imps:
  q=[];inf=[]
  for label,lo,hi in [('15-19',15,20),('20-27',20,28)]:
   yes=imp.phq9.ge(lo)&imp.phq9.lt(hi);r=fit(imp,yes.to_numpy(float),[imp.o3i_z],['O3I'],mask=yes|imp.phq9.lt(5),need_if=True);j=r['names'].index('O3I');q.append(r['b'][j]);inf.append(r['IF'][:,j]);book.add('severity_extra/'+label,r,meta={'category':label,'cases':int(yes.sum())})
  U=vcov_if(np.column_stack(inf),imp);c=np.array([-1,1]);book.add_est('severity_extra/20-27 versus15-19 coefficient contrast',c@np.array(q),c@U@c,'OR',{'n':len(imp)})
 for y in ['dep','dep_sev']:
  r=fit(base,y,[base.o3i_z],['O3I'],model='m0');book.add('unadjusted/'+y,r,meta={'outcome':y})
 extra,_=book.save('extra')
 c=pd.read_csv(P/'core.csv');ev=[]
 for _,r in c[c.key.str.match(r'model/(dep|dep_sev)/m3/(SD|Q4)$')].iterrows():
  v,lim=evalue(r.estimate,r.lo,r.hi);con,clim=evalue(np.sqrt(r.estimate),np.sqrt(r.lo),np.sqrt(r.hi));ev.append(dict(key=r.key,OR=r.estimate,lo=r.lo,hi=r.hi,E_value_OR_as_RR=v,E_value_CI=lim,E_value_sqrt_OR=con,E_value_sqrt_CI=clim))
 pd.DataFrame(ev).to_csv(P/'evalues.csv',index=False)
 t=c[c.key.str.startswith('threshold/')].copy();t['p_bonferroni_7']=multipletests(t.p,method='bonferroni')[1];t['p_BH_7']=multipletests(t.p,method='fdr_bh')[1];t.to_csv(P/'threshold_multiplicity.csv',index=False)
 t=pd.read_csv(P/'interactions_tests.csv');sg=t[t.key.str.startswith('subgroup_interaction/')].copy();sg['p_bonferroni_7']=multipletests(sg.p,method='bonferroni')[1];sg['p_BH_7']=multipletests(sg.p,method='fdr_bh')[1];sg.to_csv(P/'subgroup_multiplicity.csv',index=False)
 for y in ['dep','dep_sev']:
  reference=float(c.loc[c.key.eq(f'model/{y}/m3/SD'),'estimate'].iloc[0]);ss=pd.read_csv(P/'sensitivity.csv');checks['duplicate_'+y]=max(abs(float(v)-reference) for v in [c.loc[c.key.eq('secondary/'+y),'estimate'].iloc[0],ss.loc[ss.key.eq('sensitivity/Main MI/'+y),'estimate'].iloc[0]])
 for tag in ['full','diet']:
  ob=json.loads((P/('imputation_'+tag+'.json')).read_text());arr=np.array([[list(v.values()) for v in chain[50:]] for chain in ob['traces']]);W=arr.var(axis=1,ddof=1).mean(axis=0);B=arr.shape[1]*arr.mean(axis=1).var(axis=0,ddof=1);checks['constant_trace_'+tag]={'zero_within_nonzero_between':int(((W==0)&(B>1e-14)).sum()),'rhat_max':max(ob['rhat'].values()),'unique_datasets':len(set(ob['fingerprints']))}
 # Observed cells and design variables must be preserved in every completed data set.
 targets=['pir','bmi','egfr','log_crp','sleep_h','insured','food_insec','pa_active','diabetes','cvd','cancer','htn','rx','edu','marital','smoke','alc']
 for v in targets:
  obs=base[v].notna();checks['observed_'+v]=all(np.array_equal(i.loc[obs,v].to_numpy(),base.loc[obs,v].to_numpy()) for i in imps)
 checks['design_unchanged']=all(i[['SEQN','w','SDMVSTRA','SDMVPSU']].equals(base[['SEQN','w','SDMVSTRA','SDMVPSU']]) for i in imps)
 allrows=pd.concat([pd.read_csv(P/(f+'.csv')) for f in ['core','interactions','sensitivity','extra']],ignore_index=True);checks['finite_estimates']=bool(np.isfinite(allrows[['estimate','lo','hi','p']]).all().all());checks['ordered_CIs']=bool(((allrows.lo<=allrows.estimate)&(allrows.estimate<=allrows.hi)).all());checks['P_range']=bool(allrows.p.between(0,1).all())
 (P/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2),flush=True);print(pd.DataFrame(extra)[['key','estimate','lo','hi','p']].to_string(index=False),flush=True)
if __name__=='__main__':main()
