from engine import *
from statsmodels.stats.multitest import multipletests
def main():
 imps=load();base=raw();book=Book();curves=Book();q=np.column_stack([base.o3i_q.eq(k) for k in [2,3,4]]).astype(float);trend=base.o3i_qmed.to_numpy();z=base.o3i_z.to_numpy();cl=[base.o3i.lt(4).to_numpy(float),base.o3i.between(4,8,inclusive='left').to_numpy(float)]
 fa=[('o3i_z','Omega-3 Index'),('epa_z','EPA'),('dha_z','DHA'),('dpa_z','DPA n-3'),('ala_z','ALA'),('aa_z','Arachidonic acid'),('la_z','Linoleic acid'),('hufa_z','n-3 HUFA score'),('n6n3_z','n-6:n-3'),('aaepa_z','log AA:EPA'),('sfa_z','Saturated fatty acids')]
 sec=[('dep','OR'),('dep_sev','OR'),('dep_mild','OR'),('si','OR'),('anhedonia','OR'),('impair','OR'),('anx_freq','OR'),('anx_severe','OR'),('dep_wg_severe','OR'),('mh_visit','OR'),('phq9','beta'),('phq_som','beta'),('phq_cog','beta')]
 grids={};splines={}
 for y in ['dep','dep_sev']:
  kn=META['knots_dep' if y=='dep' else 'knots_sev'];B=rcs(base.o3i,kn);E=np.column_stack([base.o3i,B]);names=['spline_linear']+[f'spline_nonlinear{j+1}' for j in range(B.shape[1])];lo,hi=np.quantile(base.o3i,[.01,.99]);grid=np.unique(np.r_[np.linspace(lo,hi,160),[v for v in [2.,4.,6.,8.] if lo<=v<=hi]]);C=np.column_stack([grid,rcs(grid,kn)])-np.r_[4.,rcs([4.],kn)[0]];splines[y]=(E,names,C);grids[y]=grid
 for m,d in enumerate(imps):
  for y in ['dep','dep_sev']:
   for model in ['m1','m2','m3','m4']:
    if model=='m1' and m>0:continue
    meta={'outcome':y,'model':model};r=fit(d,y,[z],['O3I'],model=model);book.add(f'model/{y}/{model}/SD',r,meta=meta)
    if model=='m3':
     for term in r['names']:
      if term!='Intercept':book.add(f'fullcoef/{y}/{term}',r,term,meta={'outcome':y,'covariate':term})
    rq=fit(d,y,[q[:,j] for j in range(3)],['Q2','Q3','Q4'],model=model,need_if=(model=='m3'))
    for t in ['Q2','Q3','Q4']:book.add(f'model/{y}/{model}/{t}',rq,t,meta=meta)
    rt=fit(d,y,[trend],['trend'],model=model);book.add(f'model/{y}/{model}/trend',rt,'trend',meta=meta)
    if model=='m3':
     R,V=marginal(rq,d,[{'Q2':0,'Q3':0,'Q4':0},{'Q2':1,'Q3':0,'Q4':0},{'Q2':0,'Q3':1,'Q4':0},{'Q2':0,'Q3':0,'Q4':1}]);add_risks(book,f'marginal_quartile/{y}',R,V,{'n':len(d),'outcome':y});c=np.array([1,0,0,-1]);book.add_est(f'marginal_quartile/{y}/Q1-Q4',c@R,c@V@c,'beta',{'n':len(d),'outcome':y})
   rc=fit(d,y,cl,['O3I <4','O3I 4-<8'])
   for term in ['O3I <4','O3I 4-<8']:book.add(f'clinical/{y}/{term}',rc,term,meta={'outcome':y})
   rb=fit(d,y,[cl[0]],['low O3I'],need_if=True);R,V=marginal(rb,d,[{'low O3I':0},{'low O3I':1}]);add_risks(book,f'marginal_binary/{y}',R,V,{'n':len(d),'outcome':y});c=np.array([-1,1]);book.add_est(f'marginal_binary/{y}/low-high',c@R,c@V@c,'beta',{'n':len(d),'outcome':y})
   for v,lab in fa:
    rr=fit(d,y,[d[v]],['O3I']);book.add(f'fatty/{y}/{lab}',rr,meta={'outcome':y,'exposure':lab})
   rr=fit(d,y,[d.epa_z,d.dha_z],['EPA','DHA'])
   for t in ['EPA','DHA']:book.add(f'mutual/{y}/{t}',rr,t,meta={'outcome':y})
   c=np.zeros(len(rr['b']));c[rr['names'].index('EPA')]=1;c[rr['names'].index('DHA')]=-1;book.contrast(f'mutual/{y}/EPA-DHA',rr,c,'beta',{'outcome':y})
   E,names,C=splines[y];rs=fit(d,y,[E[:,j] for j in range(E.shape[1])],names);book.test(f'spline/{y}/overall',rs,names,{'outcome':y});book.test(f'spline/{y}/nonlinear',rs,names[1:],{'outcome':y})
   idx=[rs['names'].index(t) for t in names]
   for t in names:book.add(f'splinecoef/{y}/{t}',rs,t,'beta',{'outcome':y})
   for j,v in enumerate(grids[y]):curves.add_est(f'{y}/{j}',C[j]@rs['b'][idx],C[j]@rs['V'][np.ix_(idx,idx)]@C[j],'OR',{'outcome':y,'o3i':float(v)})
  for t in [3,5,8,10,12,15,20]:
   yy=d.phq9.ge(t).to_numpy(float);r=fit(d,yy,[z],['O3I']);book.add(f'threshold/{t}',r,meta={'threshold':t,'cases':int(yy.sum())})
  # Pairwise baseline-category models, jointly correlated through PSU influences.
  qs=[];ifs=[]
  for label,lo,hi in [('5-9',5,10),('10-14',10,15),('>=15',15,100)]:
   yes=d.phq9.ge(lo)&d.phq9.lt(hi);mask=d.phq9.lt(5)|yes;rr=fit(d,yes.to_numpy(float),[z],['O3I'],mask=mask,need_if=True);j=rr['names'].index('O3I');qs.append(rr['b'][j]);ifs.append(rr['IF'][:,j]);book.add('severity_category/'+label,rr,meta={'category':label})
  Q=np.array(qs);V=vcov_if(np.column_stack(ifs),d);C=np.array([[1,0,-1],[0,1,-1]]);book.add_vector('severity_heterogeneity',C@Q,C@V@C.T,{'n':len(d)});c=np.array([-1,0,1]);book.add_est('severity_endpoint',c@Q,c@V@c,'OR',{'n':len(d)})
  s=pd.concat([d]*3,ignore_index=True);yy=np.concatenate([d.phq9.ge(t).to_numpy(float) for t in [5,10,15]]);j=np.repeat([0.,1.,2.],len(d));zs=np.tile(z,3);rr=fit(s,yy,[(j==1).astype(float),(j==2).astype(float),zs,zs*j],['cut10','cut15','O3I','O3I x cutpoint']);book.add('cumulative_trend',rr,'O3I x cutpoint',meta={'n':len(d),'stacked_n':len(s)})
  for i in range(1,10):
   name=f'DPQ{i*10:03d}';yy=d[name].ge(2).to_numpy(float);rr=fit(d,yy,[z],['O3I']);book.add('item/'+name,rr,meta={'item':name,'cases':int(yy.sum())})
  for y,kind in sec:
   rr=fit(d,y,[z],['O3I'],kind=kind);book.add('secondary/'+y,rr,meta={'outcome':y})
  for label,cut,thr in [('>=15 vs5-14',5,15),('>=15 vs10-14',10,15),('>=10 vs5-9',5,10)]:
   rr=fit(d,d.phq9.ge(thr).to_numpy(float),[z],['O3I'],mask=d.phq9.ge(cut));book.add('conditional/'+label,rr)
  for nm,kind in [('mel_any','OR'),('non_any','OR'),('mel_z','beta'),('non_z','beta')]:
   rr=fit(d,nm,[z],['O3I'],kind=kind);book.add('cluster/'+nm,rr,meta={'outcome':nm})
  for kind,ys in [('OR',['mel_any','non_any']),('beta',['mel_z','non_z'])]:
   s=pd.concat([d,d],ignore_index=True);yy=np.r_[d[ys[0]],d[ys[1]]];mel=np.r_[np.ones(len(d)),np.zeros(len(d))];zz=np.tile(z,2);rr=fit(s,yy,[mel,zz,zz*mel],['Melancholic outcome','O3I','cluster interaction'],kind=kind);book.add('cluster_comparison/'+kind,rr,'cluster interaction',meta={'n':len(d),'stacked_n':len(s),'post_hoc':True})
  if (m+1)%5==0:print('CORE',m+1,'/40',flush=True);(P/'progress_core.json').write_text(json.dumps({'imputations':m+1}))
 rows,tests=book.save('core');cr,_=curves.save('spline_curves');pd.DataFrame(cr).to_csv(P/'spline_curves.csv',index=False)
 for y in ['dep','dep_sev']:pd.DataFrame(cr).query('outcome==@y').to_csv(P/('spline_'+y+'.csv'),index=False)
 R=pd.DataFrame(rows)
 for prefix,label in [('item/','items'),('fatty/dep/','fatty_dep'),('fatty/dep_sev/','fatty_dep_sev')]:
  it=R[R.key.str.startswith(prefix)].copy();it['p_bonferroni']=multipletests(it.p,method='bonferroni')[1];it['p_BH']=multipletests(it.p,method='fdr_bh')[1];it.to_csv(P/(label+'_multiplicity.csv'),index=False)
 print('CORE COMPLETE',len(rows),'scalar results',len(tests),'joint tests',flush=True)
if __name__=='__main__':main()
