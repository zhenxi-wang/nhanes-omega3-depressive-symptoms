from engine import *
def ipw_setup():
 full=pd.read_pickle(P/'full_v2.pkl');frame=full.RIDSTATR.eq(2)&full.age.ge(18)&full.RIDEXPRG.ne(1)&full.w.gt(0)&full.o3i.notna();response=full.phq9.notna().to_numpy(float);a=fit(full,response,model='m2',missing=True,mask=frame,need_if=True);pi=expit(a['X']@a['b']);pc=np.clip(pi,.05,1);mapping={v:i for i,v in enumerate(full.SEQN)}
 return full,a,pi,pc,mapping
def ipw_fit(d,y,kind,setup):
 full,a,pi,pc,mapping=setup;rows=np.array([mapping[v] for v in d.SEQN]);dd=d.copy();dd['w_ipw']=dd.w.to_numpy()/pc[rows];r=fit(dd,y,[dd.o3i_z],['O3I'],weight='w_ipw',kind=kind,need_if=True);X=r['X'];b=r['b'];yy=dd[y].to_numpy();mu=expit(X@b) if kind=='OR' else X@b;active=r['valid'];deriv=(pi[rows]>.05)&(pi[rows]<1);v=np.where(active,dd.w_ipw,0)*(1-pi[rows])*(yy-mu)*deriv;C=X.T@(a['X'][rows]*v[:,None]);infl=np.zeros((len(full),len(b)));infl[rows]=r['IF'];infl-=a['IF']@C.T@r['Hinv'].T;r['V']=vcov_if(infl,full);return r
def main():
 imps=load();dietimps=load(True);base=raw();book=Book();z=base.o3i_z.to_numpy();complete=correct_missing_mask(base);dis=base[['cvd','diabetes','cancer']].eq(0).all(axis=1)&base.age.ge(20);ipw=ipw_setup()
 masks=[('Main MI',np.ones(len(base),bool),'w'),('Excluding CRP>10',base.crp.le(10),'w'),('No omega-3 supplement',base.o3sup.eq(0)&base.WTDRD1.gt(0),'WTDRD1'),('No mental-health contact',base.mh_visit.eq(0),'w'),('No CVD cancer diabetes',dis,'w'),('No current smokers',base.smoke.notna()&base.smoke.ne('Current'),'w'),('BMI<40',base.bmi.lt(40),'w'),('Age<65',base.age.lt(65),'w'),('Age>=20',base.age.ge(20),'w')]
 for m,d in enumerate(imps):
  for label,mask,wt in masks:
   for y,kind in [('dep','OR'),('dep_sev','OR'),('phq9','beta')]:
    rr=fit(d,y,[z],['O3I'],mask=mask,kind=kind,weight=wt);book.add(f'sensitivity/{label}/{y}',rr,meta={'analysis':label,'outcome':y,'weight':wt})
  for y,kind in [('dep','OR'),('dep_sev','OR'),('phq9','beta')]:
   rr=fit(d,y,[z,d.sleep_h],['O3I','sleep hours'],kind=kind);book.add(f'sensitivity/+ sleep duration/{y}',rr,meta={'analysis':'+ sleep duration','outcome':y,'weight':'w'})
   rr=ipw_fit(d,y,kind,ipw);book.add(f'sensitivity/PHQ nonresponse IPW/{y}',rr,meta={'analysis':'PHQ nonresponse IPW','outcome':y,'weight':'IPW; response-model uncertainty included'})
   rr=fit(d,y,[z,d.rx],['O3I','any prescription'],kind=kind);book.add(f'medication/+ prescription/{y}',rr,meta={'analysis':'+ prescription','outcome':y})
   rr=fit(d,y,[z],['O3I'],kind=kind,mask=base.RXQ033.eq(2));book.add(f'medication/no prescription/{y}',rr,meta={'analysis':'no prescription','outcome':y})
  for label,dd,wt in [('Phlebotomy',d,'w'),('Dietary day1',dietimps[m],'WTDRD1')]:
   mask=dd.hei.notna()&dd.kcal.gt(0)&dd.WTDRD1.gt(0);sub=base[base.hei.notna()&base.kcal.gt(0)&base.WTDRD1.gt(0)];lm=np.log1p(sub.diet_epa_dha);dz=(np.log1p(dd.diet_epa_dha)-lm.mean())/lm.std();energy=np.log(dd.kcal.where(dd.kcal>0));ld=np.log1p(dd.diet_epa_dha)
   for y in ['dep','dep_sev']:
    configs=[('biomarker',[dd.o3i_z],['O3I'],'O3I'),('dietary EPA+DHA',[dz],['diet SD'],'diet SD'),('biomarker + diet',[dd.o3i_z,energy,ld],['O3I','log energy','log dietary EPA+DHA'],'O3I')]
    for model,ex,names,term in configs:
     rr=fit(dd,y,ex,names,mask=mask,weight=wt);book.add(f'diet/{label}/{y}/{model}',rr,term,meta={'analysis':model,'outcome':y,'weight':wt})
  dd=dietimps[m];energy=np.log(dd.kcal);ld=np.log1p(dd.diet_epa_dha)
  for y,kind in [('dep','OR'),('dep_sev','OR'),('phq9','beta')]:
   for label,extra,names in [('Diet subset baseline',[],[]),('+ HEI',[dd.hei/10],['HEI per10']),('+ prescription',[dd.rx],['any prescription']),('+ HEI and prescription',[dd.hei/10,dd.rx],['HEI per10','any prescription']),('+ energy and EPA+DHA',[energy,ld],['log energy','log dietary EPA+DHA'])]:
    rr=fit(dd,y,[dd.o3i_z]+extra,['O3I']+names,weight='WTDRD1',kind=kind);book.add(f'diet_sensitivity/{label}/{y}',rr,meta={'analysis':label,'outcome':y,'weight':'WTDRD1'})
  if (m+1)%5==0:print('SENSITIVITY',m+1,'/40',flush=True);(P/'progress_sensitivity.json').write_text(json.dumps({'imputations':m+1}))
 for label,mask,missing in [('Corrected missing indicator',None,True),('No item nonresponse',complete,False),('Age>=20 fully observed',complete&base.age.ge(20).to_numpy(),False)]:
  for y,kind in [('dep','OR'),('dep_sev','OR'),('phq9','beta')]:
   rr=fit(base,y,[z],['O3I'],mask=mask,missing=missing,kind=kind);book.add(f'sensitivity/{label}/{y}',rr,meta={'analysis':label,'outcome':y,'weight':'w'})
 rows,tests=book.save('sensitivity');print('SENSITIVITY COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
