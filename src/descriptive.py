from engine import *
from scipy.stats import spearmanr
def summary(d,y,mask=None,wt='w',kind='pct'):
 y=np.asarray(y,float);w=d[wt].fillna(0).to_numpy();ok=np.isfinite(y)&(w>0)
 if mask is not None:ok &=np.asarray(mask,bool)
 ww=np.where(ok,w,0);yy=np.nan_to_num(y);W=ww.sum();n=int(ok.sum())
 if not n:return {'text':'NA','n':0}
 mu=np.dot(ww,yy)/W
 if kind=='median':
  v=y[ok];vwt=w[ok];o=np.argsort(v);cw=np.cumsum(vwt[o])/vwt.sum();q=[v[o][min(np.searchsorted(cw,t),len(v)-1)] for t in [.25,.5,.75]];return {'text':f'{q[1]:.2f} ({q[0]:.2f}–{q[2]:.2f})','n':n,'estimate':q[1],'q1':q[0],'q3':q[2]}
 influence=(ww*(yy-mu)/W)[:,None];se=float(np.sqrt(vcov_if(influence,d)[0,0]));t=stats.t.ppf(.975,15)
 if kind=='pct':
  if 0<mu<1:l,h=expit(np.log(mu/(1-mu))+np.array([-1,1])*t*se/(mu*(1-mu)))
  else:l=h=mu
  return {'text':f'{100*mu:.2f}','n':n,'estimate':mu,'lo':l,'hi':h,'se':se,'weighted_N':W,'weighted_events':np.dot(ww,yy)}
 sd=np.sqrt(np.dot(ww,(yy-mu)**2)/W);return {'text':f'{mu:.2f} ({sd:.2f})','n':n,'estimate':mu,'lo':mu-t*se,'hi':mu+t*se,'se':se,'SD':sd}
def main():
 full=pd.read_pickle(P/'full_v2.pkl');d=raw();masks=[np.ones(len(d),bool)]+[d.o3i_q.eq(q).to_numpy() for q in [1,2,3,4]];names=['Overall','Q1','Q2','Q3','Q4'];rows=[];den=[]
 def add(label,v,kind='pct',wt='w'):
  row={'Characteristic':label,'Weight':wt};nrow={'Characteristic':label}
  for nm,mask in zip(names,masks):r=summary(d,v,mask,wt,kind);row[nm]=r['text'];nrow[nm]=r['n']
  rows.append(row);den.append(nrow)
 rows.append({'Characteristic':'Unweighted n','Weight':'none',**{nm:int(mask.sum()) for nm,mask in zip(names,masks)}})
 rows.append({'Characteristic':'Weighted N, million','Weight':'w',**{nm:float(d.loc[mask,'w'].sum()/1e6) for nm,mask in zip(names,masks)}})
 for v,label,kind in [('o3i','Omega-3 Index, %','median'),('epa','EPA, %','median'),('dha','DHA, %','median'),('age','Age, mean (SD)','mean'),('female','Female, %','pct')]:add(label,d[v],kind)
 for lv in ['NH White','NH Black','Mexican American','Other Hispanic','NH Asian','Other/Multiracial']:add(lv+', %',d.race.eq(lv))
 for lv in ['<High school','High school/GED','Some college','College graduate+']:add('Education: '+lv+', %',d.edu.eq(lv).astype(float).where(d.edu.notna()))
 for lv in ['<1.3','1.3-3.5','>3.5']:add('PIR '+{'<1.3':'<=1.3','1.3-3.5':'>1.3–3.5','>3.5':'>3.5'}[lv]+', %',d.pir_cat.eq(lv).astype(float).where(d.pir_cat.notna()))
 add('Married/partnered, %',d.marital.eq('Married/partnered').astype(float).where(d.marital.notna()))
 for v,label in [('insured','Health insurance, %'),('food_insec','Food insecurity, %'),('usborn','US-born, %')]:add(label,d[v])
 for lv in ['Never','Former','Current']:add('Smoking: '+lv+', %',d.smoke.eq(lv).astype(float).where(d.smoke.notna()))
 for lv in ['None','Moderate (<=7/wk)','Heavy (>7/wk)']:add('Alcohol: '+lv+', %',d.alc.eq(lv).astype(float).where(d.alc.notna()))
 add('MVPA >=150 min/wk, %',d.pa_active);add('BMI, mean (SD)',d.bmi,'mean');add('BMI >=30, %',d.obese)
 for v,label in [('diabetes','Diabetes, %'),('cvd','Cardiovascular disease (age>=20), %'),('cancer','Cancer (age>=20), %'),('htn','Hypertension, %')]:add(label,d[v])
 add('eGFR, mean (SD)',d.egfr,'mean');add('hsCRP, median (IQR)',d.crp,'median');add('hsCRP >=3, %',d.crp.ge(3).astype(float).where(d.crp.notna()))
 add('Omega-3 supplement, %',d.o3sup,wt='WTDRD1');add('Fish/shellfish in past30 days, %',d.fish30,wt='WTDRD1');add('Dietary EPA+DHA mg/d, median (IQR)',d.diet_epa_dha,'median','WTDRD1');add('PHQ-9, median (IQR)',d.phq9,'median')
 for v,label in [('dep','PHQ-9>=10, %'),('dep_sev','PHQ-9>=15, %'),('si','Suicidal ideation, %'),('anx_freq','Anxiety daily/weekly, %'),('anx_severe','Severe anxiety, %'),('mh_visit','Mental-health contact, %')]:add(label,d[v])
 pd.DataFrame(rows).to_csv(P/'table1.csv',index=False);pd.DataFrame(den).to_csv(P/'table1_denominators.csv',index=False)
 # National distribution, outcome prevalence and subgroup descriptive quantities.
 descr=[]
 def add_desc(key,s,kind='pct',mask=None,wt='w'):descr.append(dict(key=key,kind=kind,weight=wt,**summary(d,s,mask,wt,kind)))
 add_desc('O3I mean',d.o3i,'mean');add_desc('Age mean',d.age,'mean');add_desc('Female',d.female)
 for nm,m in [('<4',d.o3i.lt(4)),('4-<8',d.o3i.between(4,8,inclusive='left')),('>=8',d.o3i.ge(8))]:add_desc('O3I category '+nm,m)
 for v in ['dep','dep_sev','dep_mild','si','anhedonia','impair','anx_freq','anx_severe','dep_wg_severe','mh_visit']:add_desc('Prevalence '+v,d[v])
 for t in [3,5,8,10,12,15,20]:add_desc('Threshold '+str(t),d.phq9.ge(t))
 for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,28)]:add_desc(f'Severity {lo}-{hi-1}',d.phq9.between(lo,hi,inclusive='left'))
 for q in [1,2,3,4]:
  for y in ['dep','dep_sev']:add_desc(f'Quartile {q} {y}',d[y],mask=d.o3i_q.eq(q))
 for i in range(1,10):add_desc(f'Item DPQ{i*10:03d}',d[f'DPQ{i*10:03d}'].ge(2))
 hi=d.crp.ge(3);low=d.o3i.lt(4);ok=d.crp.notna()
 for group in [0,1,2,3]:
  mask=ok&((low.astype(int)*2+hi.astype(int))==group)
  for y in ['dep','dep_sev']:add_desc(f'Joint {group} {y}',d[y],mask=mask)
 for wt in ['w','WTDRD1']:
  for v in ['o3sup','fish30']:
   for val in [0,1]:add_desc(f'O3I mean {v}={val}',d.o3i,'mean',d[v].eq(val),wt)
 add_desc('Low O3I among PHQ>=15',d.o3i.lt(4),mask=d.phq9.ge(15));add_desc('High CRP',d.crp.ge(3).astype(float).where(ok))
 pd.DataFrame(descr).to_csv(P/'descriptive.csv',index=False)
 totals=[]
 for label,y in [('Population',np.ones(len(d))),('PHQ>=10',d.dep),('PHQ>=15',d.dep_sev),('PHQ>=15 with O3I<4',d.dep_sev*d.o3i.lt(4))]:
  influence=(d.w.to_numpy()*np.asarray(y))[:,None];T=influence.sum();se=np.sqrt(vcov_if(influence,d)[0,0]);crit=stats.t.ppf(.975,15);totals.append(dict(label=label,total=T,se=se,lo=T-crit*se,hi=T+crit*se))
 pd.DataFrame(totals).to_csv(P/'national_totals.csv',index=False)
 corr=[]
 for label,mask in [('All reliable recalls n4278',d.diet_epa_dha.notna()),('Positive-energy recalls n4273',d.diet_epa_dha.notna()&d.kcal.gt(0))]:
  s=d[mask];r,p=spearmanr(s.o3i,s.diet_epa_dha);corr.append(dict(label=label,n=len(s),rho=r,p_naive=p))
 pd.DataFrame(corr).to_csv(P/'dietary_correlation.csv',index=False)
 eligible=full.RIDSTATR.eq(2)&full.age.ge(18)&full.RIDEXPRG.ne(1)&full.w.gt(0)&full.o3i.notna();inc=full[eligible&full.phq9.notna()];exc=full[eligible&full.phq9.isna()];comp=[]
 for label,kind,func in [('n','n',None),('Age','mean',lambda x:x.age),('Female','pct',lambda x:x.female),('O3I','mean',lambda x:x.o3i),('EPA','mean',lambda x:x.epa),('DHA','mean',lambda x:x.dha),('BMI','mean',lambda x:x.bmi),('CRP','median',lambda x:x.crp),('PIR','mean',lambda x:x.pir),('NH White','pct',lambda x:x.race.eq('NH White')),('NH Black','pct',lambda x:x.race.eq('NH Black')),('NH Asian','pct',lambda x:x.race.eq('NH Asian')),('Hispanic','pct',lambda x:x.race.isin(['Mexican American','Other Hispanic'])),('College graduate','pct',lambda x:x.edu.eq('College graduate+').astype(float).where(x.edu.notna())),('Current smoker','pct',lambda x:x.smoke.eq('Current').astype(float).where(x.smoke.notna())),('Omega-3 supplement','pct',lambda x:x.o3sup)]:
  row={'Characteristic':label,'Type':kind}
  for nm,s in [('Included',inc),('Excluded',exc)]:
   v=func(s) if func else None;row[nm]=len(s) if kind=='n' else (float(v.median()) if kind=='median' else float(v.mean()*(100 if kind=='pct' else 1)));row[nm+'_observed_n']=len(s) if v is None else int(v.notna().sum())
  comp.append(row)
 pd.DataFrame(comp).to_csv(P/'tableS2.csv',index=False)
 # Preserve the source inventory and add the newly used files without asserting an unknown original download date.
 s1=pd.read_csv(ROOT/'resources/tableS1_datafiles.csv');s1=pd.concat([s1,pd.DataFrame([{'NHANES file':'RXQ_RX_L','Content':'Prescription medication use','Variables used':'RXQ033 and RXQ050; drug names unavailable'},{'NHANES file':'FPED 2021–2023 day1 totals','Content':'USDA food-pattern equivalents','Variables used':'13 HEI-2020 components combined with DR1TOT_L nutrients'}])],ignore_index=True);s1.to_csv(P/'tableS1.csv',index=False)
 print('DESCRIPTIVES COMPLETE table1 rows',len(rows),'n',len(d),'supplement unknown',d.o3sup.isna().sum(),flush=True)
if __name__=='__main__':main()
