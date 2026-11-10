"""Observed item nonresponse, structural non-applicability and joint patterns."""
from engine import *
def main():
 b=raw();T=P/'deliverables/tables';T.mkdir(parents=True,exist_ok=True);mm=pd.DataFrame({v:b[v].isna()&(~b.age.lt(20) if v in STRUCT else True) for v in VS});rows=[]
 for k in sorted(mm.sum(axis=1).unique()):
  m=mm.sum(axis=1).eq(k);rows.append({'Number of missing Model3 covariates':int(k),'n':int(m.sum()),'Unweighted %':100*m.mean(),'Weighted %':100*b.loc[m,'w'].sum()/b.w.sum()})
 pd.DataFrame(rows).to_csv(T/'Missingness_number.csv',index=False);pat=mm.apply(lambda r:', '.join(mm.columns[r]) if r.any() else 'None',axis=1);rows=[]
 for s,n in pat.value_counts().items():
  m=pat.eq(s);rows.append({'Joint missingness pattern':s,'n':int(n),'Unweighted %':100*m.mean(),'Weighted %':100*b.loc[m,'w'].sum()/b.w.sum()})
 pd.DataFrame(rows).to_csv(T/'Missingness_patterns.csv',index=False);rows=[]
 for label,m in [('All original marital missing',b.marital.isna()),('Structural age18-19',b.age.lt(20)),('True marital item missing age>=20',b.marital.isna()&b.age.ge(20))]:rows.append({'Group':label,'n':int(m.sum()),'PHQ>=10 cases':int(b.loc[m,'dep'].sum()),'PHQ>=15 cases':int(b.loc[m,'dep_sev'].sum()),'Weighted %':100*b.loc[m,'w'].sum()/b.w.sum()})
 pd.DataFrame(rows).to_csv(T/'Marital_missing_diagnosis.csv',index=False)
 path=P/'deliverables/table_catalog.json'
 if path.exists():
  cat=json.loads(path.read_text());names={t['name'] for t in cat}
  for n in ['Missingness_number','Missingness_patterns','Marital_missing_diagnosis']:
   if n not in names:cat.append({'name':n,'title':n.replace('_',' '),'note':'Observed data; structural non-applicability is not item nonresponse.','rows':len(pd.read_csv(T/(n+'.csv')))})
  path.write_text(json.dumps(cat,indent=2))
 assert int(mm.any(axis=1).sum())==971
 print('JOINT MISSINGNESS VERIFIED',flush=True)
if __name__=='__main__':main()
