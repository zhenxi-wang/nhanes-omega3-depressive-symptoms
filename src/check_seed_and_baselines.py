from engine import *
def main():
 base=raw();imps=load();alt=pd.read_pickle(P/'imputed_full_20261121.pkl');book=Book();masks=[('Supplement eligible baseline',base.o3sup.notna()&base.WTDRD1.gt(0),'WTDRD1'),('Known chronic status age>=20 baseline',base.age.ge(20)&(base[['cvd','diabetes','cancer']].eq(1).any(axis=1)|base[['cvd','diabetes','cancer']].eq(0).all(axis=1)),'w')]
 for d,da in zip(imps,alt):
  for label,mask,wt in masks:
   for y,kind in [('dep','OR'),('dep_sev','OR'),('phq9','beta')]:
    r=fit(d,y,[d.o3i_z],['O3I'],mask=mask,weight=wt,kind=kind);book.add('matched/'+label+'/'+y,r,meta={'analysis':label,'outcome':y,'weight':wt})
  for y in ['dep','dep_sev']:
   r=fit(da,y,[da.o3i_z],['O3I']);book.add('seed2/'+y,r,meta={'outcome':y})
  r=fit(da,da.phq9.ge(20).to_numpy(float),[da.o3i_z],['O3I']);book.add('seed2/PHQ>=20',r)
 rows,tests=book.save('seed_baselines');print(pd.DataFrame(rows)[['key','n','estimate','lo','hi','p']].to_string(index=False),flush=True)
if __name__=='__main__':main()
