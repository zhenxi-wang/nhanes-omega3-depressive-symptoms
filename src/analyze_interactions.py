from engine import *
def main():
 imps=load();base=raw();book=Book();z=base.o3i_z.to_numpy();validcrp=base.crp.notna().to_numpy();hi=base.crp.ge(3).to_numpy(float);low=base.o3i.lt(4).to_numpy(float)
 domains=[('Age',np.select([base.age.lt(45),base.age.lt(65)],[0,1],default=2),['18-44','45-64','65+'],np.ones(len(base),bool),'w'),('Sex',base.female.to_numpy(int),['Male','Female'],np.ones(len(base),bool),'w'),('BMI',base.bmi.ge(30).to_numpy(int),['<30','>=30'],base.bmi.notna().to_numpy(),'w'),('Smoking',base.smoke.ne('Never').to_numpy(int),['Never','Ever'],base.smoke.notna().to_numpy(),'w'),('Omega-3 supplement',base.o3sup.eq(1).to_numpy(int),['Non-user','User'],(base.o3sup.notna()&base.WTDRD1.gt(0)).to_numpy(),'WTDRD1'),('PIR',base.pir.ge(1.3).to_numpy(int),['<1.3','>=1.3'],base.pir.notna().to_numpy(),'w')]
 disease=base[['cvd','diabetes','cancer']];yes=disease.eq(1).any(axis=1);no=disease.eq(0).all(axis=1);domains.append(('Chronic disease',yes.to_numpy(int),['None','>=1 condition'],(base.age.ge(20)&(yes|no)).to_numpy(),'w'))
 outcomes=['dep','dep_sev','si','anx_severe','anx_freq','phq9']
 for m,d in enumerate(imps):
  for y in outcomes:
   kind='beta' if y=='phq9' else 'OR'
   for lev in [0,1]:
    rr=fit(d,y,[z],['O3I'],kind=kind,mask=validcrp&(hi==lev));book.add(f'crp_binary/{y}/{lev}',rr,meta={'outcome':y,'stratum':'<3' if lev==0 else '>=3'})
   rr=fit(d,y,[z,hi,z*hi],['O3I','CRP high','interaction'],kind=kind,mask=validcrp);book.add(f'crp_interaction/{y}',rr,'interaction',meta={'outcome':y});book.test(f'crp_interaction/{y}',rr,['interaction'],{'outcome':y})
  for y in ['dep','dep_sev']:
   g3=np.select([base.crp.lt(1),base.crp.lt(3)],[0,1],default=2)
   for lev in [0,1,2]:
    rr=fit(d,y,[z],['O3I'],mask=validcrp&(g3==lev));book.add(f'crp_three/{y}/{lev}',rr,meta={'outcome':y,'stratum':['<1','1-<3','>=3'][lev]})
   dummy=[(g3==k).astype(float) for k in [1,2]];rr=fit(d,y,[z]+dummy+[z*a for a in dummy],['O3I','CRP mid','CRP high','interaction mid','interaction high'],mask=validcrp);book.test(f'crp_three_interaction/{y}',rr,['interaction mid','interaction high'],{'outcome':y})
   groups=[(1-low)*hi,low*(1-hi),low*hi];terms=['CRP only','low O3I only','both'];rr=fit(d,y,groups,terms,mask=validcrp,need_if=True)
   for term in terms:book.add(f'joint/{y}/{term}',rr,term,meta={'outcome':y})
   R,V=marginal(rr,d,[dict(zip(terms,v)) for v in [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]]);add_risks(book,f'risk_interaction/{y}',R,V,{'outcome':y,'n':rr['n']})
   c=np.array([1,-1,-1,1]);rd=c@R;uv=c@V@c;book.add_est(f'risk_interaction/{y}/RD',rd,uv,'beta',{'outcome':y,'n':rr['n']});G=np.array([(R[1]+R[2]-R[3])/R[0]**2,-1/R[0],-1/R[0],1/R[0]]);book.add_est(f'risk_interaction/{y}/RERI',rd/R[0],G@V@G,'beta',{'outcome':y,'n':rr['n']})
  for name,group,labels,mask,wt in domains:
   for k,lab in enumerate(labels):
    rr=fit(d,'dep',[z],['O3I'],mask=mask&(group==k),weight=wt);book.add(f'subgroup/{name}/{lab}',rr,meta={'domain':name,'stratum':lab,'weight':wt})
   dm=[(group==k).astype(float) for k in range(1,len(labels))];terms=[f'interaction{k}' for k in range(1,len(labels))];rr=fit(d,'dep',[z]+dm+[z*v for v in dm],['O3I']+[f'group{k}' for k in range(1,len(labels))]+terms,mask=mask,weight=wt);book.test(f'subgroup_interaction/{name}',rr,terms,{'domain':name,'weight':wt})
   if len(labels)==2:book.add(f'subgroup_interaction/{name}/ratio',rr,terms[0],meta={'domain':name,'weight':wt})
  female=base.female.to_numpy(float)
  for y in ['dep','dep_sev','anhedonia','si']:
   for sex in [0,1]:
    rr=fit(d,y,[z],['O3I'],mask=female==sex);book.add(f'sex/{y}/{sex}',rr,meta={'outcome':y,'sex':'Male' if sex==0 else 'Female'})
   rr=fit(d,y,[z,female,z*female],['O3I','sex indicator','interaction']);book.add(f'sex_interaction/{y}',rr,'interaction',meta={'outcome':y})
  if (m+1)%5==0:print('INTERACTIONS',m+1,'/40',flush=True);(P/'progress_interactions.json').write_text(json.dumps({'imputations':m+1}))
 rows,tests=book.save('interactions');print('INTERACTIONS COMPLETE',len(rows),len(tests),flush=True)
if __name__=='__main__':main()
