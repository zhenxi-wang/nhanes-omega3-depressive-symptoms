"""Derive the corrected NHANES cohort directly from public XPT files.

No input file is modified. Structural eligibility, unknown responses and true zeros
are handled explicitly before any imputation. See resources and the supplement.
"""
import sys,json,hashlib,platform
from pathlib import Path
import numpy as np,pandas as pd
from config import DATA,FPED,P
from model_spec import VS,STRUCT
sys.dont_write_bytecode=True
def rd(name,cols=None):
    f=DATA/(name+'.xpt')
    d=pd.read_sas(f,format='xport').replace(5.397605346934028e-79,0.0)
    return d[cols] if cols else d
def yn(s):return s.map({1:1.,2:0.})
def valid(s,codes):return s.where(~s.isin(codes))
def merge(d,t):
    if not t.SEQN.is_unique:raise ValueError('Duplicate participant identifiers in an input module')
    return d.merge(t,on='SEQN',how='left',validate='one_to_one')
def hei(d):
    f=pd.read_sas(FPED,format='sas7bdat')
    n=rd('DR1TOT_L')[['SEQN','DR1DRSTZ','DR1TKCAL','DR1TSFAT','DR1TMFAT','DR1TPFAT','DR1TSODI']]
    h=f.merge(n,on='SEQN',how='inner',validate='one_to_one',suffixes=('','_nut'));k=h.DR1TKCAL;den=k/1000;leg=h.DR1T_V_LEGUMES;pl=h.DR1T_PF_LEGUMES;c={}
    for name,val,mx,pts in [('fruit',h.DR1T_F_TOTAL,.8,5),('wholefruit',h.DR1T_F_CITMLB+h.DR1T_F_OTHER,.4,5),('vegetables',h.DR1T_V_TOTAL+leg,1.1,5),('greens_beans',h.DR1T_V_DRKGR+leg,.2,5),('wholegrains',h.DR1T_G_WHOLE,1.5,10),('dairy',h.DR1T_D_TOTAL,1.3,10),('protein',h.DR1T_PF_MPS_TOTAL+h.DR1T_PF_EGGS+h.DR1T_PF_NUTSDS+h.DR1T_PF_SOY+pl,2.5,5),('sea_plant',h.DR1T_PF_SEAFD_HI+h.DR1T_PF_SEAFD_LOW+h.DR1T_PF_NUTSDS+h.DR1T_PF_SOY+pl,.8,5)]:c[name]=(pts*val/den/mx).clip(0,pts)
    ratio=(h.DR1TMFAT+h.DR1TPFAT)/h.DR1TSFAT;c['fattyacids']=(10*(ratio-1.2)/1.3).clip(0,10)
    c['fattyacids']=c['fattyacids'].mask(h.DR1TSFAT.eq(0)&(h.DR1TMFAT+h.DR1TPFAT).gt(0),10).mask(h.DR1TSFAT.eq(0)&(h.DR1TMFAT+h.DR1TPFAT).eq(0),0)
    for name,val,lo,hi in [('sodium',h.DR1TSODI/k,1.1,2),('refinedgrains',h.DR1T_G_REFINED/den,1.8,4.3),('satfat',100*h.DR1TSFAT*9/k,8,16),('addedsugar',100*h.DR1T_ADD_SUGARS*16/k,6.5,26)]:c[name]=(10*(hi-val)/(hi-lo)).clip(0,10)
    c=pd.DataFrame(c);h['hei']=c.sum(axis=1,min_count=13).where(h.DR1DRSTZ.eq(1)&k.gt(0))
    assert h.hei.dropna().between(0,100).all()
    h[['SEQN','hei']].join(c).to_csv(P/'hei_components.csv',index=False)
    return merge(d,h[['SEQN','hei']])
def build():
    d=rd('DEMO_L')[['SEQN','RIDSTATR','RIAGENDR','RIDAGEYR','RIDRETH3','DMDEDUC2','DMDMARTZ','INDFMPIR','RIDEXPRG','WTINT2YR','WTMEC2YR','SDMVSTRA','SDMVPSU','DMDBORN4']].copy()
    d['age']=d.RIDAGEYR;d['female']=d.RIAGENDR.eq(2).astype(float);d['race']=d.RIDRETH3.map({1:'Mexican American',2:'Other Hispanic',3:'NH White',4:'NH Black',6:'NH Asian',7:'Other/Multiracial'})
    d['edu']=d.DMDEDUC2.map({1:'<High school',2:'<High school',3:'High school/GED',4:'Some college',5:'College graduate+'});d['marital']=d.DMDMARTZ.map({1:'Married/partnered',2:'Widowed/divorced/separated',3:'Never married'});d['pir']=d.INDFMPIR
    # Internal labels retain the original fit coding; display labels state the actual boundaries.
    d['pir_cat']=pd.cut(d.pir,[-.01,1.3,3.5,99],labels=['<1.3','1.3-3.5','>3.5']);d['usborn']=d.DMDBORN4.eq(1).astype(float)
    fa={'LBXPPE':'epa','LBXPHA':'dha','LBXPD3':'dpa_n3','LBXPAN':'ala','LBXPRA':'aa','LBXPNL':'la','LBXPGH':'dgla','LBXPD6':'dpa_n6','LBXPTA':'ada','LBXPST':'stearic','LBXPPM':'palmitic','LBXPOL':'oleic','LBXPPL':'palmitoleic','LBXPMR':'myristic','LBXPED':'eda','LBXPLG':'gla'}
    t=rd('FAR_L')[['SEQN','WTPH2YR']+list(fa)].rename(columns=fa);t['o3i']=t.epa+t.dha;t['n3_total']=t.epa+t.dha+t.dpa_n3+t.ala;t['n6_total']=t.la+t.aa+t.dgla+t.dpa_n6+t.ada+t.gla+t.eda;t['n6_n3']=t.n6_total/t.n3_total;t['aa_epa']=t.aa/t.epa;t['hufa_n3']=100*(t.epa+t.dpa_n3+t.dha)/(t.epa+t.dpa_n3+t.dha+t.aa+t.dgla+t.dpa_n6+t.ada);t['sfa']=t.stearic+t.palmitic+t.myristic;d=merge(d,t)
    t=rd('DPQ_L');items=[f'DPQ{i:03d}' for i in range(10,100,10)]
    for v in items:t[v]=t[v].where(t[v].between(0,3))
    t['phq9']=t[items].sum(axis=1,min_count=9)
    for v,k in [('dep',10),('dep_sev',15),('dep_mild',5)]:t[v]=t.phq9.ge(k).astype(float).where(t.phq9.notna())
    t['si']=t.DPQ090.ge(1).astype(float).where(t.DPQ090.notna());t['anhedonia']=t.DPQ010.ge(2).astype(float).where(t.DPQ010.notna());t['phq_som']=t[['DPQ030','DPQ040','DPQ050','DPQ080']].sum(axis=1,min_count=4);t['phq_cog']=t[['DPQ010','DPQ020','DPQ060','DPQ070','DPQ090']].sum(axis=1,min_count=5);im=t.DPQ100.where(t.DPQ100.between(0,3));t['impair']=im.ge(2).astype(float).where(im.notna());d=merge(d,t[['SEQN','phq9','dep','dep_sev','dep_mild','si','anhedonia','phq_som','phq_cog','impair']+items])
    t=rd('FNQ_L');f=t.FNQ510.where(t.FNQ510.isin([1,2,3,4,5]));t['anx_freq']=f.le(2).astype(float).where(f.notna())
    for fr,it,name in [('FNQ510','FNQ520','anx_severe'),('FNQ530','FNQ540','dep_wg_severe')]:t[name]=np.select([t[fr].eq(1)&t[it].eq(2),t[fr].isin([2,3,4,5])|(t[fr].eq(1)&t[it].isin([1,3]))],[1.,0.],default=np.nan)
    d=merge(d,t[['SEQN','anx_freq','anx_severe','dep_wg_severe']])
    for name,ren in [('HSCRP_L',{'LBXHSCRP':'crp'}),('BMX_L',{'BMXBMI':'bmi','BMXWAIST':'waist'}),('GHB_L',{'LBXGH':'hba1c'}),('BIOPRO_L',{'LBXSCR':'scr','LBXSAL':'alb','LBXSGTSI':'ggt','LBXSATSI':'alt'}),('HDL_L',{'LBDHDD':'hdl'}),('TCHOL_L',{'LBXTC':'tc'}),('TRIGLY_L',{'LBXTLG':'tg'}),('COT_L',{'LBXCOT':'cotinine'})]:d=merge(d,rd(name)[['SEQN']+list(ren)].rename(columns=ren))
    t=rd('SMQ_L');t['smoke']=np.select([t.SMQ020.eq(2),t.SMQ020.eq(1)&t.SMQ040.eq(3),t.SMQ020.eq(1)&t.SMQ040.isin([1,2])],['Never','Former','Current'],default=None);d=merge(d,t[['SEQN','smoke']])
    t=rd('ALQ_L');fr=valid(t.ALQ121,[77,99]);amount=valid(t.ALQ130,[777,999]);freq=fr.map({0:0,1:365,2:310,3:182,4:104,5:52,6:30,7:12,8:9,9:4.5,10:1.5});t['drinks_wk']=freq*amount/52;t['alc']=np.select([t.ALQ111.eq(2)|fr.eq(0),t.drinks_wk.le(7),t.drinks_wk.gt(7)],['None','Moderate (<=7/wk)','Heavy (>7/wk)'],default=None);d=merge(d,t[['SEQN','alc','drinks_wk']])
    t=rd('PAQ_L')
    def pa(q,u,mins):
        freq=valid(t[q],[7777,9999]);unit=t[u].str.decode('ascii').map({'D':7,'W':1,'M':1/4.345,'Y':1/52.14});duration=valid(t[mins],[7777,9999]);return (freq*unit*duration).where(freq.ne(0),0.)
    mod=pa('PAD790Q','PAD790U','PAD800');vig=pa('PAD810Q','PAD810U','PAD820');t['mvpa']=mod+2*vig;t['pa_active']=np.select([(mod.fillna(0)+2*vig.fillna(0)).ge(150),mod.notna()&vig.notna()],[1.,0.],default=np.nan);d=merge(d,t[['SEQN','mvpa','pa_active']])
    t=rd('DIQ_L')[['SEQN','DIQ010']];d=merge(d,t);d['diabetes']=np.select([d.DIQ010.eq(1)|d.hba1c.ge(6.5),d.DIQ010.isin([2,3])&d.hba1c.lt(6.5)],[1.,0.],default=np.nan)
    t=rd('MCQ_L');q=t[['MCQ160B','MCQ160C','MCQ160E','MCQ160F']];t['cvd']=np.select([q.eq(1).any(axis=1),q.eq(2).all(axis=1)],[1.,0.],default=np.nan);t['cancer']=yn(t.MCQ220);d=merge(d,t[['SEQN','cvd','cancer']])
    for name,raw,v in [('BPQ_L','BPQ020','htn'),('HIQ_L','HIQ011','insured'),('HUQ_L','HUQ090','mh_visit')]:t=rd(name);t[v]=yn(t[raw]);d=merge(d,t[['SEQN',v]])
    t=rd('FSQ_L');t['food_insec']=t.FSDAD.ge(3).astype(float).where(t.FSDAD.notna());d=merge(d,t[['SEQN','food_insec']]);d=merge(d,rd('SLQ_L')[['SEQN','SLD012']].rename(columns={'SLD012':'sleep_h'}));d=merge(d,rd('RXQ_RX_L')[['SEQN','RXQ033','RXQ050']]);d['rx']=yn(d.RXQ033)
    t=rd('DR1TOT_L');f=t[['DRD340','DRD360']];t['fish30']=np.select([f.eq(1).any(axis=1),f.eq(2).all(axis=1)],[1.,0.],default=np.nan);t['kcal']=t.DR1TKCAL;t['diet_epa_dha']=(t.DR1TP205+t.DR1TP226)*1000;d=merge(d,t.loc[t.DR1DRSTZ.eq(1),['SEQN','WTDRD1','kcal','diet_epa_dha','fish30']])
    ids=rd('DSQIDS_L');ids['o3sup']=ids.DSDSUPP.astype(str).str.upper().str.contains(r'OMEGA|FISH OIL|COD LIVER|KRILL|SALMON OIL|MENHADEN|ALGAL|\bEPA\b|\bDHA\b',regex=True,na=False).astype(float);d=merge(d,ids.groupby('SEQN',as_index=False).o3sup.max());t=rd('DSQTOT_L');t['any_sup']=yn(t.DSD010);d=merge(d,t[['SEQN','DSD010','any_sup']]);known=d.SEQN.isin(ids.SEQN)|d.DSD010.isin([1,2]);d['o3sup']=d.o3sup.fillna(0).where(known)
    k=np.where(d.female.eq(1),.7,.9);a=np.where(d.female.eq(1),-.241,-.302);r=d.scr/k;d['egfr']=142*np.minimum(r,1)**a*np.maximum(r,1)**(-1.200)*.9938**d.age*np.where(d.female.eq(1),1.012,1.)
    d['under20']=d.age.lt(20).astype(float);d.loc[d.age.lt(20),STRUCT]=np.nan;d['w']=d.WTPH2YR.fillna(0);d['insample']=(d.RIDSTATR.eq(2)&d.age.ge(18)&d.RIDEXPRG.ne(1)&d.w.gt(0)&d.o3i.notna()&d.phq9.notna()).astype(int);d['log_crp']=np.log(d.crp.clip(lower=.01));d['crp_hi']=d.crp.ge(3).astype(float).where(d.crp.notna());d['obese']=d.bmi.ge(30).astype(float).where(d.bmi.notna())
    d=hei(d);analytic=d.insample.eq(1);b=d[analytic];cuts=np.quantile(b.o3i,[.25,.5,.75]);d['o3i_q']=np.select([d.o3i.le(cuts[0]),d.o3i.le(cuts[1]),d.o3i.le(cuts[2]),d.o3i.gt(cuts[2])],[1,2,3,4],default=np.nan);med=d.loc[analytic].groupby('o3i_q').o3i.median();d['o3i_qmed']=d.o3i_q.map(med);scales={}
    for new,v,log in [('o3i_z','o3i',False),('epa_z','epa',False),('dha_z','dha',False),('dpa_z','dpa_n3',False),('ala_z','ala',False),('aa_z','aa',False),('la_z','la',False),('hufa_z','hufa_n3',False),('n6n3_z','n6_n3',False),('aaepa_z','aa_epa',True),('sfa_z','sfa',False)]:
        val=np.log(d[v]) if log else d[v];mean=val[analytic].mean();sd=val[analytic].std();d[new]=(val-mean)/sd;scales[new]={'mean':mean,'sd':sd,'n':int(val[analytic].notna().sum())}
    for name,items in [('mel',['DPQ010','DPQ020','DPQ080','DPQ050']),('non',['DPQ030','DPQ040','DPQ070','DPQ060','DPQ090'])]:
        d[name+'_sum']=d[items].sum(axis=1,min_count=len(items));v=d.loc[analytic,name+'_sum'];d[name+'_z']=(d[name+'_sum']-v.mean())/v.std();d[name+'_any']=d[items].max(axis=1).ge(2).astype(float).where(d[items].notna().all(axis=1))
    b=d[analytic];meta={'quartile_cutpoints':cuts.tolist(),'quartile_medians':{str(k):v for k,v in med.items()},'scales':scales,'knots_dep':np.quantile(b.o3i,[.05,.35,.65,.95]).tolist(),'knots_sev':np.quantile(b.o3i,[.10,.50,.90]).tolist()};(P/'metadata.json').write_text(json.dumps(meta,indent=2));d.to_pickle(P/'full_v2.pkl')
    rows=[]
    for v in VS:
        m=b[v].isna()&(~b.age.lt(20) if v in STRUCT else True);rows.append({'variable':v,'N':len(b),'missing_n':int(m.sum()),'missing_pct':100*m.mean(),'weighted_missing_pct':100*b.loc[m,'w'].sum()/b.w.sum(),'structural_n':int((b.age.lt(20)&b[v].isna()).sum()) if v in STRUCT else 0})
    pd.DataFrame(rows).to_csv(P/'missing_corrected.csv',index=False)
    provenance={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'inputs':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in DATA.glob('*.xpt')},'fped_sha256':hashlib.sha256(FPED.read_bytes()).hexdigest()};(P/'input_provenance.json').write_text(json.dumps(provenance,indent=2))
    missing=pd.DataFrame({v:b[v].isna()&(~b.age.lt(20) if v in STRUCT else True) for v in VS}).any(axis=1)
    assert len(b)==5120 and int(b.dep.sum())==671 and int(b.dep_sev.sum())==246
    assert int(missing.sum())==971
    print('DERIVATION VERIFIED',len(b),'participants;',int(missing.sum()),'with Model3 item nonresponse',flush=True)
    return d
if __name__=='__main__':build()
