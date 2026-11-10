from engine import *
from shutil import copyfile
_read_csv=pd.read_csv
def read_csv_preserving_categories(*args,**kwargs):
 kwargs.setdefault('keep_default_na',False);kwargs.setdefault('na_values',['']);return _read_csv(*args,**kwargs)
O=P/'deliverables';T=O/'tables';T.mkdir(parents=True,exist_ok=True)
C=read_csv_preserving_categories(P/'core.csv');I=read_csv_preserving_categories(P/'interactions.csv');S=read_csv_preserving_categories(P/'sensitivity.csv');X=read_csv_preserving_categories(P/'extra.csv');IT=read_csv_preserving_categories(P/'interactions_tests.csv');CT=read_csv_preserving_categories(P/'core_tests.csv');D=read_csv_preserving_categories(P/'descriptive.csv');M=read_csv_preserving_categories(P/'seed_baselines.csv');LOOK={r.key:r for _,r in pd.concat([C,I,S,X,M]).iterrows()}
ITEMS={'DPQ010':'Anhedonia','DPQ020':'Depressed mood','DPQ030':'Sleep disturbance','DPQ040':'Fatigue/low energy','DPQ050':'Appetite change','DPQ060':'Worthlessness','DPQ070':'Concentration difficulty','DPQ080':'Psychomotor change','DPQ090':'Self-harm thoughts'}
YL={'dep':'PHQ-9 >=10','dep_sev':'PHQ-9 >=15','dep_mild':'PHQ-9 >=5','si':'Suicidal ideation (item9 >=1)','anhedonia':'Anhedonia (item1 >=2)','impair':'Functional impairment (very/extremely difficult)','anx_freq':'Anxiety daily/weekly','anx_severe':'Severe anxiety','dep_wg_severe':'Severe depressed affect','mh_visit':'Mental-health contact','phq9':'PHQ-9 total score','phq_som':'Somatic subscale','phq_cog':'Cognitive-affective subscale'}
def num(v,dp=3,null=1):
 if pd.isna(v):return '—'
 for k in range(dp,7):
  s=f'{v:.{k}f}'
  if float(s)!=null or v==null:return s
 return s
def pv(v):
 if pd.isna(v):return '—'
 if v<.0001:return '<0.0001'
 if v<.001:return f'{v:.4f}'
 s=f'{v:.3f}'
 return f'{v:.4f}' if (v<.05)!=(float(s)<.05) else s
def ci(r,scale=1,dp=3):
 null=1 if r.get('kind','OR')=='OR' else 0
 return f'{num(r.estimate*scale,dp,null)} ({num(r.lo*scale,dp,null)}–{num(r.hi*scale,dp,null)})'
def desc(key):return D.loc[D.key.eq(key)].iloc[0]
def label(r):
 a=r.key.split('/')
 if a[0]=='mutual':return YL[a[1]]+'; '+({'EPA':'EPA adjusted for DHA','DHA':'DHA adjusted for EPA','EPA-DHA':'EPA minusDHA log-OR coefficient contrast'}[a[2]])
 if a[0]=='crp_three':return YL[a[1]]+'; hsCRP '+str(r.stratum)+' mg/L'
 if a[0]=='cluster':return {'mel_any':'Any melancholic-like symptom >=2','non_any':'Any remaining symptom >=2','mel_z':'Standardised melancholic-like score','non_z':'Standardised remaining-symptom score'}[a[1]]
 if a[0]=='cluster_comparison':return 'Post hoc between-cluster '+('ratio of ORs' if a[1]=='OR' else 'standardised beta difference')
 if a[0]=='clinical':return YL[a[1]]+'; '+a[2]+' versus>=8%'
 if a[0]=='conditional':return 'PHQ-9 '+a[1]
 if a[0]=='diet':return YL[a[2]]+'; '+a[3]+'; '+a[1]+' weights'
 if a[0]=='severity_extra':return 'PHQ-9 '+a[1]+(' versus0–4' if a[1] in ['15-19','20-27'] else '')
 if a[0]=='unadjusted':return YL[a[1]]+'; unadjusted'
 if a[0]=='seed2':return YL.get(a[1],a[1])+'; alternative random seed'
 if a[0]=='splinecoef':return YL[a[1]]+'; '+a[2].replace('spline_','')
 if a[0] in ['dep','dep_sev']:return YL[a[0]]+f'; O3I={r.o3i:g}% versus4%'
 return r.key
TABLES=[]
def put(name,title,df,note=''):
 df=df.map(lambda v:v.replace('Heavy (>7/wk)','>7 drinks/week').replace('Moderate (<=7/wk)','<=7 drinks/week') if isinstance(v,str) else v)
 df.to_csv(T/(name+'.csv'),index=False);TABLES.append({'name':name,'title':title,'note':note,'rows':len(df)});return df
def general_rows(df,labels=None):
 rows=[]
 for _,r in df.iterrows():rows.append({'Analysis':labels.get(r.key,r.key) if labels else label(r),'n':int(r.n) if pd.notna(r.get('n')) else '—','Estimate (95% CI)':ci(r),'P':'—' if r.get('kind')=='risk' else pv(r.p)})
 return pd.DataFrame(rows)
def main():
 t=read_csv_preserving_categories(P/'table1.csv');t['Weight']=t.Weight.replace({'w':'WTPH2YR'});t['Characteristic']=t.Characteristic.replace({'Alcohol: Moderate (<=7/wk), %':'Alcohol: <=7 drinks/week, %','Alcohol: Heavy (>7/wk), %':'Alcohol: >7 drinks/week, %'});idx=t.Characteristic.eq('Weighted N, million')
 for q in ['Overall','Q1','Q2','Q3','Q4']:t.loc[idx,q]=t.loc[idx,q].astype(float).map(lambda v:f'{v:.1f}')
 put('Table1','Table 1. Observed weighted characteristics by Omega-3 Index quartile',t,'Available-case denominators; not MI-completed descriptions. Adult-only education, marital status, CVD and cancer variables exclude structural non-applicability. Supplement, fish and dietary rows use WTDRD1; other rows use WTPH2YR. PIR covariate categories are <=1.3, >1.3–3.5 and >3.5. Quartiles are unweighted empirical categories.')
 put('Table1_denominators','Available-case denominators for Table 1',read_csv_preserving_categories(P/'table1_denominators.csv'))
 for y,tag in [('dep','A'),('dep_sev','B')]:
  rows=[]
  for model in ['m1','m2','m3','m4']:
   r=LOOK[f'model/{y}/{model}/SD'];row={'Model':model.upper(),'n':int(r.n),'Per-SD OR (95% CI)':ci(r),'P per SD':pv(r.p)}
   for q in ['Q2','Q3','Q4']:row[q+' vs Q1']=ci(LOOK[f'model/{y}/{model}/{q}']);row['P '+q]=pv(LOOK[f'model/{y}/{model}/{q}'].p)
   row['P trend']=pv(LOOK[f'model/{y}/{model}/trend'].p);rows.append(row)
  put('Table2'+tag,f'Table 2{tag}. '+YL[y],pd.DataFrame(rows),'Model 1 uses observed complete covariates and 15 design df. Models 2–4 use expanded M=40 MI; Model 4 includes imputed log hsCRP. All exposure SDs use 1.576007 percentage points. Q1 is the reference; quartile median values are used for trend tests.')
 cat=C[C.key.str.startswith('severity_category/')].copy();labels={r.key:r.key.split('/')[-1]+' versus 0–4' for _,r in cat.iterrows()};d=general_rows(cat,labels);d=pd.concat([d,general_rows(C[C.key.isin(['severity_endpoint','cumulative_trend'])],{'severity_endpoint':'>=15 versus5–9 coefficient contrast: ratio of ORs','cumulative_trend':'Cumulative-logit cut-point multiplier'})],ignore_index=True);put('Table2C','Table 2C / S12. Post hoc severity heterogeneity',d,'Nested threshold estimates are separate from mutually exclusive-category analyses. Joint MI D1 test and its finite denominator df are in TableS12_tests. The ordered trend is not statistically significant.')
 put('Table3A','Table 3A / S7. Mental-health outcomes',general_rows(C[C.key.str.startswith('secondary/')],{f'secondary/{v}':k for v,k in YL.items()}),'OR for binary outcomes and beta in original score points for continuous outcomes. Outcomes were not imputed. Functional impairment has a conditional denominator of 3,853. Raw P values are exploratory except where multiplicity correction is specified.')
 rows=[]
 for y in ['dep','dep_sev','si','anx_severe','anx_freq','phq9']:
  a=LOOK[f'crp_binary/{y}/0'];b=LOOK[f'crp_binary/{y}/1'];r=LOOK[f'crp_interaction/{y}'];rows.append({'Outcome':YL[y],'n CRP<3':int(a.n),'CRP<3 estimate':ci(a),'P CRP<3':pv(a.p),'n CRP>=3':int(b.n),'CRP>=3 estimate':ci(b),'P CRP>=3':pv(b.p),'Interaction ratio of ORs / beta difference':ci(r),'P interaction':pv(r.p)})
 put('Table3B','Table 3B / S8. Effect modification by observed hsCRP',pd.DataFrame(rows),'Stratum membership is fixed using observed hsCRP; 125 participants without observed hsCRP are excluded from stratification. Other covariates are multiply imputed. Continuous PHQ-9 uses beta and an interaction beta difference.')
 put('TableS1','Table S1. Data-source inventory',read_csv_preserving_categories(P/'tableS1.csv'),'Original NHANES files were supplied locally; FPED and the HEI scoring specification were retrieved from USDA/NCI. The original NHANES download date was not independently established.')
 put('TableS2','Table S2. Included versus excluded participants',read_csv_preserving_categories(P/'tableS2.csv'),'Unweighted available-case descriptions. Included n=5,120; excluded for incomplete PHQ-9 n=796. Observed denominators are supplied because unknown responses are not classified as negative.')
 for y,suffix in [('dep_sev',''),('dep','b')]:
  sub=C[C.key.str.startswith('fullcoef/'+y+'/')].copy();lab={r.key:str(r.covariate).replace('pir_cat: <1.3','PIR <=1.3').replace('pir_cat: 1.3-3.5','PIR >1.3–3.5') for _,r in sub.iterrows()};put('TableS3'+suffix,'Table S3'+suffix+'. Complete MI-pooled Model 3 coefficients: '+YL[y],general_rows(sub,lab),'No item-missingness indicators are fitted in the MI model. A shared structural non-applicability indicator identifies age18–19; its coefficient is a nuisance adjustment and must not be interpreted as a disease effect.')
 th=read_csv_preserving_categories(P/'threshold_multiplicity.csv');rows=[]
 for _,r in th.iterrows():
  pr=desc('Threshold '+str(int(r.threshold)));rows.append({'PHQ-9 threshold':int(r.threshold),'n':int(r.n),'Cases':int(r.cases),'Weighted prevalence %':f'{100*pr.estimate:.2f}','OR (95% CI)':ci(r),'Raw P':pv(r.p),'Bonferroni P (7 exploratory thresholds)':pv(r.p_bonferroni_7),'BH P (7 thresholds)':pv(r.p_BH_7)})
 put('TableS4','Table S4. Nested PHQ-9 thresholds',pd.DataFrame(rows),'Thresholds are nested and overlapping. Multiplicity columns for the seven thresholds are additional exploratory checks; they do not convert these analyses into prospectively confirmatory tests.')
 it=read_csv_preserving_categories(P/'items_multiplicity.csv');rows=[]
 for _,r in it.iterrows():
  pr=desc('Item '+r['item']);rows.append({'Item':ITEMS[r['item']],'n':int(r.n),'Cases':int(r.cases),'Weighted prevalence %':f'{100*pr.estimate:.2f}','OR (95% CI)':ci(r),'Raw P':pv(r.p),'Bonferroni P':pv(r.p_bonferroni),'BH P':pv(r.p_BH),'Survives Bonferroni':bool(r.p_bonferroni<.05)})
 items=pd.DataFrame(rows);put('TableS5','Table S5. PHQ-9 item associations',items.iloc[:,:6]);put('TableS13','Table S13. Multiplicity across nine PHQ-9 items',items,'Only psychomotor change and depressed mood survive either Bonferroni or BH correction.')
 for y,suffix in [('dep',''),('dep_sev','b')]:
  f=read_csv_preserving_categories(P/('fatty_'+y+'_multiplicity.csv'));rows=[{'Exposure':r.exposure.replace('Saturated fatty acids','Selected saturated fatty acids (14:0+16:0+18:0)'),'n':int(r.n),'OR (95% CI)':ci(r),'Raw P':pv(r.p),'Bonferroni P (11)':pv(r.p_bonferroni),'BH P (11)':pv(r.p_BH)} for _,r in f.iterrows()];put('TableS6'+suffix,'Table S6'+suffix+'. Fatty-acid associations: '+YL[y],pd.DataFrame(rows),'Each outcome uses a separate exploratory family of eleven measures. Missing alternative fatty-acid exposures were not imputed; n therefore varies. Figure S5 displays the original primary-outcome family.')
 put('TableS6c','Mutually adjusted EPA and DHA and coefficient contrasts',general_rows(C[C.key.str.startswith('mutual/')]),'The EPA-minus-DHA contrast is a difference in log-OR coefficients for separately standardised exposures. A non-significant contrast does not establish equivalence.')
 put('TableS7','Table S7. Secondary mental-health outcomes',read_csv_preserving_categories(T/'Table3A.csv'))
 put('TableS8','Table S8. Binary hsCRP strata and interaction',read_csv_preserving_categories(T/'Table3B.csv'))
 rows=[]
 for y in ['dep','dep_sev']:
  for k,label in enumerate(['O3I>=4 / CRP<3','O3I>=4 / CRP>=3','O3I<4 / CRP<3','O3I<4 / CRP>=3']):
   v=desc(f'Joint {k} {y}');r=None if k==0 else LOOK[f'joint/{y}/'+['','CRP only','low O3I only','both'][k]];rows.append({'Outcome':YL[y],'Group':label,'n':int(v.n),'Weighted prevalence % (95% CI)':f'{100*v.estimate:.2f} ({100*v.lo:.2f}–{100*v.hi:.2f})','Adjusted OR (95% CI)':'1.000 [reference]' if r is None else ci(r),'P':'—' if r is None else pv(r.p)})
 put('TableS8b','Table S8b. Joint Omega-3 Index and hsCRP categories',pd.DataFrame(rows),'Observed prevalence and model-standardised risks are different quantities. Model-standardised risks and risk-scale interaction are in S16.')
 sub=I[I.key.str.startswith('crp_three/')];put('TableS8c','Table S8c. Three observed hsCRP strata',general_rows(sub),'Joint 2-df MI interaction tests are listed below. Stratum-specific P values do not test interaction.');put('TableS8c_tests','Three-category hsCRP joint tests',IT[IT.key.str.startswith('crp_three_interaction/')])
 sg=read_csv_preserving_categories(P/'subgroup_multiplicity.csv');rows=[]
 for _,r in I[I.key.str.startswith('subgroup/')].iterrows():
  tr=sg[sg.domain.eq(r.domain)].iloc[0];rows.append({'Domain':r.domain,'Stratum':r.stratum,'n':int(r.n),'Weight':r.weight.replace('w','WTPH2YR'),'OR (95% CI)':ci(r),'P within stratum':pv(r.p),'Interaction F':f'{tr.F:.3f}','df1':int(tr.df1),'df2':f'{tr.df2:.3f}','Raw P interaction':pv(tr.p),'Bonferroni P interaction (7)':pv(tr.p_bonferroni_7),'BH P interaction (7)':pv(tr.p_BH_7)})
 put('TableS9','Table S9. Exploratory subgroups and domain interaction tests',pd.DataFrame(rows),'Modifier membership is based on observed data; missing modifiers are not assigned to a group. BMI, smoking and PIR domains exclude 42, 9 and 604 unknowns. Chronic-disease classification is restricted to age>=20 and observed sufficient status (n=4,891). Supplement analyses use known status and positive WTDRD1 (n=4,261). No domain interaction survives correction across seven domains. PIR subgroup boundary is strictly <1.3 versus >=1.3.')
 sources=pd.concat([S[S.key.str.startswith('sensitivity/')|S.key.str.startswith('medication/')|S.key.str.startswith('diet_sensitivity/')],M[M.key.str.startswith('matched/')]])
 sources=sources.copy();sources.loc[sources.key.str.startswith('diet_sensitivity/+ prescription/'),'analysis']='Diet + prescription'
 rows=[]
 for label,g in sources.groupby('analysis',sort=False):
  row={'Analysis':label};
  for y in ['dep','dep_sev','phq9']:
   rr=g[g.outcome.eq(y)]
   if rr.empty:continue
   r=rr.iloc[0];row['n']=int(r.n);wt=r.get('weight','w');row['Weight']='WTPH2YR' if pd.isna(wt) or wt=='w' else wt;row[YL[y]+' estimate (95% CI)']=ci(r);row[YL[y]+' P']=pv(r.p)
  rows.append(row)
 put('TableS10','Table S10. Updated sensitivity analyses for binary and continuous PHQ-9 outcomes',pd.DataFrame(rows),'All three outcome columns are recalculated. Restrictions use observed exclusion variables. No-item-nonresponse records retain structural non-applicability categories (n=4,149); the age>=20 fully observed analysis has n=3,989. IPW variances additionally account for estimation of the response model. Supplement restriction is compared with its matched eligibility baseline under WTDRD1.')
 rows=[]
 for y in ['dep','dep_sev','anhedonia','si']:
  row={'Outcome':YL[y]}
  for sex in [0,1]:
   r=LOOK[f'sex/{y}/{sex}'];lab='Men' if sex==0 else 'Women';row[lab+' n']=int(r.n);row[lab+' OR (95% CI)']=ci(r);row[lab+' P']=pv(r.p)
  r=LOOK['sex_interaction/'+y];row['Female/male ratio of ORs']=ci(r);row['Raw P interaction']=pv(r.p);rows.append(row)
 put('TableS11','Table S11. Sex-stratified associations',pd.DataFrame(rows),'Exploratory raw P values. The primary-outcome sex interaction has Bonferroni/BH P=0.122 across the seven subgroup domains.')
 put('TableS12','Table S12. Formal post hoc severity analyses',read_csv_preserving_categories(T/'Table2C.csv'));put('TableS12_tests','Table S12. Joint heterogeneity and spline tests',CT)
 put('TableS12b','Additional post hoc separation of PHQ15–19 and20–27',general_rows(X[X.key.str.startswith('severity_extra/')]),'The 20–27 versus15–19 coefficient contrast is exploratory, not adjusted for the decision to inspect these categories. It does not reach P<.05.')
 put('TableS14','Table S14. Post hoc symptom-cluster descriptions and formal comparisons',general_rows(C[C.key.str.startswith('cluster/')|C.key.str.startswith('cluster_comparison/')]),'The grouping was selected from the same item-level results. Formal comparison P values are descriptive and are not selection-adjusted or independent confirmation. Omit them from main-text emphasis and Figure3 annotations.')
 mi=C[C.key.str.match(r'model/(dep|dep_sev)/m3/(SD|Q4)$')].copy();put('TableS15','Table S15. Expanded MI results and variance decomposition',mi[['key','n','estimate','lo','hi','p','M','U','B','T','lambda_','FMI','df','MCSE']],'U, B and T are on the log-OR scale. lambda and FMI are fractions, not percentages. FMI includes the finite-df adjustment; it is not the fraction of participants with missing data.')
 diag=[]
 for tag in ['full','diet','full_20261121']:
  ob=json.loads((P/('imputation_'+tag+'.json')).read_text());diag.append({'Set':tag,'M':len(ob['fingerprints']),'Distinct data sets':len(set(ob['fingerprints'])),'Iterations/chain':ob['cycles'],'Burn-in':ob['burn'],'Thinning':ob['thin'],'Maximum split Rhat':max(ob['rhat'].values()),'Maximum conditional iterations':max(ob['max_iterations'])})
 put('TableS15_diagnostics','MI convergence and reproducibility diagnostics',pd.DataFrame(diag));put('TableS15_seed_check','Alternative random-seed estimates',general_rows(M[M.key.str.startswith('seed2/')]))
 rows=[]
 for _,r in I[I.key.str.startswith('risk_interaction/')].iterrows():
  typ=r.key.split('/')[-1];rows.append({'Outcome':YL[r.outcome],'Estimand':{'risk0':'Risk R00: O3I>=4 / CRP<3','risk1':'Risk R01: CRP>=3 only','risk2':'Risk R10: O3I<4 only','risk3':'Risk R11: both','RD':'Additive risk-difference interaction','RERI':'Risk-scale RERI'}[typ],'n':int(r.n),'Estimate (95% CI)':ci(r,100 if typ.startswith('risk') or typ=='RD' else 1,2 if typ.startswith('risk') or typ=='RD' else 3),'Unit':'%' if typ.startswith('risk') else ('percentage points' if typ=='RD' else 'risk-ratio scale'),'P':'—' if typ.startswith('risk') else pv(r.p)})
 put('TableS16','Table S16. Model-standardised risks and additive interaction',pd.DataFrame(rows),'Within each imputation, the four risks were standardised over the same observed-CRP sample. RERI=(R11−R10−R01+R00)/R00. Variances include regression and weighted standardisation uncertainty; estimates and variances were pooled, not confidence limits.')
 put('TableS17','Table S17. Dietary EPA+DHA versus the erythrocyte biomarker',general_rows(S[S.key.str.startswith('diet/')]),'All twelve comparisons use the same 4,273 positive-energy reliable recalls. Dietary exposure is log(1+EPA+DHA mg/d), standardised within this subset. The diet-adjusted biomarker model includes log energy and log(1+EPA+DHA). Weight-specific MI sets are used; the biomarker SD is always the full-sample SD. The former n=4,278 included five zero-energy recalls.')
 missing=read_csv_preserving_categories(AUD/'missing_corrected.csv');missing=pd.concat([missing,pd.DataFrame([dict(variable='At least one Model3 covariate missing',N=5120,missing_n=971,missing_pct=971/5120*100,weighted_missing_pct=19.199014802367376,structural_n=0)])],ignore_index=True);put('TableS18','Table S18. Model3 item nonresponse and structural non-applicability',missing,'Structural counts refer to the same 211 age18–19 participants across rows and must not be summed. No item nonresponse: n=4,149. Strictly fully observed adults age>=20: n=3,989.')
 b=raw();aux=[]
 for v in ['log_crp','sleep_h','o3sup','fish30','mh_visit','anx_freq','anx_severe','dep_wg_severe','hei']:
  m=b[v].isna();aux.append({'Variable':v,'N':len(b),'Unavailable_n':int(m.sum()),'Unavailable_%':100*m.mean(),'Weighted_unavailable_%':100*b.loc[m,'w'].sum()/b.w.sum()})
 put('TableS18b','Additional-variable availability audit',pd.DataFrame(aux),'Unavailable does not always mean item nonresponse: dietary/supplement modules and positive-energy eligibility contribute. Outcomes, supplement status and alternative fatty-acid exposures were not imputed. log hsCRP and sleep duration were imputed as auxiliary/model covariates.')
 wanted=[('Revised principal Model3','sensitivity/Main MI'),('+ prescription use','medication/+ prescription'),('No prescription users','medication/no prescription'),('Diet subset baseline','diet_sensitivity/Diet subset baseline'),('Diet + prescription','diet_sensitivity/+ prescription'),('Diet + HEI','diet_sensitivity/+ HEI'),('Diet + HEI + prescription','diet_sensitivity/+ HEI and prescription')];rows=[]
 for label,key in wanted:
  row={'Analysis':label}
  for y in ['dep','dep_sev']:
   r=LOOK[key+'/'+y];row['n']=int(r.n);row[YL[y]+' OR (95% CI)']=ci(r);row[YL[y]+' P']=pv(r.p)
  rows.append(row)
 put('TableS19','Table S19. Prescription medication and HEI-2020 sensitivity analyses',pd.DataFrame(rows),'Full-sample models use WTPH2YR; dietary models use WTDRD1 and the same n=4,273. Five unknown prescription-use responses are imputed; restriction to non-users uses only explicit no responses. These analyses cannot identify antidepressant-specific confounding.')
 put('Clinical_categories','Clinical reference-category associations',general_rows(C[C.key.str.startswith('clinical/')]),'Reference category is O3I>=8%; these cardiovascular-derived categories are not depression-specific thresholds.')
 marg=[]
 for _,r in C[C.key.str.startswith('marginal_')].iterrows():
  a=r.key.split('/');s=a[2]
  if a[0]=='marginal_quartile':lab='Q'+str(int(s[-1])+1) if s.startswith('risk') else 'Q1 minusQ4'
  else:lab={'risk0':'O3I>=4%','risk1':'O3I<4%','low-high':'O3I<4% minus>=4%'}[s]
  marg.append({'Outcome':YL[a[1]],'Category/contrast':lab,'n':int(r.n),'Estimate (95% CI)':ci(r,100,2),'Unit':'%' if s.startswith('risk') else 'percentage points','P':'—' if s.startswith('risk') else pv(r.p)})
 put('Marginal_prevalence','Model-standardised prevalence and differences',pd.DataFrame(marg),'Q1–Q4 refers to Q1 minusQ4; low-high refers to O3I<4 minus>=4. Confidence intervals incorporate the sample distribution and model fitting. These are model-standardised descriptions, not causal intervention effects.')
 put('Conditional_severity','Conditional severity among symptomatic respondents',general_rows(C[C.key.str.startswith('conditional/')]),'Conditioning on symptoms can induce selection effects. These estimates are not independent confirmation.')
 for name,title in [('national_totals','Weighted national totals'),('descriptive','National and subgroup descriptive estimates'),('dietary_correlation','Unweighted Spearman dietary correlation'),('evalues','E-values'),('extra','Additional severity and unadjusted results')]:put(name,title,read_csv_preserving_categories(P/(name+'.csv')))
 put('Spline_coefficients','Restricted cubic spline coefficients',general_rows(C[C.key.str.startswith('splinecoef/')]));put('Spline_tests','Spline overall and nonlinearity tests',CT[CT.key.str.startswith('spline/')]);put('Spline_anchor_OR','Spline ORs at2%,4%,6%,8%',general_rows(read_csv_preserving_categories(P/'spline_curves.csv').query('o3i in [2,4,6,8]')))
 for y in ['dep','dep_sev']:copyfile(P/('spline_'+y+'.csv'),T/('spline_'+y+'_all_points.csv'))
 put('Exposure_scaling','Exposure standardisation parameters',pd.DataFrame([dict(exposure=k,**v) for k,v in META['scales'].items()]));(O/'table_catalog.json').write_text(json.dumps(TABLES,indent=2));print('TABLES COMPLETE',len(TABLES),'aggregate rows',sum(t['rows'] for t in TABLES),flush=True)
if __name__=='__main__':main()
