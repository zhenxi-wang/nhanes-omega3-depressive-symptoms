"""Publication figures from full-precision aggregate estimates (600 dpi + vector PDF)."""
from pathlib import Path
import os,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,FixedLocator,FuncFormatter,NullFormatter
from matplotlib.patches import FancyBboxPatch
from statsmodels.stats.multitest import multipletests
from config import P,ROOT
_read_csv=pd.read_csv
def read_csv_preserving_categories(*args,**kwargs):
 kwargs.setdefault('keep_default_na',False);kwargs.setdefault('na_values',['']);return _read_csv(*args,**kwargs)
REF=ROOT/'reference_results';OUT=Path(os.environ.get('NHANES_FIGURE_DIR',P/'figures'));OUT.mkdir(parents=True,exist_ok=True)
def source(name):return P/name if (P/name).exists() else REF/name
def estimates(name):
 f=P/(name+'.csv')
 if f.exists():return read_csv_preserving_categories(f)
 d=read_csv_preserving_categories(REF/'Pooled_estimates_full_precision.csv');return d[d.Result_source.eq(name)].copy()
C=estimates('core');I=estimates('interactions');S=estimates('sensitivity');M=estimates('seed_baselines');D=read_csv_preserving_categories(source('descriptive.csv'))
f=P/'core_tests.csv';CT=read_csv_preserving_categories(f) if f.exists() else read_csv_preserving_categories(REF/'Joint_tests_full_precision.csv').query("Result_source=='core_tests'")
L={r.key:r for _,r in pd.concat([C,I,S,M]).iterrows()};NAVY='#173E58';RED='#B51F32';GRAY='#6D747B';LIGHT='#E9EDF0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':12,'axes.titleweight':'bold','pdf.fonttype':42,'ps.fonttype':42})
ITEMS=['Anhedonia','Depressed mood','Sleep disturbance','Fatigue or low energy','Appetite change','Worthlessness','Concentration difficulty','Psychomotor change','Self-harm thoughts']
def pval(p):return '<.001' if p<.001 else f'{p:.3f}'.removeprefix('0')
def save(fig,name):
 for obj in fig.findobj(match=matplotlib.text.Text):obj.set_text(obj.get_text().replace('>=','≥').replace('<=','≤'))
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white')
 fig.savefig(OUT/(name+'.png'),dpi=600,bbox_inches='tight',facecolor='white')
 fig.savefig(OUT/(name+'.tiff'),dpi=600,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
 plt.close(fig);print('FIGURE',name,flush=True)
def forest(ax,rows,labels,fill=None,color=NAVY):
 n=len(rows)
 for j,r in enumerate(rows):
  y=n-j-1;col=RED if fill is not None and fill[j] else color;ax.plot([r.lo,r.hi],[y,y],color=col,lw=1.3);ax.plot(r.estimate,y,'s',ms=5.2,mec=col,mfc=col if fill is None or fill[j] else 'white')
 ax.set_yticks(range(n),[str(x).replace('>=','≥').replace('<=','≤') for x in labels[::-1]]);ax.set_ylim(-.8,n-.2);ax.axvline(1,ls=':',color=GRAY,lw=1.1);ax.set_xscale('log');ax.set_xticks([.4,.6,.8,1,1.5,2]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(axis='x',color=LIGHT,lw=.7);ax.set_axisbelow(True);ax.set_xlabel('Adjusted OR per 1-SD higher Omega-3 Index')
def spline(ax,y,title):
 f=source('spline_'+y+'_all_points.csv')
 if not f.exists():f=P/('spline_'+y+'.csv')
 r=read_csv_preserving_categories(f).sort_values('o3i');col=RED if y=='dep_sev' else NAVY;ax.fill_between(r.o3i,r.lo,r.hi,color=col,alpha=.15);ax.plot(r.o3i,r.estimate,color=col,lw=1.8);ax.axhline(1,color=GRAY,ls=':',lw=1);ax.axvline(4,color=GRAY,ls='--',lw=.9);ax.set_yscale('log');ax.yaxis.set_major_locator(FixedLocator([.1,.25,.5,1,2,4]));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'));ax.yaxis.set_minor_formatter(NullFormatter());ax.set_xlabel('Erythrocyte Omega-3 Index (%)');ax.set_ylabel('Adjusted OR (reference 4%)');ax.set_title(title,loc='left')
 overall=CT[CT.key.eq('spline/'+y+'/overall')].iloc[0];non=CT[CT.key.eq('spline/'+y+'/nonlinear')].iloc[0];ax.text(.97,.97,f'P overall = {overall.p:.4f}\nP nonlinearity = {non.p:.3f}',ha='right',va='top',transform=ax.transAxes,fontsize=9)
def main():
 fig,axs=plt.subplots(1,2,figsize=(11.6,4.7),gridspec_kw={'wspace':.28});ax=axs[0]
 for lo,hi,col in [(0,4,'#F5E9EB'),(4,8,'#F8F0E2'),(8,15,'#E7F1F0')]:ax.axvspan(lo,hi,color=col)
 if (P/'full_v2.pkl').exists():
  d=pd.read_pickle(P/'full_v2.pkl');d=d[d.insample.eq(1)];bins=np.arange(0,15.25,.25);values,_=np.histogram(d.o3i,bins=bins,weights=100*d.w/d.w.sum());h=pd.DataFrame({'lower':bins[:-1],'upper':bins[1:],'weighted_percent':values})
 else:h=read_csv_preserving_categories(REF/'distribution_bins.csv')
 ax.bar(h.lower,h.weighted_percent/(h.upper-h.lower),width=h.upper-h.lower,align='edge',color=NAVY,alpha=.83,lw=0);ax.axvline(4,color=RED,ls='--',lw=1.1);ax.axvline(8,color='#237D7C',ls='--',lw=1.1);ax.set_xlim(0,15);ax.set_ylim(0,40);ax.set_ylabel('Weighted density (% per percentage point)');ax.set_xlabel('Erythrocyte Omega-3 Index (%)');ax.set_title('A  Omega-3 Index distribution',loc='left')
 for x,txt,col in [(1.6,'<4%\n50.14%',RED),(6,'4 to <8%\n47.54%','#98672E'),(11.5,'>=8%\n2.31%','#237D7C')]:ax.text(x,38,txt,ha='center',va='top',color=col,fontsize=10)
 ax.text(8.5,24,'Cardiovascular reference ranges;\nno depression-specific cut-point',fontsize=8.5,color=GRAY)
 ax=axs[1]
 for y,off,col,marker,lab in [('dep',-.10,NAVY,'o','PHQ-9 >=10'),('dep_sev',.10,RED,'s','PHQ-9 >=15')]:
  r=[D[D.key.eq(f'Quartile {q} {y}')].iloc[0] for q in [1,2,3,4]];v=np.array([t.estimate for t in r])*100;lo=np.array([t.lo for t in r])*100;hi=np.array([t.hi for t in r])*100;ax.errorbar(np.arange(1,5)+off,v,yerr=[v-lo,hi-v],fmt=marker+'-',color=col,lw=1.7,capsize=3,label=lab)
 ax.set_xticks([1,2,3,4],['Q1','Q2','Q3','Q4']);ax.set_ylim(0,23);ax.set_ylabel('Observed weighted prevalence (%)');ax.set_xlabel('Omega-3 Index quartile');ax.set_title('B  Depressive symptom prevalence',loc='left');ax.legend(frameon=False,fontsize=9);save(fig,'Figure1')
 fig,axs=plt.subplots(1,2,figsize=(11.5,4.9),gridspec_kw={'wspace':.5});rr=[L[f'threshold/{k}'] for k in [3,5,8,10,12,15,20]];forest(axs[0],rr,[f'PHQ-9 >={int(r.threshold)}  ({int(r.cases)} cases)' for r in rr]);axs[0].set_xlim(.5,1.6);axs[0].set_title('A  Nested PHQ-9 thresholds',loc='left');t=CT[CT.key.eq('severity_heterogeneity')].iloc[0];axs[0].text(0,-.22,f'Post hoc category heterogeneity: F(2, {t.df2:.2f}) = {t.F:.2f}, P = {pval(t.p)}\nOrdered cut-point trend: P = {pval(L["cumulative_trend"].p)}',transform=axs[0].transAxes,fontsize=8.4);spline(axs[1],'dep_sev','B  PHQ-9 >=15 spline');save(fig,'Figure2')
 rr=[L[f'item/DPQ{k:03d}'] for k in range(10,100,10)];bp=multipletests([r.p for r in rr],method='bonferroni')[1];fig=plt.figure(figsize=(11.6,5.5));gs=fig.add_gridspec(1,2,width_ratios=[1.15,1],wspace=.10);ax=fig.add_subplot(gs[0]);forest(ax,rr,ITEMS,[p<.05 for p in bp]);ax.set_xlim(.5,1.3);ax.set_title('Individual PHQ-9 symptoms',loc='left');tx=fig.add_subplot(gs[1]);tx.set_xlim(0,1);tx.set_ylim(ax.get_ylim());tx.axis('off')
 for x,txt in [(.02,'OR (95% CI)'),(.63,'Raw P'),(.88,'Bonferroni P')]:tx.text(x,8.55,txt,ha='left' if x<.1 else 'center',fontweight='bold',fontsize=9)
 for j,(r,p) in enumerate(zip(rr,bp)):
  y=8-j;col=RED if p<.05 else NAVY;tx.text(.02,y,f'{r.estimate:.3f} ({r.lo:.3f}–{r.hi:.3f})',va='center',fontsize=9,color=col);tx.text(.63,y,pval(r.p),ha='center',va='center',fontsize=9);tx.text(.88,y,pval(p),ha='center',va='center',fontsize=9)
 fig.text(.02,.005,'Filled red squares indicate Bonferroni-adjusted P<.05 across nine symptoms. Items are shown in questionnaire order.',fontsize=8,color=GRAY);save(fig,'Figure3')
 fig,ax=plt.subplots(figsize=(7.4,8.4));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1);steps=[('NHANES participants','11,933'),('Examined at MEC','8,860'),('Age >=18 years','6,337'),('After excluding identified pregnancies','6,296'),('Valid erythrocyte fatty acids','5,916'),('Complete PHQ-9 analytic sample','5,120')];ex=['Not examined: 3,073','Age <18 years: 2,523','Identified pregnancies: 41','No valid fatty-acid result: 380','Incomplete PHQ-9: 796']
 for j,(label,n) in enumerate(steps):
  y=.92-.145*j;ax.add_patch(FancyBboxPatch((.10,y-.043),.59,.085,boxstyle='round,pad=.008',fc=LIGHT,ec=NAVY,lw=1));ax.text(.395,y,label+'\nn = '+n,ha='center',va='center',fontsize=10)
  if j<5:ax.annotate('',xy=(.395,y-.098),xytext=(.395,y-.049),arrowprops={'arrowstyle':'->','color':NAVY});ax.text(.72,y-.073,ex[j],fontsize=8,va='center')
 ax.text(.395,.065,'M=40 MI retains all 5,120 participants.\n971 have >=1 Model 3 item missing; 4,149 have none.\nStructural non-applicability is handled separately.',ha='center',va='center',fontsize=9);save(fig,'FigureS1')
 sg=I[I.key.str.startswith('subgroup/')];rr=[r for _,r in sg.iterrows()];fig,ax=plt.subplots(figsize=(9.5,7.5));forest(ax,rr,[f'{r.domain}: {r.stratum} (n={int(r.n)})' for r in rr]);ax.set_xlim(.45,1.7);ax.set_title('Exploratory subgroups for PHQ-9 >=10',loc='left');fig.text(.02,.005,'Observed modifier membership. Supplement strata use WTDRD1; other strata use WTPH2YR.\nNo domain survives multiplicity correction; adjusted sex-interaction P = .122.',fontsize=8,color=GRAY);save(fig,'FigureS2')
 order=['sensitivity/Main MI','sensitivity/Corrected missing indicator','sensitivity/No item nonresponse','sensitivity/Age>=20 fully observed','diet_sensitivity/+ energy and EPA+DHA','sensitivity/Excluding CRP>10','matched/Supplement eligible baseline','sensitivity/No omega-3 supplement','sensitivity/No mental-health contact','sensitivity/No CVD cancer diabetes','sensitivity/No current smokers','sensitivity/BMI<40','sensitivity/Age<65','sensitivity/Age>=20','sensitivity/+ sleep duration','sensitivity/PHQ nonresponse IPW','medication/+ prescription','medication/no prescription','diet_sensitivity/Diet subset baseline','diet_sensitivity/+ prescription','diet_sensitivity/+ HEI','diet_sensitivity/+ HEI and prescription'];rr=[L[k+'/dep_sev'] for k in order];labels=[k.split('/',1)[1]+f' (n={int(r.n)})' for k,r in zip(order,rr)];fig,ax=plt.subplots(figsize=(10.4,10.7));forest(ax,rr,labels);ax.set_xlim(.45,2.15);ax.set_title('Sensitivity analyses for PHQ-9 >=15',loc='left');fig.text(.02,.005,'Dietary and supplement analyses use WTDRD1; the corresponding eligible baselines are shown.\nRestriction estimates address different selected populations and are not all statistically significant.',fontsize=8,color=GRAY);save(fig,'FigureS3')
 fig,axs=plt.subplots(1,2,figsize=(11.6,4.9),gridspec_kw={'wspace':.62});rr=[L[f'crp_binary/{y}/{h}'] for y in ['dep','dep_sev'] for h in [0,1]];labels=[f'PHQ-9 >={t}; hsCRP {s}' for t in [10,15] for s in ['<3','>=3']];forest(axs[0],rr,labels);axs[0].set_xlim(.45,1.6);axs[0].set_title('A  Observed hsCRP strata',loc='left');axs[0].text(0,-.22,'Interaction P: PHQ-9 >=10 = .507; PHQ-9 >=15 = .657',transform=axs[0].transAxes,fontsize=8)
 rr=[pd.Series({'estimate':1.,'lo':1.,'hi':1.})]+[L['joint/dep/'+k] for k in ['CRP only','low O3I only','both']];forest(axs[1],rr,['O3I>=4 / CRP<3 (reference)','O3I>=4 / CRP>=3','O3I<4 / CRP<3','O3I<4 / CRP>=3']);axs[1].set_xlim(.6,3.5);axs[1].set_title('B  Joint categories for PHQ-9 >=10',loc='left');axs[1].set_xlabel('Adjusted OR versus reference group');save(fig,'FigureS4')
 fa=C[C.key.str.startswith('fatty/dep/')];rr=[r for _,r in fa.iterrows()];fig,ax=plt.subplots(figsize=(9,6));forest(ax,rr,[r.exposure.replace('Saturated fatty acids','Selected saturated fatty acids')+f' (n={int(r.n)})' for r in rr]);ax.set_xlim(.58,1.5);ax.set_xlabel('Adjusted OR per 1-SD higher fatty-acid measure');ax.set_title('Fatty-acid measures and PHQ-9 >=10',loc='left');fig.text(.02,.005,'Each exposure uses its own observed SD. Selected SFAs are 14:0+16:0+18:0. None survives correction across eleven measures.',fontsize=8,color=GRAY);save(fig,'FigureS5')
 fig,ax=plt.subplots(figsize=(6.8,4.5));spline(ax,'dep','Four-knot spline for PHQ-9 >=10');save(fig,'FigureS6')
 for tag in ['full','diet']:
  f=P/('imputation_'+tag+'.json')
  if not f.exists():f=REF/('imputation_'+tag+'_diagnostics.json')
  ob=json.loads(f.read_text());keys=sorted(ob['rhat'],key=ob['rhat'].get,reverse=True)[:8];fig,axs=plt.subplots(4,2,figsize=(10.5,10));cols=[NAVY,RED,'#398782','#826698']
  for ax,k in zip(axs.flat,keys):
   for c,ch in enumerate(ob['traces']):ax.plot(np.arange(1,len(ch)+1),[it[k] for it in ch],lw=.8,alpha=.8,color=cols[c],label=f'Chain {c+1}')
   ax.axvline(50,color=GRAY,ls='--',lw=.8);ax.set_title(f'{k}: split R-hat {ob["rhat"][k]:.3f}',fontsize=10);ax.set_xlabel('Iteration');ax.set_ylabel('Imputed-cell summary',fontsize=9)
  axs.flat[0].legend(frameon=False,ncol=2,fontsize=7);fig.suptitle('Multiple-imputation diagnostics: '+tag+' sample',fontsize=14);fig.tight_layout(rect=(0,0,1,.97));save(fig,'FigureS7' if tag=='full' else 'FigureS8')
if __name__=='__main__':main()
