import numpy as np,pandas as pd
RACE_REF = 'NH White'
RACES = ['Mexican American', 'Other Hispanic', 'NH Black', 'NH Asian', 'Other/Multiracial']
EDU = ['<High school', 'High school/GED', 'Some college']          # ref = College graduate+
PIR = ['<1.3', '1.3-3.5']                                          # ref = >3.5
MAR = ['Widowed/divorced/separated', 'Never married']              # ref = Married/partnered
SMK = ['Former', 'Current']                                        # ref = Never
ALC = ['None', 'Heavy (>7/wk)']                                    # ref = Moderate

VS=['age', 'female', 'race', 'edu', 'pir_cat', 'marital', 'insured', 'food_insec', 'usborn', 'smoke', 'alc', 'pa_active', 'bmi', 'diabetes', 'cvd', 'cancer', 'htn', 'egfr']
STRUCT=['edu', 'marital', 'cvd', 'cancer']
def Xcorr(d,missing=False):
 cols=[np.ones(len(d)),(d.age.to_numpy()-50)/10,((d.age.to_numpy()-50)/10)**2,d.female.to_numpy(float),d.age.lt(20).to_numpy(float)]
 names=['Intercept','Age/10','Age squared','Female','Not assessed age18-19']
 for v,levs in [('race',RACES),('edu',EDU),('pir_cat',PIR),('marital',MAR),('smoke',SMK),('alc',ALC)]:
  for lv in levs: cols.append(d[v].eq(lv).fillna(False).to_numpy(float));names.append(v+': '+lv)
  m=d[v].isna() & (~d.age.lt(20) if v in STRUCT else True)
  if missing and m.any():cols.append(m.to_numpy(float));names.append(v+' missing')
 for v in ['insured','food_insec','usborn','pa_active','bmi','diabetes','cvd','cancer','htn','egfr']:
  val=d[v].to_numpy(float)
  if v=='bmi':val=(val-28)/5
  if v=='egfr':val=(val-90)/10
  cols.append(np.nan_to_num(val));names.append(v)
  m=d[v].isna() & (~d.age.lt(20) if v in STRUCT else True)
  if missing and m.any():cols.append(m.to_numpy(float));names.append(v+' missing')
 return np.column_stack(cols),names

def rcs(x, knots):
    """Restricted cubic spline basis (Harrell), returns k-2 columns."""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    K = len(k)
    out = []
    den = (k[-1] - k[0]) ** 2
    for j in range(K - 2):
        t = (np.maximum(x - k[j], 0) ** 3
             - np.maximum(x - k[K - 2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[K - 2])
             + np.maximum(x - k[-1], 0) ** 3 * (k[K - 2] - k[j]) / (k[-1] - k[K - 2]))
        out.append(t / den)
    return np.column_stack(out)
