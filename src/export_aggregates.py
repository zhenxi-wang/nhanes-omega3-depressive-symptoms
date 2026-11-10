from config import P
import pandas as pd
T=P/'deliverables/tables';T.mkdir(parents=True,exist_ok=True)
pd.concat([pd.read_csv(P/(n+'.csv'),keep_default_na=False,na_values=['']).assign(Result_source=n) for n in ['core','interactions','sensitivity','extra','seed_baselines']],ignore_index=True).to_csv(T/'Pooled_estimates_full_precision.csv',index=False)
pd.concat([pd.read_csv(P/(n+'.csv'),keep_default_na=False,na_values=['']).assign(Result_source=n) for n in ['core_tests','interactions_tests']],ignore_index=True).to_csv(T/'Joint_tests_full_precision.csv',index=False)
print('FULL PRECISION AGGREGATES EXPORTED')
