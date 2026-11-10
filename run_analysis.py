"""Run the corrected workflow with explicit input/output paths."""
import argparse,os,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--raw-data',type=Path);ap.add_argument('--fped',type=Path);ap.add_argument('--output',type=Path,default=Path('outputs'));ap.add_argument('--stage',choices=['all','prepare','impute','models','tables','figures'],default='all');a=ap.parse_args()
root=Path(__file__).resolve().parent;env=os.environ.copy();env['NHANES_OUTPUT_DIR']=str(a.output.resolve());env['PYTHONDONTWRITEBYTECODE']='1'
if a.raw_data:env['NHANES_RAW_DATA']=str(a.raw_data.resolve())
if a.fped:env['NHANES_FPED']=str(a.fped.resolve())
def run(script,*args):subprocess.run([sys.executable,'-B',str(root/'src'/script),*args],env=env,check=True)
stages=['prepare','impute','models','tables','figures'] if a.stage=='all' else [a.stage]
for stage in stages:
 if stage=='prepare':run('derive.py')
 elif stage=='impute':
  run('impute.py');run('impute.py','--diet');run('impute.py','--seed','20261121')
 elif stage=='models':
  for f in ['analyze_core.py','analyze_interactions.py','analyze_sensitivity.py','check_seed_and_baselines.py','descriptive.py','validate_extra.py']:run(f)
 elif stage=='tables':run('make_tables.py');run('missingness_audit.py');run('export_aggregates.py')
 elif stage=='figures':run('publication_figures.py')
