"""Portable paths. Input files are never modified."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('NHANES_OUTPUT_DIR',ROOT/'outputs')).resolve()
DATA=Path(os.environ.get('NHANES_RAW_DATA',ROOT/'data')).resolve()
FPED=Path(os.environ.get('NHANES_FPED',DATA/'fped_dr1tot_2123.sas7bdat')).resolve()
AUD=P
P.mkdir(parents=True,exist_ok=True)
