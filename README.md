# NHANES erythrocyte omega-3 and depressive symptoms
This is the public code repository accompanying **Erythrocyte Omega-3 Status and Depressive Symptom Burden in US Adults**. See [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) for a map from the reviewer concerns to code and numerical evidence. The submission snapshot is version **v1.0.0** (11 November 2026).

This repository contains the corrected analysis for NHANES August 2021–August 2023. The principal model uses M=40 chained-equations imputation and design-based survey variance. The primary outcome for reporting is PHQ-9 >=10; PHQ-9 >=15 is secondary. No prospective registration or independently verifiable early analysis plan is claimed.

## Inputs and environment

Use Python 3.12. Install the tested versions with `python -m pip install -r requirements-tested.txt` (or use the compatible ranges in requirements.txt). Obtain the public XPT files listed in resources/input_file_manifest.csv from CDC, and extract `fped_dr1tot_2123.sas7bdat` from the USDA FPED 2021–2023 day-1 total-intake download. FPED source: https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fped-doc-and-datasets/

Original input files are read-only. Participant-level files, imputations and intermediate outputs are excluded from Git. Only aggregate reference results are included.

## Reproduce the analysis

Run from this repository directory, supplying your local input paths:

```
python run_analysis.py --raw-data PATH_TO_XPT_DIRECTORY --fped PATH_TO_FPED_SAS7BDAT --output outputs --stage all
```

Stages can be run separately: prepare, impute, models, tables, figures. The impute stage runs both main and dietary M=40 imputations and the alternative main-sample seed. The models stage includes core outcomes, interactions, all sensitivity analyses, post hoc additional checks, multiplicity and seed/eligibility baselines. The figures stage exports 600-dpi PNG/TIFF and vector PDF. A full run is computationally intensive; do not replace the stochastic M=40 stage with one completed dataset.

Figures can also be recreated directly from the supplied aggregate reference values without individual data:

```
python src/publication_figures.py
```

The figure script uses outputs when present and falls back to reference_results otherwise. The publication label “selected saturated fatty acids” refers to 14:0+16:0+18:0, not every laboratory SFA. Some internal historical category strings are retained for exact model reproducibility; the displayed PIR boundaries are <=1.3, >1.3–3.5 and >3.5. The PIR subgroup uses <1.3 versus >=1.3.

## R survey validation

The package includes an actual comparison with R 4.5.2 and survey 4.5 across 162 regression fits. Install survey in your R environment with `install.packages("survey")`. After producing the completed datasets, run:

```
python validation/export_r_validation.py
Rscript validation/validate_survey.R outputs/r_validation
```

If you use a custom R library directory, supply it as a second argument to validate_survey.R. Inputs for validation are generated under the ignored outputs directory. The per-fit aggregate comparison, pooled principal estimates and original session information are supplied in validation/. Coefficient, full covariance and standard-error differences are checked with absolute tolerance 1e-6. Strict R convergence was needed for comparison of the full covariance matrices. Scalar MI tests use a common complete-data design df of 15 and Barnard–Rubin adjustment, rather than R svyglm’s default regression residual df. Multi-parameter tests use the documented finite-sample D1 calculation.

## Statistical scope

Survey strata and PSUs are retained in domain variances. Observed modifier/restriction membership is not assigned from imputations. Structural age 18–19 non-applicability is separate from item nonresponse. Main sample imputation includes log phlebotomy weight and PSU-by-stratum predictors; dietary imputation uses the day-1 weight. Most restrictions reuse the main imputations and then use their stated analysis weights. Missing outcomes, supplement status and alternative fatty-acid exposures are not imputed in the substantive analysis. Posterior draws are approximate conditional-model draws; convergence does not verify MAR or remove residual confounding.

The supplied aggregate reference tables include exploratory analyses and necessary repeated presentation across main and supplementary tables. They are not independent confirmatory hypotheses. Cluster comparisons are descriptive and not selection-adjusted. Any general-prescription sensitivity analysis is not antidepressant-specific control. R validation checks numerical regression/variance agreement, not all scientific assumptions.

## Files

- src/derive.py: raw-file decoding, eligibility, covariates, outcomes, HEI and scaling.
- src/impute.py: explicit expanded chained equations and diagnostics.
- src/engine.py and src/survey.py: survey estimation, MI pooling, contrasts and standardisation.
- src/analyze_*.py: all substantive models and restrictions.
- src/descriptive.py, src/make_tables.py and src/missingness_audit.py: descriptive, publication and missingness tables.
- src/publication_figures.py: high-resolution figures from aggregate outputs.
- src/validate_extra.py and validation/: independent numerical checks and R comparison.
- resources/: public data manifest and source inventory.
- reference_results/: aggregate final results and imputation-summary traces, without participant data.

No code in this repository uploads data or publishes a GitHub repository.

## Versioned access

Repository: https://github.com/zhenxi-wang/nhanes-omega3-depressive-symptoms

Submission snapshot: https://github.com/zhenxi-wang/nhanes-omega3-depressive-symptoms/tree/v1.0.0

Release and source download: https://github.com/zhenxi-wang/nhanes-omega3-depressive-symptoms/releases/tag/v1.0.0

The SHA-256 manifest `FILES_SHA256.csv` records every file in this snapshot except the manifest itself. All files from the supplied code directory are included; the README has been supplemented for public access. Manuscript documents and individual-level datasets are outside this repository.
