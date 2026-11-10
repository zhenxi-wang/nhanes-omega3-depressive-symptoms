# Guide for reviewers

This map identifies the code and aggregate evidence relevant to the statistical and reproducibility requests. It describes the corrected analysis; no prospective registration or independently verifiable early statistical analysis plan is claimed.

| Concern | Implementation | Included aggregate evidence |
|---|---|---|
| Raw-file decoding, eligibility and questionnaire skips | `src/derive.py`, `src/model_spec.py` | `resources/input_file_manifest.csv`, `reference_results/TableS1.csv`, `reference_results/TableS2.csv` |
| Variable-specific missingness and complete-case loss | `src/derive.py`, `src/missingness_audit.py` | `reference_results/TableS18.csv`, `Missingness_number.csv`, `Missingness_patterns.csv` in `reference_results/` |
| Former unstable marital-missing category | `src/missingness_audit.py` | `reference_results/Marital_missing_diagnosis.csv`, `validation/historical_separation_diagnosis.json` |
| Chained-equations specification, distinct draws, convergence and seed checks | `src/impute.py`, `src/check_seed_and_baselines.py` | `reference_results/TableS15.csv`, `TableS15_diagnostics.csv`, `TableS15_seed_check.csv`, `imputation_full_diagnostics.json`, `imputation_diet_diagnostics.json` |
| Within/between variance, Rubin pooling and small-sample inference | `src/engine.py` | `reference_results/TableS15.csv`, `Pooled_estimates_full_precision.csv` |
| Weights, strata, PSUs and domain variances | `src/survey.py`, `src/engine.py` | `validation/R_python_comparison.csv`, `R_validation_summary.csv`, `R_sessionInfo.txt` |
| Actual validation against R survey | `validation/export_r_validation.py`, `validation/validate_survey.R` | 162 per-fit comparisons, full covariance checks and `validation/R_pooled_results.csv` |
| Prescription-use and HEI sensitivity analyses with matched dietary populations | `src/derive.py`, `src/analyze_sensitivity.py` | `reference_results/TableS10.csv`, `TableS17.csv`, `TableS19.csv` |
| Highest symptom threshold and non-monotonic severity pattern | `src/analyze_core.py`, `src/validate_extra.py` | `reference_results/Table2C.csv`, `TableS4.csv`, `TableS12.csv`, `TableS12b.csv` |
| Nine-item and seven-domain multiplicity; descriptive post hoc clusters | `src/analyze_core.py`, `src/analyze_interactions.py`, `src/make_tables.py` | `reference_results/TableS9.csv`, `TableS13.csv`, `TableS14.csv` |
| E-values as limited sensitivity benchmarks | `src/validate_extra.py` | `reference_results/evalues.csv` |
| Reproducible publication figures | `src/publication_figures.py` | Aggregate reference results suffice to regenerate the figures without participant data |

Paths shown without a directory in an evidence cell refer to the directory explicitly named in that cell. `validation/historical_separation_diagnosis.json` records the audit of the former model; it is not a claim that the former missing-indicator model is the principal analysis.

## Numerical validation actually performed

The regression implementation was compared with R 4.5.2 and survey 4.5 across 162 fits. The largest absolute coefficient difference was 9.193493e-9; the largest full covariance entry difference was 6.415594e-7. Both were below the absolute tolerance of 1e-6. Scalar pooling of the two principal per-SD models and the dietary HEI model was also reproduced from R-derived coefficients and variances. The scope, convergence tolerances and reference degrees of freedom are documented in the README, R script and session record.

This checks selected numerical estimation and variance calculations. It does not establish MAR, absence of residual confounding or causal validity, and it does not claim independent R replication of every imputation or supplementary contrast.

## Important scientific corrections

- True missingness in at least one Model 3 covariate is 971/5,120 (19.0%; weighted 19.2%). The former 2,875 complete-case count mixed processing errors, skip patterns and structural non-applicability. There are 4,149 records without item nonresponse and 3,989 fully observed adults aged at least 20.
- The principal per-SD estimates are OR 0.863694 (95% CI 0.719998–1.036067) for PHQ-9 at least 10 and 0.725899 (0.621662–0.847614) for PHQ-9 at least 15.
- The PHQ-9 at least 20 point estimate is near the null. Same-subset HEI adjustment attenuates the PHQ-9 at least 15 association and its confidence interval includes 1. General prescription use cannot identify antidepressant or psychotropic drug use in this cycle.
- Sex-interaction and symptom-cluster interpretations are exploratory. No subgroup domain survives correction across seven domains. Cluster comparisons are descriptive and not adjusted for same-data selection.

## Reproduction

The README gives full instructions, input sources, package versions and stage commands. A full run requires downloading the public NHANES and FPED inputs. The repository supplies aggregate results and trace summaries, not individual records or completed imputation datasets. The full stochastic M=40 run is computationally intensive.

A download-only review can inspect the CSV/JSON evidence immediately and regenerate the 600-dpi PNG/TIFF and vector PDF figures using:

```
python -m pip install -r requirements-tested.txt
python src/publication_figures.py
```
