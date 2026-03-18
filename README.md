# Omics_ADRD_Integrated_Modeling

Integrated prediction of **Alzheimer’s disease and related dementias (ADRD)** using **polygenic risk scores (PRS)**, **multi-domain biological aging measures**, and **multi-omics aging clocks**, with **machine learning** prediction and **competing-risk (Fine–Gray)** risk stratification in **UK Biobank**.

> **Important:** UK Biobank individual-level data are not distributed in this repository. This repo provides analysis code, expected input formats, and figure/table generation workflows.

---

## Study snapshot

- **Cohort:** UK Biobank participants with complete genetics + clinical biomarkers + proteomics + metabolomics data
- **Final analytic sample:** **N = 16,215** ADRD-free at baseline
- **Incident ADRD cases:** **N = 397**
- **Follow-up:** baseline → earliest of ADRD diagnosis, death, or censoring  
  **End of follow-up:** **March 2023**  
  **Median follow-up:** **10.08 years**
- **Outcome (ADRD):** ICD-10 codes from inpatient and death registry sources (see manuscript/Supplementary Appendix A for code list)

---

## Predictors used

### Core predictors and covariates
- **Chronological age (CA)**
- **Sex**
- **Lifestyle covariates:** smoking, alcohol intake frequency, BMI, education

### Genetic risk
- **PRS including the APOE locus**  
  - PRS computed using the DDML Bayesian variational autoencoder implementation (https://link.springer.com/article/10.1186/s13195-026-02011-w):  
    https://github.com/shayanmostafaei/DDML_PRS_ADRD

### Biological aging measures (clinical)
- **PhenoAge**
- **Frailty Index (FI)** (49-item Rockwood deficit accumulation; range 0–1)
- **Telomere Length (TL)**
- Additional measures used for univariate/correlation analyses: **KDM**, **HD**

### Omics clocks
- **ProtAge (proteomics clock)**  
  https://github.com/shayanmostafaei/Proteomic-Aging-Clock-ProtAge
- **MetaboAge (metabolomics clock)**  
  https://github.com/shayanmostafaei/Metabolomic-Aging-Clock-MetaboAge

---

## What this repository does

### 1) Builds the analysis dataset
- harmonizes predictors/covariates
- handles missingness (training-only preprocessing; applied to held-out test split)
- exports a modeling-ready dataset (**no UKB raw fields distributed**)

### 2) Stepwise ADRD prediction with XGBoost (Table 2)
We fit a stepwise set of **XGBoost classification models** and evaluate discrimination on a held-out test set:

- **Base model:** Age + Sex
- **Model 1:** Base + Lifestyle (smoking, alcohol intake frequency, BMI, education)
- **Model 2:** Model 1 + PRS
- **Model 3:** Model 2 + PhenoAge + FI + TL
- **Model 4:** Model 3 + ProtAge
- **Model 5 (final):** Model 4 + MetaboAge

**Validation design:** stratified **70/30 train–test split** (performance reported on the test set).  
**Leakage control:** preprocessing is fit in training and applied to test.

**Primary metrics:** ROC-AUC (with 95% CI) and AUPRC.  
**Model comparisons:** pairwise ROC AUC comparisons on the same test set (DeLong test; see Supplementary Table A.3).

> If your workflow re-trains omics clocks within the training split, document that explicitly in `01_data_prep/` to avoid any leakage concerns. If clocks are computed using fixed pipelines, state that no outcome information is used during clock computation.

### 3) Calibration of the final model (Supplementary Figure A.6)
We generate a decile-binned calibration plot in the held-out test set and report:
- calibration slope and intercept
- Brier score 

### 4) Interpretability (Supplementary Figure A.7)
We compute SHAP values for the final integrated model to summarize global feature contributions.

### 5) Survival + competing-risk risk stratification (Fine–Gray)
Using predictors from the final integrated model, we:
- encode time-to-event as earliest of **ADRD** or **death**
- treat **death as a competing event**
- fit **Fine–Gray competing-risk regression**
- predict **absolute ADRD risk at 5 years** (primary horizon) and optionally 9 years
- stratify participants into:
  - **Highest-risk:** top 25% of predicted 5-year risk
  - **Lower-risk:** bottom 75%
- estimate separation using **subdistribution hazard ratio (sHR)** and visualize cumulative incidence (manuscript Figure 5; Supplementary Figures A.8–A.10)

---

## Results at a glance (held-out test set)

Stepwise XGBoost discrimination:
- **Base (Age+Sex):** AUC ≈ 0.79  
- **+ Lifestyle:** AUC ≈ 0.82  
- **+ PRS:** AUC ≈ 0.86  
- **+ PhenoAge + FI + TL:** AUC ≈ 0.89  
- **+ ProtAge:** AUC ≈ 0.90  
- **+ MetaboAge (final):** AUC ≈ 0.90 (AUPRC also reported)

Competing-risk stratification shows strong separation between predicted risk groups (see manuscript Figure 5).

---

## Repository layout (recommended)

- `01_data_prep/`
  - build analytic dataset; variable harmonization; train/test split
- `02_univariate/`
  - univariate AUCs (Figure 2), distributions 
- `03_correlations_associations/`
  - correlation matrix (Figure 3) and lifestyle associations 
- `04_stepwise_xgboost/`
  - Table 2 stepwise models + AUC comparisons + AUPRC + SHAP 
- `05_calibration/`
  - calibration plot for Model 5 
- `06_competing_risk/`
  - Fine–Gray modeling + CIF plots 
- `tables/` and `figures/`
  - manuscript-ready exports

---

## Reproducibility

### R environment
This project was run using R (see the manuscript for the exact version) and common packages including:
`xgboost`, `caret`, `pROC`, `riskRegression`, `cmprsk`, `prodlim`, `ggplot2`, `dplyr`, `tidyr`.

### No raw UKB data
This repository does **not** include UK Biobank data.  
You should provide:
- an input template (column names + types)
- scripts that read a user-provided dataset and generate figures/tables

---

## Citation

If you use this code, please cite:

Mostafaei S, Gustavsson K, Mak JKL, Vu TN, Karlsson IK, Hägg S.  
**Precision Prediction of Alzheimer’s Disease and Related Dementias Using Integrative Multi-Omics Aging Clocks and Genetic Data** (manuscript, Under Review).

---

## License
MIT License (see `LICENSE`).

---

## Contact
- Shayan Mostafaei — shayan.mostafaei@ki.se  
- Sara Hägg — sara.hagg@ki.se
