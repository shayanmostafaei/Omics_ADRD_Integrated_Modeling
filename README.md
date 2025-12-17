# Omics_ADRD_Integrated_Modeling

Integrated prediction of **Alzheimer’s disease and related dementias (ADRD)** using **multi-omics biological aging clocks**, **polygenic risk scores (PRS)**, and **machine learning**, with **survival/competing-risk** risk stratification in **UK Biobank**.

> **Important:** UK Biobank data are not distributed in this repository. This repo provides the analysis code and expected input formats.

---

## Study snapshot 

- **Cohort:** UK Biobank participants with complete data across genetics, clinical biomarkers, proteomics, and metabolomics  
- **Final analytic sample:** **N = 16,215** ADRD-free at baseline  
- **Follow-up:** baseline → earliest of ADRD diagnosis, death, or censoring (**end of follow-up: March 2023**)  
- **Outcome (ADRD):** ICD-10 codes from hospital inpatient + death registry data (see manuscript for code list)

---

## What this repository does

### 1) Builds predictors used in ADRD modeling
This workflow integrates:

- **Chronological age (CA)** and **sex**
- **Lifestyle covariates:** smoking, alcohol consumption, BMI, education
- **Genetic risk:** **PRS (including APOE locus)**  
  - PRS is computed using the **DDML Bayesian variational autoencoder** implementation:
  - Code: https://github.com/shayanmostafaei/DDML_PRS_ADRD
- **Biological aging measures (BA):** PhenoAge, Frailty Index (FI), Telomere Length (TL)
- **Omics clocks:**
  - **ProtAge (proteomics clock)**  
    Code: https://github.com/shayanmostafaei/Proteomic-Aging-Clock-ProtAge-
  - **MetaboAge (metabolomics clock)**  
    Code: https://github.com/shayanmostafaei/Metabolomic-Aging-Clock-MetaboAge-

### 2) Predicts incident ADRD using stepwise XGBoost models
We fit a stepwise set of **XGBoost classification models** and compare performance on a held-out test set:

- **Crude:** Age + Sex  
- **Model 1:** Crude + Lifestyle  
- **Model 2:** Model 1 + PRS  
- **Model 3:** Model 2 + PhenoAge + FI + TL  
- **Model 4:** Model 3 + ProtAge  
- **Model 5:** Model 4 + MetaboAge (final integrated model)

**Validation design:** stratified **70/30 train–test split** (test set used only for evaluation).  
**Leakage control:** all preprocessing is done in the training split and applied to test; critically, **ProtAge and MetaboAge predictions are generated using clock models trained only in the training split**, then applied to the held-out test split.

**Primary metric:** AUC (ROC), with **pairwise AUC tests** for stepwise comparisons on the *same* held-out test set.  
**Imbalanced classification:** we additionally report **AUPRC** for Model 5.

> Tip: If you want thresholds/clinical operating points, this repo can also output confusion-matrix metrics; however, thresholding is not the primary focus because ADRD is rare.

### 3) Performs survival + competing-risk risk stratification (Fine–Gray)
Using predictors from the final integrated model, we:

- Encode time-to-event as earliest of **ADRD diagnosis** or **death**
- Treat **death as a competing event**
- Fit **Fine–Gray competing risk regression**
- Predict **absolute risks at 5 and 9 years**
- Stratify participants into:
  - **High-risk:** top 25% of predicted risk
  - **Low-risk:** bottom 75%
- Estimate separation by **subdistribution hazard ratio (sHR)** between groups

---

## Results at a glance (from the manuscript)

Stepwise XGBoost AUCs on the held-out test set:

- **Crude (Age+Sex):** AUC 0.79  
- **+ Lifestyle:** AUC 0.82  
- **+ PRS:** AUC 0.86  
- **+ PhenoAge + FI + TL:** AUC 0.89  
- **+ ProtAge:** AUC 0.90  
- **+ MetaboAge:** AUC 0.90 (final model; AUPRC also reported)

Competing-risk stratification (Fine–Gray) shows strong separation between high-risk vs low-risk groups (see manuscript).

---

## Citation

If you use this code, please cite:

Mostafaei S, et al. (2025).  
“Precision Prediction of Alzheimer's Disease and Related Dementias Using Integrative Multi-Omics Aging Clocks and Genetic Data” (manuscript).

---

## License

MIT License (see `LICENSE`).

---

## Contact

- Dr. Shayan Mostafaei — shayan.mostafaei@ki.se  
- Dr. Sara Hägg — sara.hagg@ki.se
