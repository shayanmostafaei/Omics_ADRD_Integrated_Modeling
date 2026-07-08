Omics_ADRD_Integrated_Modeling

Code repository for the study, entitled: 

Integrative Prediction of Alzheimer’s Disease and Related Dementias Using Multi-Omics Aging Clocks and Genetic Data

This repository contains analysis scripts for integrated prediction of Alzheimer’s disease and related dementias (ADRD) using polygenic risk scores (PRS), multi-domain biological aging measures, frailty, telomere length, proteomic aging, and metabolomic aging.

The workflow includes:

1. Internal model development and held-out evaluation in UK Biobank
2. Calibration, precision–recall, threshold, decision-curve, and competing-risk analyses
3. Robustness checks, repeated train–test splits, benchmarking, and missing-data sensitivity analyses
4. External validation/replication in TwinGene

Important: UK Biobank and TwinGene individual-level data are not distributed in this repository. This repo provides analysis code, expected input formats, and manuscript figure/table generation workflows.

⸻

Study snapshot

UK Biobank internal evaluation

* Cohort: UK Biobank participants with complete genetics, clinical biomarkers, proteomics, metabolomics, and biological aging measures
* Final complete-case analytic sample: N = 16,215 ADRD-free participants at baseline
* Incident ADRD cases: N = 397
* Median follow-up: 10.08 years
* Follow-up definition: baseline assessment to earliest of ADRD diagnosis, death, or censoring
* Primary evaluation design: stratified 70/30 train–test split
* Training set: N = 11,351, including 275 incident ADRD cases
* Held-out test set: N = 4,864, including 122 incident ADRD cases

TwinGene external validation/replication

* External replication cohort: N = 3,772
* Incident ADRD cases: N = 331
* Non-cases: N = 3,441
* Competing death events: N = 841
* Mean baseline age: 64.2 years
* Median follow-up: 16.7 years
* Purpose: external transportability and methodological replication of the integrated prediction framework

⸻

Outcome definition

The primary outcome was incident Alzheimer’s disease and related dementias (ADRD), identified using registry-based ICD-10 codes from inpatient and death registry sources.

Follow-up was defined from baseline assessment to the earliest of:

* incident ADRD diagnosis
* death
* censoring

For competing-risk analyses, death before ADRD was treated as a competing event.

⸻

Predictors used

Core predictors and covariates

* Chronological age
* Sex
* Body mass index
* Smoking
* Alcohol intake frequency
* Education
* Top genetic principal components (10 genetic PCs)

Genetic risk

* ADRD polygenic risk score, including the APOE locus
* APOE genotype was not entered separately in the primary prediction models
* APOE-e4 carrier status was used only for stratified sensitivity analyses

PRS construction was based on the DDML Bayesian variational autoencoder implementation:

* https://link.springer.com/article/10.1186/s13195-026-02011-w
Clinical and functional biological aging measures

Primary multivariable biological aging predictors:

* PhenoAge
* Frailty Index
* Telomere Length

Additional biological aging measures evaluated in univariate and correlation analyses:

* Klemera–Doubal Method
* Homeostatic Dysregulation

Related biological aging reference:

* https://www.nature.com/articles/s41416-023-02288-w

Omics aging clocks

* ProtAge: proteomic aging clock derived from Olink protein data
    https://github.com/shayanmostafaei/Proteomic-Aging-Clock-ProtAge
* MetaboAge: metabolomic aging clock derived from Nightingale NMR metabolomics
    https://github.com/shayanmostafaei/Metabolomic-Aging-Clock-MetaboAge

⸻

Main modeling workflow

1. Dataset construction and harmonization

The workflow harmonizes:

* outcome variables
* follow-up time
* competing death events
* demographic covariates
* lifestyle covariates
* genetic principal components
* PRS
* clinical biological aging measures
* ProtAge
* MetaboAge

All preprocessing steps requiring fitted parameters are estimated using training data only and then applied to held-out test data.

⸻

2. Stepwise ADRD prediction with XGBoost

The main prediction models were fitted using XGBoost classification and evaluated in the held-out UK Biobank test set.

Stepwise model sequence:

* Base model: age + sex
* Model 1: base model + lifestyle covariates
* Model 2: Model 1 + ADRD PRS
* Model 3: Model 2 + PhenoAge + Frailty Index + Telomere Length
* Model 4: Model 3 + ProtAge
* Model 5: Model 4 + MetaboAge

Primary metrics:

* ROC-AUC with 95% confidence intervals
* AUPRC with bootstrap 95% confidence intervals
* Pairwise ROC-AUC comparisons between consecutive models

⸻

3. Calibration, threshold performance, and decision-curve analysis

For the final integrated model, the workflow evaluates:

* decile-based calibration
* calibration intercept
* calibration slope
* Brier score
* threshold-based operating characteristics
* decision-curve analysis

Threshold operating points include:

* top 5% predicted risk
* top 10% predicted risk
* illustrative default threshold of 0.5

The 0.5 threshold is included only as an illustrative operating point and is not proposed as a clinical decision threshold.

⸻

4. SHAP interpretability

SHAP values are computed for the final integrated XGBoost model in the held-out test set to summarize:

* global feature importance
* direction of predictor contributions
* relative contribution of PRS, age, ProtAge, PhenoAge, frailty, telomere length, lifestyle factors, and MetaboAge

⸻

5. Fine–Gray competing-risk risk stratification

Using the final integrated predictor set, the workflow fits Fine–Gray competing-risk regression with death before ADRD treated as a competing event.

The model estimates:

* subdistribution hazard ratios for individual predictors
* predicted absolute ADRD risk at 5 and 9 years
* cumulative incidence by predicted-risk group
* risk-group separation comparing the top 25% predicted-risk group versus the lower 75%

⸻

6. Robustness and sensitivity analyses

The revised workflow includes:

* alternative predictor-entry orders
* 100 repeated stratified 70/30 train–test splits
* logistic-regression benchmark models
* subgroup analyses by sex
* subgroup analyses by APOE-e4 carrier status
* PRS with and without the APOE locus
* models with and without lifestyle covariates
* prediction horizons at 5 and 9 years
* early-event exclusion
* complete-case versus multiple-imputation sensitivity analysis
* cause-specific Cox versus Fine–Gray comparison
* time-since-baseline versus attained-age sensitivity

⸻

7. TwinGene external validation/replication

TwinGene was used to evaluate whether the integrated signal transported beyond UK Biobank.

The TwinGene workflow includes:

* harmonization of available predictors
* PRS using the same selected SNP panel and external GWAS weights 
* harmonized clinical biological aging measures
* Olink-based ProtAge replication
* UKB-portable MetaboAge implementation 
* XGBoost discrimination analysis
* logistic-regression replication benchmark
* Fine–Gray competing-risk risk stratification
* cumulative incidence by predicted-risk group

⸻

Main results at a glance

UK Biobank held-out test set

Stepwise XGBoost discrimination:

* Base model, age + sex: ROC-AUC ≈ 0.79
* + lifestyle covariates: ROC-AUC ≈ 0.82
* + PRS: ROC-AUC ≈ 0.86
* + PhenoAge + Frailty Index + Telomere Length: ROC-AUC ≈ 0.89
* + ProtAge: ROC-AUC ≈ 0.90
* + MetaboAge, final Model 5: ROC-AUC ≈ 0.90, AUPRC ≈ 0.24

Fine–Gray competing-risk model:

* PRS: sHR ≈ 2.24
* ProtAge: sHR ≈ 2.01
* Top 25% predicted-risk group versus lower 75%: sHR ≈ 16.73

Repeated split robustness

Across 100 repeated stratified 70/30 splits:

* Final XGBoost Model 5 mean ROC-AUC ≈ 0.903
* Final XGBoost Model 5 mean AUPRC ≈ 0.239
* Logistic-regression full-model benchmark mean ROC-AUC ≈ 0.868
* Logistic-regression full-model benchmark mean AUPRC ≈ 0.228


TwinGene external validation/replication

* ROC-AUC ≈ 0.757
* AUPRC ≈ 0.223
* Top 25% predicted-risk group versus lower 75%: sHR ≈ 4.02
* Event proportions: 19.9% in top 25% versus 5.1% in lower 75%

TwinGene showed lower discrimination than UK Biobank, as expected, but preserved meaningful risk enrichment.

⸻

Repository layout

Omics_ADRD_Integrated_Modeling/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── README_data_dictionary.md
├── docs/
│   └── manuscript_output_map.md
├── scripts/
│   ├── 01_Univariate_Analysis_ROCs.R
│   ├── 02_Interaction_Heatmaps_BioAge.R
│   ├── 03_Multivariable_XGBoost_Modeling.R
│   ├── 04_SHAP_Analysis.py
│   ├── 05_Survival_Competing_Risks_Analysis.R
│   ├── 06_Sensitivity_Analyses.R
│   ├── 07_Calibration_DCA_Thresholds.R
│   ├── 08_Repeated_Split_Stability.R
│   ├── 09_Benchmarking_Reclassification.R
│   ├── 10_Multiple_Imputation_Sensitivity.R
│   ├── 11_TwinGene_External_Validation.R
│   └── 99_session_info.R
└── results/
    └── README_outputs.md

⸻

Manuscript output map

Manuscript item	Script	Main output
Figure 2	scripts/01_Univariate_Analysis_ROCs.R	univariate ROC-AUC forest plot
Figure 3	scripts/02_Interaction_Heatmaps_BioAge.R	predictor correlation matrix
Table 2	scripts/03_Multivariable_XGBoost_Modeling.R	stepwise ROC-AUC and AUPRC table
Figure A.5	scripts/03_Multivariable_XGBoost_Modeling.R	precision–recall curves
Figure A.6	scripts/07_Calibration_DCA_Thresholds.R	calibration plot
Figure A.7	scripts/07_Calibration_DCA_Thresholds.R	decision-curve analysis
Figure A.8	scripts/04_SHAP_Analysis.py	SHAP summary plot
Figure 4	scripts/05_Survival_Competing_Risks_Analysis.R	Fine–Gray predictor forest plot
Figure 5	scripts/05_Survival_Competing_Risks_Analysis.R	cumulative incidence by predicted-risk group
Figure A.9	scripts/06_Sensitivity_Analyses.R	sensitivity-analysis AUC forest plot
Figure A.10	scripts/06_Sensitivity_Analyses.R	early-event exclusion analysis
Table A.5	scripts/08_Repeated_Split_Stability.R	repeated-split XGBoost stability
Table A.6	scripts/08_Repeated_Split_Stability.R	repeated-split logistic-regression benchmark
Table A.8	scripts/09_Benchmarking_Reclassification.R	simpler-model benchmarking and reclassification
Table A.10	scripts/11_TwinGene_External_Validation.R	TwinGene predictor harmonization
Table A.11	scripts/11_TwinGene_External_Validation.R	TwinGene validation performance
Table A.12	scripts/11_TwinGene_External_Validation.R	TwinGene risk stratification
Figure A.11	scripts/11_TwinGene_External_Validation.R	TwinGene ROC and precision–recall curves
Figure A.12	scripts/11_TwinGene_External_Validation.R	TwinGene cumulative incidence curves

⸻

Expected input data

Raw individual-level UK Biobank and TwinGene data are not included.

UK Biobank input


Key required columns include:

* f.eid
* Dementia_status
* Time_to_Dementia
* death_status
* time_to_death
* length_followup
* CA
* sex
* bmi
* smoking
* alcohol_intake_frequency
* education
* PRS_ADRD
* PhenoAge
* FI
* TL
* ProtAge
* MetaboAge
* PC1 to PC10
* APOEe4_status
* age_at_dementia_onset

TwinGene input


Key required columns include:

* twin_id
* pair_id
* zygosity
* age
* sex
* dementia
* time_to_adrd
* death_status
* time_to_death
* followup_time
* PRS_ADRD
* PhenoAge
* FI
* TL
* ProtAge
* MetaboAge
* * PC1 to PC10
* APOEe4_status
* age_at_dementia_onset


⸻

Reproducibility

R environment

This project was run using R version 4.4.3. Core R packages include:

* dplyr
* tidyr
* readr
* caret
* xgboost
* pROC
* PRROC
* riskRegression
* cmprsk
* prodlim
* survival
* survminer
* ggplot2
* patchwork
* mice

Python environment

Python was used for SHAP and selected machine-learning utilities. Core packages include:

* python >= 3.10
* numpy
* pandas
* scikit-learn
* xgboost
* shap
* matplotlib

A full session information script is provided in:

scripts/99_session_info.R

⸻

Data governance

This repository does not include raw or individual-level UK Biobank or TwinGene data.

The following files should not be committed:

* raw .rds, .RData, .csv, .tsv, or .parquet files
* individual-level predictions
* model objects containing individual-level metadata
* derived files that could identify participants

Researchers must obtain access to UK Biobank and TwinGene data through the appropriate governed access procedures.

⸻

Interpretation note

This repository supports a prediction and risk-stratification study. The analyses are not designed to estimate causal effects of PRS, biological aging measures, ProtAge, or MetaboAge on ADRD risk.

The final interpretation is that integrated genetic and biological aging information improves ADRD risk ranking and risk enrichment, especially for prevention-oriented research and trial enrichment. The model is not proposed as a diagnostic classifier or ready for clinical deployment.

⸻

Citation

If you use this code, please cite:

Mostafaei S, Gustavsson K, Hagelin H, Mak JKL, Vu TN, Karlsson IK, Hägg S.
Integrative Prediction of Alzheimer’s Disease and Related Dementias Using Multi-Omics Aging Clocks and Genetic Data. Manuscript under review.

Related PRS method:

Mostafaei S, Shemer DW, Mak JKL, Karlsson IK, Hägg S.
Improved polygenic risk prediction for Alzheimer’s disease and related dementias using deep learning: age and APOE-stratified analysis. Alzheimer’s Research & Therapy. 2026.

⸻

License

MIT License. See LICENSE.

⸻

Contact

* Shayan Mostafaei — shayan.mostafaei@ki.se
* Sara Hägg — sara.hagg@ki.se
