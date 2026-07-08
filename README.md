Integrative Prediction of Alzheimer’s Disease and Related Dementias Using Multi-Omics Aging Clocks and Genetic Data

Overview

This project evaluates whether combining polygenic risk scores (PRS) with multi-domain biological aging markers improves prediction and risk stratification for Alzheimer’s disease and related dementias (ADRD).

The analysis uses a multicohort prediction/validation design:

* Development and internal evaluation cohort: UK Biobank
    Analytic n = 16,215; incident ADRD cases = 397; median follow-up = 10.08 years
* External validation/replication cohort: Swedish TwinGene
    Analytic n = 3,772; incident ADRD cases = 331; competing death events = 841; median follow-up = 16.7 years

The workflow includes:

* Stepwise XGBoost prediction models
* ROC-AUC and AUPRC evaluation for low-incidence ADRD prediction
* Calibration analysis and Brier score
* Threshold-based operating characteristics
* Decision-curve analysis
* SHAP-based model interpretation
* Fine–Gray competing-risk regression with death as a competing event
* Repeated 70/30 split robustness analyses
* Logistic-regression benchmark models
* Multiple-imputation sensitivity analysis
* External validation/replication in TwinGene

⸻

Main analytic design

UK Biobank

The primary UK Biobank analysis used participants free of ADRD at baseline with available genetic, clinical, proteomic, metabolomic, and biological aging measures.

* Final complete-case sample: 16,215
* Incident ADRD cases: 397
* Training set: 11,351 participants, including 275 ADRD cases
* Held-out test set: 4,864 participants, including 122 ADRD cases
* Split design: stratified 70/30 train–test split
* Outcome: incident ADRD during follow-up
* Competing event: death before ADRD

All preprocessing steps that required fitted parameters were estimated using training data only and then applied to the held-out test set.

TwinGene

TwinGene was used as an independent Swedish population-based cohort to evaluate external transportability and methodological replication.

* Final complete-case replication sample: 3,772
* Incident ADRD cases: 331
* Non-cases: 3,441
* Competing death events: 841
* Mean baseline age: 64.2 years
* Men: 47.0%
* Median follow-up: 16.7 years

⸻

Predictors

Core covariates

* Chronological age
* Sex
* Body mass index
* Smoking
* Alcohol intake frequency
* Education
* Genetic principal components

Genetic risk

* ADRD PRS including the APOE locus
* APOE genotype was not entered separately in the primary prediction models
* APOE-e4 carrier status was used only for stratified sensitivity analyses

Related PRS repository:

* https://github.com/shayanmostafaei/DDML_PRS_ADRD

Biological aging markers

Main biological aging predictors in multivariable models:

* PhenoAge
* Frailty Index
* Telomere Length

Additional biological aging markers evaluated in univariate/correlation analyses:

* Klemera–Doubal Method
* Homeostatic Dysregulation

Omics aging clocks

* ProtAge: proteomic aging clock derived from Olink proteomics
    https://github.com/shayanmostafaei/Proteomic-Aging-Clock-ProtAge
* MetaboAge: metabolomic aging clock derived from Nightingale NMR metabolomics
    https://github.com/shayanmostafaei/Metabolomic-Aging-Clock-MetaboAge

⸻

Stepwise prediction models

The primary prediction framework used stepwise XGBoost models:

* Base model: age + sex
* Model 1: base model + lifestyle covariates
* Model 2: Model 1 + PRS
* Model 3: Model 2 + PhenoAge + Frailty Index + Telomere Length
* Model 4: Model 3 + ProtAge
* Model 5: Model 4 + MetaboAge

Primary metrics:

* ROC-AUC with 95% confidence intervals
* AUPRC with bootstrap 95% confidence intervals
* Pairwise ROC-AUC comparisons between consecutive models

⸻

Key analytic choices

This project is designed for prediction and risk stratification, not causal inference. Associations of PRS, biological aging measures, ProtAge, or MetaboAge with ADRD should not be interpreted as causal effects.

Why AUPRC is included

Because ADRD incidence is low, ROC-AUC alone can overstate practical performance. AUPRC is reported to better evaluate enrichment of future ADRD cases among individuals predicted to be at higher risk.

Competing-risk analysis

Death before ADRD was treated as a competing event using Fine–Gray models. This is important in aging cohorts because participants may die before receiving an ADRD diagnosis.

TwinGene interpretation

TwinGene validation evaluates whether the integrated signal transports beyond UK Biobank. It is not interpreted as strict frozen-coefficient validation of all UKB-derived omics clocks because cohort design, omics platforms, and feature availability differed.

⸻

Results at a glance

UK Biobank held-out test set

Stepwise XGBoost performance:

* Age + sex: ROC-AUC ≈ 0.79
* ￼	lifestyle: ROC-AUC ≈ 0.82
* ￼	PRS: ROC-AUC ≈ 0.86
* ￼	PhenoAge + Frailty Index + Telomere Length: ROC-AUC ≈ 0.89
* ￼	ProtAge: ROC-AUC ≈ 0.90
* Final Model 5, + MetaboAge: ROC-AUC ≈ 0.90, AUPRC ≈ 0.24

Fine–Gray competing-risk model:

* PRS: sHR ≈ 2.24
* ProtAge: sHR ≈ 2.01
* Top 25% predicted-risk group versus lower 75%: sHR ≈ 16.73

Robustness analyses

Across 100 repeated stratified 70/30 splits:

* Final XGBoost Model 5 mean ROC-AUC ≈ 0.903
* Final XGBoost Model 5 mean AUPRC ≈ 0.239
* Logistic-regression full-model benchmark mean ROC-AUC ≈ 0.868
* Logistic-regression full-model benchmark mean AUPRC ≈ 0.228

TwinGene external validation/replication

* ROC-AUC ≈ 0.757
* AUPRC ≈ 0.223
* Top 25% predicted-risk group versus lower 75%: sHR ≈ 4.02
* ADRD event proportion: 19.9% in the top 25% versus 5.1% in the lower 75%

⸻

Scripts

The main analysis scripts are stored in scripts/.

Script	Purpose
01_Univariate_Analysis_ROCs.R	Univariate ROC-AUC analyses for individual predictors
02_Interaction_Heatmaps_BioAge.R	Correlations and associations among biological aging markers, PRS, and covariates
03_Multivariable_XGBoost_Modeling.R	Stepwise XGBoost prediction models, ROC-AUC, AUPRC, DeLong tests, and prediction export
04_SHAP_Analysis.py	SHAP interpretation for the final XGBoost model
05_Survival_Competing_Risks_Analysis.R	Fine–Gray competing-risk analysis and cumulative incidence by predicted-risk group
06_Sensitivity_Analyses.R	Subgroup, prediction-horizon, early-event, and robustness analyses
07_Calibration_DCA_Thresholds.R	Calibration, Brier score, threshold operating characteristics, and decision-curve analysis
08_Repeated_Split_Stability.R	100 repeated stratified train–test split analyses for XGBoost and logistic regression
09_Benchmarking_Reclassification.R	Benchmarking against simpler predictor sets, IDI, and continuous NRI
10_Multiple_Imputation_Sensitivity.R	Complete-case versus multiple-imputation sensitivity analysis
11_TwinGene_External_Validation.R	TwinGene external validation/replication, ROC-AUC, AUPRC, and competing-risk risk stratification
99_session_info.R	R session information and package versions

⸻

Manuscript output map

Manuscript output	Main script
Figure 2: univariate predictor ROC-AUCs	01_Univariate_Analysis_ROCs.R
Figure 3: predictor correlation matrix	02_Interaction_Heatmaps_BioAge.R
Table 2: stepwise XGBoost performance	03_Multivariable_XGBoost_Modeling.R
Figure A.5: precision–recall curves	03_Multivariable_XGBoost_Modeling.R
Figure A.6: calibration	07_Calibration_DCA_Thresholds.R
Figure A.7: decision-curve analysis	07_Calibration_DCA_Thresholds.R
Figure A.8: SHAP summary plot	04_SHAP_Analysis.py
Figure 4: Fine–Gray predictor sHRs	05_Survival_Competing_Risks_Analysis.R
Figure 5: cumulative incidence by predicted-risk group	05_Survival_Competing_Risks_Analysis.R
Figure A.9: sensitivity analyses	06_Sensitivity_Analyses.R
Figure A.10: early-event exclusion	06_Sensitivity_Analyses.R
Table A.5: repeated-split XGBoost stability	08_Repeated_Split_Stability.R
Table A.6: repeated-split logistic-regression benchmark	08_Repeated_Split_Stability.R
Table A.8: simpler-model benchmarking and reclassification	09_Benchmarking_Reclassification.R
Table A.10: TwinGene predictor harmonization	11_TwinGene_External_Validation.R
Table A.11: TwinGene validation performance	11_TwinGene_External_Validation.R
Table A.12: TwinGene risk stratification	11_TwinGene_External_Validation.R
Figure A.11: TwinGene ROC and precision–recall curves	11_TwinGene_External_Validation.R
Figure A.12: TwinGene cumulative incidence curves	11_TwinGene_External_Validation.R

⸻

Inputs not included

Individual-level UK Biobank and TwinGene data are not included in this repository.


⸻

Requirements

Core R packages:

dplyr
tidyr
readr
caret
xgboost
pROC
PRROC
riskRegression
cmprsk
prodlim
survival
survminer
ggplot2
patchwork
mice

Core Python packages:

python >= 3.10
numpy
pandas
scikit-learn
xgboost
shap
matplotlib

⸻

Data governance

This repository does not distribute:

* raw UK Biobank data
* raw TwinGene data
* individual-level predictions
* identifiable participant information
* derived data requiring controlled access

Researchers must obtain access to UK Biobank and TwinGene through the appropriate governed access procedures.

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

This project is licensed under the MIT License — see the LICENSE file for details.

⸻

Contact

For questions or contributions, please contact:

* Dr. Shayan Mostafaei: shayan.mostafaei@ki.se
* Prof. Sara Hägg: sara.hagg@ki.se 
