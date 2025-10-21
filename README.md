# Omics_ADRD_Integrated_Modeling

This project integrates multi-omics biological aging clocks, polygenic risk scores (PRS), and advanced machine learning to predict Alzheimer's disease and related dementias (ADRD) using UK Biobank (UKB) data. The repository includes R and Python code for data preprocessing, model construction, survival and competing risk analysis, and visualization.

**Omics_ADRD_Integrated_Modeling** provides a comprehensive framework for integrative predictive modeling of ADRD.

---

## Overview

This study leverages ~500,000 UKB participants aged 37–73 with a median follow-up of 9.2 years. Individuals with genetically non-European ancestry, close relatives, or those who withdrew consent were excluded.  

The modeling framework integrates:

- Genetic risk via polygenic risk scores (PRS)  
- Biological aging measures from clinical biomarkers  
- Proteomic (ProtAge) and metabolomic (MetaboAge) aging clocks  
- Survival and competing risk analysis using advanced machine learning  

Ethical approval: North West Multi-Centre Research Ethics Committee (UKB) and Swedish Ethical Review Authority. All participants provided informed consent. Analyses adhere to the Declaration of Helsinki and TRIPOD reporting guidelines.

---

## Key Features

- **Genetic Data and Polygenic Risk Scores (PRS):**  
  - PRS constructed using 80 independent SNPs (including APOE) from GWAS summary statistics.  
  - Bayesian variational autoencoder trained on individual-level genotypes for latent representation of SNP effects.  
  - Code: [DDML_PRS_ADRD](https://github.com/shayanmostafaei/DDML_PRS_ADRD)

- **Biological Aging Measures:**  
  - PhenoAge, Klemera-Doubal method (KDM), homeostatic dysregulation (HD)  
  - Frailty Index (FI) and relative leucocyte telomere length (TL)  
  - Calculated using the **BioAge** R package

- **Metabolomic Aging Clock (MetaboAge):**  
  - 230,329 participants with 184 NMR-based metabolic variables  
  - Preprocessing: Box-Cox transformation, KNN imputation (k=10), Mahalanobis outlier detection  
  - Model: Stacked ensemble of XGBoost, LightGBM, CatBoost with Elastic-Net meta-learner (α=0.5)  
  - Sex-stratified models  
  - Code: [MetaboAge GitHub](https://github.com/shayanmostafaei/Metabolomic-Aging-Clock-MetaboAge-)

- **Proteomic Aging Clock (ProtAge):**  
  - 2,937 proteins from 55,327 participants (Olink platform)  
  - Preprocessing: Box-Cox transformation, KNN imputation (k=10), iForest outlier detection  
  - Model: Stacked ensemble of XGBoost, LightGBM, CatBoost with Elastic-Net meta-learner (α=0.5)  
  - Sex-stratified models  
  - Code: [ProtAge GitHub](https://github.com/shayanmostafaei/Proteomic-Aging-Clock-ProtAge-)

- **Predictive Modeling of ADRD:**  
  - Integrates PRS, BA measures, ProtAge, and MetaboAge using XGBoost ensemble  
  - Stepwise inclusion of predictors for sequential model enrichment  
  - Adjusted for age, sex, BMI, alcohol, smoking, education, and top 10 genetic PCs  
  - Competing risk modeling using Fine and Gray method to account for death  

- **Survival Analysis and Risk Stratification:**  
  - Composite risk score derived from Model 5 predictors: CA, PRS, PhenoAge, FI, TL, ProtAge, MetaboAge, covariates  
  - Time-to-event: earliest of ADRD diagnosis or death  
  - Risk groups: top 25% high-risk vs bottom 75% low-risk  
  - Subdistribution hazard ratios (sHR) estimated using Fine-Gray competing risk models  

- **Software and Tools:**  
  - **R (v4.4.3):** survival, riskRegression, cmprsk, prodlim, survminer, ggplot2, BioAge, XGBoost, LightGBM, CatBoost, glmnet  
  - **Python (v3.10):** TensorFlow, Keras, scikit-learn  

---

## Files

- R scripts for preprocessing, modeling, survival analysis, and visualization  
- Python scripts for PRS calculation  
- Example datasets and configuration files for reproducible workflows  

---

## How to Use

1. Prepare datasets in R or Python following repository guidelines.  
2. Run scripts sequentially as indicated for each analysis.  
3. Follow inline instructions in each script for details and parameter settings.  

---

## Reference

If you use this code, please cite:

> Mostafaei S, et al. (2025) "Precision Prediction of Alzheimer's Disease and Related Dementias Using Integrative Multi-Omics Aging Clocks and Genetic Data" [Manuscript].  

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or contributions, contact:  

- **Dr. Shayan Mostafaei** (shayan.mostafaei@ki.se)  
- **Dr. Sara Hägg** (sara.hagg@ki.se)  
