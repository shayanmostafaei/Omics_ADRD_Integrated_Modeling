# Omics_ADRD_Integrated_Modeling

This project combines multi-omics biological aging clocks, polygenic risk scores (PRS), and advanced machine learning techniques to predict Alzheimer's disease and related dementias (ADRD) using UK Biobank data. It includes R and Python code for data preprocessing, model construction, survival and competing risk analysis, and visualization.

**Omics_ADRD_Integrated_Modeling** is a comprehensive repository for the integrative modeling framework developed in the study:


## Overview

This project integrates multi-omics biological aging clocks and polygenic risk scores (PRS) using advanced machine learning to predict Alzheimer's disease and related dementias (ADRD) in UK Biobank data. The repository contains both R and Python code for data preprocessing, model construction, survival and competing risk analysis, and visualization.


## Key Features

- **Biological Aging Clocks:**  
  - *PhenoAge* and *KDM* clocks and *HD*, *FI*, and *TL* are constructed using "BioAge" package in R.
- **Omics_based Aging Clocks:**  
  - *MetaboAge* and *ProtAge* clocks constructed using stacked ensemble models in R.
- **Polygenic Risk Score Modeling:**  
  - Deep data-driven Bayesian variational autoencoder implemented in Python for PRS calculation.
- **Survival and Competing Risk Analysis:**  
  - Fine and Gray competing risk models using the `riskRegression` and `cmprsk` R packages.
- **Time-to-Event and ROC Analysis:**  
  - Functions for time-dependent ROC curves and survival/cumulative incidence plots.
- **Risk Stratification:**  
  - Individuals are stratified by their predicted 5-year risk of ADRD, based on integrative multi-omics and genetic models.
  - The default approach compares the top 25% ("high-risk") group to the bottom 75% ("low-risk") group, ensuring stable event counts and interpretable hazard ratios.
  - Competing risk methods (Fine-Gray models) are used to estimate subdistribution hazard ratios (sHR) for incident dementia, accounting for the competing risk of death.
  - All stratification and risk group analyses are fully reproducible and can be customized (e.g., different quantile cutoffs) using the provided R scripts.

## Files

---

## How to Use

1. Prepare your all datasets in R or Python. 
2. Run the scripts in order as listed above for each analysis. 
3. Follow instructions in each script for details.

## Reference

If you use this code, please cite:

> Mostafaei S, et al. (2025) "Precision Prediction of Alzheimer's Disease and Related Dementias Using Integrative Multi-Omics Aging Clocks and Genetic Data" [Manuscript].  

## License

This project is licensed under the MIT License

## Contact

For questions or contributions, please contact: • Dr. Shayan Mostafaei (shayan.mostafaei@ki.se) • Dr. Sara Hägg (sara.hagg@ki.se)
