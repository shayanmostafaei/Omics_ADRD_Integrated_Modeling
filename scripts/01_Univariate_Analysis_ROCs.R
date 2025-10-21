# ==================================================================================
# 01_Univariate_Analysis_ROCs.R
# Computes Univariate ROCs (Overall & By ADRD Subtypes)
# ==================================================================================

library(pROC)
library(ggplot2)
library(RColorBrewer)
library(dplyr)

# Note: Assumes 'Biomarkers' or 'Biomarkers_complete' data frames are loaded
# Ensure the outcome column 'Dementia_status' is numeric (0 = no dementia, 1 = dementia)
# Dementia subtypes columns: 'AD', 'vascular', 'Others_Unspecified' (0/1)

# List of predictors
predictors <- c("CA", "PRS_ADRD", "ProtAge", "Prot_Age_Gap", "MetaboAge", 
                "Metabo_Age_Gap", "PhenoAge", "KDM", "HD", "FI", "TL")

# --------------------------------------------
# A. Univariate ROC analysis for overall ADRD
# --------------------------------------------
auc_results_overall <- list()

for (var in predictors) {
  df <- Biomarkers_complete %>% 
    dplyr::filter(!is.na(.data[[var]]) & !is.na(Dementia_status))
  
  if(nrow(df) > 0){
    roc_obj <- pROC::roc(df$Dementia_status, df[[var]], quiet = TRUE)
    auc_val <- as.numeric(pROC::auc(roc_obj))
    ci <- pROC::ci.auc(roc_obj)
    
    auc_results_overall[[var]] <- data.frame(
      Predictor = var,
      AUC = auc_val,
      CI_low = ci[1],
      CI_high = ci[3]
    )
  }
}

auc_df_overall <- do.call(rbind, auc_results_overall)
auc_df_overall <- na.omit(auc_df_overall)
print("=== Univariate AUCs (Overall ADRD) ===")
print(auc_df_overall)

# --------------------------------------------
# B. Univariate ROC analysis by dementia type
# --------------------------------------------
dementia_types <- c("AD", "vascular", "Others_Unspecified")
auc_results_type <- list()

for (dementia in dementia_types) {
  for (var in predictors) {
    df <- Biomarkers_complete %>%
      dplyr::filter(!is.na(.data[[var]]) & !is.na(.data[[dementia]]))
    
    if(nrow(df) > 0){
      roc_obj <- pROC::roc(df[[dementia]], df[[var]], quiet = TRUE)
      auc_val <- as.numeric(pROC::auc(roc_obj))
      ci <- pROC::ci.auc(roc_obj)
      
      auc_results_type[[paste(dementia, var, sep = "_")]] <- data.frame(
        Dementia_Type = dementia,
        Predictor = var,
        AUC = auc_val,
        CI_low = ci[1],
        CI_high = ci[3]
      )
    }
  }
}

auc_df_type <- do.call(rbind, auc_results_type)
auc_df_type <- na.omit(auc_df_type)
print("=== Univariate AUCs by Dementia Type ===")
print(auc_df_type)

# --------------------------------------------
# Optional: Save results
# --------------------------------------------
if(!dir.exists("results")) dir.create("results")
save(auc_df_overall, auc_df_type, file = "results/univariate_ROC_metrics.RData")
