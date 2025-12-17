# ==================================================================================
# 03_Multivariable_XGBoost_Modeling.R
# Stepwise multivariable ADRD prediction (XGBoost) with:
# - Stratified 70/30 holdout split
# - Train-only preprocessing 
# - Consistent test-set evaluation across stepwise models
# - AUC + 95% CI (DeLong) + pairwise DeLong tests between consecutive models
# - Final model AUPRC (recommended for imbalanced outcome)
# - Clean outputs: predictions + metrics tables + saved model objects
#
# Manuscript-aligned stepwise sequence:
#   Model 0: Age + Sex
#   Model 1: + Lifestyle 
#   Model 2: + PRS
#   Model 3: + PhenoAge + FI + TL
#   Model 4: + ProtAge
#   Model 5: + MetaboAge (final integrated model)
# ==================================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(xgboost)
  library(pROC)
  library(PRROC)    # for AUPRC
})

# --------------------------
# USER SETTINGS
# --------------------------

set.seed(20250101)

# Input: analysis table as .rds (recommended) OR load Biomarkers_complete before sourcing this script
INPUT_RDS <- "data/biomarkers_complete.rds"   

OUT_DIR <- "results/multivariable_xgboost"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Required columns
OUTCOME_COL <- "Dementia_status"   # 0/1 or factor
ID_COL <- "f.eid"                  # used for tracking predictions (change to your ID)
# If you don't have f.eid in this table, set ID_COL <- NULL

# Predictors (edit if your column names differ)
AGE_COL <- "CA"
SEX_COL <- "sex"
BMI_COL <- "bmi"
SMOKE_COL <- "smoking"
ALC_COL <- "alcohol"
EDU_COL <- "education"                 
PRS_COL <- "DDML_PRS_With_APOE"
PHENO_COL <- "PhenoAge"
FI_COL <- "FI"
TL_COL <- "TL"
PROTAGE_COL <- "ProtAge"
METABOAGE_COL <- "MetaboAge"

PC_COLS <- paste0("PC", 1:10)  # Genetic PCs

# XGBoost parameters (reasonable defaults; keep stable across models)
XGB_PARAMS <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 4,
  eta = 0.05,
  subsample = 0.9,
  colsample_bytree = 0.9,
  min_child_weight = 1
)
NROUNDS_MAX <- 2000
EARLY_STOPPING <- 50

# --------------------------
# LOAD DATA
# --------------------------

if (nzchar(INPUT_RDS)) {
  Biomarkers_complete <- readRDS(INPUT_RDS)
} else {
  if (!exists("Biomarkers_complete")) stop("No input found: set INPUT_RDS or load Biomarkers_complete before running.")
}

df0 <- Biomarkers_complete

# --------------------------
# VALIDATION + CLEANING
# --------------------------

required_base <- c(OUTCOME_COL, AGE_COL, SEX_COL, BMI_COL, SMOKE_COL, ALC_COL, EDU_COL,
                   PRS_COL, PHENO_COL, FI_COL, TL_COL, PROTAGE_COL, METABOAGE_COL)

missing_required <- setdiff(required_base, names(df0))
if (length(missing_required) > 0) {
  stop("Missing required columns in input data: ", paste(missing_required, collapse = ", "))
}

PC_COLS <- PC_COLS[PC_COLS %in% names(df0)]

# Outcome as 0/1 numeric (stable for xgboost + ROC tools)
y_raw <- df0[[OUTCOME_COL]]
if (is.factor(y_raw)) {
  # assume levels correspond to 0/1; make numeric robustly
  y_num <- as.numeric(as.character(y_raw))
  if (any(is.na(y_num))) {
    # fallback: use factor order
    y_num <- as.numeric(y_raw) - 1
  }
} else {
  y_num <- suppressWarnings(as.numeric(y_raw))
}
y_num <- ifelse(y_num == 1, 1, 0)

df1 <- df0 %>%
  mutate(
    .outcome = y_num,
    # standardize sex as factor
    sex_std = as.factor(.data[[SEX_COL]])
  )

# Optional: harmonize education as factor (keep as-is; model.matrix will dummy-code)
df1 <- df1 %>%
  mutate(education_std = as.factor(.data[[EDU_COL]]))

# --------------------------
# TRAIN/TEST SPLIT (70/30 stratified)
# --------------------------

train_idx <- caret::createDataPartition(df1$.outcome, p = 0.70, list = FALSE)
train_df <- df1[train_idx, , drop = FALSE]
test_df  <- df1[-train_idx, , drop = FALSE]

# --------------------------
# PREPROCESSING (TRAIN ONLY)
# --------------------------

# Identify numeric columns used across any model
numeric_candidates <- c(
  AGE_COL, BMI_COL, PRS_COL, PHENO_COL, FI_COL, TL_COL, PROTAGE_COL, METABOAGE_COL, PC_COLS
)
numeric_candidates <- numeric_candidates[numeric_candidates %in% names(df1)]

# Coerce these to numeric safely (in both splits)
for (col in numeric_candidates) {
  train_df[[col]] <- suppressWarnings(as.numeric(train_df[[col]]))
  test_df[[col]]  <- suppressWarnings(as.numeric(test_df[[col]]))
}

# Fit imputer on training numeric columns only
preproc <- caret::preProcess(train_df[, numeric_candidates, drop = FALSE], method = c("medianImpute"))

train_df_imp <- train_df
test_df_imp  <- test_df

train_df_imp[, numeric_candidates] <- predict(preproc, train_df[, numeric_candidates, drop = FALSE])
test_df_imp[, numeric_candidates]  <- predict(preproc, test_df[, numeric_candidates, drop = FALSE])

# Use standardized factor columns for modeling
train_df_imp$sex <- train_df_imp$sex_std
test_df_imp$sex  <- test_df_imp$sex_std
train_df_imp$education <- train_df_imp$education_std
test_df_imp$education  <- test_df_imp$education_std

# --------------------------
# STEPWISE MODEL DEFINITIONS 
# --------------------------

model_formulas <- list(
  Model0_Crude = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex")),
  Model1_Lifestyle = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + ",
                                      SMOKE_COL, " + ", ALC_COL, " + ", BMI_COL, " + education")),
  Model2_Add_PRS = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + ",
                                    SMOKE_COL, " + ", ALC_COL, " + ", BMI_COL, " + education + ",
                                    PRS_COL,
                                    if (length(PC_COLS) > 0) paste0(" + ", paste(PC_COLS, collapse = " + ")) else "")),
  Model3_Add_BA = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + ",
                                   SMOKE_COL, " + ", ALC_COL, " + ", BMI_COL, " + education + ",
                                   PRS_COL,
                                   if (length(PC_COLS) > 0) paste0(" + ", paste(PC_COLS, collapse = " + ")) else "",
                                   " + ", PHENO_COL, " + ", FI_COL, " + ", TL_COL)),
  Model4_Add_ProtAge = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + ",
                                        SMOKE_COL, " + ", ALC_COL, " + ", BMI_COL, " + education + ",
                                        PRS_COL,
                                        if (length(PC_COLS) > 0) paste0(" + ", paste(PC_COLS, collapse = " + ")) else "",
                                        " + ", PHENO_COL, " + ", FI_COL, " + ", TL_COL,
                                        " + ", PROTAGE_COL)),
  Model5_Add_MetaboAge = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + ",
                                          SMOKE_COL, " + ", ALC_COL, " + ", BMI_COL, " + education + ",
                                          PRS_COL,
                                          if (length(PC_COLS) > 0) paste0(" + ", paste(PC_COLS, collapse = " + ")) else "",
                                          " + ", PHENO_COL, " + ", FI_COL, " + ", TL_COL,
                                          " + ", PROTAGE_COL, " + ", METABOAGE_COL))
)

# --------------------------
# TRAIN + EVALUATE FUNCTION
# --------------------------

fit_xgb_step <- function(formula, trainData, testData, params, nrounds_max, early_stop) {
  # Use model.matrix for stable dummy encoding (train/test aligned by formula)
  X_train <- model.matrix(formula, data = trainData)[, -1, drop = FALSE]
  X_test  <- model.matrix(formula, data = testData)[, -1, drop = FALSE]
  y_train <- trainData$.outcome
  y_test  <- testData$.outcome

  dtrain <- xgboost::xgb.DMatrix(data = X_train, label = y_train)
  dtest  <- xgboost::xgb.DMatrix(data = X_test, label = y_test)

  model <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds_max,
    watchlist = list(train = dtrain, test = dtest),
    early_stopping_rounds = early_stop,
    verbose = 0
  )

  pred <- as.numeric(predict(model, newdata = dtest))

  roc_obj <- pROC::roc(response = y_test, predictor = pred, levels = c(0, 1), direction = "auto", quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci <- as.numeric(pROC::ci.auc(roc_obj, method = "delong"))

  list(
    model = model,
    pred = pred,
    y_test = y_test,
    roc = roc_obj,
    auc = auc_val,
    ci_low = ci[1],
    ci_high = ci[3]
  )
}

# --------------------------
# RUN STEPWISE MODELS
# --------------------------

model_results <- list()
pred_table <- data.frame(
  ID = if (!is.null(ID_COL) && ID_COL %in% names(test_df_imp)) as.character(test_df_imp[[ID_COL]]) else seq_len(nrow(test_df_imp)),
  y_test = test_df_imp$.outcome
)

for (nm in names(model_formulas)) {
  res <- fit_xgb_step(
    formula = model_formulas[[nm]],
    trainData = train_df_imp,
    testData = test_df_imp,
    params = XGB_PARAMS,
    nrounds_max = NROUNDS_MAX,
    early_stop = EARLY_STOPPING
  )
  model_results[[nm]] <- res
  pred_table[[paste0("pred_", nm)]] <- res$pred
  saveRDS(res$model, file.path(OUT_DIR, paste0(nm, "_xgb_model.rds")))
}

# --------------------------
# SUMMARIZE AUCs + PAIRWISE TESTS 
# --------------------------

auc_summary <- bind_rows(lapply(names(model_results), function(nm) {
  res <- model_results[[nm]]
  data.frame(
    Model = nm,
    AUC = res$auc,
    CI_low = res$ci_low,
    CI_high = res$ci_high,
    Best_nrounds = res$model$best_iteration,
    stringsAsFactors = FALSE
  )
})) %>% arrange(Model)

write.csv(auc_summary, file.path(OUT_DIR, "stepwise_auc_summary.csv"), row.names = FALSE)

pairwise_tests <- list()
model_names <- names(model_results)

for (i in 2:length(model_names)) {
  m_prev <- model_names[i - 1]
  m_curr <- model_names[i]
  test <- pROC::roc.test(model_results[[m_prev]]$roc, model_results[[m_curr]]$roc, method = "delong")
  pairwise_tests[[paste0(m_prev, "_vs_", m_curr)]] <- data.frame(
    Comparison = paste0(m_prev, " vs ", m_curr),
    p_value = as.numeric(test$p.value),
    stringsAsFactors = FALSE
  )
}

pairwise_df <- bind_rows(pairwise_tests)
write.csv(pairwise_df, file.path(OUT_DIR, "pairwise_auc_delong_tests.csv"), row.names = FALSE)

# --------------------------
# AUPRC for final model (Model5)
# --------------------------

final_name <- "Model5_Add_MetaboAge"
if (final_name %in% names(model_results)) {
  y <- model_results[[final_name]]$y_test
  s <- model_results[[final_name]]$pred

  # PRROC expects scores for class 1 and 0 separately
  fg <- PRROC::pr.curve(scores.class0 = s[y == 1], scores.class1 = s[y == 0], curve = FALSE)
  auprc <- fg$auc.integral

  auprc_df <- data.frame(Model = final_name, AUPRC = as.numeric(auprc))
  write.csv(auprc_df, file.path(OUT_DIR, "final_model_auprc.csv"), row.names = FALSE)
}

# --------------------------
# SAVE PREDICTIONS (TEST SET)
# --------------------------

write.csv(pred_table, file.path(OUT_DIR, "test_set_predictions_stepwise.csv"), row.names = FALSE)

cat("\n=== Stepwise AUC Summary (TEST set) ===\n")
print(auc_summary)

cat("\n=== Pairwise DeLong tests (consecutive models) ===\n")
print(pairwise_df)

if (file.exists(file.path(OUT_DIR, "final_model_auprc.csv"))) {
  cat("\n=== Final model AUPRC written to final_model_auprc.csv ===\n")
}

cat("\nDONE ✅ Multivariable XGBoost stepwise modeling completed.\n")
cat("Output folder:", OUT_DIR, "\n\n")
