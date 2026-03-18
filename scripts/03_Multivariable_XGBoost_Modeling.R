# ==============================================================================
# 03_Multivariable_XGBoost_Modeling.R
# Stepwise multivariable ADRD prediction using XGBoost:
# - Stratified holdout split
# - Train-only imputation/scaling
# - Consistent stepwise test evaluation
# - ROC-AUC + 95% CI, DeLong tests between models
# - Final model AUPRC + bootstrap CI
# - Clean exports: predictions, metrics, model objects, SHAP inputs
# AUTHOR: Shayan Mostafaei
# DATE CREATED: 2026-03-18
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(xgboost)
  library(pROC)
  library(PRROC)
  library(readr)
})

# --------------------------
# USER CONFIGURATION
# --------------------------

set.seed(20250101)
INPUT_RDS <- "data/biomarkers_complete.rds"
OUT_DIR <- file.path("results", "04_stepwise_xgboost")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

OUTCOME_COL <- "Dementia_status"
ID_COL <- "f.eid"  # set to NULL if not present

AGE_COL <- "CA"
SEX_COL <- "sex"
BMI_COL <- "bmi"
SMOKE_COL <- "smoking"
ALC_COL <- "alcohol_intake_frequency"
EDU_COL <- "education"
PRS_COL <- "PRS_ADRD"
PHENO_COL <- "PhenoAge"
FI_COL <- "FI"
TL_COL <- "TL"
PROTAGE_COL <- "ProtAge"
METABOAGE_COL <- "MetaboAge"
PC_COLS <- paste0("PC", 1:10)  # will be filtered by presence

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

BOOT_AUPRC <- TRUE
BOOT_N <- 2000

# --------------------------
# LOAD/VALIDATE DATA
# --------------------------

if (!nzchar(INPUT_RDS) || !file.exists(INPUT_RDS))
  stop("❌ Input RDS not found: ", INPUT_RDS)
df0 <- readRDS(INPUT_RDS)

required_cols <- c(
  OUTCOME_COL, AGE_COL, SEX_COL, BMI_COL, SMOKE_COL, ALC_COL, EDU_COL,
  PRS_COL, PHENO_COL, FI_COL, TL_COL, PROTAGE_COL, METABOAGE_COL
)
missing <- setdiff(required_cols, names(df0))
if (length(missing))
  stop("❌ Missing required columns: ", paste(missing, collapse = ", "))

PC_COLS <- PC_COLS[PC_COLS %in% names(df0)]

# Re-encode outcome as 0/1
y_raw <- df0[[OUTCOME_COL]]
y_num <- if (is.factor(y_raw)) {
  tmp <- suppressWarnings(as.numeric(as.character(y_raw)))
  if (any(is.na(tmp))) as.numeric(y_raw) - 1 else tmp
} else suppressWarnings(as.numeric(y_raw))
y_num <- ifelse(y_num == 1, 1, 0)

df1 <- df0 %>%
  mutate(
    .outcome = y_num,
    sex_std = as.factor(.data[[SEX_COL]]),
    education_std = as.factor(.data[[EDU_COL]]),
    smoking_std = as.factor(.data[[SMOKE_COL]]),
    alcohol_std = as.factor(.data[[ALC_COL]])
  )

# --------------------------
# STRATIFIED 70/30 DATA SPLIT
# --------------------------

train_idx <- caret::createDataPartition(df1$.outcome, p = 0.70, list = FALSE)
train_df <- df1[train_idx, , drop = FALSE]
test_df <- df1[-train_idx, , drop = FALSE]

# --------------------------
# PREPROCESSING: TRAIN MEDIAN IMPUTATION
# --------------------------

num_candidates <- c(
  AGE_COL, BMI_COL, PRS_COL, PHENO_COL, FI_COL, TL_COL, PROTAGE_COL, METABOAGE_COL, PC_COLS
)
num_candidates <- num_candidates[num_candidates %in% names(df1)]
convert_numeric <- function(x) suppressWarnings(as.numeric(x))
for (col in num_candidates) {
  train_df[[col]] <- convert_numeric(train_df[[col]])
  test_df[[col]] <- convert_numeric(test_df[[col]])
}

preproc <- caret::preProcess(train_df[, num_candidates, drop=FALSE], method = c("medianImpute"))
train_df_imp <- train_df
test_df_imp <- test_df
train_df_imp[, num_candidates] <- predict(preproc, train_df[, num_candidates, drop = FALSE])
test_df_imp[, num_candidates] <- predict(preproc, test_df[, num_candidates, drop = FALSE])

# Harmonize factors (imputation above can't affect these)
factor_cols <- list(
  sex = "sex_std", education = "education_std",
  smoking = "smoking_std", alcohol_intake_frequency = "alcohol_std"
)
for (nm in names(factor_cols)) {
  train_df_imp[[nm]] <- train_df_imp[[factor_cols[[nm]]]]
  test_df_imp[[nm]] <- test_df_imp[[factor_cols[[nm]]]]
}

# --------------------------
# STEPWISE MODEL FORMULAS
# --------------------------

pc_term <- if (length(PC_COLS)) paste0(" + ", paste(PC_COLS, collapse = " + ")) else ""
formulas <- list(
  Model0_Base = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex")),
  Model1_Lifestyle = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL, " + education")),
  Model2_Add_PRS = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL, " + education + ",", PRS_COL, pc_term)),
  Model3_Add_BA  = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL, " + education + ",", PRS_COL, pc_term, " + ", PHENO_COL, " + ", FI_COL, " + ", TL_COL)),
  Model4_Add_ProtAge = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL, " + education + ",", PRS_COL, pc_term, " + ", PHENO_COL, " + ", FI_COL, " + ", TL_COL, " + ", PROTAGE_COL)),
  Model5_Add_MetaboAge = as.formula(paste0(".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL, " + education + ",", PRS_COL, pc_term, " + ", PHENO_COL, " + ", FI_COL, " + ", TL_COL, " + ", PROTAGE_COL, " + ", METABOAGE_COL))
)

# --------------------------
# XGBOOST TRAIN + EVAL FUNCTION
# --------------------------

fit_xgb <- function(formula, trainData, testData, params, nrounds_max, early_stop) {
  X_train <- model.matrix(formula, data = trainData)[, -1, drop=FALSE]
  X_test <- model.matrix(formula, data = testData)[, -1, drop=FALSE]
  y_train <- trainData$.outcome
  y_test <- testData$.outcome

  dtrain <- xgboost::xgb.DMatrix(data = X_train, label = y_train)
  dtest <- xgboost::xgb.DMatrix(data = X_test, label = y_test)

  model <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds_max,
    watchlist = list(train = dtrain, test = dtest),
    early_stopping_rounds = early_stop,
    verbose = 0
  )

  pred <- predict(model, newdata = dtest)
  roc_obj <- pROC::roc(response = y_test, predictor = pred, levels = c(0, 1), direction = "auto", quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_auc <- as.numeric(pROC::ci.auc(roc_obj, method = "delong"))
  list(
    model = model,
    pred = pred,
    y_test = y_test,
    roc = roc_obj,
    auc = auc_val,
    ci_low = ci_auc[1],
    ci_high = ci_auc[3],
    best_nrounds = model$best_iteration
  )
}

# --------------------------
# TRAIN STEPWISE MODELS
# --------------------------

model_results <- list()
pred_table <- tibble(
  ID = if (!is.null(ID_COL) && ID_COL %in% names(test_df_imp)) as.character(test_df_imp[[ID_COL]]) else as.character(seq_along(test_df_imp$.outcome)),
  y_test = test_df_imp$.outcome
)
for (nm in names(formulas)) {
  res <- fit_xgb(formulas[[nm]], train_df_imp, test_df_imp, XGB_PARAMS, NROUNDS_MAX, EARLY_STOPPING)
  model_results[[nm]] <- res
  pred_table[[paste0("pred_", nm)]] <- res$pred
  saveRDS(res$model, file.path(OUT_DIR, paste0(nm, "_xgb_model.rds")))
}

# ---- Export Model 5 matrices/labels for Python SHAP ----
m5_name <- "Model5_Add_MetaboAge"
X_train_m5 <- model.matrix(formulas[[m5_name]], data = train_df_imp)[, -1, drop = FALSE]
X_test_m5  <- model.matrix(formulas[[m5_name]], data = test_df_imp)[, -1, drop = FALSE]
shap_input <- list(
  X_train = as.data.frame(X_train_m5),
  X_test  = as.data.frame(X_test_m5),
  y_train = as.integer(train_df_imp$.outcome),
  y_test  = as.integer(test_df_imp$.outcome)
)
saveRDS(shap_input, file.path(OUT_DIR, "model5_shap_input.rds"))
xgboost::xgb.save(model_results[[m5_name]]$model, file.path(OUT_DIR, "Model5_Add_MetaboAge_xgb_model.json"))

# --------------------------
# SUMMARIZE AUC AND TESTS
# --------------------------

auc_summary <- bind_rows(lapply(names(model_results), function(nm) {
  res <- model_results[[nm]]
  tibble(
    Model = nm,
    AUC = res$auc,
    CI_low = res$ci_low,
    CI_high = res$ci_high,
    Best_nrounds = res$best_nrounds
  )
}))
write_csv(auc_summary, file.path(OUT_DIR, "stepwise_auc_summary.csv"))

# DeLong pairwise model comparisons (consecutive)
model_names <- names(model_results)
pairwise_tests <- list()
if (length(model_names) >= 2) {
  for (i in 2:length(model_names)) {
    m_prev <- model_names[i - 1]
    m_curr <- model_names[i]
    test <- pROC::roc.test(model_results[[m_prev]]$roc, model_results[[m_curr]]$roc, method = "delong")
    pairwise_tests[[paste0(m_prev, "_vs_", m_curr)]] <- tibble(
      Comparison = paste0(m_prev, " vs ", m_curr),
      p_value = as.numeric(test$p.value)
    )
  }
}
pairwise_df <- bind_rows(pairwise_tests)
write_csv(pairwise_df, file.path(OUT_DIR, "pairwise_auc_delong_tests.csv"))

# --------------------------
# FINAL MODEL (Model 5) AUPRC (+ BOOTSTRAP)
# --------------------------

if (m5_name %in% names(model_results)) {
  y <- model_results[[m5_name]]$y_test
  s <- model_results[[m5_name]]$pred

  pr <- PRROC::pr.curve(scores.class0 = s[y == 1], scores.class1 = s[y == 0], curve = FALSE)
  auprc <- as.numeric(pr$auc.integral)
  auprc_df <- tibble(Model = m5_name, AUPRC = auprc)

  if (BOOT_AUPRC) {
    set.seed(20250101)
    idx_cases <- which(y == 1)
    idx_ctrls <- which(y == 0)
    boot_vals <- replicate(BOOT_N, {
      bs_idx <- c(sample(idx_cases, length(idx_cases), replace = TRUE),
                  sample(idx_ctrls, length(idx_ctrls), replace = TRUE))
      yb <- y[bs_idx]; sb <- s[bs_idx]
      prb <- PRROC::pr.curve(scores.class0 = sb[yb == 1], scores.class1 = sb[yb == 0], curve = FALSE)
      as.numeric(prb$auc.integral)
    })
    auprc_df <- auprc_df %>%
      mutate(
        AUPRC_CI_low = quantile(boot_vals, 0.025, na.rm = TRUE),
        AUPRC_CI_high = quantile(boot_vals, 0.975, na.rm = TRUE),
        AUPRC_boot_N = BOOT_N
      )
  }
  write_csv(auprc_df, file.path(OUT_DIR, "final_model_auprc.csv"))
}

# --------------------------
# EXPORT TEST SET PREDICTIONS
# --------------------------

write_csv(pred_table, file.path(OUT_DIR, "test_set_predictions_stepwise.csv"))

cat("\n=== Stepwise AUC Summary (TEST set) ===\n")
print(auc_summary)
cat("\n=== Pairwise DeLong tests (consecutive models) ===\n")
print(pairwise_df)
if (file.exists(file.path(OUT_DIR, "final_model_auprc.csv"))) {
  cat("\n=== Final model AUPRC written to final_model_auprc.csv ===\n")
}
cat("\n✅ DONE: Multivariable XGBoost stepwise modeling completed.\n")
cat("Output folder:", OUT_DIR, "\n\n")
