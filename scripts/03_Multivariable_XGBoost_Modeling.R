# ==============================================================================
# 03_Multivariable_XGBoost_Modeling.R
# Stepwise multivariable ADRD prediction using XGBoost:
# - Stratified 70/30 holdout split (final evaluation on held-out test set)
# - Repeated stratified k-fold CV (5 × 3) on training split for hyperparameter
#   tuning and internal AUC estimation (no leakage into test set)
# - Hyperparameter grid search via caret::train() on Model 5 (full model);
#   best params reused for Models 0–4 (same dataset, computationally practical)
# - scale_pos_weight to address class imbalance (ADRD is rare)
# - Train-only imputation/scaling applied to test set
# - Consistent stepwise test evaluation
# - ROC-AUC + 95% CI (DeLong), pairwise DeLong tests between models
# - Final model AUPRC + bootstrap CI
# - Clean exports: predictions, metrics, model objects, SHAP inputs (.rds + .parquet)
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
  library(arrow)
})

# --------------------------
# USER CONFIGURATION
# --------------------------

set.seed(20250101)
INPUT_RDS <- "data/biomarkers_complete.rds"
OUT_DIR <- file.path("results", "04_stepwise_xgboost")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

OUTCOME_COL <- "Dementia_status"
ID_COL      <- "f.eid"  # set to NULL if not present

AGE_COL      <- "CA"
SEX_COL      <- "sex"
BMI_COL      <- "bmi"
SMOKE_COL    <- "smoking"
ALC_COL      <- "alcohol_intake_frequency"
EDU_COL      <- "education"
PRS_COL      <- "PRS_ADRD"
PHENO_COL    <- "PhenoAge"
FI_COL       <- "FI"
TL_COL       <- "TL"
PROTAGE_COL  <- "ProtAge"
METABOAGE_COL <- "MetaboAge"
PC_COLS      <- paste0("PC", 1:10)  # will be filtered by presence

# CV settings (training split only; test set is never seen during tuning)
CV_FOLDS   <- 5
CV_REPEATS <- 3

# Hyperparameter tuning grid (applied to Model 5 only via caret::train)
TUNE_GRID <- expand.grid(
  nrounds          = c(500, 1000, 1500),
  max_depth        = c(3, 4, 6),
  eta              = c(0.01, 0.05, 0.1),
  subsample        = c(0.8, 0.9),
  colsample_bytree = c(0.8, 0.9),
  min_child_weight = c(1, 5),
  gamma            = c(0, 0.1)
)

BOOT_AUPRC <- TRUE
BOOT_N     <- 2000

# --------------------------
# LOAD / VALIDATE DATA
# --------------------------

if (!nzchar(INPUT_RDS) || !file.exists(INPUT_RDS))
  stop("❌ Input RDS not found: ", INPUT_RDS)
df0 <- readRDS(INPUT_RDS)

required_cols <- c(
  OUTCOME_COL, AGE_COL, SEX_COL, BMI_COL, SMOKE_COL, ALC_COL, EDU_COL,
  PRS_COL, PHENO_COL, FI_COL, TL_COL, PROTAGE_COL, METABOAGE_COL
)
missing_req <- setdiff(required_cols, names(df0))
if (length(missing_req))
  stop("❌ Missing required columns: ", paste(missing_req, collapse = ", "))

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
    .outcome      = y_num,
    sex_std       = as.factor(.data[[SEX_COL]]),
    education_std = as.factor(.data[[EDU_COL]]),
    smoking_std   = as.factor(.data[[SMOKE_COL]]),
    alcohol_std   = as.factor(.data[[ALC_COL]])
  )

# --------------------------
# STRATIFIED 70/30 DATA SPLIT
# --------------------------

train_idx <- caret::createDataPartition(df1$.outcome, p = 0.70, list = FALSE)
train_df  <- df1[ train_idx, , drop = FALSE]
test_df   <- df1[-train_idx, , drop = FALSE]

# Save test-set IDs explicitly so downstream scripts can reconstruct the split
test_ids <- tibble(
  row_idx = which(seq_len(nrow(df1)) %in% setdiff(seq_len(nrow(df1)), train_idx)),
  ID      = if (!is.null(ID_COL) && ID_COL %in% names(df1))
              as.character(df1[-train_idx, ID_COL, drop = TRUE])
            else
              as.character(setdiff(seq_len(nrow(df1)), train_idx))
)
write_csv(test_ids, file.path(OUT_DIR, "test_ids.csv"))

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
  test_df[[col]]  <- convert_numeric(test_df[[col]])
}

# Missingness report on training set (before imputation)
missing_rates <- data.frame(
  Column       = num_candidates,
  N_missing    = sapply(num_candidates, function(col) sum(is.na(train_df[[col]]))),
  Missing_rate = sapply(num_candidates, function(col) mean(is.na(train_df[[col]]))),
  stringsAsFactors = FALSE
)
write_csv(missing_rates, file.path(OUT_DIR, "missing_rates_train.csv"))
cat("ℹ️  Missingness report saved to missing_rates_train.csv\n")

preproc <- caret::preProcess(train_df[, num_candidates, drop = FALSE], method = "medianImpute")
train_df_imp <- train_df
test_df_imp  <- test_df
train_df_imp[, num_candidates] <- predict(preproc, train_df[, num_candidates, drop = FALSE])
test_df_imp[,  num_candidates] <- predict(preproc, test_df[,  num_candidates, drop = FALSE])

# Harmonize factor columns under model-friendly names
factor_map <- list(
  sex = "sex_std", education = "education_std",
  smoking = "smoking_std", alcohol_intake_frequency = "alcohol_std"
)
for (nm in names(factor_map)) {
  train_df_imp[[nm]] <- train_df_imp[[factor_map[[nm]]]]
  test_df_imp[[nm]]  <- test_df_imp[[factor_map[[nm]]]]
}

# Class-imbalance weight (used in XGBoost params below)
n_pos <- sum(train_df_imp$.outcome == 1)
n_neg <- sum(train_df_imp$.outcome == 0)
scale_pos_weight <- n_neg / n_pos
cat(sprintf("ℹ️  Training set: %d positives, %d negatives → scale_pos_weight = %.2f\n",
            n_pos, n_neg, scale_pos_weight))

# --------------------------
# STEPWISE MODEL FORMULAS
# --------------------------

pc_term <- if (length(PC_COLS)) paste0(" + ", paste(PC_COLS, collapse = " + ")) else ""

formulas <- list(
  Model0_Base = as.formula(paste0(
    ".outcome ~ ", AGE_COL, " + sex")),
  Model1_Lifestyle = as.formula(paste0(
    ".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL, " + education")),
  Model2_Add_PRS = as.formula(paste0(
    ".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL,
    " + education + ", PRS_COL, pc_term)),
  Model3_Add_BA = as.formula(paste0(
    ".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL,
    " + education + ", PRS_COL, pc_term, " + ", PHENO_COL, " + ", FI_COL, " + ", TL_COL)),
  Model4_Add_ProtAge = as.formula(paste0(
    ".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL,
    " + education + ", PRS_COL, pc_term, " + ", PHENO_COL, " + ", FI_COL, " + ", TL_COL,
    " + ", PROTAGE_COL)),
  Model5_Add_MetaboAge = as.formula(paste0(
    ".outcome ~ ", AGE_COL, " + sex + smoking + alcohol_intake_frequency + ", BMI_COL,
    " + education + ", PRS_COL, pc_term, " + ", PHENO_COL, " + ", FI_COL, " + ", TL_COL,
    " + ", PROTAGE_COL, " + ", METABOAGE_COL))
)

m5_name <- "Model5_Add_MetaboAge"

# --------------------------
# HYPERPARAMETER TUNING (MODEL 5 ONLY, TRAINING SPLIT)
# --------------------------
# caret::train() with repeated stratified CV is used here solely to find the
# best hyperparameters and obtain an internal CV-AUC estimate. The final model
# for each step is re-trained on the full training set using the best params.
# Models 0–4 share the same dataset, so using Model 5's best params is a
# computationally practical and documented approximation.

cat("\n--- Hyperparameter tuning via", CV_FOLDS, "×", CV_REPEATS,
    "repeated stratified CV (Model 5 only) ---\n")

# Build design matrix for Model 5 training data (caret needs a data.frame)
X_train_m5_tune <- as.data.frame(
  model.matrix(formulas[[m5_name]], data = train_df_imp)[, -1, drop = FALSE]
)
y_train_factor <- factor(ifelse(train_df_imp$.outcome == 1, "case", "ctrl"),
                         levels = c("ctrl", "case"))

cv_ctrl <- caret::trainControl(
  method          = "repeatedcv",
  number          = CV_FOLDS,
  repeats         = CV_REPEATS,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter     = FALSE
)

set.seed(20250101)
caret_fit <- caret::train(
  x          = X_train_m5_tune,
  y          = y_train_factor,
  method     = "xgbTree",
  trControl  = cv_ctrl,
  tuneGrid   = TUNE_GRID,
  metric     = "ROC",
  nthread    = 1
)

best_params <- caret_fit$bestTune
cat("✅ Best hyperparameters found:\n")
print(best_params)

# Save best hyperparameters
write_csv(best_params, file.path(OUT_DIR, "best_hyperparams.csv"))

# CV AUC summary per fold/repeat for Model 5
cv_results <- caret_fit$results
best_row   <- merge(best_params, cv_results)
cv_auc_df  <- tibble(
  Model    = m5_name,
  CV_AUC   = best_row$ROC[1],
  CV_folds = CV_FOLDS,
  CV_reps  = CV_REPEATS
)

# --------------------------
# BUILD FINAL XGB PARAMS FROM TUNED VALUES
# --------------------------

xgb_params_final <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = best_params$max_depth,
  eta              = best_params$eta,
  subsample        = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma            = best_params$gamma,
  scale_pos_weight = scale_pos_weight
)
nrounds_best <- best_params$nrounds

# --------------------------
# XGBOOST TRAIN + EVAL FUNCTION
# --------------------------

fit_xgb <- function(formula, trainData, testData, params, nrounds) {
  X_train <- model.matrix(formula, data = trainData)[, -1, drop = FALSE]
  X_test  <- model.matrix(formula, data = testData)[, -1, drop = FALSE]
  y_train <- trainData$.outcome
  y_test  <- testData$.outcome

  dtrain <- xgboost::xgb.DMatrix(data = X_train, label = y_train)
  dtest  <- xgboost::xgb.DMatrix(data = X_test,  label = y_test)

  model <- xgboost::xgb.train(
    params  = params,
    data    = dtrain,
    nrounds = nrounds,
    verbose = 0
  )

  pred    <- predict(model, newdata = dtest)
  roc_obj <- pROC::roc(response = y_test, predictor = pred,
                       levels = c(0, 1), direction = "auto", quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_auc  <- as.numeric(pROC::ci.auc(roc_obj, method = "delong"))

  list(
    model     = model,
    pred      = pred,
    y_test    = y_test,
    roc       = roc_obj,
    auc       = auc_val,
    ci_low    = ci_auc[1],
    ci_high   = ci_auc[3],
    nrounds   = nrounds
  )
}

# --------------------------
# TRAIN STEPWISE MODELS (FULL TRAINING SET, BEST PARAMS)
# --------------------------
# Note: Models 0–4 use the hyperparameters tuned on Model 5. This is a practical
# approximation — all models share the same training set and the hyperparameters
# primarily regulate regularization strength, which is dataset-level rather than
# formula-level. A separate tuning run per model would be ~6× more expensive.

cat("\n--- Training stepwise models on full training set ---\n")

model_results <- list()
pred_table <- tibble(
  ID     = if (!is.null(ID_COL) && ID_COL %in% names(test_df_imp))
             as.character(test_df_imp[[ID_COL]])
           else
             as.character(seq_len(nrow(test_df_imp))),
  y_test = test_df_imp$.outcome
)

cv_auc_rows <- list()

for (nm in names(formulas)) {
  cat("  Fitting", nm, "...\n")
  res <- fit_xgb(formulas[[nm]], train_df_imp, test_df_imp,
                 xgb_params_final, nrounds_best)
  model_results[[nm]] <- res
  pred_table[[paste0("pred_", nm)]] <- res$pred
  saveRDS(res$model, file.path(OUT_DIR, paste0(nm, "_xgb_model.rds")))

  # For Models 0–4, internal CV AUC comes from the best-tune row of Model 5 CV;
  # for Model 5, we already have it from caret_fit.
  cv_auc_rows[[nm]] <- tibble(
    Model  = nm,
    CV_AUC = if (nm == m5_name) cv_auc_df$CV_AUC else NA_real_,
    Note   = if (nm == m5_name) "from repeated-CV tuning" else "not separately tuned"
  )
}

# Save CV AUC summary
cv_auc_summary <- bind_rows(cv_auc_rows) %>%
  mutate(CV_folds = CV_FOLDS, CV_reps = CV_REPEATS)
write_csv(cv_auc_summary, file.path(OUT_DIR, "cv_auc_summary.csv"))

# --------------------------
# EXPORT MODEL 5 MATRICES FOR PYTHON SHAP
# --------------------------

X_train_m5 <- model.matrix(formulas[[m5_name]], data = train_df_imp)[, -1, drop = FALSE]
X_test_m5  <- model.matrix(formulas[[m5_name]], data = test_df_imp)[, -1, drop = FALSE]

shap_input <- list(
  X_train = as.data.frame(X_train_m5),
  X_test  = as.data.frame(X_test_m5),
  y_train = as.integer(train_df_imp$.outcome),
  y_test  = as.integer(test_df_imp$.outcome)
)
saveRDS(shap_input, file.path(OUT_DIR, "model5_shap_input.rds"))

# Parquet export so Script 04 (Python) can load without pyreadr
arrow::write_parquet(as.data.frame(X_train_m5),
                     file.path(OUT_DIR, "model5_X_train.parquet"))
arrow::write_parquet(as.data.frame(X_test_m5),
                     file.path(OUT_DIR, "model5_X_test.parquet"))
arrow::write_parquet(
  data.frame(y_train = as.integer(train_df_imp$.outcome)),
  file.path(OUT_DIR, "model5_y_train.parquet")
)
arrow::write_parquet(
  data.frame(y_test = as.integer(test_df_imp$.outcome)),
  file.path(OUT_DIR, "model5_y_test.parquet")
)

xgboost::xgb.save(model_results[[m5_name]]$model,
                  file.path(OUT_DIR, "Model5_Add_MetaboAge_xgb_model.json"))

# --------------------------
# SUMMARIZE AUC AND TESTS
# --------------------------

auc_summary <- bind_rows(lapply(names(model_results), function(nm) {
  res <- model_results[[nm]]
  tibble(
    Model      = nm,
    AUC        = res$auc,
    CI_low     = res$ci_low,
    CI_high    = res$ci_high,
    Nrounds    = res$nrounds
  )
}))
write_csv(auc_summary, file.path(OUT_DIR, "stepwise_auc_summary.csv"))

# DeLong pairwise model comparisons (consecutive steps)
model_names   <- names(model_results)
pairwise_tests <- list()
if (length(model_names) >= 2) {
  for (i in 2:length(model_names)) {
    m_prev <- model_names[i - 1]
    m_curr <- model_names[i]
    delong <- pROC::roc.test(model_results[[m_prev]]$roc,
                             model_results[[m_curr]]$roc, method = "delong")
    pairwise_tests[[paste0(m_prev, "_vs_", m_curr)]] <- tibble(
      Comparison = paste0(m_prev, " vs ", m_curr),
      p_value    = as.numeric(delong$p.value)
    )
  }
}
pairwise_df <- bind_rows(pairwise_tests)
write_csv(pairwise_df, file.path(OUT_DIR, "pairwise_auc_delong_tests.csv"))

# --------------------------
# FINAL MODEL (MODEL 5) AUPRC + BOOTSTRAP CI
# --------------------------

if (m5_name %in% names(model_results)) {
  y <- model_results[[m5_name]]$y_test
  s <- model_results[[m5_name]]$pred

  pr    <- PRROC::pr.curve(scores.class0 = s[y == 1], scores.class1 = s[y == 0], curve = FALSE)
  auprc <- as.numeric(pr$auc.integral)
  auprc_df <- tibble(Model = m5_name, AUPRC = auprc)

  if (BOOT_AUPRC) {
    set.seed(20250101)
    idx_cases <- which(y == 1)
    idx_ctrls <- which(y == 0)
    boot_vals <- replicate(BOOT_N, {
      bs_idx <- c(sample(idx_cases, length(idx_cases), replace = TRUE),
                  sample(idx_ctrls, length(idx_ctrls), replace = TRUE))
      yb  <- y[bs_idx]; sb <- s[bs_idx]
      prb <- PRROC::pr.curve(scores.class0 = sb[yb == 1], scores.class1 = sb[yb == 0], curve = FALSE)
      as.numeric(prb$auc.integral)
    })
    auprc_df <- auprc_df %>%
      mutate(
        AUPRC_CI_low  = quantile(boot_vals, 0.025, na.rm = TRUE),
        AUPRC_CI_high = quantile(boot_vals, 0.975, na.rm = TRUE),
        AUPRC_boot_N  = BOOT_N
      )
  }
  write_csv(auprc_df, file.path(OUT_DIR, "final_model_auprc.csv"))
}

# --------------------------
# EXPORT TEST SET PREDICTIONS
# --------------------------

write_csv(pred_table, file.path(OUT_DIR, "test_set_predictions_stepwise.csv"))

# --------------------------
# CONSOLE SUMMARY
# --------------------------

cat("\n=== Stepwise AUC Summary (TEST set) ===\n")
print(auc_summary)
cat("\n=== CV AUC Summary (training split) ===\n")
print(cv_auc_summary)
cat("\n=== Pairwise DeLong tests (consecutive models) ===\n")
print(pairwise_df)
if (file.exists(file.path(OUT_DIR, "final_model_auprc.csv"))) {
  cat("\n=== Final model AUPRC written to final_model_auprc.csv ===\n")
}
cat("\n✅ DONE: Multivariable XGBoost stepwise modeling completed.\n")
cat("Output folder:", OUT_DIR, "\n\n")
