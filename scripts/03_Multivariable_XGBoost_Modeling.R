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
  library(tibble)
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

# CV settings (Fix 1)
CV_FOLDS <- 5
CV_REPEATS <- 3

# Hyperparameter tuning (Fix 2) on Model 5 only; best reused for Models 0–4
TUNE_ON_MODEL <- "Model5_Add_MetaboAge"

# XGBoost defaults (will be overwritten by best tuned params for final fit)
XGB_BASE_PARAMS <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc"
)

NROUNDS_MAX <- 4000
EARLY_STOPPING <- 50

# Final-fit internal split (used ONLY within TRAIN to enable early stopping safely)
FINALFIT_VALID_PROP <- 0.15

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
# STRATIFIED 70/30 DATA SPLIT (TEST remains untouched for final reporting)
# --------------------------

train_idx <- caret::createDataPartition(df1$.outcome, p = 0.70, list = FALSE)
train_df <- df1[train_idx, , drop = FALSE]
test_df <- df1[-train_idx, , drop = FALSE]

# --------------------------
# PREPROCESSING: TRAIN MEDIAN IMPUTATION (fit on TRAIN only)
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

preproc <- caret::preProcess(train_df[, num_candidates, drop = FALSE], method = c("medianImpute"))
train_df_imp <- train_df
test_df_imp <- test_df
train_df_imp[, num_candidates] <- predict(preproc, train_df[, num_candidates, drop = FALSE])
test_df_imp[, num_candidates] <- predict(preproc, test_df[, num_candidates, drop = FALSE])

# Harmonize factors
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

if (!TUNE_ON_MODEL %in% names(formulas)) {
  stop("❌ TUNE_ON_MODEL not found in formulas: ", TUNE_ON_MODEL)
}

# --------------------------
# UTILITIES
# --------------------------

make_model_matrix <- function(formula, data) {
  model.matrix(formula, data = data)[, -1, drop = FALSE]
}

calc_scale_pos_weight <- function(y01) {
  n_pos <- sum(y01 == 1, na.rm = TRUE)
  n_neg <- sum(y01 == 0, na.rm = TRUE)
  if (n_pos == 0) return(1)
  n_neg / n_pos
}

auc_from_preds <- function(y01, pred) {
  roc_obj <- pROC::roc(response = y01, predictor = pred, levels = c(0, 1), direction = "auto", quiet = TRUE)
  as.numeric(pROC::auc(roc_obj))
}

# --------------------------
# FIX 1: REPEATED STRATIFIED K-FOLD CV (TRAIN ONLY)
#   - early stopping uses each fold's internal watchlist (train vs val)
#   - test set is NOT used here
# --------------------------

make_repeated_folds <- function(y, k = 5, repeats = 3, seed = 20250101) {
  set.seed(seed)
  folds <- caret::createMultiFolds(y = y, k = k, times = repeats)
  # folds: named list of training indices; validation = setdiff
  folds
}

cv_eval_one_paramset <- function(X, y, folds_train_idx, params_base, nrounds_max, early_stop) {
  aucs <- numeric(length(folds_train_idx))
  best_iters <- integer(length(folds_train_idx))

  for (i in seq_along(folds_train_idx)) {
    tr_idx <- folds_train_idx[[i]]
    va_idx <- setdiff(seq_len(nrow(X)), tr_idx)

    X_tr <- X[tr_idx, , drop = FALSE]
    y_tr <- y[tr_idx]
    X_va <- X[va_idx, , drop = FALSE]
    y_va <- y[va_idx]

    spw <- calc_scale_pos_weight(y_tr)

    dtr <- xgboost::xgb.DMatrix(data = X_tr, label = y_tr)
    dva <- xgboost::xgb.DMatrix(data = X_va, label = y_va)

    params <- modifyList(params_base, list(scale_pos_weight = spw))

    model <- xgboost::xgb.train(
      params = params,
      data = dtr,
      nrounds = nrounds_max,
      watchlist = list(train = dtr, valid = dva),
      early_stopping_rounds = early_stop,
      verbose = 0
    )

    pred <- predict(model, newdata = dva)
    aucs[i] <- auc_from_preds(y_va, pred)
    best_iters[i] <- model$best_iteration
  }

  list(
    auc_mean = mean(aucs),
    auc_sd = sd(aucs),
    best_iter_mean = mean(best_iters),
    best_iter_median = as.integer(stats::median(best_iters)),
    aucs = aucs,
    best_iters = best_iters
  )
}

# --------------------------
# FIX 2: HYPERPARAMETER GRID TUNING (Model 5 only, TRAIN ONLY)
# --------------------------

# A compact but real grid; expand if you want (kept moderate for runtime).
tune_grid <- tidyr::expand_grid(
  max_depth = c(3, 4, 6),
  eta = c(0.01, 0.05, 0.1),
  subsample = c(0.8, 0.9, 1.0),
  colsample_bytree = c(0.8, 0.9, 1.0),
  min_child_weight = c(1, 5)
) %>%
  dplyr::mutate(grid_id = dplyr::row_number())

# Prepare Model 5 design matrix ONCE (TRAIN only)
X_train_m5_all <- make_model_matrix(formulas[[TUNE_ON_MODEL]], train_df_imp)
y_train_all <- train_df_imp$.outcome

folds_train_idx <- make_repeated_folds(y_train_all, k = CV_FOLDS, repeats = CV_REPEATS, seed = 20250101)

message("\n🔧 Hyperparameter tuning on ", TUNE_ON_MODEL, " using ",
        CV_FOLDS, "-fold × ", CV_REPEATS, "-repeat CV (TRAIN only). Grid size = ", nrow(tune_grid), "\n")

tune_results <- vector("list", nrow(tune_grid))
for (g in seq_len(nrow(tune_grid))) {
  gp <- tune_grid[g, ]
  params_g <- modifyList(XGB_BASE_PARAMS, list(
    max_depth = gp$max_depth,
    eta = gp$eta,
    subsample = gp$subsample,
    colsample_bytree = gp$colsample_bytree,
    min_child_weight = gp$min_child_weight
  ))

  cv_res <- cv_eval_one_paramset(
    X = X_train_m5_all,
    y = y_train_all,
    folds_train_idx = folds_train_idx,
    params_base = params_g,
    nrounds_max = NROUNDS_MAX,
    early_stop = EARLY_STOPPING
  )

  tune_results[[g]] <- tibble(
    grid_id = gp$grid_id,
    max_depth = gp$max_depth,
    eta = gp$eta,
    subsample = gp$subsample,
    colsample_bytree = gp$colsample_bytree,
    min_child_weight = gp$min_child_weight,
    cv_auc_mean = cv_res$auc_mean,
    cv_auc_sd = cv_res$auc_sd,
    best_iter_median = cv_res$best_iter_median,
    best_iter_mean = cv_res$best_iter_mean
  )
}

tune_df <- dplyr::bind_rows(tune_results) %>%
  dplyr::arrange(dplyr::desc(cv_auc_mean), cv_auc_sd)

best_row <- tune_df[1, , drop = FALSE]
write_csv(tune_df, file.path(OUT_DIR, "hyperparam_grid_results_model5.csv"))
write_csv(best_row, file.path(OUT_DIR, "best_hyperparams.csv"))

message("✅ Best hyperparams selected (by highest CV mean AUC):\n",
        paste(capture.output(print(best_row)), collapse = "\n"), "\n")

BEST_PARAMS <- modifyList(XGB_BASE_PARAMS, list(
  max_depth = best_row$max_depth,
  eta = best_row$eta,
  subsample = best_row$subsample,
  colsample_bytree = best_row$colsample_bytree,
  min_child_weight = best_row$min_child_weight
))
BEST_NROUNDS <- as.integer(best_row$best_iter_median)

# --------------------------
# CV AUC SUMMARY FOR ALL MODELS (using BEST_PARAMS, TRAIN ONLY)  [Fix 1]
# --------------------------

message("\n📌 Computing CV AUC summary for all stepwise models using best hyperparams from Model 5.\n",
        "NOTE: scale_pos_weight is recomputed within each fold.\n")

cv_summary_list <- list()
for (nm in names(formulas)) {
  X_tr <- make_model_matrix(formulas[[nm]], train_df_imp)

  cv_res <- cv_eval_one_paramset(
    X = X_tr,
    y = y_train_all,
    folds_train_idx = folds_train_idx,
    params_base = BEST_PARAMS,
    nrounds_max = NROUNDS_MAX,
    early_stop = EARLY_STOPPING
  )

  cv_summary_list[[nm]] <- tibble(
    Model = nm,
    CV_folds = CV_FOLDS,
    CV_repeats = CV_REPEATS,
    CV_AUC_mean = cv_res$auc_mean,
    CV_AUC_sd = cv_res$auc_sd,
    CV_best_iter_median = cv_res$best_iter_median,
    CV_best_iter_mean = cv_res$best_iter_mean
  )
}

cv_auc_summary <- bind_rows(cv_summary_list)
write_csv(cv_auc_summary, file.path(OUT_DIR, "cv_auc_summary.csv"))

# --------------------------
# FINAL TRAINING ON FULL TRAIN (with internal valid split for early stopping)
#   - This is the only place we fit "final" models
#   - TEST remains untouched until evaluation
# --------------------------

final_fit_xgb <- function(formula, trainData, params_base, nrounds_max, early_stop, valid_prop = 0.15, seed = 20250101) {
  set.seed(seed)
  X_all <- make_model_matrix(formula, trainData)
  y_all <- trainData$.outcome

  # Stratified internal split for early stopping (within TRAIN only)
  tr_idx <- caret::createDataPartition(y_all, p = 1 - valid_prop, list = FALSE)
  va_idx <- setdiff(seq_len(nrow(X_all)), tr_idx)

  X_tr <- X_all[tr_idx, , drop = FALSE]; y_tr <- y_all[tr_idx]
  X_va <- X_all[va_idx, , drop = FALSE]; y_va <- y_all[va_idx]

  spw <- calc_scale_pos_weight(y_tr)
  params <- modifyList(params_base, list(scale_pos_weight = spw))

  dtr <- xgboost::xgb.DMatrix(data = X_tr, label = y_tr)
  dva <- xgboost::xgb.DMatrix(data = X_va, label = y_va)

  model <- xgboost::xgb.train(
    params = params,
    data = dtr,
    nrounds = nrounds_max,
    watchlist = list(train = dtr, valid = dva),
    early_stopping_rounds = early_stop,
    verbose = 0
  )

  list(
    model = model,
    best_iteration = model$best_iteration
  )
}

eval_on_test <- function(model, formula, testData) {
  X_test <- make_model_matrix(formula, testData)
  y_test <- testData$.outcome
  dtest <- xgboost::xgb.DMatrix(data = X_test, label = y_test)

  pred <- predict(model, newdata = dtest)
  roc_obj <- pROC::roc(response = y_test, predictor = pred, levels = c(0, 1), direction = "auto", quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_auc <- as.numeric(pROC::ci.auc(roc_obj, method = "delong"))

  list(
    pred = pred,
    y_test = y_test,
    roc = roc_obj,
    auc = auc_val,
    ci_low = ci_auc[1],
    ci_high = ci_auc[3]
  )
}

# --------------------------
# FIT FINAL STEPWISE MODELS + TEST EVALUATION (TEST is used ONLY here)
# --------------------------

model_results <- list()
pred_table <- tibble(
  ID = if (!is.null(ID_COL) && ID_COL %in% names(test_df_imp)) as.character(test_df_imp[[ID_COL]]) else as.character(seq_along(test_df_imp$.outcome)),
  y_test = test_df_imp$.outcome
)

for (nm in names(formulas)) {
  message("🧠 Final fitting: ", nm, " (train-only early stopping, then evaluate on test)")

  final_fit <- final_fit_xgb(
    formula = formulas[[nm]],
    trainData = train_df_imp,
    params_base = BEST_PARAMS,
    nrounds_max = NROUNDS_MAX,
    early_stop = EARLY_STOPPING,
    valid_prop = FINALFIT_VALID_PROP,
    seed = 20250101
  )

  test_eval <- eval_on_test(final_fit$model, formulas[[nm]], test_df_imp)

  model_results[[nm]] <- list(
    model = final_fit$model,
    pred = test_eval$pred,
    y_test = test_eval$y_test,
    roc = test_eval$roc,
    auc = test_eval$auc,
    ci_low = test_eval$ci_low,
    ci_high = test_eval$ci_high,
    best_nrounds = final_fit$best_iteration
  )

  pred_table[[paste0("pred_", nm)]] <- test_eval$pred
  saveRDS(final_fit$model, file.path(OUT_DIR, paste0(nm, "_xgb_model.rds")))
}

# ---- Export Model 5 matrices/labels for Python SHAP ----
m5_name <- "Model5_Add_MetaboAge"
X_train_m5 <- make_model_matrix(formulas[[m5_name]], train_df_imp)
X_test_m5  <- make_model_matrix(formulas[[m5_name]], test_df_imp)

shap_input <- list(
  X_train = as.data.frame(X_train_m5),
  X_test  = as.data.frame(X_test_m5),
  y_train = as.integer(train_df_imp$.outcome),
  y_test  = as.integer(test_df_imp$.outcome)
)
saveRDS(shap_input, file.path(OUT_DIR, "model5_shap_input.rds"))
xgboost::xgb.save(model_results[[m5_name]]$model, file.path(OUT_DIR, "Model5_Add_MetaboAge_xgb_model.json"))

# --------------------------
# SUMMARIZE TEST AUC AND TEST-SET DeLong TESTS (FINAL REPORTING)
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

# DeLong pairwise model comparisons (consecutive) on TEST
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
# FINAL MODEL (Model 5) AUPRC (+ BOOTSTRAP) ON TEST ONLY
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

# --------------------------
# CONSOLE OUTPUT
# --------------------------

cat("\n=== CV AUC Summary (TRAIN only; 5-fold × 3-repeat) ===\n")
print(cv_auc_summary)

cat("\n=== Best hyperparameters (tuned on Model 5; TRAIN only) ===\n")
print(best_row)

cat("\n=== Stepwise AUC Summary (TEST set; final reporting) ===\n")
print(auc_summary)

cat("\n=== Pairwise DeLong tests (consecutive models; TEST set) ===\n")
print(pairwise_df)

if (file.exists(file.path(OUT_DIR, "final_model_auprc.csv"))) {
  cat("\n=== Final model AUPRC written to final_model_auprc.csv (TEST set) ===\n")
}

cat("\n✅ DONE: Multivariable XGBoost stepwise modeling completed.\n")
cat("Output folder:", OUT_DIR, "\n\n")
