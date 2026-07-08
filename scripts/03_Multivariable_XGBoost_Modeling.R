# ==============================================================================
# 03_Multivariable_XGBoost_Modeling.R
#
# Integrative Prediction of Alzheimer’s Disease and Related Dementias Using
# Multi-Omics Aging Clocks and Genetic Data
#
# Purpose:
# - Fit stepwise XGBoost models for incident ADRD prediction.
# - Use stratified 70/30 train-test split.
# - Fit preprocessing parameters in training data only.
# - Tune XGBoost hyperparameters using repeated cross-validation inside training.
# - Evaluate final models only once in the held-out test set.
# - Report ROC-AUC and AUPRC for all stepwise models.
# - Export prediction files for SHAP, calibration, DCA, threshold analyses, and
#   downstream sensitivity analyses.
#
# Main manuscript output:
# - Table 2: stepwise ROC-AUC and AUPRC performance
#
# Supplementary output:
# - Figure A.5: precision-recall curves
# - model5_shap_input.rds for script 04
#
# Author: Shayan Mostafaei
# Updated for revision: 2026-06-08
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(caret)
  library(xgboost)
  library(pROC)
  library(PRROC)
  library(ggplot2)
})

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------

set.seed(20250101)

INPUT_RDS <- "data/biomarkers_complete.rds"
OUT_DIR <- file.path("results", "04_stepwise_xgboost")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

ID_COL <- "f.eid"
OUTCOME_COL <- "Dementia_status"

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
PC_COLS <- paste0("PC", 1:10)

CV_FOLDS <- 5
CV_REPEATS <- 3
NROUNDS_MAX <- 4000
EARLY_STOPPING <- 50
BOOT_N <- 2000

TUNE_ON_MODEL <- "Model5_Add_MetaboAge"

XGB_BASE_PARAMS <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc"
)

# ------------------------------------------------------------------------------
# LOAD AND VALIDATE DATA
# ------------------------------------------------------------------------------

if (!file.exists(INPUT_RDS)) {
  stop("Input RDS not found: ", INPUT_RDS)
}

df0 <- readRDS(INPUT_RDS)

required_cols <- c(
  OUTCOME_COL, AGE_COL, SEX_COL, BMI_COL, SMOKE_COL, ALC_COL, EDU_COL,
  PRS_COL, PHENO_COL, FI_COL, TL_COL, PROTAGE_COL, METABOAGE_COL
)

missing_cols <- setdiff(required_cols, names(df0))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

if (!ID_COL %in% names(df0)) {
  df0[[ID_COL]] <- seq_len(nrow(df0))
}

PC_COLS <- PC_COLS[PC_COLS %in% names(df0)]

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

coerce_binary01 <- function(x) {
  if (is.logical(x)) return(as.numeric(x))
  if (is.factor(x)) x <- as.character(x)
  out <- suppressWarnings(as.numeric(x))
  ifelse(out == 1, 1, ifelse(out == 0, 0, NA_real_))
}

mode_value <- function(x) {
  x <- as.factor(x)
  tab <- table(x, useNA = "no")
  if (length(tab) == 0) return(NA_character_)
  names(tab)[which.max(tab)]
}

build_formula <- function(rhs_terms) {
  as.formula(paste(".outcome ~", paste(rhs_terms, collapse = " + ")))
}

make_model_matrix <- function(formula, data) {
  mm <- model.matrix(formula, data = data)
  mm[, colnames(mm) != "(Intercept)", drop = FALSE]
}

calc_scale_pos_weight <- function(y01) {
  n_pos <- sum(y01 == 1, na.rm = TRUE)
  n_neg <- sum(y01 == 0, na.rm = TRUE)
  if (n_pos == 0) return(1)
  n_neg / n_pos
}

auc_from_preds <- function(y01, pred) {
  roc_obj <- pROC::roc(
    response = y01,
    predictor = pred,
    levels = c(0, 1),
    direction = "auto",
    quiet = TRUE
  )
  as.numeric(pROC::auc(roc_obj))
}

calc_auprc <- function(y, pred) {
  if (length(unique(y)) < 2) return(NA_real_)
  pr <- PRROC::pr.curve(
    scores.class0 = pred[y == 1],
    scores.class1 = pred[y == 0],
    curve = FALSE
  )
  as.numeric(pr$auc.integral)
}

calc_auprc_boot <- function(y, pred, boot_n = 2000, seed = 20250101) {
  auprc <- calc_auprc(y, pred)

  idx_cases <- which(y == 1)
  idx_ctrls <- which(y == 0)

  if (length(idx_cases) < 2 || length(idx_ctrls) < 2) {
    return(tibble(
      AUPRC = auprc,
      AUPRC_CI_low = NA_real_,
      AUPRC_CI_high = NA_real_,
      AUPRC_boot_N = 0
    ))
  }

  set.seed(seed)

  boot_vals <- replicate(boot_n, {
    bs_idx <- c(
      sample(idx_cases, length(idx_cases), replace = TRUE),
      sample(idx_ctrls, length(idx_ctrls), replace = TRUE)
    )
    yb <- y[bs_idx]
    pb <- pred[bs_idx]
    calc_auprc(yb, pb)
  })

  tibble(
    AUPRC = auprc,
    AUPRC_CI_low = as.numeric(quantile(boot_vals, 0.025, na.rm = TRUE)),
    AUPRC_CI_high = as.numeric(quantile(boot_vals, 0.975, na.rm = TRUE)),
    AUPRC_boot_N = boot_n
  )
}

make_repeated_folds <- function(y, k = 5, repeats = 3, seed = 20250101) {
  set.seed(seed)
  caret::createMultiFolds(y = factor(y), k = k, times = repeats)
}

cv_eval_one_paramset <- function(X, y, folds_train_idx, params_base,
                                 nrounds_max = 4000, early_stop = 50) {
  aucs <- numeric(length(folds_train_idx))
  best_iters <- integer(length(folds_train_idx))

  for (i in seq_along(folds_train_idx)) {
    tr_idx <- folds_train_idx[[i]]
    va_idx <- setdiff(seq_len(nrow(X)), tr_idx)

    X_tr <- X[tr_idx, , drop = FALSE]
    y_tr <- y[tr_idx]
    X_va <- X[va_idx, , drop = FALSE]
    y_va <- y[va_idx]

    params <- modifyList(
      params_base,
      list(scale_pos_weight = calc_scale_pos_weight(y_tr))
    )

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

    pred <- predict(model, newdata = dva)
    aucs[i] <- auc_from_preds(y_va, pred)
    best_iters[i] <- model$best_iteration
  }

  list(
    auc_mean = mean(aucs, na.rm = TRUE),
    auc_sd = sd(aucs, na.rm = TRUE),
    best_iter_median = as.integer(stats::median(best_iters, na.rm = TRUE)),
    best_iter_mean = mean(best_iters, na.rm = TRUE),
    aucs = aucs,
    best_iters = best_iters
  )
}

fit_xgb_full_train <- function(formula, train_data, params_base, nrounds) {
  X <- make_model_matrix(formula, train_data)
  y <- train_data$.outcome

  params <- modifyList(
    params_base,
    list(scale_pos_weight = calc_scale_pos_weight(y))
  )

  dtrain <- xgboost::xgb.DMatrix(data = X, label = y)

  xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
}

eval_on_test <- function(model, formula, test_data) {
  X_test <- make_model_matrix(formula, test_data)
  y_test <- test_data$.outcome
  dtest <- xgboost::xgb.DMatrix(data = X_test, label = y_test)

  pred <- predict(model, newdata = dtest)

  roc_obj <- pROC::roc(
    response = y_test,
    predictor = pred,
    levels = c(0, 1),
    direction = "auto",
    quiet = TRUE
  )

  ci_auc <- as.numeric(pROC::ci.auc(roc_obj, method = "delong"))

  list(
    pred = pred,
    y_test = y_test,
    roc = roc_obj,
    auc = as.numeric(pROC::auc(roc_obj)),
    ci_low = ci_auc[1],
    ci_high = ci_auc[3]
  )
}

# ------------------------------------------------------------------------------
# PREPARE OUTCOME AND STRATIFIED TRAIN-TEST SPLIT
# ------------------------------------------------------------------------------

df1 <- df0 %>%
  mutate(
    .id = as.character(.data[[ID_COL]]),
    .outcome = coerce_binary01(.data[[OUTCOME_COL]]),
    sex_std = as.factor(.data[[SEX_COL]]),
    education_std = as.factor(.data[[EDU_COL]]),
    smoking_std = as.factor(.data[[SMOKE_COL]]),
    alcohol_std = as.factor(.data[[ALC_COL]])
  ) %>%
  filter(!is.na(.outcome))

set.seed(20250101)
train_idx <- caret::createDataPartition(df1$.outcome, p = 0.70, list = FALSE)

train_df <- df1[train_idx, , drop = FALSE]
test_df <- df1[-train_idx, , drop = FALSE]

split_summary <- tibble(
  Split = c("Training", "Held-out test"),
  N = c(nrow(train_df), nrow(test_df)),
  ADRD_cases = c(sum(train_df$.outcome == 1), sum(test_df$.outcome == 1)),
  Non_cases = c(sum(train_df$.outcome == 0), sum(test_df$.outcome == 0))
)

write_csv(split_summary, file.path(OUT_DIR, "train_test_split_summary.csv"))
write_csv(train_df %>% select(.id), file.path(OUT_DIR, "train_ids.csv"))
write_csv(test_df %>% select(.id), file.path(OUT_DIR, "test_ids.csv"))

cat("\n=== Train-test split summary ===\n")
print(split_summary)

# ------------------------------------------------------------------------------
# TRAIN-ONLY PREPROCESSING
# ------------------------------------------------------------------------------

numeric_cols <- c(
  AGE_COL, BMI_COL, PRS_COL, PHENO_COL, FI_COL, TL_COL,
  PROTAGE_COL, METABOAGE_COL, PC_COLS
)
numeric_cols <- numeric_cols[numeric_cols %in% names(df1)]

for (v in numeric_cols) {
  train_df[[v]] <- suppressWarnings(as.numeric(train_df[[v]]))
  test_df[[v]] <- suppressWarnings(as.numeric(test_df[[v]]))
}

preproc <- caret::preProcess(
  train_df[, numeric_cols, drop = FALSE],
  method = c("medianImpute")
)

train_df_imp <- train_df
test_df_imp <- test_df

train_df_imp[, numeric_cols] <- predict(preproc, train_df[, numeric_cols, drop = FALSE])
test_df_imp[, numeric_cols] <- predict(preproc, test_df[, numeric_cols, drop = FALSE])

factor_map <- c(
  sex = "sex_std",
  education = "education_std",
  smoking = "smoking_std",
  alcohol_intake_frequency = "alcohol_std"
)

for (new_name in names(factor_map)) {
  old_name <- factor_map[[new_name]]

  train_df_imp[[new_name]] <- as.factor(train_df_imp[[old_name]])
  test_df_imp[[new_name]] <- as.factor(test_df_imp[[old_name]])

  mode_train <- mode_value(train_df_imp[[new_name]])
  train_df_imp[[new_name]][is.na(train_df_imp[[new_name]])] <- mode_train
  train_df_imp[[new_name]] <- droplevels(train_df_imp[[new_name]])

  test_df_imp[[new_name]][is.na(test_df_imp[[new_name]])] <- mode_train
  test_df_imp[[new_name]] <- factor(test_df_imp[[new_name]], levels = levels(train_df_imp[[new_name]]))
}

saveRDS(
  list(
    preproc_numeric = preproc,
    numeric_cols = numeric_cols,
    train_ids = train_df_imp$.id,
    test_ids = test_df_imp$.id,
    split_summary = split_summary
  ),
  file.path(OUT_DIR, "xgboost_preprocess_and_split_objects.rds")
)

# ------------------------------------------------------------------------------
# STEPWISE MODEL FORMULAS
# ------------------------------------------------------------------------------

pc_terms <- PC_COLS

rhs_base <- c(AGE_COL, "sex")
rhs_lifestyle <- c(rhs_base, "smoking", "alcohol_intake_frequency", BMI_COL, "education")
rhs_prs <- c(rhs_lifestyle, PRS_COL, pc_terms)
rhs_ba <- c(rhs_prs, PHENO_COL, FI_COL, TL_COL)
rhs_prot <- c(rhs_ba, PROTAGE_COL)
rhs_full <- c(rhs_prot, METABOAGE_COL)

formulas <- list(
  Model0_Base = build_formula(rhs_base),
  Model1_Lifestyle = build_formula(rhs_lifestyle),
  Model2_Add_PRS = build_formula(rhs_prs),
  Model3_Add_BA = build_formula(rhs_ba),
  Model4_Add_ProtAge = build_formula(rhs_prot),
  Model5_Add_MetaboAge = build_formula(rhs_full)
)

if (!TUNE_ON_MODEL %in% names(formulas)) {
  stop("TUNE_ON_MODEL not found in formulas: ", TUNE_ON_MODEL)
}

model_labels <- tibble(
  Model = names(formulas),
  Model_label = c(
    "Age + sex",
    "Age + sex + lifestyle",
    "Age + sex + lifestyle + PRS",
    "Age + sex + lifestyle + PRS + PhenoAge + FI + TL",
    "Age + sex + lifestyle + PRS + PhenoAge + FI + TL + ProtAge",
    "Age + sex + lifestyle + PRS + PhenoAge + FI + TL + ProtAge + MetaboAge"
  )
)

write_csv(model_labels, file.path(OUT_DIR, "stepwise_model_labels.csv"))

# ------------------------------------------------------------------------------
# HYPERPARAMETER TUNING ON MODEL 5 INSIDE TRAINING DATA ONLY
# ------------------------------------------------------------------------------

tune_grid <- tidyr::expand_grid(
  max_depth = c(3, 4, 6),
  eta = c(0.01, 0.05, 0.10),
  subsample = c(0.8, 0.9, 1.0),
  colsample_bytree = c(0.8, 0.9, 1.0),
  min_child_weight = c(1, 5)
) %>%
  mutate(grid_id = row_number())

X_train_m5_all <- make_model_matrix(formulas[[TUNE_ON_MODEL]], train_df_imp)
y_train_all <- train_df_imp$.outcome

folds_train_idx <- make_repeated_folds(
  y = y_train_all,
  k = CV_FOLDS,
  repeats = CV_REPEATS,
  seed = 20250101
)

message(
  "\nHyperparameter tuning on ", TUNE_ON_MODEL,
  " using ", CV_FOLDS, "-fold x ", CV_REPEATS,
  "-repeat CV inside training data only."
)

tune_results <- vector("list", nrow(tune_grid))

for (g in seq_len(nrow(tune_grid))) {
  gp <- tune_grid[g, ]

  params_g <- modifyList(
    XGB_BASE_PARAMS,
    list(
      max_depth = gp$max_depth,
      eta = gp$eta,
      subsample = gp$subsample,
      colsample_bytree = gp$colsample_bytree,
      min_child_weight = gp$min_child_weight
    )
  )

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

tune_df <- bind_rows(tune_results) %>%
  arrange(desc(cv_auc_mean), cv_auc_sd)

best_row <- tune_df %>% slice(1)

write_csv(tune_df, file.path(OUT_DIR, "hyperparam_grid_results_model5.csv"))
write_csv(best_row, file.path(OUT_DIR, "best_hyperparams.csv"))

BEST_PARAMS <- modifyList(
  XGB_BASE_PARAMS,
  list(
    max_depth = best_row$max_depth,
    eta = best_row$eta,
    subsample = best_row$subsample,
    colsample_bytree = best_row$colsample_bytree,
    min_child_weight = best_row$min_child_weight
  )
)

BEST_NROUNDS <- max(as.integer(best_row$best_iter_median), 10)

cat("\n=== Best hyperparameters selected using training CV ===\n")
print(best_row)

# ------------------------------------------------------------------------------
# CV SUMMARY FOR ALL STEPWISE MODELS, TRAINING DATA ONLY
# ------------------------------------------------------------------------------

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

cv_auc_summary <- bind_rows(cv_summary_list) %>%
  left_join(model_labels, by = "Model")

write_csv(cv_auc_summary, file.path(OUT_DIR, "cv_auc_summary_train_only.csv"))

# ------------------------------------------------------------------------------
# FINAL MODEL FITTING ON FULL TRAINING DATA AND TEST EVALUATION
# ------------------------------------------------------------------------------

model_results <- list()

pred_table <- tibble(
  ID = test_df_imp$.id,
  y_test = test_df_imp$.outcome
)

for (nm in names(formulas)) {
  message("Final fitting on full training set: ", nm)

  fitted_model <- fit_xgb_full_train(
    formula = formulas[[nm]],
    train_data = train_df_imp,
    params_base = BEST_PARAMS,
    nrounds = BEST_NROUNDS
  )

  test_eval <- eval_on_test(fitted_model, formulas[[nm]], test_df_imp)

  model_results[[nm]] <- list(
    model = fitted_model,
    pred = test_eval$pred,
    y_test = test_eval$y_test,
    roc = test_eval$roc,
    auc = test_eval$auc,
    ci_low = test_eval$ci_low,
    ci_high = test_eval$ci_high,
    nrounds = BEST_NROUNDS
  )

  pred_table[[paste0("pred_", nm)]] <- test_eval$pred

  saveRDS(
    fitted_model,
    file.path(OUT_DIR, paste0(nm, "_xgb_model.rds"))
  )
}

write_csv(pred_table, file.path(OUT_DIR, "test_set_predictions_stepwise.csv"))

# ------------------------------------------------------------------------------
# AUC SUMMARY AND DELONG TESTS
# ------------------------------------------------------------------------------

auc_summary <- bind_rows(lapply(names(model_results), function(nm) {
  res <- model_results[[nm]]

  tibble(
    Model = nm,
    AUC = res$auc,
    AUC_CI_low = res$ci_low,
    AUC_CI_high = res$ci_high,
    Nrounds = res$nrounds
  )
})) %>%
  left_join(model_labels, by = "Model") %>%
  select(Model, Model_label, everything())

write_csv(auc_summary, file.path(OUT_DIR, "stepwise_auc_summary.csv"))

model_names <- names(model_results)
pairwise_tests <- list()

if (length(model_names) >= 2) {
  for (i in 2:length(model_names)) {
    m_prev <- model_names[i - 1]
    m_curr <- model_names[i]

    test <- pROC::roc.test(
      model_results[[m_prev]]$roc,
      model_results[[m_curr]]$roc,
      method = "delong"
    )

    pairwise_tests[[paste0(m_prev, "_vs_", m_curr)]] <- tibble(
      Comparison = paste0(m_prev, " vs ", m_curr),
      Previous_model = m_prev,
      Current_model = m_curr,
      p_value = as.numeric(test$p.value)
    )
  }
}

pairwise_df <- bind_rows(pairwise_tests)
write_csv(pairwise_df, file.path(OUT_DIR, "pairwise_auc_delong_tests.csv"))

# ------------------------------------------------------------------------------
# AUPRC FOR ALL STEPWISE MODELS
# ------------------------------------------------------------------------------

auprc_summary <- bind_rows(lapply(names(model_results), function(nm) {
  res <- model_results[[nm]]

  calc_auprc_boot(
    y = res$y_test,
    pred = res$pred,
    boot_n = BOOT_N,
    seed = 20250101
  ) %>%
    mutate(Model = nm, .before = 1)
})) %>%
  left_join(model_labels, by = "Model") %>%
  select(Model, Model_label, everything())

write_csv(
  auprc_summary,
  file.path(OUT_DIR, "stepwise_auprc_summary_all_models.csv")
)

# ------------------------------------------------------------------------------
# MANUSCRIPT TABLE 2 EXPORT
# ------------------------------------------------------------------------------

table2 <- auc_summary %>%
  left_join(
    auprc_summary %>%
      select(Model, AUPRC, AUPRC_CI_low, AUPRC_CI_high, AUPRC_boot_N),
    by = "Model"
  ) %>%
  left_join(
    pairwise_df %>%
      select(Model = Current_model, P_value_vs_previous = p_value),
    by = "Model"
  ) %>%
  mutate(
    AUC_95CI = sprintf("%.3f (%.3f-%.3f)", AUC, AUC_CI_low, AUC_CI_high),
    AUPRC_95CI = sprintf("%.3f (%.3f-%.3f)", AUPRC, AUPRC_CI_low, AUPRC_CI_high)
  )

write_csv(table2, file.path(OUT_DIR, "Table2_stepwise_auc_auprc.csv"))

# ------------------------------------------------------------------------------
# PRECISION-RECALL CURVES: FIGURE A.5
# ------------------------------------------------------------------------------

pr_curve_rows <- list()

for (nm in names(model_results)) {
  y <- model_results[[nm]]$y_test
  p <- model_results[[nm]]$pred

  pr <- PRROC::pr.curve(
    scores.class0 = p[y == 1],
    scores.class1 = p[y == 0],
    curve = TRUE
  )

  pr_curve_rows[[nm]] <- as_tibble(pr$curve) %>%
    setNames(c("Recall", "Precision", "Threshold")) %>%
    mutate(Model = nm)
}

pr_curve_df <- bind_rows(pr_curve_rows) %>%
  left_join(model_labels, by = "Model")

write_csv(
  pr_curve_df,
  file.path(OUT_DIR, "precision_recall_curve_points_all_models.csv")
)

p_pr <- ggplot(pr_curve_df, aes(x = Recall, y = Precision, group = Model_label)) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ Model_label, ncol = 2) +
  labs(
    title = "Precision-recall curves for stepwise ADRD prediction models",
    x = "Recall",
    y = "Precision"
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    strip.text = element_text(size = 9)
  )

ggsave(
  filename = file.path(OUT_DIR, "FigureA5_precision_recall_curves.png"),
  plot = p_pr,
  width = 9,
  height = 8,
  dpi = 500
)

# ------------------------------------------------------------------------------
# EXPORT MODEL 5 INPUTS FOR SHAP
# ------------------------------------------------------------------------------

m5_name <- "Model5_Add_MetaboAge"

X_train_m5 <- make_model_matrix(formulas[[m5_name]], train_df_imp)
X_test_m5 <- make_model_matrix(formulas[[m5_name]], test_df_imp)

shap_input <- list(
  X_train = as.data.frame(X_train_m5),
  X_test = as.data.frame(X_test_m5),
  y_train = as.integer(train_df_imp$.outcome),
  y_test = as.integer(test_df_imp$.outcome)
)

saveRDS(shap_input, file.path(OUT_DIR, "model5_shap_input.rds"))

xgboost::xgb.save(
  model_results[[m5_name]]$model,
  file.path(OUT_DIR, "Model5_Add_MetaboAge_xgb_model.json")
)

saveRDS(
  list(
    formulas = formulas,
    model_labels = model_labels,
    best_params = BEST_PARAMS,
    best_nrounds = BEST_NROUNDS,
    model_results = model_results,
    train_ids = train_df_imp$.id,
    test_ids = test_df_imp$.id,
    split_summary = split_summary
  ),
  file.path(OUT_DIR, "stepwise_xgboost_results_bundle.rds")
)

# ------------------------------------------------------------------------------
# CONSOLE OUTPUT
# ------------------------------------------------------------------------------

cat("\n=== Table 2: held-out test performance ===\n")
print(table2)

cat("\n=== Pairwise DeLong tests ===\n")
print(pairwise_df)

cat("\nDONE: Stepwise XGBoost modeling completed.\n")
cat("Outputs written to:", OUT_DIR, "\n\n")
