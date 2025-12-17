# ==================================================================================
# 06_Sensitivity_Analyses.R
#
# Inputs required (produced by previous scripts):
#   - results/survival_competing_risk/finegray_model_train.rds   (Script 05)
#   - results/survival_competing_risk/test_predicted_risks_5y_9y.csv (Script 05) [we will recompute 3y/7y/9y too]
#   - results/multivariable_xgboost/test_set_predictions_stepwise.csv (Script 03)
#   - data/biomarkers_complete.rds (for subgroup variables + time-to-event columns)
#
# Outputs:
#   results/sensitivity/
#     - shr_by_horizon_test.csv
#     - cif_horizon_plots.png
#     - early_event_exclusion_shr_cif.png
#     - auc_subgroups_from_holdout_preds.csv
#     - auc_sensitivity_forestplot.png
# ==================================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(riskRegression)
  library(cmprsk)
  library(ggplot2)
  library(scales)
  library(pROC)
  library(patchwork)
})

# --------------------------
# USER SETTINGS
# --------------------------
set.seed(20250101)

INPUT_RDS <- "data/biomarkers_complete.rds"

# Outputs from Script 05
FG_MODEL_RDS <- "results/survival_competing_risk/finegray_model_train.rds"

# Outputs from Script 03
XGB_PRED_CSV <- "results/multivariable_xgboost/test_set_predictions_stepwise.csv"

OUT_DIR <- "results/sensitivity"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Column name settings
ID_COL <- "f.eid"
OUTCOME_COL <- "Dementia_status"
SEX_COL <- "sex"

# Survival columns (from Script 05)
T_DEM_COL <- "Time_to_Dementia"
DEATH_COL <- "death_status"
T_DEATH_COL <- "time_to_death"

# Optional subgroup columns for AUC sensitivity
AGE_ONSET_COL <- "age_at_dementia_onset"   # needed for early vs late onset AUC
APOE4_COL <- "APOEe4_status"               # needed for carrier/non-carrier AUC

# Risk group cutpoint
HIGH_RISK_Q <- 0.75

# Horizons (days)
HORIZONS <- c("3y" = 1096, "7y" = 2555, "9y" = 3287)

# Early event exclusion window (days)
EARLY_EVENT_DAYS <- 30

# Which XGBoost prediction column corresponds to the FINAL model in Script 03
FINAL_XGB_PRED_COL <- "pred_Model5_Add_MetaboAge"

# --------------------------
# LOAD INPUTS
# --------------------------
if (!file.exists(INPUT_RDS)) stop("Missing input: ", INPUT_RDS)
if (!file.exists(FG_MODEL_RDS)) stop("Missing Fine–Gray model: ", FG_MODEL_RDS)
if (!file.exists(XGB_PRED_CSV)) stop("Missing XGBoost predictions: ", XGB_PRED_CSV)

Biomarkers_complete <- readRDS(INPUT_RDS)
fg_fit <- readRDS(FG_MODEL_RDS)
xgb_preds <- read.csv(XGB_PRED_CSV, stringsAsFactors = FALSE)

# --------------------------
# Helper: build TEST set for survival outcomes consistent with Script 05
# --------------------------
build_survival_test <- function(df0) {
  if (!ID_COL %in% names(df0)) df0[[ID_COL]] <- seq_len(nrow(df0))

  y_raw <- df0[[OUTCOME_COL]]
  y_num <- if (is.factor(y_raw)) suppressWarnings(as.numeric(as.character(y_raw))) else suppressWarnings(as.numeric(y_raw))
  if (any(is.na(y_num))) y_num <- if (is.factor(y_raw)) as.numeric(y_raw) - 1 else ifelse(is.na(y_num), 0, y_num)
  y_num <- ifelse(y_num == 1, 1, 0)

  df1 <- df0 %>%
    transmute(
      id = as.character(.data[[ID_COL]]),
      y_class = y_num,
      sex = as.factor(.data[[SEX_COL]]),
      t_dem = suppressWarnings(as.numeric(.data[[T_DEM_COL]])),
      dem_event = ifelse(y_num == 1, 1, 0),
      death_event = suppressWarnings(as.numeric(as.character(.data[[DEATH_COL]]))),
      death_event = ifelse(death_event == 1, 1, 0),
      t_death = suppressWarnings(as.numeric(.data[[T_DEATH_COL]])),

      # optional subgroup variables
      age_onset = if (AGE_ONSET_COL %in% names(df0)) suppressWarnings(as.numeric(.data[[AGE_ONSET_COL]])) else NA_real_,
      apoe4 = if (APOE4_COL %in% names(df0)) as.factor(.data[[APOE4_COL]]) else NA
    ) %>%
    filter(!is.na(y_class))

  df1 <- df1 %>%
    mutate(
      has_dem = dem_event == 1 & !is.na(t_dem),
      has_death = death_event == 1 & !is.na(t_death),

      event = case_when(
        has_dem & (!has_death | t_dem <= t_death) ~ 1,
        has_death & (!has_dem | t_death < t_dem)  ~ 2,
        TRUE ~ 0
      ),
      time = case_when(
        event == 1 ~ t_dem,
        event == 2 ~ t_death,
        TRUE ~ pmin(t_dem, t_death, na.rm = TRUE)
      )
    ) %>%
    filter(is.finite(time) & !is.na(time) & time > 0) %>%
    mutate(event = as.integer(event))

  train_idx <- caret::createDataPartition(df1$y_class, p = 0.70, list = FALSE)
  test_df <- df1[-train_idx, , drop = FALSE]
  test_df
}

test_surv <- build_survival_test(Biomarkers_complete)

# --------------------------
# A) Horizon robustness: 3y/7y/9y
# - Predict absolute risk in TEST using TRAIN Fine–Gray model
# - Create risk group at each horizon and compute:
#     CIF (cause 1) + sHR High vs Low + sHR per +1% absolute risk
# --------------------------

predict_risk_vec <- function(fg_fit, newdata, times) {
  r <- riskRegression::predictRisk(fg_fit, newdata = newdata, times = times)
  as.numeric(if (is.matrix(r)) r[, 1] else r)
}

compute_shr_group <- function(df, group_col) {
  mm <- model.matrix(reformulate(group_col), data = df)[, -1, drop = FALSE]
  fit <- cmprsk::crr(ftime = df$time, fstatus = df$event, cov1 = mm, failcode = 1, cencode = 0)
  b <- as.numeric(fit$coef)
  se <- sqrt(diag(fit$var))
  data.frame(
    beta = b,
    se = se,
    sHR = exp(b),
    CI_low = exp(b - 1.96 * se),
    CI_high = exp(b + 1.96 * se),
    p_value = 2 * pnorm(-abs(b / se))
  )
}

compute_shr_continuous_per1pct <- function(df, risk_col) {
  cov_mat <- matrix(df[[risk_col]], ncol = 1)
  fit <- cmprsk::crr(ftime = df$time, fstatus = df$event, cov1 = cov_mat, failcode = 1, cencode = 0)
  b <- as.numeric(fit$coef)
  se <- sqrt(diag(fit$var))

  # per +1% absolute risk increase == +0.01
  data.frame(
    beta = b,
    se = se,
    sHR_per_1pct = exp(b * 0.01),
    CI_low = exp((b - 1.96 * se) * 0.01),
    CI_high = exp((b + 1.96 * se) * 0.01),
    p_value = 2 * pnorm(-abs(b / se))
  )
}

make_cif_df <- function(df, group_col) {
  ci <- cmprsk::cuminc(ftime = df$time, fstatus = df$event, group = df[[group_col]])
  cause1_names <- names(ci)[grepl(" 1$", names(ci))]
  out <- bind_rows(lapply(cause1_names, function(nm) {
    e <- ci[[nm]]
    grp <- sub(" 1$", "", nm)
    data.frame(time = e$time, est = as.numeric(e$est), var = as.numeric(e$var), group = grp)
  })) %>%
    mutate(
      se = sqrt(var),
      lower = pmax(0, est - 1.96 * se),
      upper = pmin(1, est + 1.96 * se),
      group = factor(group, levels = c("Low", "High"))
    )
  out
}

shr_rows <- list()
plots <- list()

for (nm in names(HORIZONS)) {
  t_days <- as.numeric(HORIZONS[[nm]])

  risk_col <- paste0("risk_", nm)
  test_tmp <- test_surv %>%
    mutate(!!risk_col := predict_risk_vec(fg_fit, newdata = test_surv, times = t_days))

  q75 <- as.numeric(quantile(test_tmp[[risk_col]], probs = HIGH_RISK_Q, na.rm = TRUE))
  grp_col <- paste0("risk_group_", nm)

  test_tmp <- test_tmp %>%
    mutate(
      !!grp_col := factor(ifelse(.data[[risk_col]] > q75, "High", "Low"), levels = c("Low", "High"))
    )

  # sHR High vs Low
  shr_g <- compute_shr_group(test_tmp, grp_col)
  # sHR per 1% absolute-risk increase
  shr_c <- compute_shr_continuous_per1pct(test_tmp, risk_col)

  shr_rows[[nm]] <- data.frame(
    Horizon = nm,
    Days = t_days,
    Group_sHR = shr_g$sHR,
    Group_CI_low = shr_g$CI_low,
    Group_CI_high = shr_g$CI_high,
    Group_p = shr_g$p_value,
    Cont_sHR_per_1pct = shr_c$sHR_per_1pct,
    Cont_CI_low = shr_c$CI_low,
    Cont_CI_high = shr_c$CI_high,
    Cont_p = shr_c$p_value
  )

  # CIF plot
  cif_df <- make_cif_df(test_tmp, grp_col)
  ann <- paste0(
    "High vs Low sHR = ", sprintf("%.2f", shr_g$sHR),
    " (", sprintf("%.2f", shr_g$CI_low), "–", sprintf("%.2f", shr_g$CI_high), ")\n",
    "sHR per +1% risk = ", sprintf("%.2f", shr_c$sHR_per_1pct),
    " (", sprintf("%.2f", shr_c$CI_low), "–", sprintf("%.2f", shr_c$CI_high), ")"
  )

  p <- ggplot(cif_df, aes(x = time, y = est, color = group, fill = group)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.18, color = NA) +
    geom_step(linewidth = 1) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title = paste0("Cumulative Incidence of ADRD by Risk Group (", nm, " predicted risk; TEST)"),
      subtitle = ann,
      x = "Time (days)",
      y = "Cumulative incidence",
      color = "Risk group",
      fill = "Risk group"
    ) +
    theme_classic(base_size = 13) +
    theme(legend.position = "top", plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))

  plots[[nm]] <- p
}

shr_by_horizon <- bind_rows(shr_rows)
write.csv(shr_by_horizon, file.path(OUT_DIR, "shr_by_horizon_test.csv"), row.names = FALSE)

# Save combined CIF plots
cif_panel <- plots[["3y"]] / plots[["7y"]] / plots[["9y"]]
ggsave(file.path(OUT_DIR, "cif_horizon_plots.png"), cif_panel, width = 9, height = 16, dpi = 300)

# --------------------------
# B) Early event exclusion robustness (<=30 days)
# - Use 9y risk group from A) recomputed inside this block (TEST only)
# - Remove ADRD events with time <= 30 days, then recompute CIF + sHR
# --------------------------

# Build 9y risk group again 
risk9 <- predict_risk_vec(fg_fit, newdata = test_surv, times = HORIZONS[["9y"]])
q75_9 <- as.numeric(quantile(risk9, probs = HIGH_RISK_Q, na.rm = TRUE))

test_9 <- test_surv %>%
  mutate(
    risk_9y = risk9,
    risk_group_9y = factor(ifelse(risk_9y > q75_9, "High", "Low"), levels = c("Low","High"))
  )

# Exclude early ADRD events
test_9_sens <- test_9 %>%
  filter(!(event == 1 & time <= EARLY_EVENT_DAYS))

shr_g_sens <- compute_shr_group(test_9_sens, "risk_group_9y")
cif_sens <- make_cif_df(test_9_sens, "risk_group_9y")
cif_full <- make_cif_df(test_9, "risk_group_9y")

ann_sens <- paste0(
  "Sensitivity: Excluding ADRD events ≤", EARLY_EVENT_DAYS, " days\n",
  "sHR (High vs Low) = ", sprintf("%.2f", shr_g_sens$sHR),
  " (", sprintf("%.2f", shr_g_sens$CI_low), "–", sprintf("%.2f", shr_g_sens$CI_high), ")"
)

p_full <- ggplot(cif_full, aes(x=time, y=est, color=group)) +
  geom_step(linewidth=0.8, alpha=0.4) +
  labs(title="CIF (TEST): Full data (faint)") +
  theme_classic(base_size = 12) +
  theme(legend.position="none")

p_sens <- ggplot(cif_sens, aes(x=time, y=est, color=group, fill=group)) +
  geom_ribbon(aes(ymin=lower, ymax=upper), alpha=0.18, color=NA) +
  geom_step(linewidth=1) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = paste0("CIF (TEST): Excluding ADRD events ≤", EARLY_EVENT_DAYS, " days"),
    subtitle = ann_sens,
    x = "Time (days)", y = "Cumulative incidence"
  ) +
  theme_classic(base_size = 12) +
  theme(legend.position="top", plot.title = element_text(face="bold", hjust=0.5),
        plot.subtitle = element_text(hjust=0.5))

panel_early <- p_full / p_sens
ggsave(file.path(OUT_DIR, "early_event_exclusion_shr_cif.png"), panel_early, width = 9, height = 10, dpi = 300)

# --------------------------
# C) AUC subgroup sensitivity using Script 03 holdout predictions
# (NO re-training; uses same holdout predictions)
# --------------------------

# Merge subgroup variables into predictions (by row order fallback if ID mismatch)
# xgb_preds has: ID, y_test, pred_Model...
pred_df <- xgb_preds

# Ensure we have a usable ID
if (!"ID" %in% names(pred_df)) stop("XGB predictions file must contain 'ID' column.")
pred_df$ID <- as.character(pred_df$ID)

# Attach subgroup vars from Biomarkers_complete
sub_df <- Biomarkers_complete %>%
  mutate(
    ID = if (ID_COL %in% names(Biomarkers_complete)) as.character(.data[[ID_COL]]) else as.character(seq_len(nrow(.))),
    sex = as.factor(.data[[SEX_COL]]),
    age_onset = if (AGE_ONSET_COL %in% names(.)) suppressWarnings(as.numeric(.data[[AGE_ONSET_COL]])) else NA_real_,
    apoe4 = if (APOE4_COL %in% names(.)) as.factor(.data[[APOE4_COL]]) else NA
  ) %>%
  select(ID, sex, age_onset, apoe4)

pred_df <- pred_df %>%
  left_join(sub_df, by = c("ID" = "ID"))

if (!FINAL_XGB_PRED_COL %in% names(pred_df)) stop("Missing final prediction column: ", FINAL_XGB_PRED_COL)

auc_ci <- function(y, p) {
  ok <- !is.na(y) & !is.na(p)
  y <- y[ok]; p <- p[ok]
  if (length(unique(y)) < 2) return(c(AUC=NA, L=NA, U=NA))
  r <- pROC::roc(response = y, predictor = p, levels = c(0,1), direction = "auto", quiet = TRUE)
  ci <- as.numeric(pROC::ci.auc(r, method = "delong"))
  c(AUC = as.numeric(pROC::auc(r)), L = ci[1], U = ci[3])
}

auc_rows <- list()

# Sex strata
if (!all(is.na(pred_df$sex))) {
  for (s in levels(droplevels(pred_df$sex))) {
    dd <- pred_df %>% filter(sex == s)
    vals <- auc_ci(dd$y_test, dd[[FINAL_XGB_PRED_COL]])
    auc_rows[[paste0("Sex_", s)]] <- data.frame(
      Analysis = "Sex subgroup",
      Group = as.character(s),
      AUC = vals["AUC"], Lower_CI = vals["L"], Upper_CI = vals["U"]
    )
  }
}

# Early vs late onset 
if (AGE_ONSET_COL %in% names(Biomarkers_complete)) {
  dd_all <- pred_df
  dd_all <- dd_all %>% mutate(
    early_case = ifelse(!is.na(age_onset) & age_onset < 65, 1, 0),
    late_case  = ifelse(!is.na(age_onset) & age_onset >= 65, 1, 0)
  )

  # Early-onset: controls + early cases
  dd_early <- dd_all %>% filter(y_test == 0 | early_case == 1)
  v <- auc_ci(dd_early$y_test, dd_early[[FINAL_XGB_PRED_COL]])
  auc_rows[["EarlyOnset"]] <- data.frame(
    Analysis = "Age of onset",
    Group = "Early-onset (<65)",
    AUC = v["AUC"], Lower_CI = v["L"], Upper_CI = v["U"]
  )

  # Late-onset: controls + late cases
  dd_late <- dd_all %>% filter(y_test == 0 | late_case == 1)
  v <- auc_ci(dd_late$y_test, dd_late[[FINAL_XGB_PRED_COL]])
  auc_rows[["LateOnset"]] <- data.frame(
    Analysis = "Age of onset",
    Group = "Late-onset (≥65)",
    AUC = v["AUC"], Lower_CI = v["L"], Upper_CI = v["U"]
  )
}

# APOE ε4 carrier/non-carrier
if (APOE4_COL %in% names(Biomarkers_complete) && !all(is.na(pred_df$apoe4))) {
  for (g in levels(droplevels(pred_df$apoe4))) {
    dd <- pred_df %>% filter(apoe4 == g)
    v <- auc_ci(dd$y_test, dd[[FINAL_XGB_PRED_COL]])
    auc_rows[[paste0("APOE4_", g)]] <- data.frame(
      Analysis = "APOE-ε4 carrier status",
      Group = as.character(g),
      AUC = v["AUC"], Lower_CI = v["L"], Upper_CI = v["U"]
    )
  }
}

auc_subgroups <- bind_rows(auc_rows)
write.csv(auc_subgroups, file.path(OUT_DIR, "auc_subgroups_from_holdout_preds.csv"), row.names = FALSE)

# --------------------------
# D) Forest plot for AUC sensitivity 
# --------------------------
if (nrow(auc_subgroups) > 0) {
  auc_plot_df <- auc_subgroups %>%
    mutate(
      Analysis = factor(Analysis, levels = unique(Analysis)),
      Group = factor(Group, levels = rev(unique(Group)))
    )

  p_auc <- ggplot(auc_plot_df, aes(x = AUC, y = Group, xmin = Lower_CI, xmax = Upper_CI)) +
    facet_wrap(~ Analysis, ncol = 1, scales = "free_y", strip.position = "right") +
    geom_errorbar(width = 0.15, linewidth = 0.9) +
    geom_point(size = 2.8) +
    labs(
      title = "Sensitivity Analyses: AUC of Final XGBoost Model (Holdout TEST predictions)",
      x = "AUC (DeLong 95% CI)",
      y = ""
    ) +
    scale_x_continuous(limits = c(0.5, 1.0), breaks = seq(0.5, 1.0, by = 0.05)) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      strip.text.y.right = element_text(angle = 0, face = "bold"),
      strip.background = element_rect(fill = "gray90", color = NA),
      strip.placement = "outside",
      panel.spacing.y = unit(0.8, "lines")
    )

  ggsave(file.path(OUT_DIR, "auc_sensitivity_forestplot.png"), p_auc, width = 9, height = 10, dpi = 300)
}

cat("\nDONE ✅ Sensitivity analyses completed.\n")
cat("Outputs written to:", OUT_DIR, "\n\n")
