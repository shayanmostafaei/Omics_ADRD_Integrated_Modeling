# ==================================================================================
# 06_Sensitivity_Analyses.R
#
# Inputs required:
#   - results/survival_competing_risk/finegray_model_train.rds (Script 05)
#   - results/multivariable_xgboost/test_set_predictions_stepwise.csv (Script 03)
#   - data/biomarkers_complete.rds
#
# Outputs:
#   results/sensitivity/
#     - shr_by_horizon_test.csv
#     - cif_horizon_plots.png
#     - early_event_exclusion_shr_cif.png  (only if needed)
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
  library(grid)
})

# --------------------------
# USER SETTINGS
# --------------------------
set.seed(20250101)

INPUT_RDS <- "data/biomarkers_complete.rds"
FG_MODEL_RDS <- "results/survival_competing_risk/finegray_model_train.rds"
XGB_PRED_CSV <- "results/multivariable_xgboost/test_set_predictions_stepwise.csv"

OUT_DIR <- "results/sensitivity"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

ID_COL <- "f.eid"
OUTCOME_COL <- "Dementia_status"
SEX_COL <- "sex"

# survival columns
T_DEM_COL <- "Time_to_Dementia"     # DAYS
DEATH_COL <- "death_status"         # 0/1
T_DEATH_COL <- "time_to_death"      # YEARS in your dataset
CENS_COL <- "length_followup"       # YEARS in your dataset

T_DEATH_IS_YEARS <- TRUE
CENS_IS_YEARS <- TRUE

AGE_ONSET_COL <- "age_at_dementia_onset"
APOE4_COL <- "APOEe4_status"

HIGH_RISK_Q <- 0.75

# Horizons in DAYS (consistent with Script 05)
HORIZONS <- c(
  "3y" = 3 * 365.25,
  "7y" = 7 * 365.25,
  "9y" = 9 * 365.25
)

EARLY_EVENT_DAYS <- 30

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
# Helper: build TEST set consistent with Script 05
# --------------------------
build_survival_test <- function(df0) {

  if (!ID_COL %in% names(df0)) df0[[ID_COL]] <- seq_len(nrow(df0))

  # y_class for stratified splitting
  y_raw <- df0[[OUTCOME_COL]]
  y_num <- if (is.factor(y_raw)) suppressWarnings(as.numeric(as.character(y_raw))) else suppressWarnings(as.numeric(y_raw))
  y_num <- ifelse(is.na(y_num), 0, y_num)
  y_num <- ifelse(y_num == 1, 1, 0)

  df1 <- df0 %>%
    transmute(
      id = as.character(.data[[ID_COL]]),
      y_class = y_num,

      # raw times
      t_dem_days = suppressWarnings(as.numeric(.data[[T_DEM_COL]])),  # DAYS
      death_event = suppressWarnings(as.numeric(as.character(.data[[DEATH_COL]]))),
      death_event = ifelse(is.na(death_event), 0, ifelse(death_event == 1, 1, 0)),
      t_death_raw = suppressWarnings(as.numeric(.data[[T_DEATH_COL]])), # likely YEARS
      t_cens_raw  = suppressWarnings(as.numeric(.data[[CENS_COL]])),    # likely YEARS

      # subgroup vars
      sex = as.factor(.data[[SEX_COL]]),
      age_onset = if (AGE_ONSET_COL %in% names(df0)) suppressWarnings(as.numeric(.data[[AGE_ONSET_COL]])) else NA_real_,
      apoe4 = if (APOE4_COL %in% names(df0)) as.factor(.data[[APOE4_COL]]) else NA
    ) %>%
    filter(!is.na(y_class))

  # Convert death & censoring to DAYS
  df1 <- df1 %>%
    mutate(
      t_death_days = ifelse(
        death_event == 1 & !is.na(t_death_raw),
        if (T_DEATH_IS_YEARS) t_death_raw * 365.25 else t_death_raw,
        Inf
      ),

      t_cens_days = ifelse(
        !is.na(t_cens_raw),
        if (CENS_IS_YEARS) t_cens_raw * 365.25 else t_cens_raw,
        Inf
      ),

      # ADRD time only active if case
      t_dem_event_days = ifelse(y_class == 1 & !is.na(t_dem_days), t_dem_days, Inf),

      time = pmin(t_dem_event_days, t_death_days, t_cens_days, na.rm = TRUE),

      event = case_when(
        is.finite(t_dem_event_days) & t_dem_event_days <= t_death_days & t_dem_event_days <= t_cens_days ~ 1L,
        is.finite(t_death_days)     & t_death_days     <  t_dem_event_days & t_death_days <= t_cens_days ~ 2L,
        TRUE ~ 0L
      )
    ) %>%
    filter(is.finite(time), !is.na(time), time > 0) %>%
    mutate(event = as.integer(event))

  # Same split mechanism as Script 05
  train_idx <- caret::createDataPartition(df1$y_class, p = 0.70, list = FALSE)
  test_df <- df1[-train_idx, , drop = FALSE]
  test_df
}

test_surv <- build_survival_test(Biomarkers_complete)

message("TEST event counts (0=cens,1=ADRD,2=death):")
print(table(test_surv$event))

# --------------------------
# A) Horizon robustness: 3y/7y/9y
# --------------------------
predict_risk_vec <- function(fg_fit, newdata, times) {
  r <- riskRegression::predictRisk(fg_fit, newdata = newdata, times = times)
  as.numeric(if (is.matrix(r)) r[, 1] else r)
}

compute_shr_group <- function(df, group_var) {
  mm <- model.matrix(reformulate(group_var), data = df)[, -1, drop = FALSE]
  fit <- cmprsk::crr(ftime = df$time, fstatus = df$event, cov1 = mm, failcode = 1, cencode = 0)
  b <- as.numeric(fit$coef)
  se <- sqrt(diag(fit$var))
  data.frame(
    sHR = exp(b),
    CI_low = exp(b - 1.96 * se),
    CI_high = exp(b + 1.96 * se),
    p_value = 2 * pnorm(-abs(b / se))
  )
}

compute_shr_continuous_per1pct <- function(df, risk_vec) {
  cov_mat <- matrix(risk_vec, ncol = 1)
  fit <- cmprsk::crr(ftime = df$time, fstatus = df$event, cov1 = cov_mat, failcode = 1, cencode = 0)
  b <- as.numeric(fit$coef)
  se <- sqrt(diag(fit$var))
  data.frame(
    sHR_per_1pct = exp(b * 0.01),
    CI_low = exp((b - 1.96 * se) * 0.01),
    CI_high = exp((b + 1.96 * se) * 0.01),
    p_value = 2 * pnorm(-abs(b / se))
  )
}

make_cif_df <- function(df, group_vec) {
  ci <- cmprsk::cuminc(ftime = df$time, fstatus = df$event, group = group_vec)
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

  risk_vec <- predict_risk_vec(fg_fit, newdata = test_surv, times = t_days)
  q75 <- as.numeric(quantile(risk_vec, probs = HIGH_RISK_Q, na.rm = TRUE))
  group_vec <- factor(ifelse(risk_vec > q75, "High", "Low"), levels = c("Low","High"))

  shr_g <- compute_shr_group(test_surv %>% mutate(group_tmp = group_vec), "group_tmp")
  shr_c <- compute_shr_continuous_per1pct(test_surv, risk_vec)

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

  cif_df <- make_cif_df(test_surv, group_vec)

  ann <- paste0(
    "High vs Low sHR = ", sprintf("%.2f", shr_g$sHR),
    " (", sprintf("%.2f", shr_g$CI_low), "–", sprintf("%.2f", shr_g$CI_high), ")\n",
    "sHR per +1% risk = ", sprintf("%.2f", shr_c$sHR_per_1pct),
    " (", sprintf("%.2f", shr_c$CI_low), "–", sprintf("%.2f", shr_c$CI_high), ")"
  )

  plots[[nm]] <- ggplot(cif_df, aes(x = time, y = est, color = group, fill = group)) +
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
    theme(legend.position = "top",
          plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
}

shr_by_horizon <- bind_rows(shr_rows)
write.csv(shr_by_horizon, file.path(OUT_DIR, "shr_by_horizon_test.csv"), row.names = FALSE)

cif_panel <- plots[["3y"]] / plots[["7y"]] / plots[["9y"]]
ggsave(file.path(OUT_DIR, "cif_horizon_plots.png"), cif_panel, width = 9, height = 16, dpi = 300)

# --------------------------
# B) Early event exclusion robustness (<=30 days)
# Uses 5-year risk grouping (aligned with Figure 5 logic)
# Only runs if such early events exist
# --------------------------
risk5 <- predict_risk_vec(fg_fit, newdata = test_surv, times = 5 * 365.25)
q75_5 <- as.numeric(quantile(risk5, probs = HIGH_RISK_Q, na.rm = TRUE))
group5 <- factor(ifelse(risk5 > q75_5, "High", "Low"), levels = c("Low","High"))

has_early <- any(test_surv$time <= EARLY_EVENT_DAYS & test_surv$event %in% c(1,2))
if (has_early) {

  test_sens <- test_surv %>%
    mutate(group5 = group5) %>%
    filter(!(event %in% c(1,2) & time <= EARLY_EVENT_DAYS))

  shr_sens <- compute_shr_group(test_sens, "group5")
  cif_sens <- make_cif_df(test_sens, test_sens$group5)

  ann_sens <- paste0(
    "Sensitivity: Excluding events ≤", EARLY_EVENT_DAYS, " days\n",
    "sHR (High vs Low) = ", sprintf("%.2f", shr_sens$sHR),
    " (", sprintf("%.2f", shr_sens$CI_low), "–", sprintf("%.2f", shr_sens$CI_high), ")"
  )

  p_sens <- ggplot(cif_sens, aes(x=time, y=est, color=group, fill=group)) +
    geom_ribbon(aes(ymin=lower, ymax=upper), alpha=0.18, color=NA) +
    geom_step(linewidth=1) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title = paste0("CIF (TEST): Excluding events ≤", EARLY_EVENT_DAYS, " days"),
      subtitle = ann_sens,
      x = "Time (days)",
      y = "Cumulative incidence",
      color = "Risk group",
      fill = "Risk group"
    ) +
    theme_classic(base_size = 12) +
    theme(legend.position="top",
          plot.title = element_text(face="bold", hjust=0.5),
          plot.subtitle = element_text(hjust=0.5))

  ggsave(file.path(OUT_DIR, "early_event_exclusion_shr_cif.png"),
         p_sens, width = 9, height = 6, dpi = 300)

} else {
  message("Early-event sensitivity skipped: no events within ", EARLY_EVENT_DAYS, " days in TEST.")
}

# --------------------------
# C) AUC subgroup sensitivity (uses holdout predictions; no retraining)
# --------------------------
pred_df <- xgb_preds
if (!"ID" %in% names(pred_df)) stop("XGB predictions file must contain 'ID' column.")
pred_df$ID <- as.character(pred_df$ID)

sub_df <- Biomarkers_complete %>%
  mutate(
    ID = if (ID_COL %in% names(.)) as.character(.data[[ID_COL]]) else as.character(seq_len(nrow(.))),
    sex = as.factor(.data[[SEX_COL]]),
    age_onset = if (AGE_ONSET_COL %in% names(.)) suppressWarnings(as.numeric(.data[[AGE_ONSET_COL]])) else NA_real_,
    apoe4 = if (APOE4_COL %in% names(.)) as.factor(.data[[APOE4_COL]]) else NA
  ) %>%
  select(ID, sex, age_onset, apoe4)

pred_df <- pred_df %>% left_join(sub_df, by = "ID")

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

# Early vs late onset (cases only have onset age)
if (AGE_ONSET_COL %in% names(Biomarkers_complete)) {
  dd_all <- pred_df %>%
    mutate(
      early_case = ifelse(!is.na(age_onset) & age_onset < 65, 1, 0),
      late_case  = ifelse(!is.na(age_onset) & age_onset >= 65, 1, 0)
    )

  dd_early <- dd_all %>% filter(y_test == 0 | early_case == 1)
  v <- auc_ci(dd_early$y_test, dd_early[[FINAL_XGB_PRED_COL]])
  auc_rows[["EarlyOnset"]] <- data.frame(
    Analysis = "Age of onset",
    Group = "Early-onset (<65)",
    AUC = v["AUC"], Lower_CI = v["L"], Upper_CI = v["U"]
  )

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

# Forest plot
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

  ggsave(file.path(OUT_DIR, "auc_sensitivity_forestplot.png"),
         p_auc, width = 9, height = 10, dpi = 300)
}

cat("\nDONE ✅ Sensitivity analyses completed.\n")
cat("Outputs written to:", OUT_DIR, "\n\n")
