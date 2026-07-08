# ==============================================================================
# 05_Survival_Competing_Risks_Analysis.R
#
# Integrative Prediction of Alzheimer’s Disease and Related Dementias Using
# Multi-Omics Aging Clocks and Genetic Data
#
# Purpose:
# - Construct time-to-event data for ADRD with death as a competing event.
# - Fit Fine-Gray competing-risk regression in the training split.
# - Estimate predictor-level subdistribution hazard ratios.
# - Predict absolute ADRD risk in the held-out test set.
# - Stratify test-set participants into top 25% predicted 5-year risk versus
#   lower 75%.
# - Generate manuscript Figure 4 and Figure 5.
# - Save processed train/test survival objects for script 06 sensitivity analyses.
#
# Main manuscript outputs:
# - Figure 4: Fine-Gray predictor forest plot
# - Figure 5: cumulative incidence by predicted 5-year risk group
#
# Author: Shayan Mostafaei
# Updated for revision: 2026-06-15
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(caret)
  library(riskRegression)
  library(cmprsk)
  library(prodlim)
  library(ggplot2)
  library(scales)
  library(forcats)
})

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------

set.seed(20250101)

INPUT_RDS <- "data/biomarkers_complete.rds"
OUT_DIR <- file.path("results", "06_competing_risk")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

ID_COL <- "f.eid"

OUTCOME_COL <- "Dementia_status"
T_DEM_COL <- "Time_to_Dementia"
DEATH_COL <- "death_status"
T_DEATH_COL <- "time_to_death"
CENS_COL <- "length_followup"

T_DEATH_IS_YEARS <- TRUE
CENS_IS_YEARS <- TRUE

AGE_COL <- "CA"
SEX_COL <- "sex"
SMOKE_COL <- "smoking"
ALC_COL <- "alcohol_intake_frequency"
BMI_COL <- "bmi"
EDU_COL <- "education"
PRS_COL <- "PRS_ADRD"
PHENO_COL <- "PhenoAge"
FI_COL <- "FI"
TL_COL <- "TL"
PROTAGE_COL <- "ProtAge"
METABOAGE_COL <- "MetaboAge"
PC_COLS <- paste0("PC", 1:10)

TIME_5Y <- 5 * 365.25
TIME_9Y <- 9 * 365.25
HIGH_RISK_Q <- 0.75

# ------------------------------------------------------------------------------
# LOAD AND VALIDATE DATA
# ------------------------------------------------------------------------------

if (!file.exists(INPUT_RDS)) {
  stop("Input RDS not found: ", INPUT_RDS)
}

df0 <- readRDS(INPUT_RDS)

required_cols <- c(
  OUTCOME_COL, T_DEM_COL, DEATH_COL, T_DEATH_COL, CENS_COL,
  AGE_COL, SEX_COL, SMOKE_COL, ALC_COL, BMI_COL, EDU_COL,
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

convert_time_to_days <- function(x, is_years = FALSE) {
  x <- suppressWarnings(as.numeric(x))
  ifelse(is.na(x) | x <= 0, Inf, ifelse(is_years, x * 365.25, x))
}

extract_fgr_terms <- function(fit) {
  beta <- fit$crrFit$coef
  var_mat <- fit$crrFit$var
  se <- sqrt(diag(var_mat))

  tibble(
    term = names(beta),
    beta = as.numeric(beta),
    se = as.numeric(se),
    sHR = exp(beta),
    CI_low = exp(beta - 1.96 * se),
    CI_high = exp(beta + 1.96 * se),
    p_value = 2 * pnorm(-abs(beta / se))
  )
}

clean_term_labels <- function(x) {
  dplyr::case_when(
    x == "CA" ~ "Chronological age",
    grepl("^sex", x) ~ "Sex",
    grepl("^smoking", x) ~ "Smoking",
    grepl("^alcohol_intake_frequency", x) ~ "Alcohol intake",
    x == "bmi" ~ "BMI",
    grepl("^education", x) ~ "Education",
    x == "PRS_ADRD" ~ "ADRD PRS",
    x == "PhenoAge" ~ "PhenoAge",
    x == "FI" ~ "Frailty index",
    x == "TL" ~ "Telomere length",
    x == "ProtAge" ~ "ProtAge",
    x == "MetaboAge" ~ "MetaboAge",
    grepl("^PC", x) ~ x,
    TRUE ~ x
  )
}

compute_group_shr <- function(df, group_var) {
  mm <- model.matrix(reformulate(group_var), data = df)[, -1, drop = FALSE]

  fit <- cmprsk::crr(
    ftime = df$time,
    fstatus = df$event,
    cov1 = mm,
    failcode = 1,
    cencode = 0
  )

  beta <- as.numeric(fit$coef)
  se <- sqrt(diag(fit$var))

  tibble(
    Comparison = "High vs lower risk group",
    beta = beta,
    se = se,
    sHR = exp(beta),
    CI_low = exp(beta - 1.96 * se),
    CI_high = exp(beta + 1.96 * se),
    p_value = 2 * pnorm(-abs(beta / se))
  )
}

make_cif_df <- function(time, event, group) {
  ci_obj <- cmprsk::cuminc(ftime = time, fstatus = event, group = group)

  cause1_names <- names(ci_obj)[grepl(" 1$", names(ci_obj))]

  bind_rows(lapply(cause1_names, function(nm) {
    e <- ci_obj[[nm]]
    grp <- sub(" 1$", "", nm)

    tibble(
      time = e$time,
      est = as.numeric(e$est),
      var = as.numeric(e$var),
      group = grp
    )
  })) %>%
    mutate(
      se = sqrt(var),
      lower = pmax(0, est - 1.96 * se),
      upper = pmin(1, est + 1.96 * se),
      group = factor(group, levels = c("Lower 75%", "Top 25%"))
    )
}

# ------------------------------------------------------------------------------
# 1. BUILD COMPETING-RISK DATASET
# ------------------------------------------------------------------------------

df1 <- df0 %>%
  transmute(
    id = as.character(.data[[ID_COL]]),
    y_class = coerce_binary01(.data[[OUTCOME_COL]]),
    t_dem_raw = suppressWarnings(as.numeric(.data[[T_DEM_COL]])),
    death_event = coerce_binary01(.data[[DEATH_COL]]),
    t_death_raw = suppressWarnings(as.numeric(.data[[T_DEATH_COL]])),
    t_cens_raw = suppressWarnings(as.numeric(.data[[CENS_COL]])),
    CA = suppressWarnings(as.numeric(.data[[AGE_COL]])),
    sex = as.factor(.data[[SEX_COL]]),
    smoking = as.factor(.data[[SMOKE_COL]]),
    alcohol_intake_frequency = as.factor(.data[[ALC_COL]]),
    bmi = suppressWarnings(as.numeric(.data[[BMI_COL]])),
    education = as.factor(.data[[EDU_COL]]),
    PRS_ADRD = suppressWarnings(as.numeric(.data[[PRS_COL]])),
    PhenoAge = suppressWarnings(as.numeric(.data[[PHENO_COL]])),
    FI = suppressWarnings(as.numeric(.data[[FI_COL]])),
    TL = suppressWarnings(as.numeric(.data[[TL_COL]])),
    ProtAge = suppressWarnings(as.numeric(.data[[PROTAGE_COL]])),
    MetaboAge = suppressWarnings(as.numeric(.data[[METABOAGE_COL]])),
    across(all_of(PC_COLS), ~ suppressWarnings(as.numeric(.x)))
  ) %>%
  filter(!is.na(y_class)) %>%
  mutate(
    death_event = ifelse(is.na(death_event), 0, death_event),
    t_dem_event_days = ifelse(
      y_class == 1 & !is.na(t_dem_raw) & t_dem_raw > 0,
      t_dem_raw,
      Inf
    ),
    t_death_days = ifelse(
      death_event == 1,
      convert_time_to_days(t_death_raw, is_years = T_DEATH_IS_YEARS),
      Inf
    ),
    t_cens_days = convert_time_to_days(t_cens_raw, is_years = CENS_IS_YEARS),
    time = pmin(t_dem_event_days, t_death_days, t_cens_days, na.rm = TRUE),
    event = case_when(
      is.finite(t_dem_event_days) &
        t_dem_event_days <= t_death_days &
        t_dem_event_days <= t_cens_days ~ 1L,
      is.finite(t_death_days) &
        t_death_days < t_dem_event_days &
        t_death_days <= t_cens_days ~ 2L,
      TRUE ~ 0L
    )
  ) %>%
  filter(is.finite(time), !is.na(time), time > 0)

event_summary_all <- df1 %>%
  summarise(
    n = n(),
    n_ADRD = sum(event == 1),
    n_death = sum(event == 2),
    n_censored = sum(event == 0),
    median_followup_days = median(time),
    median_followup_years = median(time) / 365.25
  )

write_csv(event_summary_all, file.path(OUT_DIR, "time_event_summary_all.csv"))

cat("\n=== Event summary, full analytic sample ===\n")
print(event_summary_all)
print(table(df1$event))

# ------------------------------------------------------------------------------
# 2. STRATIFIED TRAIN-TEST SPLIT
# ------------------------------------------------------------------------------

set.seed(20250101)
train_idx <- caret::createDataPartition(df1$y_class, p = 0.70, list = FALSE)

train_df <- df1[train_idx, , drop = FALSE]
test_df <- df1[-train_idx, , drop = FALSE]

split_summary <- tibble(
  Split = c("Training", "Held-out test"),
  N = c(nrow(train_df), nrow(test_df)),
  ADRD_cases = c(sum(train_df$event == 1), sum(test_df$event == 1)),
  Competing_deaths = c(sum(train_df$event == 2), sum(test_df$event == 2)),
  Censored = c(sum(train_df$event == 0), sum(test_df$event == 0))
)

write_csv(split_summary, file.path(OUT_DIR, "survival_train_test_split_summary.csv"))

cat("\n=== Survival train-test split summary ===\n")
print(split_summary)

# ------------------------------------------------------------------------------
# 3. TRAIN-ONLY PREPROCESSING
# ------------------------------------------------------------------------------

numeric_cols <- c(
  "CA", "bmi", "PRS_ADRD", "PhenoAge", "FI", "TL",
  "ProtAge", "MetaboAge", PC_COLS
)
numeric_cols <- numeric_cols[numeric_cols %in% names(df1)]

pp <- caret::preProcess(
  train_df[, numeric_cols, drop = FALSE],
  method = c("medianImpute", "center", "scale")
)

train_df_pp <- train_df
test_df_pp <- test_df

train_df_pp[, numeric_cols] <- predict(pp, train_df[, numeric_cols, drop = FALSE])
test_df_pp[, numeric_cols] <- predict(pp, test_df[, numeric_cols, drop = FALSE])

cat_cols <- c("sex", "smoking", "alcohol_intake_frequency", "education")

for (v in cat_cols) {
  train_df_pp[[v]] <- as.factor(train_df_pp[[v]])
  m <- mode_value(train_df_pp[[v]])

  train_df_pp[[v]][is.na(train_df_pp[[v]])] <- m
  train_df_pp[[v]] <- droplevels(train_df_pp[[v]])

  test_df_pp[[v]] <- as.factor(test_df_pp[[v]])
  test_df_pp[[v]][is.na(test_df_pp[[v]])] <- m
  test_df_pp[[v]] <- factor(test_df_pp[[v]], levels = levels(train_df_pp[[v]]))
}

saveRDS(train_df_pp, file.path(OUT_DIR, "survival_train_processed.rds"))
saveRDS(test_df_pp, file.path(OUT_DIR, "survival_test_processed.rds"))
saveRDS(pp, file.path(OUT_DIR, "survival_preprocess_object.rds"))

# ------------------------------------------------------------------------------
# 4. FIT FINE-GRAY MODEL IN TRAINING DATA
# ------------------------------------------------------------------------------

rhs_terms <- c(
  "CA",
  "sex",
  "smoking",
  "alcohol_intake_frequency",
  "bmi",
  "education",
  "PRS_ADRD",
  "PhenoAge",
  "FI",
  "TL",
  "ProtAge",
  "MetaboAge",
  PC_COLS
)

rhs_terms <- rhs_terms[rhs_terms %in% names(train_df_pp)]

fg_formula <- as.formula(
  paste0("Hist(time, event) ~ ", paste(rhs_terms, collapse = " + "))
)

fg_fit <- riskRegression::FGR(
  formula = fg_formula,
  data = train_df_pp,
  cause = 1
)

saveRDS(fg_fit, file.path(OUT_DIR, "finegray_model_train.rds"))

# ------------------------------------------------------------------------------
# 5. PREDICTOR-LEVEL sHRs FOR FIGURE 4
# ------------------------------------------------------------------------------

fg_terms <- extract_fgr_terms(fg_fit) %>%
  mutate(
    term_clean = clean_term_labels(term),
    p_fdr = p.adjust(p_value, method = "fdr")
  )

write_csv(fg_terms, file.path(OUT_DIR, "finegray_predictor_shr_table.csv"))

# Plot primary biological/genetic predictors only for readability.
figure4_terms <- fg_terms %>%
  filter(term %in% c("PRS_ADRD", "PhenoAge", "FI", "TL", "ProtAge", "MetaboAge")) %>%
  mutate(
    term_clean = factor(term_clean, levels = rev(c(
      "ADRD PRS", "PhenoAge", "Frailty index", "Telomere length", "ProtAge", "MetaboAge"
    ))),
    label = sprintf("%.2f (%.2f-%.2f)", sHR, CI_low, CI_high)
  )

if (nrow(figure4_terms) > 0) {
  p_forest <- ggplot(figure4_terms, aes(x = sHR, y = term_clean)) +
    geom_vline(xintercept = 1, linetype = "dashed", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.18, linewidth = 0.8) +
    geom_point(size = 2.8) +
    geom_text(aes(label = label), hjust = -0.05, size = 3.4) +
    scale_x_continuous(trans = "log10") +
    labs(
      title = "Fine-Gray associations with incident ADRD",
      subtitle = "Subdistribution hazard ratios per standardized predictor",
      x = "Subdistribution hazard ratio, log scale",
      y = NULL
    ) +
    theme_classic(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )

  ggsave(
    filename = file.path(OUT_DIR, "Figure4_FineGray_predictor_forest.png"),
    plot = p_forest,
    width = 8.5,
    height = 5.5,
    dpi = 500
  )
}

# ------------------------------------------------------------------------------
# 6. PREDICT ABSOLUTE ADRD RISK IN HELD-OUT TEST SET
# ------------------------------------------------------------------------------

risk_5y <- as.numeric(riskRegression::predictRisk(fg_fit, newdata = test_df_pp, times = TIME_5Y))
risk_9y <- as.numeric(riskRegression::predictRisk(fg_fit, newdata = test_df_pp, times = TIME_9Y))

pred_df <- test_df_pp %>%
  transmute(
    id,
    event,
    time,
    y_class,
    risk_5y = risk_5y,
    risk_9y = risk_9y
  )

write_csv(pred_df, file.path(OUT_DIR, "test_predicted_risks_5y_9y.csv"))

# ------------------------------------------------------------------------------
# 7. RISK GROUP STRATIFICATION: TOP 25% VERSUS LOWER 75%
# ------------------------------------------------------------------------------

q75_5y <- as.numeric(quantile(pred_df$risk_5y, probs = HIGH_RISK_Q, na.rm = TRUE))

pred_df <- pred_df %>%
  mutate(
    risk_group_5y = factor(
      ifelse(risk_5y > q75_5y, "Top 25%", "Lower 75%"),
      levels = c("Lower 75%", "Top 25%")
    )
  )

write_csv(
  tibble(q75_5y_threshold = q75_5y),
  file.path(OUT_DIR, "riskgroup_thresholds.csv")
)

riskgroup_event_summary <- pred_df %>%
  group_by(risk_group_5y) %>%
  summarise(
    n = n(),
    n_ADRD = sum(event == 1),
    n_death = sum(event == 2),
    n_censored = sum(event == 0),
    ADRD_percent = 100 * n_ADRD / n(),
    death_percent = 100 * n_death / n(),
    .groups = "drop"
  )

write_csv(
  riskgroup_event_summary,
  file.path(OUT_DIR, "riskgroup_event_summary_5y.csv")
)

# ------------------------------------------------------------------------------
# 8. TEST-SET RISK-GROUP sHR
# ------------------------------------------------------------------------------

shr_df <- compute_group_shr(pred_df, "risk_group_5y") %>%
  mutate(
    Comparison = "Top 25% vs lower 75% by predicted 5-year ADRD risk, test set"
  )

write_csv(shr_df, file.path(OUT_DIR, "riskgroup_shr_high_vs_low_5y.csv"))

shr_text <- paste0(
  "sHR = ", sprintf("%.2f", shr_df$sHR),
  " (95% CI ", sprintf("%.2f", shr_df$CI_low),
  "-", sprintf("%.2f", shr_df$CI_high), ")"
)

# ------------------------------------------------------------------------------
# 9. FIGURE 5: CUMULATIVE INCIDENCE CURVES
# ------------------------------------------------------------------------------

cif_df <- make_cif_df(
  time = pred_df$time,
  event = pred_df$event,
  group = pred_df$risk_group_5y
)

write_csv(cif_df, file.path(OUT_DIR, "Figure5_CIF_5yRiskGroup_TEST_data.csv"))

p_cif <- ggplot(cif_df, aes(x = time / 365.25, y = est, group = group)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.18) +
  geom_step(linewidth = 1) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Cumulative incidence of ADRD by predicted risk group",
    subtitle = paste0("Top 25% vs lower 75% predicted 5-year risk; ", shr_text),
    x = "Follow-up time, years",
    y = "Cumulative incidence",
    color = "Risk group",
    fill = "Risk group"
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(
  filename = file.path(OUT_DIR, "Figure5_CIF_5yRiskGroup_TEST.png"),
  plot = p_cif,
  width = 8,
  height = 6,
  dpi = 500
)

# ------------------------------------------------------------------------------
# SAVE RESULT BUNDLE
# ------------------------------------------------------------------------------

saveRDS(
  list(
    fg_fit = fg_fit,
    fg_formula = fg_formula,
    train_df_pp = train_df_pp,
    test_df_pp = test_df_pp,
    pred_df = pred_df,
    fg_terms = fg_terms,
    riskgroup_shr = shr_df,
    riskgroup_event_summary = riskgroup_event_summary,
    split_summary = split_summary,
    q75_5y = q75_5y
  ),
  file.path(OUT_DIR, "competing_risk_results_bundle.rds")
)

cat("\nDONE: Competing-risk analysis completed.\n")
cat("Fine-Gray model fit in training data; risks, CIF, and risk-group sHR computed in held-out test set.\n")
cat("Outputs written to:", OUT_DIR, "\n\n")
