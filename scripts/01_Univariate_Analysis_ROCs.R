# ==============================================================================
# 01_Univariate_Analysis_ROCs.R
#
# Integrative Prediction of Alzheimer’s Disease and Related Dementias Using
# Multi-Omics Aging Clocks and Genetic Data
#
# Purpose:
# - Compute univariate ROC-AUC for chronological age, PRS, biological aging
#   measures, ProtAge, and MetaboAge.
# - Compute AUPRC for each univariate predictor because ADRD incidence is low.
# - Generate manuscript Figure 2: univariate ROC-AUC forest plot.
# - Export ROC point sets for reproducibility.
#
# Main manuscript output:
# - Figure 2: univariate predictor discrimination
#
# Supplementary/diagnostic outputs:
# - univariate_auc_auprc_overall.csv
# - univariate_auc_by_subtype.csv, if subtype columns are present
# - univariate_roc_points_overall.csv
#
# Notes:
# - This is a discrimination-only analysis and ignores censoring.
# - Time-to-event analyses are handled in scripts 05 and 06.
# - Raw UK Biobank data are not distributed in this repository.
#
# Author: Shayan Mostafaei
# Updated for revision: 2026-06-08
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(pROC)
  library(PRROC)
  library(ggplot2)
  library(forcats)
})

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------

set.seed(20250101)

INPUT_RDS <- "data/biomarkers_complete.rds"
OUT_DIR <- file.path("results", "01_univariate")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

OUTCOME_COL <- "Dementia_status"

# Optional subtype columns. These are skipped if absent.
SUBTYPE_COLS <- c("AD", "vascular", "Others_Unspecified")

PREDICTORS <- c(
  "CA",
  "PRS_ADRD",
  "ProtAge",
  "MetaboAge",
  "PhenoAge",
  "KDM",
  "HD",
  "FI",
  "TL"
)

PREDICTOR_LABELS <- c(
  CA = "Chronological age",
  PRS_ADRD = "ADRD PRS",
  ProtAge = "ProtAge",
  MetaboAge = "MetaboAge",
  PhenoAge = "PhenoAge",
  KDM = "KDM",
  HD = "Homeostatic dysregulation",
  FI = "Frailty index",
  TL = "Telomere length"
)

CI_METHOD <- "delong"
MIN_N <- 50
TOP_N_ROC_CURVES <- 6

# ------------------------------------------------------------------------------
# LOAD AND VALIDATE DATA
# ------------------------------------------------------------------------------

if (!file.exists(INPUT_RDS)) {
  stop("Input RDS not found: ", INPUT_RDS)
}

df0 <- readRDS(INPUT_RDS)

if (!OUTCOME_COL %in% names(df0)) {
  stop("Missing outcome column: ", OUTCOME_COL)
}

available_predictors <- intersect(PREDICTORS, names(df0))
missing_predictors <- setdiff(PREDICTORS, available_predictors)

if (length(missing_predictors) > 0) {
  message("Skipping missing predictors: ", paste(missing_predictors, collapse = ", "))
}

if (length(available_predictors) == 0) {
  stop("None of the requested predictors were found.")
}

SUBTYPE_COLS <- intersect(SUBTYPE_COLS, names(df0))

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

coerce_binary01 <- function(x) {
  if (is.logical(x)) return(as.numeric(x))
  if (is.factor(x)) x <- as.character(x)
  if (is.character(x)) {
    x <- trimws(tolower(x))
    x[x %in% c("case", "yes", "y", "true", "1")] <- "1"
    x[x %in% c("control", "no", "n", "false", "0")] <- "0"
  }
  out <- suppressWarnings(as.numeric(x))
  ifelse(out == 1, 1, ifelse(out == 0, 0, NA_real_))
}

safe_ci_auc <- function(roc_obj, method = "delong") {
  tryCatch({
    ci <- as.numeric(pROC::ci.auc(roc_obj, method = method))
    c(ci_low = ci[1], ci_high = ci[3])
  }, error = function(e) {
    c(ci_low = NA_real_, ci_high = NA_real_)
  })
}

compute_auprc <- function(y, p) {
  ok <- !is.na(y) & !is.na(p)
  y <- y[ok]
  p <- p[ok]

  if (length(unique(y)) < 2) return(NA_real_)
  if (sum(y == 1) < 2 || sum(y == 0) < 2) return(NA_real_)

  pr <- PRROC::pr.curve(
    scores.class0 = p[y == 1],
    scores.class1 = p[y == 0],
    curve = FALSE
  )
  as.numeric(pr$auc.integral)
}

compute_univariate_metrics <- function(outcome, predictor) {
  dd <- tibble(y = outcome, x = predictor) %>%
    filter(!is.na(y), !is.na(x))

  if (nrow(dd) < MIN_N || length(unique(dd$y)) < 2) return(NULL)

  dd <- dd %>% mutate(y = ifelse(y == 1, 1, 0))

  roc_obj <- pROC::roc(
    response = dd$y,
    predictor = dd$x,
    levels = c(0, 1),
    direction = "auto",
    quiet = TRUE
  )

  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_vals <- safe_ci_auc(roc_obj, CI_METHOD)
  auprc_val <- compute_auprc(dd$y, dd$x)

  list(
    roc = roc_obj,
    n = nrow(dd),
    n_cases = sum(dd$y == 1),
    n_controls = sum(dd$y == 0),
    auc = auc_val,
    ci_low = ci_vals["ci_low"],
    ci_high = ci_vals["ci_high"],
    auprc = auprc_val,
    direction = roc_obj$direction
  )
}

roc_to_df <- function(roc_obj, predictor_name) {
  tibble(
    Predictor = predictor_name,
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities
  )
}

pretty_label <- function(x) {
  out <- PREDICTOR_LABELS[x]
  out[is.na(out)] <- x[is.na(out)]
  unname(out)
}

# ------------------------------------------------------------------------------
# OVERALL ADRD UNIVARIATE DISCRIMINATION
# ------------------------------------------------------------------------------

y_overall <- coerce_binary01(df0[[OUTCOME_COL]])

overall_rocs <- list()
overall_rows <- list()

for (var in available_predictors) {
  x <- suppressWarnings(as.numeric(df0[[var]]))
  res <- compute_univariate_metrics(y_overall, x)

  if (!is.null(res)) {
    overall_rocs[[var]] <- res$roc
    overall_rows[[var]] <- tibble(
      Outcome = "Overall_ADRD",
      Predictor = var,
      Predictor_label = pretty_label(var),
      N = res$n,
      Cases = res$n_cases,
      Controls = res$n_controls,
      AUC = res$auc,
      CI_low = res$ci_low,
      CI_high = res$ci_high,
      AUPRC = res$auprc,
      ROC_direction = as.character(res$direction)
    )
  }
}

auc_df_overall <- bind_rows(overall_rows) %>%
  arrange(desc(AUC))

write_csv(
  auc_df_overall,
  file.path(OUT_DIR, "univariate_auc_auprc_overall.csv")
)

cat("\n=== Univariate ROC-AUC and AUPRC: overall ADRD ===\n")
print(auc_df_overall)

# ------------------------------------------------------------------------------
# MANUSCRIPT FIGURE 2: FOREST PLOT OF UNIVARIATE AUC
# ------------------------------------------------------------------------------

if (nrow(auc_df_overall) > 0) {
  fig2_df <- auc_df_overall %>%
    mutate(
      Predictor_label = fct_reorder(Predictor_label, AUC),
      AUC_label = sprintf("%.2f (%.2f-%.2f)", AUC, CI_low, CI_high)
    )

  p_fig2 <- ggplot(fig2_df, aes(x = AUC, y = Predictor_label)) +
    geom_vline(xintercept = 0.50, linetype = "dashed", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.18, linewidth = 0.8) +
    geom_point(size = 2.8) +
    geom_text(aes(label = AUC_label), hjust = -0.05, size = 3.4) +
    scale_x_continuous(limits = c(0.45, 1.00), breaks = seq(0.50, 1.00, by = 0.10)) +
    labs(
      title = "Univariate prediction of incident ADRD",
      subtitle = "ROC-AUC with DeLong 95% confidence intervals",
      x = "ROC-AUC",
      y = NULL
    ) +
    theme_classic(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )

  ggsave(
    filename = file.path(OUT_DIR, "Figure2_univariate_auc_forest.png"),
    plot = p_fig2,
    width = 8.5,
    height = 5.8,
    dpi = 500
  )
}

# ------------------------------------------------------------------------------
# TOP ROC CURVES AND POINT EXPORT
# ------------------------------------------------------------------------------

if (length(overall_rocs) > 0) {
  top_vars <- auc_df_overall %>%
    slice_head(n = min(TOP_N_ROC_CURVES, n())) %>%
    pull(Predictor)

  roc_df <- bind_rows(lapply(top_vars, function(v) {
    roc_to_df(overall_rocs[[v]], v)
  })) %>%
    mutate(Predictor_label = pretty_label(Predictor))

  write_csv(
    roc_df,
    file.path(OUT_DIR, "univariate_roc_points_overall_top.csv")
  )

  p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR, group = Predictor_label)) +
    geom_line(linewidth = 0.9) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    facet_wrap(~ Predictor_label, ncol = 3) +
    coord_equal() +
    labs(
      title = "Top univariate ROC curves",
      x = "False positive rate",
      y = "True positive rate"
    ) +
    theme_classic(base_size = 12) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))

  ggsave(
    filename = file.path(OUT_DIR, "univariate_rocs_overall_top.png"),
    plot = p_roc,
    width = 9,
    height = 6,
    dpi = 500
  )
}

# ------------------------------------------------------------------------------
# OPTIONAL ADRD SUBTYPE ANALYSES
# ------------------------------------------------------------------------------

subtype_rows <- list()

if (length(SUBTYPE_COLS) > 0) {
  for (sub in SUBTYPE_COLS) {
    y_sub <- coerce_binary01(df0[[sub]])

    for (var in available_predictors) {
      x <- suppressWarnings(as.numeric(df0[[var]]))
      res <- compute_univariate_metrics(y_sub, x)

      if (!is.null(res)) {
        subtype_rows[[paste(sub, var, sep = "::")]] <- tibble(
          Outcome = sub,
          Predictor = var,
          Predictor_label = pretty_label(var),
          N = res$n,
          Cases = res$n_cases,
          Controls = res$n_controls,
          AUC = res$auc,
          CI_low = res$ci_low,
          CI_high = res$ci_high,
          AUPRC = res$auprc,
          ROC_direction = as.character(res$direction)
        )
      }
    }
  }
}

auc_df_subtype <- bind_rows(subtype_rows)

if (nrow(auc_df_subtype) > 0) {
  auc_df_subtype <- auc_df_subtype %>% arrange(Outcome, desc(AUC))
  write_csv(
    auc_df_subtype,
    file.path(OUT_DIR, "univariate_auc_auprc_by_subtype.csv")
  )
}

# ------------------------------------------------------------------------------
# SAVE SETTINGS AND RESULT BUNDLE
# ------------------------------------------------------------------------------

saveRDS(
  list(
    auc_overall = auc_df_overall,
    auc_by_subtype = auc_df_subtype,
    settings = list(
      INPUT_RDS = INPUT_RDS,
      OUTCOME_COL = OUTCOME_COL,
      SUBTYPE_COLS = SUBTYPE_COLS,
      PREDICTORS_REQUESTED = PREDICTORS,
      PREDICTORS_AVAILABLE = available_predictors,
      CI_METHOD = CI_METHOD,
      MIN_N = MIN_N
    )
  ),
  file.path(OUT_DIR, "univariate_roc_results.rds")
)

cat("\nDONE: Univariate ROC-AUC/AUPRC analysis completed.\n")
cat("Outputs written to:", OUT_DIR, "\n\n")
