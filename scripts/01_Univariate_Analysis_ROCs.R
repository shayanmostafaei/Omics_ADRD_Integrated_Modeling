# ========================================================================
# 01_Univariate_Analysis_ROCs.R
# Univariate ROC/AUC analysis for ADRD and its subtypes.
# Computes AUC + 95% CI (DeLong) for each predictor worth analyzing.
# Outputs CSV summary, ROC plots (optional), and ROC point sets.
# AUTHOR: Shayan Mostafaei
# DATE CREATED: 2026-03-08
# ========================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(pROC)
  library(ggplot2)
  library(readr)
})

# --------------------------
# USER CONFIGURATION
# --------------------------

set.seed(20250101)
INPUT_RDS <- "data/biomarkers_complete.rds"
OUT_DIR <- file.path("results", "02_univariate", "rocs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
OUTCOME_COL <- "Dementia_status"
SUBTYPE_COLS <- c("AD", "vascular", "Others_Unspecified")
PREDICTORS <- c("CA", "PRS_ADRD", "ProtAge", "Prot_Age_Gap", "MetaboAge", "Metabo_Age_Gap", "PhenoAge", "KDM", "HD", "FI", "TL")
CI_METHOD <- "delong"
MIN_N <- 50
MAKE_PLOTS <- TRUE
TOP_N_PLOTS <- 6
SAVE_ROC_POINTS <- TRUE


# --------------------------
# LOAD DATA AND CHECK INPUTS
# --------------------------

if (!nzchar(INPUT_RDS) || !file.exists(INPUT_RDS)) {
  stop("❌ Input RDS not found. Set INPUT_RDS correctly (currently: ", INPUT_RDS, ")")
}
df0 <- readRDS(INPUT_RDS)

if (!OUTCOME_COL %in% names(df0))
  stop("❌ Missing OUTCOME_COL in data: ", OUTCOME_COL)

missing_preds <- setdiff(PREDICTORS, names(df0))
if (length(missing_preds) > 0)
  stop("❌ Missing predictor columns: ", paste(missing_preds, collapse = ", "))

# Only keep subtype columns available
subtypes_found <- SUBTYPE_COLS[SUBTYPE_COLS %in% names(df0)]
if (length(subtypes_found) == 0) {
  message("ℹ️ No subtype columns found. Subtype analysis skipped.")
}
SUBTYPE_COLS <- subtypes_found

# --------------------------
# HELPER FUNCTIONS
# --------------------------

#' Coerce binary vectors to numeric 0/1 (NA allowed)
coerce_binary01 <- function(x) {
  if (is.logical(x)) return(as.numeric(x))
  if (is.factor(x)) x <- as.character(x)
  if (is.character(x)) {
    x <- trimws(tolower(x))
    x[x %in% c("case", "yes", "y", "true", "1")] <- "1"
    x[x %in% c("control", "no", "n", "false", "0")] <- "0"
  }
  suppressWarnings(as.numeric(x))
}

#' Compute CI for AUC, returns [lower, upper] or NA_real_
safe_ci_auc <- function(roc_obj, method) {
  out <- tryCatch({
    ci <- as.numeric(pROC::ci.auc(roc_obj, method=method))
    c(ci[1], ci[3])
  }, error=function(e) {
    c(NA_real_, NA_real_)
  })
  out
}

#' Conduct ROC analysis, returns list or NULL
compute_univariate_roc <- function(outcome, predictor) {
  dd <- tibble(y = outcome, x = predictor) %>% filter(!is.na(y) & !is.na(x))
  if (nrow(dd) < MIN_N || length(unique(dd$y)) < 2) return(NULL)
  dd$y <- ifelse(dd$y == 1, 1, 0)
  roc_obj <- pROC::roc(response = dd$y, predictor = dd$x, levels = c(0,1), direction="auto", quiet=TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_vals <- safe_ci_auc(roc_obj, CI_METHOD)
  list(
    roc = roc_obj,
    n = nrow(dd),
    n_cases = sum(dd$y == 1),
    n_controls = sum(dd$y == 0),
    auc = auc_val,
    ci_low = ci_vals[1],
    ci_high = ci_vals[2],
    direction = roc_obj$direction
  )
}

#' ROC points to dataframe
roc_to_df <- function(roc_obj, predictor_name) {
  tibble(
    Predictor = predictor_name,
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities
  )
}

# --------------------------
# MAIN ANALYSIS BLOCKS
# --------------------------

# A) Overall ROC Analysis
overall_rocs <- list()
overall_rows <- list()
y_overall <- coerce_binary01(df0[[OUTCOME_COL]])

for (var in PREDICTORS) {
  x <- suppressWarnings(as.numeric(df0[[var]]))
  res <- compute_univariate_roc(y_overall, x)
  if (!is.null(res)) {
    overall_rocs[[var]] <- res$roc
    overall_rows[[var]] <- tibble(
      Outcome = "Overall_ADRD",
      Predictor = var,
      N = res$n,
      Cases = res$n_cases,
      Controls = res$n_controls,
      AUC = res$auc,
      CI_low = res$ci_low,
      CI_high = res$ci_high,
      ROC_direction = res$direction
    )
  }
}

auc_df_overall <- bind_rows(overall_rows) %>% arrange(desc(AUC))
cat("\n=== Univariate AUCs (Overall ADRD) ===\n")
print(auc_df_overall)
write_csv(auc_df_overall, file.path(OUT_DIR, "univariate_auc_overall.csv"))

# Optional plotting (top predictors)
if (MAKE_PLOTS && nrow(auc_df_overall) > 0) {
  top_vars <- head(auc_df_overall$Predictor, TOP_N_PLOTS)
  roc_df <- bind_rows(lapply(top_vars, function(v) roc_to_df(overall_rocs[[v]], v)))
  p <- ggplot(roc_df, aes(x=FPR, y=TPR, color=Predictor)) +
    geom_line(linewidth=1) +
    geom_abline(intercept=0, slope=1, linetype="dashed") +
    coord_equal() +
    theme_classic(base_size=14) +
    labs(
      title = sprintf("Univariate ROC Curves (Overall ADRD) — Top %d by AUC", length(top_vars)),
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)"
    )
  ggsave(file.path(OUT_DIR, "univariate_rocs_overall_top.png"), plot=p, width=7, height=6, dpi=300)
  if (SAVE_ROC_POINTS)
    write_csv(roc_df, file.path(OUT_DIR, "univariate_rocs_overall_top_points.csv"))
}

# B) ROC Analysis by ADRD Subtype
auc_rows_subtype <- list()
if (length(SUBTYPE_COLS) > 0) {
  for (sub in SUBTYPE_COLS) {
    y_sub <- coerce_binary01(df0[[sub]])
    for (var in PREDICTORS) {
      x <- suppressWarnings(as.numeric(df0[[var]]))
      res <- compute_univariate_roc(y_sub, x)
      if (!is.null(res)) {
        auc_rows_subtype[[paste(sub, var, sep="::")]] <- tibble(
          Outcome = sub,
          Predictor = var,
          N = res$n,
          Cases = res$n_cases,
          Controls = res$n_controls,
          AUC = res$auc,
          CI_low = res$ci_low,
          CI_high = res$ci_high,
          ROC_direction = res$direction
        )
      }
    }
  }
}
auc_df_type <- bind_rows(auc_rows_subtype) %>% arrange(Outcome, desc(AUC))
cat("\n=== Univariate AUCs (By ADRD Subtype) ===\n")
print(auc_df_type)
write_csv(auc_df_type, file.path(OUT_DIR, "univariate_auc_by_subtype.csv"))

# --------------------------
# SAVE RESULTS BUNDLE
# --------------------------

saveRDS(
  list(
    auc_overall = auc_df_overall,
    auc_by_subtype = auc_df_type,
    settings = list(
      INPUT_RDS = INPUT_RDS,
      OUTCOME_COL = OUTCOME_COL,
      SUBTYPE_COLS = SUBTYPE_COLS,
      PREDICTORS = PREDICTORS,
      CI_METHOD = CI_METHOD,
      MIN_N = MIN_N,
      MAKE_PLOTS = MAKE_PLOTS,
      TOP_N_PLOTS = TOP_N_PLOTS,
      SAVE_ROC_POINTS = SAVE_ROC_POINTS
    )
  ),
  file.path(OUT_DIR, "univariate_roc_results.rds")
)

cat("\n✅ DONE: Univariate ROC analysis completed.\n")
cat("Output folder:", OUT_DIR, "\n\n")
