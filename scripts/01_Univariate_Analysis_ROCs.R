# ==================================================================================
# 01_Univariate_Analysis_ROCs.R
# Univariate ROC/AUC analysis (Overall ADRD + ADRD subtypes)
# - Computes AUC + 95% CI (DeLong) for each predictor
# - Saves clean CSV outputs + optional ROC plots
# ==================================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(pROC)
  library(ggplot2)
})

# --------------------------
# USER SETTINGS 
# --------------------------

set.seed(20250101)

# Input: analysis table as .rds (recommended) OR load Biomarkers_complete before sourcing this script
# Expected: one row per participant, with outcome + predictors already prepared
INPUT_RDS <- "data/biomarkers_complete.rds"   # set to "" if you already have Biomarkers_complete in memory

# Output directory
OUT_DIR <- "results/univariate_rocs"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Outcome column for overall ADRD (0/1)
OUTCOME_COL <- "Dementia_status"

# Optional ADRD subtype columns (0/1 each); set to character(0) to skip
SUBTYPE_COLS <- c("AD", "vascular", "Others_Unspecified")

# Predictors to evaluate (must exist in the input table)
PREDICTORS <- c(
  "CA", "PRS_ADRD",
  "ProtAge", "Prot_Age_Gap",
  "MetaboAge", "Metabo_Age_Gap",
  "PhenoAge", "KDM", "HD",
  "FI", "TL"
)

# AUC CI method
CI_METHOD <- "delong"   # recommended for ROC AUC in pROC

# Minimum rows required to compute a ROC
MIN_N <- 50

# Optional: plot ROC curves for the top predictors by AUC
MAKE_PLOTS <- TRUE
TOP_N_PLOTS <- 6

# --------------------------
# LOAD DATA
# --------------------------

if (nzchar(INPUT_RDS)) {
  Biomarkers_complete <- readRDS(INPUT_RDS)
} else {
  if (!exists("Biomarkers_complete")) stop("No input found: set INPUT_RDS or load Biomarkers_complete before running.")
}

df0 <- Biomarkers_complete

# Basic checks
if (!OUTCOME_COL %in% names(df0)) stop("Missing OUTCOME_COL in data: ", OUTCOME_COL)

missing_preds <- setdiff(PREDICTORS, names(df0))
if (length(missing_preds) > 0) {
  stop("Missing predictor columns in data: ", paste(missing_preds, collapse = ", "))
}

# Keep only requested subtype columns that exist
SUBTYPE_COLS <- SUBTYPE_COLS[SUBTYPE_COLS %in% names(df0)]
if (length(SUBTYPE_COLS) == 0) message("No subtype columns found (or provided). Subtype ROC analysis will be skipped.")

# --------------------------
# HELPERS
# --------------------------

coerce_binary01 <- function(x) {
  # returns numeric 0/1 with NA preserved
  if (is.logical(x)) return(as.numeric(x))
  if (is.factor(x)) x <- as.character(x)
  if (is.character(x)) {
    x <- trimws(tolower(x))
    x[x %in% c("case","yes","y","true","1")] <- "1"
    x[x %in% c("control","no","n","false","0")] <- "0"
  }
  suppressWarnings(as.numeric(x))
}

safe_roc <- function(outcome, predictor) {
  # outcome must be 0/1 numeric; predictor numeric
  dd <- dplyr::tibble(y = outcome, x = predictor) %>%
    dplyr::filter(!is.na(y) & !is.na(x))

  if (nrow(dd) < MIN_N) return(NULL)
  if (length(unique(dd$y)) < 2) return(NULL)

  # Ensure outcome encoded with controls=0, cases=1
  dd$y <- as.numeric(dd$y)
  dd$y <- ifelse(dd$y == 1, 1, 0)

  # pROC uses factor levels; set explicitly to avoid direction surprises
  roc_obj <- pROC::roc(
    response = dd$y,
    predictor = dd$x,
    levels = c(0, 1),
    direction = "auto",
    quiet = TRUE
  )

  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci <- as.numeric(pROC::ci.auc(roc_obj, method = CI_METHOD))

  list(
    roc = roc_obj,
    n = nrow(dd),
    n_cases = sum(dd$y == 1),
    n_controls = sum(dd$y == 0),
    auc = auc_val,
    ci_low = ci[1],
    ci_high = ci[3]
  )
}

roc_to_df <- function(roc_obj, predictor_name) {
  # Convert ROC curve to dataframe for ggplot
  dplyr::tibble(
    Predictor = predictor_name,
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities
  )
}

# --------------------------
# A) OVERALL UNIVARIATE ROC/AUC
# --------------------------

overall_rocs <- list()
overall_rows <- list()

y_overall <- coerce_binary01(df0[[OUTCOME_COL]])

for (var in PREDICTORS) {
  x <- suppressWarnings(as.numeric(df0[[var]]))
  res <- safe_roc(y_overall, x)
  if (is.null(res)) next

  overall_rocs[[var]] <- res$roc
  overall_rows[[var]] <- data.frame(
    Outcome = "Overall_ADRD",
    Predictor = var,
    N = res$n,
    Cases = res$n_cases,
    Controls = res$n_controls,
    AUC = res$auc,
    CI_low = res$ci_low,
    CI_high = res$ci_high,
    stringsAsFactors = FALSE
  )
}

auc_df_overall <- dplyr::bind_rows(overall_rows) %>%
  dplyr::arrange(dplyr::desc(AUC))

cat("\n=== Univariate AUCs (Overall ADRD) ===\n")
print(auc_df_overall)

# Save overall table
write.csv(auc_df_overall, file.path(OUT_DIR, "univariate_auc_overall.csv"), row.names = FALSE)

# Optional plots (top predictors)
if (MAKE_PLOTS && nrow(auc_df_overall) > 0) {
  top_vars <- head(auc_df_overall$Predictor, TOP_N_PLOTS)
  roc_df <- dplyr::bind_rows(lapply(top_vars, function(v) roc_to_df(overall_rocs[[v]], v)))

  p <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Predictor)) +
    geom_line(linewidth = 1) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    coord_equal() +
    theme_classic(base_size = 14) +
    labs(
      title = paste0("Univariate ROC Curves (Overall ADRD) — Top ", length(top_vars), " by AUC"),
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)"
    )

  ggsave(
    filename = file.path(OUT_DIR, "univariate_rocs_overall_top.png"),
    plot = p, width = 7, height = 6, dpi = 300
  )
}

# --------------------------
# B) UNIVARIATE ROC/AUC BY SUBTYPE
# --------------------------

auc_rows_subtype <- list()

if (length(SUBTYPE_COLS) > 0) {
  for (sub in SUBTYPE_COLS) {
    y_sub <- coerce_binary01(df0[[sub]])

    for (var in PREDICTORS) {
      x <- suppressWarnings(as.numeric(df0[[var]]))
      res <- safe_roc(y_sub, x)
      if (is.null(res)) next

      auc_rows_subtype[[paste(sub, var, sep = "::")]] <- data.frame(
        Outcome = sub,
        Predictor = var,
        N = res$n,
        Cases = res$n_cases,
        Controls = res$n_controls,
        AUC = res$auc,
        CI_low = res$ci_low,
        CI_high = res$ci_high,
        stringsAsFactors = FALSE
      )
    }
  }
}

auc_df_type <- dplyr::bind_rows(auc_rows_subtype) %>%
  dplyr::arrange(Outcome, dplyr::desc(AUC))

cat("\n=== Univariate AUCs (By ADRD Subtype) ===\n")
print(auc_df_type)

# Save subtype table
write.csv(auc_df_type, file.path(OUT_DIR, "univariate_auc_by_subtype.csv"), row.names = FALSE)

# --------------------------
# SAVE .RDS BUNDLE (OPTIONAL)
# --------------------------

saveRDS(
  list(
    auc_overall = auc_df_overall,
    auc_by_subtype = auc_df_type,
    settings = list(
      OUTCOME_COL = OUTCOME_COL,
      SUBTYPE_COLS = SUBTYPE_COLS,
      PREDICTORS = PREDICTORS,
      CI_METHOD = CI_METHOD,
      MIN_N = MIN_N,
      MAKE_PLOTS = MAKE_PLOTS,
      TOP_N_PLOTS = TOP_N_PLOTS
    )
  ),
  file.path(OUT_DIR, "univariate_roc_results.rds")
)

cat("\nDONE ✅ Univariate ROC analysis completed.\n")
cat("Output folder:", OUT_DIR, "\n\n")
