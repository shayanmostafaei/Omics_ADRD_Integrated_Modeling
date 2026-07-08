# ==============================================================================
# 02_Interaction_Heatmaps_BioAge.R
#
# Integrative Prediction of Alzheimer’s Disease and Related Dementias Using
# Multi-Omics Aging Clocks and Genetic Data
#
# Purpose:
# - Generate pairwise correlations among chronological age, biological aging
#   measures, omics aging clocks, and ADRD PRS.
# - Generate standardized linear associations between lifestyle/demographic
#   factors and each aging/PRS marker.
#
# Main manuscript output:
# - Figure 3: correlation matrix among predictors
#
# Supplementary output:
# - Table A.2: associations of lifestyle/demographic factors with biological
#   aging measures and PRS
#
# Notes:
# - These are descriptive/associational analyses.
# - They are not interpreted causally.
#
# Author: Shayan Mostafaei
# Updated for revision: 2026-06-08
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(broom)
  library(ggplot2)
  library(forcats)
})

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------

set.seed(20250101)

INPUT_RDS <- "data/biomarkers_complete.rds"
OUT_DIR <- file.path("results", "02_correlations_associations")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

AGING_VARS <- c(
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

VAR_LABELS <- c(
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

SEX_COL <- "sex"
EDU_COL <- "education"
SMOKE_COL <- "smoking"
ALC_COL <- "alcohol_intake_frequency"
BMI_COL <- "bmi"

FDR_ALPHA <- 0.05
MIN_N_MODEL <- 200
COR_METHOD <- "spearman"

# ------------------------------------------------------------------------------
# LOAD AND VALIDATE DATA
# ------------------------------------------------------------------------------

if (!file.exists(INPUT_RDS)) {
  stop("Input RDS not found: ", INPUT_RDS)
}

df0 <- readRDS(INPUT_RDS)

needed_cols <- unique(c(AGING_VARS, SEX_COL, EDU_COL, SMOKE_COL, ALC_COL, BMI_COL))
missing_cols <- setdiff(needed_cols, names(df0))

if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

zscale <- function(x) as.numeric(scale(x))

to_numeric_code <- function(x) {
  if (is.numeric(x)) return(x)
  if (is.logical(x)) return(as.numeric(x))
  if (is.factor(x)) return(as.numeric(x))
  if (is.character(x)) return(as.numeric(as.factor(x)))
  suppressWarnings(as.numeric(x))
}

pretty_label <- function(x) {
  out <- VAR_LABELS[x]
  out[is.na(out)] <- x[is.na(out)]
  unname(out)
}

safe_cor_test <- function(x, y, method = "spearman") {
  ok <- !is.na(x) & !is.na(y)
  x <- x[ok]
  y <- y[ok]

  if (length(x) < 10 || length(unique(x)) < 2 || length(unique(y)) < 2) {
    return(tibble(n = length(x), estimate = NA_real_, p_value = NA_real_))
  }

  ct <- suppressWarnings(stats::cor.test(x, y, method = method, exact = FALSE))

  tibble(
    n = length(x),
    estimate = unname(ct$estimate),
    p_value = ct$p.value
  )
}

mode_train <- function(x) {
  x <- as.factor(x)
  tab <- table(x, useNA = "no")
  if (length(tab) == 0) return(NA_character_)
  names(tab)[which.max(tab)]
}

# ------------------------------------------------------------------------------
# 1. PAIRWISE CORRELATION MATRIX
# ------------------------------------------------------------------------------

df_corr <- df0 %>%
  select(all_of(AGING_VARS)) %>%
  mutate(across(everything(), ~ suppressWarnings(as.numeric(.x))))

cor_rows <- list()

for (v1 in AGING_VARS) {
  for (v2 in AGING_VARS) {
    res <- safe_cor_test(df_corr[[v1]], df_corr[[v2]], method = COR_METHOD)

    cor_rows[[paste(v1, v2, sep = "::")]] <- res %>%
      mutate(
        Var1 = v1,
        Var2 = v2,
        Var1_label = pretty_label(v1),
        Var2_label = pretty_label(v2)
      )
  }
}

cor_long <- bind_rows(cor_rows) %>%
  mutate(
    p_fdr = p.adjust(p_value, method = "fdr"),
    sig_label = case_when(
      is.na(p_fdr) ~ "",
      p_fdr < 0.001 ~ "***",
      p_fdr < 0.01 ~ "**",
      p_fdr < 0.05 ~ "*",
      TRUE ~ ""
    ),
    Var1_label = factor(Var1_label, levels = pretty_label(AGING_VARS)),
    Var2_label = factor(Var2_label, levels = rev(pretty_label(AGING_VARS)))
  )

write_csv(cor_long, file.path(OUT_DIR, "pairwise_correlations_long.csv"))

cor_matrix <- cor_long %>%
  select(Var1, Var2, estimate) %>%
  pivot_wider(names_from = Var2, values_from = estimate)

write_csv(cor_matrix, file.path(OUT_DIR, "pairwise_correlations_matrix.csv"))

p_corr <- ggplot(cor_long, aes(x = Var1_label, y = Var2_label, fill = estimate)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = ifelse(is.na(estimate), "", sprintf("%.2f", estimate))), size = 3.2) +
  scale_fill_gradient2(
    low = "#3B4CC0",
    mid = "white",
    high = "#B40426",
    midpoint = 0,
    limits = c(-1, 1),
    name = paste0(COR_METHOD, " r")
  ) +
  coord_equal() +
  labs(
    title = "Correlation among genetic, biological aging, and omics aging predictors",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

ggsave(
  filename = file.path(OUT_DIR, "Figure3_correlation_matrix.png"),
  plot = p_corr,
  width = 8.5,
  height = 7.5,
  dpi = 500
)

# ------------------------------------------------------------------------------
# 2. STANDARDIZED LIFESTYLE/DEMOGRAPHIC ASSOCIATIONS
# ------------------------------------------------------------------------------

df_model <- df0 %>%
  transmute(
    across(all_of(AGING_VARS), ~ suppressWarnings(as.numeric(.x))),
    sex = as.factor(.data[[SEX_COL]]),
    education = as.factor(.data[[EDU_COL]]),
    smoking = .data[[SMOKE_COL]],
    alcohol_intake_frequency = .data[[ALC_COL]],
    bmi = suppressWarnings(as.numeric(.data[[BMI_COL]]))
  ) %>%
  mutate(
    smoking_num = to_numeric_code(smoking),
    alcohol_num = to_numeric_code(alcohol_intake_frequency),
    bmi_z = zscale(bmi),
    smoking_z = zscale(smoking_num),
    alcohol_z = zscale(alcohol_num),
    sex = droplevels(sex),
    education = droplevels(education)
  )

# Use first level as reference after factor conversion.
# If your manuscript requires a specific reference level, set it here locally.
df_model <- df_model %>%
  mutate(
    sex_male = ifelse(as.numeric(sex) == max(as.numeric(sex), na.rm = TRUE), 1, 0),
    education_num = to_numeric_code(education),
    education_z = zscale(education_num)
  )

fit_marker_lm <- function(marker) {
  dd <- df_model %>%
    filter(!is.na(.data[[marker]])) %>%
    mutate(outcome_z = zscale(.data[[marker]])) %>%
    filter(
      !is.na(outcome_z),
      !is.na(smoking_z),
      !is.na(alcohol_z),
      !is.na(bmi_z),
      !is.na(sex_male),
      !is.na(education_z)
    )

  if (nrow(dd) < MIN_N_MODEL || sd(dd$outcome_z, na.rm = TRUE) == 0) {
    return(NULL)
  }

  fit <- lm(outcome_z ~ smoking_z + alcohol_z + bmi_z + sex_male + education_z, data = dd)

  broom::tidy(fit, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      Outcome = marker,
      Outcome_label = pretty_label(marker),
      N = nrow(dd)
    )
}

assoc_df <- bind_rows(lapply(AGING_VARS, fit_marker_lm))

if (nrow(assoc_df) == 0) {
  stop("No lifestyle/demographic association models were fit.")
}

assoc_df <- assoc_df %>%
  mutate(
    p_fdr = p.adjust(p.value, method = "fdr"),
    sig_label = case_when(
      is.na(p_fdr) ~ "",
      p_fdr < 0.001 ~ "***",
      p_fdr < 0.01 ~ "**",
      p_fdr < 0.05 ~ "*",
      TRUE ~ ""
    ),
    term_clean = case_when(
      term == "smoking_z" ~ "Smoking",
      term == "alcohol_z" ~ "Alcohol intake frequency",
      term == "bmi_z" ~ "Body mass index",
      term == "sex_male" ~ "Sex",
      term == "education_z" ~ "Education",
      TRUE ~ term
    ),
    Outcome_label = factor(Outcome_label, levels = pretty_label(AGING_VARS)),
    term_clean = factor(
      term_clean,
      levels = c("Smoking", "Alcohol intake frequency", "Body mass index", "Sex", "Education")
    )
  ) %>%
  select(
    Outcome,
    Outcome_label,
    Predictor = term_clean,
    N,
    beta = estimate,
    CI_low = conf.low,
    CI_high = conf.high,
    p_value = p.value,
    p_fdr,
    sig_label
  )

write_csv(
  assoc_df,
  file.path(OUT_DIR, "TableA2_lifestyle_demographic_associations.csv")
)

p_assoc <- ggplot(assoc_df, aes(x = Outcome_label, y = Predictor, fill = beta)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = sig_label), size = 4.8) +
  scale_fill_gradient2(
    low = "#3B4CC0",
    mid = "white",
    high = "#B40426",
    midpoint = 0,
    name = "Std. beta"
  ) +
  labs(
    title = "Lifestyle and demographic associations with aging measures and PRS",
    subtitle = paste0("Standardized linear models; FDR-significant associations marked with *, **, ***"),
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(
  filename = file.path(OUT_DIR, "TableA2_lifestyle_association_heatmap.png"),
  plot = p_assoc,
  width = 9,
  height = 5.5,
  dpi = 500
)

saveRDS(
  list(
    correlation_long = cor_long,
    association_table = assoc_df,
    settings = list(
      INPUT_RDS = INPUT_RDS,
      AGING_VARS = AGING_VARS,
      COR_METHOD = COR_METHOD,
      FDR_ALPHA = FDR_ALPHA
    )
  ),
  file.path(OUT_DIR, "correlation_association_results.rds")
)

cat("\nDONE: Correlation matrix and lifestyle/demographic association analyses completed.\n")
cat("Outputs written to:", OUT_DIR, "\n\n")
