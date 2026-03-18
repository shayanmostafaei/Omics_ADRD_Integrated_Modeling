# ========================================================================
# 02_Interaction_Heatmaps_BioAge.R
# Associations between aging markers/PRS and lifestyle covariates
#
# OUTPUTS:
# 1. Pairwise association/correlation plots between aging measures/PRS
# 2. Heatmap of multivariable standardized linear associations (with FDR)
# AUTHOR: Shayan Mostafaei, CONTRIBUTORS: [Add names]
# DATE CREATED: 2026-03-18
# ========================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(broom)
  library(stringr)
  library(ggplot2)
  library(readr)
  library(tidyr)
})

# Optional: BioAge plot (if available)
HAS_BIOAGE <- requireNamespace("BioAge", quietly = TRUE)
if (HAS_BIOAGE) suppressPackageStartupMessages(library(BioAge))

# --------------------------
# USER CONFIGURATION
# --------------------------

set.seed(20250101)
INPUT_RDS <- "data/biomarkers_complete.rds"
OUT_DIR <- file.path("results", "03_correlations_associations", "bioage_lifestyle_heatmap")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

AGING_VARS <- c("CA", "ProtAge", "MetaboAge", "PhenoAge", "KDM", "HD", "FI", "TL", "PRS_ADRD")
SEX_COL   <- "sex"
EDU_COL   <- "education"
SMOKE_COL <- "smoking"
ALC_COL   <- "alcohol_intake_frequency"
BMI_COL   <- "bmi"
EDU_LOW_LABEL <- "Low"
EDU_HI_LABELS <- c("High", "Intermediate")
FDR_ALPHA <- 0.05
MIN_N_MODEL <- 200

# --------------------------
# LOAD DATA AND VALIDATE INPUT
# --------------------------

if (!nzchar(INPUT_RDS) || !file.exists(INPUT_RDS))
  stop("❌ Input RDS not found. Set INPUT_RDS correctly: ", INPUT_RDS)
df0 <- readRDS(INPUT_RDS)

needed_cols <- unique(c(AGING_VARS, SEX_COL, EDU_COL, SMOKE_COL, ALC_COL, BMI_COL))
missing_cols <- setdiff(needed_cols, names(df0))
if (length(missing_cols) > 0)
  stop("❌ Missing required columns: ", paste(missing_cols, collapse = ", "))

# --------------------------
# HELPER FUNCTIONS
# --------------------------

#' Convert ordinal data to numeric codes, preserving order
to_numeric_code <- function(x) {
  if (is.numeric(x)) return(x)
  if (is.logical(x)) return(as.numeric(x))
  if (is.factor(x)) return(as.numeric(x))
  if (is.character(x)) return(as.numeric(as.factor(x)))
  suppressWarnings(as.numeric(x))
}

#' Standardize column for regression (z-score)
zscale <- function(x) as.numeric(scale(x))

#' Fit fully standardized linear model for a single aging marker
fit_one_marker <- function(marker_name, dat) {
  if (!marker_name %in% names(dat)) return(NULL)
  dd <- dat %>% filter(!is.na(.data[[marker_name]])) %>% mutate(outcome_z = zscale(.data[[marker_name]]))
  
  if (nrow(dd) < MIN_N_MODEL) return(NULL)
  if (stats::sd(dd$outcome_z, na.rm = TRUE) == 0) return(NULL)

  dd <- dd %>% mutate(
    smoking_z = zscale(smoking_num),
    alcohol_z = zscale(alcohol_num)
  )

  m <- lm(outcome_z ~ smoking_z + alcohol_z + bmi_z + sex_male + edu_low, data = dd)
  broom::tidy(m) %>% filter(term != "(Intercept)") %>% mutate(Outcome = marker_name, N = nrow(dd))
}

# --------------------------
# 1) Pairwise Correlation Plot (Descriptive)
# --------------------------

label <- c(
  "CA"        = "Chronological Age",
  "ProtAge"   = "ProtAge\nProteomic Age",
  "MetaboAge" = "MetaboAge\nMetabolomic Age",
  "PhenoAge"  = "PhenoAge\nPhenotypic Age",
  "KDM"       = "KDM\nBiological Age",
  "HD"        = "HD\nHomeostatic Dysregulation",
  "FI"        = "FI\nFrailty Index",
  "TL"        = "Telomere Length",
  "PRS_ADRD"  = "PRS\n(DDML, incl. APOE)"
)

if (HAS_BIOAGE) {
  png(file.path(OUT_DIR, "bioage_plot_baa.png"), width = 2200, height = 1600, res = 220)
  BioAge::plot_baa(df0, AGING_VARS, label,
                   axis_type = setNames(rep("float", length(AGING_VARS)), AGING_VARS))
  dev.off()
} else {
  message("BioAge package not installed; creating correlation heatmap as fallback.")
  corr_df <- df0 %>%
    select(all_of(AGING_VARS)) %>%
    mutate(across(everything(), ~ suppressWarnings(as.numeric(.x)))) %>%
    cor(use = "pairwise.complete.obs") %>%
    as.data.frame() %>%
    tibble::rownames_to_column("Var1") %>%
    pivot_longer(-Var1, names_to = "Var2", values_to = "Correlation")

  p_corr <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile(color = "white", linewidth = 0.3) +
    coord_equal() +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Pairwise correlations among aging measures and PRS", x = NULL, y = NULL)

  ggsave(file.path(OUT_DIR, "pairwise_correlations_heatmap.png"), plot = p_corr, width = 8.5, height = 7.5, dpi = 300)
  write_csv(corr_df, file.path(OUT_DIR, "pairwise_correlations_long.csv"))
}

# --------------------------
# 2) Covariate Harmonization & Modeling Dataset Prep
# --------------------------

df1 <- df0 %>%
  mutate(
    sex_std = as.factor(.data[[SEX_COL]]),
    education_recode = case_when(
      .data[[EDU_COL]] %in% EDU_HI_LABELS ~ "High_Intermediate",
      .data[[EDU_COL]] == EDU_LOW_LABEL ~ "Low",
      TRUE ~ NA_character_
    ),
    education_recode = factor(education_recode, levels = c("High_Intermediate", "Low"))
  )

df_model <- df1 %>%
  transmute(
    across(all_of(AGING_VARS), ~ suppressWarnings(as.numeric(.x))),
    smoking = .data[[SMOKE_COL]],
    alcohol_intake_frequency = .data[[ALC_COL]],
    bmi = suppressWarnings(as.numeric(.data[[BMI_COL]])),
    sex = sex_std,
    education_recode = education_recode
  ) %>%
  filter(
    !is.na(smoking) &
    !is.na(alcohol_intake_frequency) &
    !is.na(bmi) &
    !is.na(sex) &
    !is.na(education_recode)
  )

df_std <- df_model %>%
  mutate(
    smoking_num = to_numeric_code(smoking),
    alcohol_num = to_numeric_code(alcohol_intake_frequency),
    bmi_z = zscale(bmi),
    sex_male = as.numeric(sex == levels(sex)[2]),
    edu_low = as.numeric(education_recode == "Low")
  )

# --------------------------
# 3) Standardized Linear Modeling (all outcomes)
# --------------------------

results <- lapply(AGING_VARS, fit_one_marker, dat = df_std)
assoc_df <- bind_rows(results)
if (nrow(assoc_df) == 0)
  stop("No models were fit. Check your data and column names.")

# --------------------------
# 4) FDR Correction + Labeling
# --------------------------

assoc_df <- assoc_df %>%
  mutate(
    p_fdr = p.adjust(p.value, method = "fdr"),
    sig_label = ifelse(p_fdr <= FDR_ALPHA, "★", ""),
    term_clean = case_when(
      term == "smoking_z" ~ "smoking",
      term == "alcohol_z" ~ "alcohol intake frequency",
      term == "bmi_z" ~ "BMI",
      term == "sex_male" ~ "sex (male)",
      term == "edu_low" ~ "education (low)",
      TRUE ~ term
    ),
    Outcome_clean = Outcome
  )

term_levels <- c("smoking", "alcohol intake frequency", "BMI", "sex (male)", "education (low)")
outcome_levels <- AGING_VARS

assoc_df <- assoc_df %>%
  mutate(
    term_clean = factor(term_clean, levels = term_levels),
    Outcome_clean = factor(Outcome_clean, levels = outcome_levels)
  )

write_csv(assoc_df, file.path(OUT_DIR, "lifestyle_associations_standardized_betas_fdr.csv"))

# Wide matrix export
assoc_wide <- assoc_df %>%
  select(Outcome_clean, term_clean, estimate, p_fdr) %>%
  mutate(Outcome_clean = as.character(Outcome_clean),
         term_clean = as.character(term_clean)) %>%
  pivot_wider(names_from = Outcome_clean, values_from = estimate)

write_csv(assoc_wide, file.path(OUT_DIR, "lifestyle_associations_standardized_betas_wide.csv"))

# --------------------------
# 5) Heatmap Plot
# --------------------------

p <- ggplot(assoc_df, aes(x = Outcome_clean, y = term_clean, fill = estimate)) +
  geom_tile(color = "white", linewidth = 0.5) +
  scale_fill_gradient2(midpoint = 0, low = "#3B4CC0", mid = "white", high = "#B40426", name = "Std. Beta") +
  geom_text(aes(label = sig_label), color = "black", size = 6) +
  labs(
    title = "Associations of lifestyle factors, sex, and education with aging measures and PRS",
    subtitle = paste0("Standardized linear models; FDR < ", FDR_ALPHA, " marked with ★"),
    x = "Aging measures / PRS",
    y = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(
  file.path(OUT_DIR, "heatmap_lifestyle_associations_standardized_beta.png"),
  plot = p, width = 9, height = 5.5, dpi = 300
)

cat("\n✅ DONE: BioAge pairwise plot and lifestyle association heatmap completed.\n")
cat("Output folder:", OUT_DIR, "\n\n")
