# ==================================================================================
# 02_Interaction_Heatmaps_BioAge.R
# BioAge plotting + associations between aging markers / PRS and lifestyle covariates
#
# What this script produces:
# 1) BioAge pairwise association plot across aging measures / PRS (descriptive)
# 2) Heatmap of multivariable linear associations:
#    Each aging marker (incl. PRS) ~ smoking + alcohol + BMI + sex + education
#    with FDR correction across all tests
# ==================================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(broom)
  library(stringr)
  library(ggplot2)
})

# BioAge::plot_baa is used if available; script will still run without it.
HAS_BIOAGE <- requireNamespace("BioAge", quietly = TRUE)
if (HAS_BIOAGE) suppressPackageStartupMessages(library(BioAge))

# --------------------------
# USER SETTINGS 
# --------------------------

set.seed(20250101)

# Input: analysis table as .rds (recommended) OR load Biomarkers_complete before sourcing this script
INPUT_RDS <- "data/biomarkers_complete.rds"   

# Output directory
OUT_DIR <- "results/interaction_heatmaps"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Aging measures to include (ordered as in manuscript narrative)
AGING_VARS <- c("CA", "ProtAge", "MetaboAge", "PhenoAge", "KDM", "HD", "FI", "TL", "DDML_PRS_With_APOE")

# Column names for covariates (edit if your dataset uses different names)
SEX_COL <- "sex"               # expected factor-like (Female/Male or 0/1)
EDU_COL <- "education"         # expected categories: Low / Intermediate / High (or similar)
SMOKE_COL <- "smoking"
ALC_COL   <- "alcohol"
BMI_COL   <- "bmi"

# Education recode: Low vs (Intermediate+High) (as in your original script)
EDU_LOW_LABEL <- "Low"
EDU_HI_LABELS <- c("High", "Intermediate")

# Heatmap significance threshold (FDR)
FDR_ALPHA <- 0.05

# --------------------------
# LOAD DATA
# --------------------------

if (nzchar(INPUT_RDS)) {
  Biomarkers_complete <- readRDS(INPUT_RDS)
} else {
  if (!exists("Biomarkers_complete")) stop("No input found: set INPUT_RDS or load Biomarkers_complete before running.")
}

df0 <- Biomarkers_complete

# --------------------------
# BASIC VALIDATION
# --------------------------

needed_cols <- unique(c(AGING_VARS, SEX_COL, EDU_COL, SMOKE_COL, ALC_COL, BMI_COL))
missing_cols <- setdiff(needed_cols, names(df0))
if (length(missing_cols) > 0) {
  stop("Missing required columns in input data: ", paste(missing_cols, collapse = ", "))
}

# --------------------------
# 1) BioAge plot (descriptive associations among aging measures)
# --------------------------

agevar <- AGING_VARS
axis_type <- setNames(rep("float", length(agevar)), agevar)

label <- c(
  "CA"                 = "Chronological Age",
  "ProtAge"            = "ProtAge\nProteomic Age",
  "MetaboAge"          = "MetaboAge\nMetabolomic Age",
  "PhenoAge"           = "PhenoAge\nPhenotypic Age",
  "KDM"                = "KDM\nBiological Age",
  "HD"                 = "HD\nHomeostatic Dysregulation",
  "FI"                 = "FI\nFrailty Index",
  "TL"                 = "TL\nTelomere Length",
  "DDML_PRS_With_APOE" = "PRS\n(DDML, incl. APOE)"
)

if (HAS_BIOAGE) {
  # BioAge plot (writes to active device; we save to file)
  png(file.path(OUT_DIR, "bioage_plot_baa.png"), width = 2200, height = 1600, res = 220)
  BioAge::plot_baa(df0, agevar, label, axis_type)
  dev.off()
} else {
  message("BioAge package not installed; skipping plot_baa().")
}

# --------------------------
# 2) Prepare covariates 
# --------------------------

# Robust sex formatting
df1 <- df0 %>%
  mutate(
    sex_std = as.factor(.data[[SEX_COL]])
  )

# Education recode: Low vs (High+Intermediate)
df1 <- df1 %>%
  mutate(
    education_recode = case_when(
      .data[[EDU_COL]] %in% EDU_HI_LABELS ~ "High_Intermediate",
      .data[[EDU_COL]] == EDU_LOW_LABEL ~ "Low",
      TRUE ~ NA_character_
    ),
    education_recode = factor(education_recode, levels = c("High_Intermediate", "Low"))
  )

covariate_terms <- c("smoking", "alcohol", "bmi", "sex", "education_recode")

# Build a modeling dataset with harmonized names used below
df_model <- df1 %>%
  transmute(
    # Outcomes
    across(all_of(AGING_VARS), ~ suppressWarnings(as.numeric(.x))),
    # Covariates
    smoking = .data[[SMOKE_COL]],
    alcohol = .data[[ALC_COL]],
    bmi = suppressWarnings(as.numeric(.data[[BMI_COL]])),
    sex = sex_std,
    education_recode = education_recode
  )

# Drop rows missing covariates (keeps modeling consistent across outcomes)
df_model <- df_model %>%
  filter(
    !is.na(smoking) & !is.na(alcohol) & !is.na(bmi) & !is.na(sex) & !is.na(education_recode)
  )

# --------------------------
# 3) Fit multivariable linear models for each aging marker
# --------------------------

fit_one_marker <- function(marker_name, dat) {
  # marker must exist
  if (!marker_name %in% names(dat)) return(NULL)

  dd <- dat %>% filter(!is.na(.data[[marker_name]]))
  if (nrow(dd) < 200) return(NULL)  # basic stability threshold

  fml <- as.formula(paste0(marker_name, " ~ smoking + alcohol + bmi + sex + education_recode"))
  m <- lm(fml, data = dd)

  broom::tidy(m) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      Outcome = marker_name,
      N = nrow(dd)
    )
}

results <- lapply(AGING_VARS, fit_one_marker, dat = df_model)
assoc_df <- bind_rows(results)

if (nrow(assoc_df) == 0) stop("No models were fit. Check your data and column names.")

# --------------------------
# 4) FDR correction + labeling + term harmonization
# --------------------------

assoc_df <- assoc_df %>%
  mutate(
    p.adj.fdr = p.adjust(p.value, method = "fdr"),
    sig_label = ifelse(p.adj.fdr <= FDR_ALPHA, "★", ""),  # star for significant after FDR
    # Harmonize terms for plotting readability
    term_clean = case_when(
      str_detect(term, "^sex") ~ "sex_Male",
      term == "education_recodeLow" ~ "education_Low",
      term == "smoking" ~ "smoking",
      term == "alcohol" ~ "alcohol",
      term == "bmi" ~ "bmi",
      TRUE ~ term
    ),
    Outcome_clean = ifelse(Outcome == "DDML_PRS_With_APOE", "PRS", Outcome)
  )

# Order for heatmap
term_levels <- c("smoking", "alcohol", "bmi", "sex_Male", "education_Low")
outcome_levels <- c("CA", "ProtAge", "MetaboAge", "PhenoAge", "KDM", "HD", "FI", "TL", "PRS")

assoc_df <- assoc_df %>%
  mutate(
    term_clean = factor(term_clean, levels = term_levels),
    Outcome_clean = factor(Outcome_clean, levels = outcome_levels)
  )

# Save the association table
write.csv(assoc_df, file.path(OUT_DIR, "interaction_models_coefficients_fdr.csv"), row.names = FALSE)

# --------------------------
# 5) Heatmap of standardized effect sizes (recommended)
# To compare betas across outcomes with different scales, standardize outcomes.
# Here we approximate standardized effect using:
#   beta_std ≈ beta * sd(covariate) / sd(outcome)
# This is helpful for a cross-outcome heatmap.
# --------------------------

# Compute SDs for outcomes and covariates in the modeling dataset
sd_outcome <- sapply(outcome_levels, function(o) {
  # map back PRS
  col <- if (o == "PRS") "DDML_PRS_With_APOE" else o
  stats::sd(df_model[[col]], na.rm = TRUE)
})
sd_cov <- c(
  smoking = stats::sd(as.numeric(as.factor(df_model$smoking)), na.rm = TRUE),
  alcohol = stats::sd(as.numeric(as.factor(df_model$alcohol)), na.rm = TRUE),
  bmi = stats::sd(df_model$bmi, na.rm = TRUE),
  sex_Male = stats::sd(as.numeric(df_model$sex == levels(df_model$sex)[2]), na.rm = TRUE),
  education_Low = stats::sd(as.numeric(df_model$education_recode == "Low"), na.rm = TRUE)
)

assoc_df <- assoc_df %>%
  mutate(
    outcome_sd = sd_outcome[as.character(Outcome_clean)],
    cov_sd = sd_cov[as.character(term_clean)],
    beta_std = estimate * (cov_sd / outcome_sd)
  )

# --------------------------
# 6) Heatmap plot
# --------------------------

p <- ggplot(assoc_df, aes(x = Outcome_clean, y = term_clean, fill = beta_std)) +
  geom_tile(color = "white", linewidth = 0.5) +
  scale_fill_gradient2(midpoint = 0, low = "blue", mid = "white", high = "red", name = "Std. Beta") +
  geom_text(aes(label = sig_label), color = "black", size = 6) +
  labs(
    title = "Associations of Lifestyle/Sex/Education with Aging Measures and PRS",
    subtitle = paste0("Multivariable linear models; FDR < ", FDR_ALPHA, " marked with ★"),
    x = "Aging Measures / PRS",
    y = "Covariates"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(
  filename = file.path(OUT_DIR, "heatmap_interactions_standardized_beta.png"),
  plot = p, width = 9, height = 5.5, dpi = 300
)

cat("\nDONE ✅ BioAge plot + interaction heatmap completed.\n")
cat("Output folder:", OUT_DIR, "\n\n")
