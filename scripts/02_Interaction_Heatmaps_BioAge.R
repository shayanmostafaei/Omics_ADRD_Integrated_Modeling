# ==================================================================================
# 02_Interaction_Heatmaps_BioAge.R
# BioAge plotting and associations between aging markers / PRS and lifestyle covariates
# ==================================================================================

library(BioAge)
library(broom)
library(dplyr)
library(stringr)
library(ggplot2)

# ----------------------------------------------------------------------------------
# 1. Figure 3: BioAge plot
# ----------------------------------------------------------------------------------

# Define age-related variables in the desired order
agevar <- c("CA", "ProtAge", "MetaboAge", "PhenoAge", 
            "KDM", "HD", "FI", "TL", "DDML_PRS_With_APOE")

# Axis types (all continuous floats)
axis_type <- setNames(rep("float", length(agevar)), agevar)

# Labels for plotting with line breaks
label <- c(
  "CA"                  = "Chronological Age",
  "ProtAge"             = "ProtAge\nProteomic Age",
  "MetaboAge"           = "MetaboAge\nMetabolomic Age",
  "PhenoAge"            = "PhenoAge\nLevine\nPhenotypic Age",
  "KDM"                 = "KDM\nModified-KDM\nBiological Age",
  "HD"                  = "HD\nHomeostatic Dysregulation",
  "FI"                  = "FI\nFrailty Index",
  "TL"                  = "TL\nTelomere Length",
  "DDML_PRS_With_APOE"  = "Polygenic Risk Score"
)

# Plot BioAge associations
plot_baa(Biomarkers_complete, agevar, label, axis_type)

# ----------------------------------------------------------------------------------
# 2. Associations (interaction) between aging markers / PRS and lifestyle covariates
# ----------------------------------------------------------------------------------

aging_vars <- agevar

# Recode education: Low vs (High + Intermediate)
Biomarkers_complete <- Biomarkers_complete %>%
    mutate(education_recode = case_when(
        education %in% c("High", "Intermediate") ~ "High_Intermediate",
        education == "Low" ~ "Low"
    )) %>%
    mutate(education_recode = factor(education_recode, 
                                     levels = c("High_Intermediate", "Low")))

covariates <- c("smoking", "alcohol", "bmi", "sex", "education_recode")

# Fit linear models for each aging variable
results <- lapply(aging_vars, function(marker) {
    formula_str <- paste(marker, "~", paste(covariates, collapse = " + "))
    model <- lm(as.formula(formula_str), data = Biomarkers_complete)
    
    tidy(model) %>%
        filter(term %in% c("smoking", "alcohol", "bmi") |
               str_detect(term, "sex") |
               str_detect(term, "education_recode")) %>%
        mutate(Outcome = marker)
})

# ----------------------------------------------------------------------------------
# 3. FDR Adjustment and Significance Labeling
# ----------------------------------------------------------------------------------
assoc_df <- bind_rows(results) %>%
    mutate(
        p.adj.fdr = p.adjust(p.value, method = "fdr"),
        sig_label = ifelse(p.adj.fdr > 0.05, "×", ""),
        term = case_when(
            term == "sexMen" ~ "sex_Men",
            term == "education_recodeLow" ~ "education_Low",
            TRUE ~ term
        ),
        Outcome = ifelse(Outcome == "DDML_PRS_With_APOE", "PRS", Outcome)
    )

# Order terms for heatmap
assoc_df <- assoc_df %>%
    mutate(term = factor(term, levels = c("smoking", "alcohol", "bmi", "sex_Men", "education_Low")))

# ----------------------------------------------------------------------------------
# 4. Heatmap Plot
# ----------------------------------------------------------------------------------
ggplot(assoc_df, aes(x = Outcome, y = term, fill = estimate)) +
    geom_tile(color = "white", linewidth = 0.5) +
    scale_fill_gradient2(
        midpoint = 0, low = "blue", mid = "white", high = "red",
        name = "Beta"
    ) +
    geom_text(aes(label = sig_label), color = "black", size = 6) +
    labs(
        title = "Heatmap of Effect Sizes (Beta coefficients)",
        x = "Aging Measures",
        y = "Covariates"
    ) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14), 
        axis.text.y = element_text(size = 14),
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14)
    )
