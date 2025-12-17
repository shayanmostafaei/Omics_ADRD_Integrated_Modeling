# ==================================================================================
# 05_Survival_Competing_Risks_Analysis.R
# Competing-risk time-to-event analysis for ADRD with death as a competing event.
#
# Fully aligned with the workflow:
# - Uses the SAME predictor set as the final integrated model (Model5 / Add MetaboAge)
# - Stratified 70/30 split (same seed + createDataPartition on Dementia_status)
# - Fine–Gray model (cause = ADRD; competing event = death)
# - Predict absolute risk at 5 and 9 years
# - Risk stratification: top 25% (high-risk) vs bottom 75% (low-risk) based on predicted risk
# - CIF curves + subdistribution hazard ratio (sHR) for High vs Low
# - Forest plot for key predictors (sHR per 1 SD increase for numeric predictors)
#
# Inputs:
#   data/biomarkers_complete.rds 
#
# Outputs:
#   results/survival_competing_risk/
#     - finegray_model_train.rds
#     - test_predicted_risks_5y_9y.csv
#     - cif_curve_data_9y_group.csv
#     - cif_plot_9y_group.png
#     - riskgroup_shr_high_vs_low.csv
#     - finegray_forestplot.png
# ==================================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(survival)
  library(riskRegression)
  library(cmprsk)
  library(ggplot2)
  library(scales)
  library(forestplot)
  library(grid)
})

# --------------------------
# USER SETTINGS 
# --------------------------

set.seed(20250101)

INPUT_RDS <- "data/biomarkers_complete.rds"  
OUT_DIR <- "results/survival_competing_risk"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Required columns (edit if your dataset uses different names)
ID_COL <- "f.eid"

OUTCOME_COL <- "Dementia_status"        # 0/1

# Time-to-event columns (days recommended)
T_DEM_COL <- "Time_to_Dementia"         # time to ADRD diagnosis (days)
DEATH_COL <- "death_status"             # 0/1
T_DEATH_COL <- "time_to_death"          # time to death (days)

# Predictors aligned with Model 5 (final integrated model)
AGE_COL <- "CA"
SEX_COL <- "sex"
SMOKE_COL <- "smoking"
ALC_COL <- "alcohol"
BMI_COL <- "bmi"
EDU_COL <- "education"
PRS_COL <- "DDML_PRS_With_APOE"
PHENO_COL <- "PhenoAge"
FI_COL <- "FI"
TL_COL <- "TL"
PROTAGE_COL <- "ProtAge"
METABOAGE_COL <- "MetaboAge"

PC_COLS <- paste0("PC", 1:10) # Genetic PCs

# Horizon times (days): 5 and 9 years
TIME_5Y <- 1826
TIME_9Y <- 3287

# Risk stratification (top 25% vs bottom 75%)
HIGH_RISK_Q <- 0.75

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
# VALIDATION
# --------------------------

required_cols <- c(
  OUTCOME_COL, T_DEM_COL, DEATH_COL, T_DEATH_COL,
  AGE_COL, SEX_COL, SMOKE_COL, ALC_COL, BMI_COL, EDU_COL,
  PRS_COL, PHENO_COL, FI_COL, TL_COL, PROTAGE_COL, METABOAGE_COL
)
missing_cols <- setdiff(required_cols, names(df0))
if (length(missing_cols) > 0) stop("Missing required columns: ", paste(missing_cols, collapse = ", "))

if (!ID_COL %in% names(df0)) {
  message("NOTE: ID column '", ID_COL, "' not found. Proceeding without IDs.")
  df0[[ID_COL]] <- seq_len(nrow(df0))
}

PC_COLS <- PC_COLS[PC_COLS %in% names(df0)]

# --------------------------
# 1) Build competing-risk outcome (event, time)
# event coding:
#   0 = censored
#   1 = ADRD
#   2 = death before ADRD (competing)
# time = min(time to ADRD, time to death), depending on which happens first
# --------------------------

df1 <- df0 %>%
  transmute(
    id = as.character(.data[[ID_COL]]),

    # classification outcome (used only for the 70/30 split alignment)
    y_class = suppressWarnings(as.numeric(as.character(.data[[OUTCOME_COL]]))),
    y_class = ifelse(y_class == 1, 1, 0),

    # times/events
    t_dem = suppressWarnings(as.numeric(.data[[T_DEM_COL]])),
    dem_event = ifelse(y_class == 1, 1, 0),
    death_event = suppressWarnings(as.numeric(as.character(.data[[DEATH_COL]]))),
    death_event = ifelse(death_event == 1, 1, 0),
    t_death = suppressWarnings(as.numeric(.data[[T_DEATH_COL]])),

    # predictors (raw)
    CA = suppressWarnings(as.numeric(.data[[AGE_COL]])),
    sex = as.factor(.data[[SEX_COL]]),
    smoking = as.factor(.data[[SMOKE_COL]]),
    alcohol = as.factor(.data[[ALC_COL]]),
    bmi = suppressWarnings(as.numeric(.data[[BMI_COL]])),
    education = as.factor(.data[[EDU_COL]]),

    PRS = suppressWarnings(as.numeric(.data[[PRS_COL]])),
    PhenoAge = suppressWarnings(as.numeric(.data[[PHENO_COL]])),
    FI = suppressWarnings(as.numeric(.data[[FI_COL]])),
    TL = suppressWarnings(as.numeric(.data[[TL_COL]])),
    ProtAge = suppressWarnings(as.numeric(.data[[PROTAGE_COL]])),
    MetaboAge = suppressWarnings(as.numeric(.data[[METABOAGE_COL]])),

    across(all_of(PC_COLS), ~ suppressWarnings(as.numeric(.x)))
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
  )

# --------------------------
# 2) Stratified 70/30 split 
# --------------------------

train_idx <- caret::createDataPartition(df1$y_class, p = 0.70, list = FALSE)
train_df <- df1[train_idx, , drop = FALSE]
test_df  <- df1[-train_idx, , drop = FALSE]

# --------------------------
# 3) Train-only preprocessing
# --------------------------

numeric_cols <- c("CA","bmi","PRS","PhenoAge","FI","TL","ProtAge","MetaboAge", PC_COLS)
numeric_cols <- numeric_cols[numeric_cols %in% names(df1)]

pp <- caret::preProcess(train_df[, numeric_cols, drop = FALSE], method = c("medianImpute", "center", "scale"))

train_df_pp <- train_df
test_df_pp  <- test_df

train_df_pp[, numeric_cols] <- predict(pp, train_df[, numeric_cols, drop = FALSE])
test_df_pp[, numeric_cols]  <- predict(pp, test_df[, numeric_cols, drop = FALSE])

mode_impute <- function(x) {
  x <- as.factor(x)
  tab <- table(x, useNA = "no")
  if (length(tab) == 0) return(x)
  mode_val <- names(tab)[which.max(tab)]
  x[is.na(x)] <- mode_val
  droplevels(x)
}

cat_cols <- c("sex","smoking","alcohol","education")

for (v in cat_cols) {
  train_df_pp[[v]] <- mode_impute(train_df_pp[[v]])
  test_df_pp[[v]] <- as.factor(test_df_pp[[v]])
  train_mode <- levels(train_df_pp[[v]])[1]
  if (any(is.na(test_df_pp[[v]]))) test_df_pp[[v]][is.na(test_df_pp[[v]])] <- train_mode
  test_df_pp[[v]] <- factor(test_df_pp[[v]], levels = levels(train_df_pp[[v]]))
}

# --------------------------
# 4) Fine–Gray model on TRAIN (cause=1 ADRD; competing=2 death)
# --------------------------

rhs_terms <- c(
  "CA", "sex", "smoking", "alcohol", "bmi", "education",
  "PRS", "PhenoAge", "FI", "TL", "ProtAge", "MetaboAge",
  PC_COLS
)

fg_formula <- as.formula(paste0("Hist(time, event) ~ ", paste(rhs_terms, collapse = " + ")))

fg_fit <- riskRegression::FGR(
  formula = fg_formula,
  data = train_df_pp,
  cause = 1
)

saveRDS(fg_fit, file.path(OUT_DIR, "finegray_model_train.rds"))

# --------------------------
# 5) Predict absolute ADRD risk at 5 and 9 years in TEST
# --------------------------

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

write.csv(pred_df, file.path(OUT_DIR, "test_predicted_risks_5y_9y.csv"), row.names = FALSE)

# --------------------------
# 6) Risk stratification: High (top 25%) vs Low (bottom 75%): Stratification on 9-year risk 
# --------------------------

q75_9y <- as.numeric(stats::quantile(pred_df$risk_9y, probs = HIGH_RISK_Q, na.rm = TRUE))
pred_df <- pred_df %>%
  mutate(
    risk_group_9y = factor(ifelse(risk_9y > q75_9y, "High", "Low"), levels = c("Low","High"))
  )

# --------------------------
# 7) CIF curves in TEST by risk group (9-year risk grouping)
# --------------------------

ci_obj <- cmprsk::cuminc(
  ftime = pred_df$time,
  fstatus = pred_df$event,
  group = pred_df$risk_group_9y
)

# Extract cause 1 CIF for Low/High
cause1_names <- names(ci_obj)[grepl(" 1$", names(ci_obj))]

cif_list <- lapply(cause1_names, function(nm) {
  e <- ci_obj[[nm]]
  grp <- sub(" 1$", "", nm)
  data.frame(
    time = e$time,
    est = as.numeric(e$est),
    var = as.numeric(e$var),
    group = grp,
    stringsAsFactors = FALSE
  )
})

cif_df <- bind_rows(cif_list) %>%
  mutate(
    se = sqrt(var),
    lower = pmax(0, est - 1.96 * se),
    upper = pmin(1, est + 1.96 * se),
    group = factor(group, levels = c("Low","High"))
  )

write.csv(cif_df, file.path(OUT_DIR, "cif_curve_data_9y_group.csv"), row.names = FALSE)

# --------------------------
# 8) sHR for High vs Low (Fine–Gray via crr on TEST)
# --------------------------

mm <- model.matrix(~ risk_group_9y, data = pred_df)[, -1, drop = FALSE]
crr_fit <- cmprsk::crr(
  ftime = pred_df$time,
  fstatus = pred_df$event,
  cov1 = mm,
  failcode = 1,
  cencode = 0
)

coef <- as.numeric(crr_fit$coef)
se_coef <- sqrt(diag(crr_fit$var))
sHR <- exp(coef)
lower <- exp(coef - 1.96 * se_coef)
upper <- exp(coef + 1.96 * se_coef)
pval <- 2 * pnorm(-abs(coef / se_coef))

shr_df <- data.frame(
  Comparison = "High vs Low (based on 9-year predicted risk; TEST set)",
  sHR = sHR,
  CI_low = lower,
  CI_high = upper,
  p_value = pval,
  stringsAsFactors = FALSE
)

write.csv(shr_df, file.path(OUT_DIR, "riskgroup_shr_high_vs_low.csv"), row.names = FALSE)

shr_text <- paste0(
  "sHR (High vs Low) = ", sprintf("%.2f", sHR),
  " (95% CI ", sprintf("%.2f", lower), "–", sprintf("%.2f", upper),
  "), P ", ifelse(pval < 0.001, "< 0.001", sprintf("= %.3f", pval))
)

# --------------------------
# 9) Plot CIF curves (TEST; grouped by 9-year predicted risk)
# --------------------------

p_cif <- ggplot(cif_df, aes(x = time, y = est, color = group, fill = group)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.18, color = NA) +
  geom_step(linewidth = 1) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Cumulative Incidence of ADRD by Predicted Risk Group (TEST set)",
    subtitle = shr_text,
    x = "Time (days)",
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

ggsave(file.path(OUT_DIR, "cif_plot_9y_group.png"), p_cif, width = 8, height = 6, dpi = 300)

# --------------------------
# 10) Forest plot of Fine–Gray sHR for key predictors (TRAIN model)
# --------------------------

fg_sum <- summary(fg_fit)

fg_tbl <- NULL
try({
  coef_mat <- fg_sum$coef
  conf_mat <- fg_sum$conf.int
  fg_tbl <- data.frame(
    Variable = rownames(coef_mat),
    sHR = conf_mat[, "exp(coef)"],
    CI_low = conf_mat[, "2.5%"],
    CI_high = conf_mat[, "97.5%"],
    p_value = coef_mat[, "p-value"],
    stringsAsFactors = FALSE
  )
}, silent = TRUE)

if (is.null(fg_tbl)) {
  cf <- coef(fg_fit)
  ci <- suppressWarnings(confint(fg_fit))
  fg_tbl <- data.frame(
    Variable = names(cf),
    sHR = exp(cf),
    CI_low = exp(ci[, 1]),
    CI_high = exp(ci[, 2]),
    p_value = NA_real_,
    stringsAsFactors = FALSE
  )
}

keep_vars <- c("CA", "PRS", "PhenoAge", "FI", "TL", "ProtAge", "MetaboAge")
fg_tbl$Variable <- gsub("^PRS$", "PRS", fg_tbl$Variable)  # already
fg_tbl$Variable <- ifelse(fg_tbl$Variable == "PRS", "PRS", fg_tbl$Variable)

fg_tbl$Variable <- ifelse(grepl("PRS", fg_tbl$Variable), "PRS", fg_tbl$Variable)

fg_tbl_plot <- fg_tbl %>%
  mutate(
    Variable = case_when(
      Variable == "PRS" ~ "PRS",
      TRUE ~ Variable
    )
  ) %>%
  filter(Variable %in% keep_vars) %>%
  arrange(desc(sHR))

if (nrow(fg_tbl_plot) > 0) {
  fg_tbl_plot <- fg_tbl_plot %>%
    mutate(
      FDR = p.adjust(p_value, method = "fdr"),
      sHR_label = sprintf("%.2f", sHR),
      CI_label = paste0("(", sprintf("%.2f", CI_low), "–", sprintf("%.2f", CI_high), ")"),
      FDR_label = ifelse(is.na(FDR), "", formatC(FDR, format = "e", digits = 2))
    )

  tabletext <- cbind(
    c("Variable", fg_tbl_plot$Variable),
    c("sHR", fg_tbl_plot$sHR_label),
    c("95% CI", fg_tbl_plot$CI_label),
    c("FDR", fg_tbl_plot$FDR_label)
  )

  png(file.path(OUT_DIR, "finegray_forestplot.png"), width = 2200, height = 1400, res = 200)
  forestplot::forestplot(
    labeltext = tabletext,
    mean  = c(NA, fg_tbl_plot$sHR),
    lower = c(NA, fg_tbl_plot$CI_low),
    upper = c(NA, fg_tbl_plot$CI_high),
    zero = 1,
    boxsize = 0.2,
    lineheight = "auto",
    col = forestplot::fpColors(box = "black", line = "black", zero = "gray30"),
    xlab = "Subdistribution Hazard Ratio (sHR) per 1 SD increase (95% CI)",
    title = "Fine–Gray Competing Risk Model (TRAIN set): Key Predictors"
  )
  dev.off()

  write.csv(fg_tbl_plot, file.path(OUT_DIR, "finegray_key_predictors_table.csv"), row.names = FALSE)
} else {
  message("Forest plot skipped: no matching variables found in Fine–Gray output.")
}

cat("\nDONE ✅ Survival + competing risk analysis completed.\n")
cat("Model fit on TRAIN; risks/CIF/sHR computed on TEST for leakage-safe evaluation.\n")
cat("Output folder:", OUT_DIR, "\n\n")
