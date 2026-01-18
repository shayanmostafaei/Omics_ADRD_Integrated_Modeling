# ==================================================================================
# 05_Survival_Competing_Risks_Analysis.R
# Competing-risk time-to-event analysis for ADRD with death as a competing event.
#
# Aligned with manuscript:
# - Time-to-event is earliest of ADRD, death, or censoring (end of follow-up)
# - Units harmonized to DAYS (Time_to_Dementia assumed days; death/follow-up often years)
# - Fine–Gray model trained on TRAIN split; absolute risk predicted on TEST split
# - Primary stratification uses 5-year predicted risk (top 25% vs bottom 75%) for Figure 5
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
})

# --------------------------
# USER SETTINGS
# --------------------------
set.seed(20250101)

INPUT_RDS <- "data/biomarkers_complete.rds"
OUT_DIR <- "results/survival_competing_risk"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

ID_COL <- "f.eid"

OUTCOME_COL <- "Dementia_status"        # 0/1
T_DEM_COL <- "Time_to_Dementia"         # ADRD time (DAYS)
DEATH_COL <- "death_status"             # 0/1
T_DEATH_COL <- "time_to_death"          # often YEARS in your dataset
CENS_COL <- "length_followup"           # often YEARS in your dataset

# Flags for time unit conversion (set these to match your dataset)
T_DEATH_IS_YEARS <- TRUE
CENS_IS_YEARS <- TRUE

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
PC_COLS <- paste0("PC", 1:10) # optional

# Horizons (DAYS)
TIME_5Y <- 5 * 365.25
TIME_9Y <- 9 * 365.25

HIGH_RISK_Q <- 0.75

# --------------------------
# LOAD DATA
# --------------------------
Biomarkers_complete <- readRDS(INPUT_RDS)
df0 <- Biomarkers_complete

# --------------------------
# VALIDATION
# --------------------------
required_cols <- c(
  OUTCOME_COL, T_DEM_COL, DEATH_COL, T_DEATH_COL, CENS_COL,
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
# 1) Build dataset + construct competing-risk time/event (DAYS)
# event: 0=censor, 1=ADRD, 2=death
# time: min(t_dem(if ADRD), t_death(if death), t_cens)
# --------------------------
df1 <- df0 %>%
  transmute(
    id = as.character(.data[[ID_COL]]),

    y_class = as.integer(as.character(.data[[OUTCOME_COL]])),
    y_class = ifelse(is.na(y_class), NA_integer_, ifelse(y_class == 1, 1, 0)),

    # raw times
    t_dem_raw   = suppressWarnings(as.numeric(.data[[T_DEM_COL]])),     # assumed DAYS
    death_event = as.integer(as.character(.data[[DEATH_COL]])),
    death_event = ifelse(is.na(death_event), 0L, ifelse(death_event == 1, 1L, 0L)),
    t_death_raw = suppressWarnings(as.numeric(.data[[T_DEATH_COL]])),   # often YEARS
    t_cens_raw  = suppressWarnings(as.numeric(.data[[CENS_COL]])),      # often YEARS

    # predictors
    CA = suppressWarnings(as.numeric(.data[[AGE_COL]])),
    sex = factor(.data[[SEX_COL]]),
    smoking = factor(.data[[SMOKE_COL]]),
    alcohol = factor(.data[[ALC_COL]]),
    bmi = suppressWarnings(as.numeric(.data[[BMI_COL]])),
    education = factor(.data[[EDU_COL]]),

    PRS = suppressWarnings(as.numeric(.data[[PRS_COL]])),
    PhenoAge = suppressWarnings(as.numeric(.data[[PHENO_COL]])),
    FI = suppressWarnings(as.numeric(.data[[FI_COL]])),
    TL = suppressWarnings(as.numeric(.data[[TL_COL]])),
    ProtAge = suppressWarnings(as.numeric(.data[[PROTAGE_COL]])),
    MetaboAge = suppressWarnings(as.numeric(.data[[METABOAGE_COL]])),

    across(all_of(PC_COLS), ~ suppressWarnings(as.numeric(.x)))
  ) %>%
  filter(!is.na(y_class))

# Convert times to DAYS consistently
df1 <- df1 %>%
  mutate(
    t_dem_days = t_dem_raw,  # assumed already days

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

    # dementia time is only "active" if y_class==1
    t_dem_event_days = ifelse(
      y_class == 1 & !is.na(t_dem_days),
      t_dem_days,
      Inf
    ),

    # observed time = earliest of event or censoring
    time = pmin(t_dem_event_days, t_death_days, t_cens_days, na.rm = TRUE),

    event = case_when(
      is.finite(t_dem_event_days) & t_dem_event_days <= t_death_days & t_dem_event_days <= t_cens_days ~ 1L,
      is.finite(t_death_days)     & t_death_days     <  t_dem_event_days & t_death_days <= t_cens_days ~ 2L,
      TRUE ~ 0L
    )
  ) %>%
  filter(is.finite(time), !is.na(time), time > 0)

# quick checks
message("Event counts (0=cens,1=ADRD,2=death):")
print(table(df1$event))

message("Min time among ADRD events (days): ", min(df1$time[df1$event == 1], na.rm = TRUE))
message("Min time among death events (days): ", min(df1$time[df1$event == 2], na.rm = TRUE))

# --------------------------
# 2) Stratified 70/30 split by Dementia_status (classification outcome)
# --------------------------
train_idx <- caret::createDataPartition(df1$y_class, p = 0.70, list = FALSE)
train_df <- df1[train_idx, , drop = FALSE]
test_df  <- df1[-train_idx, , drop = FALSE]

# --------------------------
# 3) Train-only preprocessing
# numeric: median impute + center/scale (=> sHR per 1 SD for numeric)
# categorical: mode impute from TRAIN and apply to TEST
# --------------------------
numeric_cols <- c("CA","bmi","PRS","PhenoAge","FI","TL","ProtAge","MetaboAge", PC_COLS)
numeric_cols <- numeric_cols[numeric_cols %in% names(df1)]

pp <- caret::preProcess(train_df[, numeric_cols, drop = FALSE],
                        method = c("medianImpute","center","scale"))

train_df_pp <- train_df
test_df_pp  <- test_df
train_df_pp[, numeric_cols] <- predict(pp, train_df[, numeric_cols, drop = FALSE])
test_df_pp[, numeric_cols]  <- predict(pp, test_df[, numeric_cols, drop = FALSE])

mode_train <- function(x) {
  x <- as.factor(x)
  tab <- table(x, useNA = "no")
  if (length(tab) == 0) return(NA_character_)
  names(tab)[which.max(tab)]
}

cat_cols <- c("sex","smoking","alcohol","education")
for (v in cat_cols) {
  # TRAIN
  train_df_pp[[v]] <- as.factor(train_df_pp[[v]])
  m <- mode_train(train_df_pp[[v]])
  train_df_pp[[v]][is.na(train_df_pp[[v]])] <- m
  train_df_pp[[v]] <- droplevels(train_df_pp[[v]])

  # TEST: align levels to TRAIN
  test_df_pp[[v]] <- as.factor(test_df_pp[[v]])
  test_df_pp[[v]][is.na(test_df_pp[[v]])] <- m
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
rhs_terms <- rhs_terms[rhs_terms %in% names(train_df_pp)]

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
    id, event, time, y_class,
    risk_5y = risk_5y,
    risk_9y = risk_9y
  )

write.csv(pred_df, file.path(OUT_DIR, "test_predicted_risks_5y_9y.csv"), row.names = FALSE)

# --------------------------
# 6) Risk stratification for Figure 5: based on 5-year predicted risk
# --------------------------
q75_5y <- as.numeric(stats::quantile(pred_df$risk_5y, probs = HIGH_RISK_Q, na.rm = TRUE))
pred_df <- pred_df %>%
  mutate(
    risk_group_5y = factor(ifelse(risk_5y > q75_5y, "High", "Low"),
                           levels = c("Low","High"))
  )

# --------------------------
# 7) CIF curves in TEST by risk group (5-year risk grouping)
# --------------------------
ci_obj <- cmprsk::cuminc(
  ftime = pred_df$time,
  fstatus = pred_df$event,
  group = pred_df$risk_group_5y
)

cause1_names <- names(ci_obj)[grepl(" 1$", names(ci_obj))]

cif_df <- bind_rows(lapply(cause1_names, function(nm) {
  e <- ci_obj[[nm]]
  grp <- sub(" 1$", "", nm)
  data.frame(
    time = e$time,
    est = as.numeric(e$est),
    var = as.numeric(e$var),
    group = grp,
    stringsAsFactors = FALSE
  )
})) %>%
  mutate(
    se = sqrt(var),
    lower = pmax(0, est - 1.96 * se),
    upper = pmin(1, est + 1.96 * se),
    group = factor(group, levels = c("Low","High"))
  )

write.csv(cif_df, file.path(OUT_DIR, "cif_curve_data_5y_group.csv"), row.names = FALSE)

# --------------------------
# 8) sHR for High vs Low in TEST (based on 5y risk group)
# --------------------------
mm <- model.matrix(~ risk_group_5y, data = pred_df)[, -1, drop = FALSE]
crr_fit <- cmprsk::crr(
  ftime = pred_df$time,
  fstatus = pred_df$event,
  cov1 = mm,
  failcode = 1,
  cencode = 0
)

beta <- as.numeric(crr_fit$coef)
se_b <- sqrt(diag(crr_fit$var))
sHR <- exp(beta)
lcl <- exp(beta - 1.96 * se_b)
ucl <- exp(beta + 1.96 * se_b)
pval <- 2 * pnorm(-abs(beta / se_b))

shr_df <- data.frame(
  Comparison = "High vs Low (based on 5-year predicted risk; TEST set)",
  sHR = sHR,
  CI_low = lcl,
  CI_high = ucl,
  p_value = pval,
  stringsAsFactors = FALSE
)
write.csv(shr_df, file.path(OUT_DIR, "riskgroup_shr_high_vs_low_5y.csv"), row.names = FALSE)

shr_text <- paste0(
  "sHR (High vs Low) = ", sprintf("%.2f", sHR),
  " (95% CI ", sprintf("%.2f", lcl), "–", sprintf("%.2f", ucl),
  "), P ", ifelse(pval < 0.001, "< 0.001", sprintf("= %.3f", pval))
)

# --------------------------
# 9) Plot CIF curves (Figure 5; TEST; grouped by 5-year predicted risk)
# --------------------------
p_cif <- ggplot(cif_df, aes(x = time, y = est, color = group, fill = group)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.18, color = NA) +
  geom_step(linewidth = 1) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Cumulative Incidence of ADRD by Predicted Risk Group",
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

ggsave(file.path(OUT_DIR, "Figure5_CIF_5yRiskGroup_TEST.png"),
       p_cif, width = 8, height = 6, dpi = 300)

cat("\nDONE ✅ Competing-risk analysis completed.\n")
cat("Fine–Gray fit on TRAIN; absolute risk + CIF + sHR computed on TEST.\n")
cat("Output folder:", OUT_DIR, "\n\n")
