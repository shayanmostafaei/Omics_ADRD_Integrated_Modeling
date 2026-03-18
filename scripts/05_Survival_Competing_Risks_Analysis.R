# ==============================================================================
# 05_Survival_Competing_Risks_Analysis.R
# Competing-risk time-to-event analysis for ADRD with death as competing event.
# Aligned with manuscript:
# - Fine–Gray model fit on train split
# - Absolute risks predicted on test split
# - Stratification by predicted 5-year risk: top 25% vs bottom 75% (Figure 5)
# AUTHOR: Shayan Mostafaei
# DATE CREATED: 2026-03-18
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(riskRegression)
  library(cmprsk)
  library(ggplot2)
  library(scales)
  library(readr)
})

# --------------------------
# USER SETTINGS
# --------------------------
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
HIGH_RISK_Q <- 0.75  # Use top 25% for high risk

# --------------------------
# LOAD DATA
# --------------------------
if (!file.exists(INPUT_RDS)) stop("Missing input RDS: ", INPUT_RDS)
df0 <- readRDS(INPUT_RDS)

# --------------------------
# VALIDATION
# --------------------------
required_cols <- c(
  OUTCOME_COL, T_DEM_COL, DEATH_COL, T_DEATH_COL, CENS_COL,
  AGE_COL, SEX_COL, SMOKE_COL, ALC_COL, BMI_COL, EDU_COL,
  PRS_COL, PHENO_COL, FI_COL, TL_COL, PROTAGE_COL, METABOAGE_COL
)
missing <- setdiff(required_cols, names(df0))
if (length(missing) > 0) stop("Missing required columns: ", paste(missing, collapse = ", "))

if (!ID_COL %in% names(df0)) {
  message("ID column '", ID_COL, "' not found: using row index as ID.")
  df0[[ID_COL]] <- seq_len(nrow(df0))
}

PC_COLS <- PC_COLS[PC_COLS %in% names(df0)]

# --------------------------
# 1) Build competing-risk dataset (time in DAYS)
# --------------------------
df1 <- df0 %>%
  transmute(
    id = as.character(.data[[ID_COL]]),
    y_class = suppressWarnings(as.numeric(.data[[OUTCOME_COL]])),
    y_class = ifelse(is.na(y_class), NA_real_, ifelse(y_class == 1, 1, 0)),
    t_dem_raw = suppressWarnings(as.numeric(.data[[T_DEM_COL]])),  # DAYS
    death_event = suppressWarnings(as.numeric(.data[[DEATH_COL]])),
    death_event = ifelse(is.na(death_event), 0, ifelse(death_event == 1, 1, 0)),
    t_death_raw = suppressWarnings(as.numeric(.data[[T_DEATH_COL]])),  # may be YEARS
    t_cens_raw  = suppressWarnings(as.numeric(.data[[CENS_COL]])),     # may be YEARS
    CA = suppressWarnings(as.numeric(.data[[AGE_COL]])),
    sex = factor(.data[[SEX_COL]]),
    smoking = factor(.data[[SMOKE_COL]]),
    alcohol_intake_frequency = factor(.data[[ALC_COL]]),
    bmi = suppressWarnings(as.numeric(.data[[BMI_COL]])),
    education = factor(.data[[EDU_COL]]),
    PRS_ADRD = suppressWarnings(as.numeric(.data[[PRS_COL]])),
    PhenoAge = suppressWarnings(as.numeric(.data[[PHENO_COL]])),
    FI = suppressWarnings(as.numeric(.data[[FI_COL]])),
    TL = suppressWarnings(as.numeric(.data[[TL_COL]])),
    ProtAge = suppressWarnings(as.numeric(.data[[PROTAGE_COL]])),
    MetaboAge = suppressWarnings(as.numeric(.data[[METABOAGE_COL]])),
    across(all_of(PC_COLS), ~ suppressWarnings(as.numeric(.x)))
  ) %>% filter(!is.na(y_class))

df1 <- df1 %>%
  mutate(
    t_dem_event_days = ifelse(y_class == 1 & !is.na(t_dem_raw) & t_dem_raw > 0, t_dem_raw, Inf),
    t_death_days = ifelse(
      death_event == 1 & !is.na(t_death_raw) & t_death_raw > 0,
      ifelse(T_DEATH_IS_YEARS, t_death_raw * 365.25, t_death_raw),
      Inf
    ),
    t_cens_days = ifelse(
      !is.na(t_cens_raw) & t_cens_raw > 0,
      ifelse(CENS_IS_YEARS, t_cens_raw * 365.25, t_cens_raw),
      Inf
    ),
    time = pmin(t_dem_event_days, t_death_days, t_cens_days, na.rm=TRUE),
    event = case_when(
      is.finite(t_dem_event_days) & t_dem_event_days <= t_death_days & t_dem_event_days <= t_cens_days ~ 1L,
      is.finite(t_death_days)     & t_death_days < t_dem_event_days & t_death_days <= t_cens_days ~ 2L,
      TRUE ~ 0L
    )
  ) %>%
  filter(is.finite(time), !is.na(time), time > 0)

message("Event counts: 0=censor, 1=ADRD, 2=death")
print(table(df1$event))

# Save time summary
write_csv(
  df1 %>%
    summarise(
      n = n(),
      n_adrd = sum(event == 1),
      n_death = sum(event == 2),
      n_cens = sum(event == 0),
      median_time_days = median(time),
      median_time_years = median(time) / 365.25
    ),
  file.path(OUT_DIR, "time_event_summary.csv")
)

# --------------------------
# 2) Stratified 70/30 split
# --------------------------
train_idx <- caret::createDataPartition(df1$y_class, p=0.7, list=FALSE)
train_df <- df1[train_idx, , drop=FALSE]
test_df  <- df1[-train_idx, , drop=FALSE]

# --------------------------
# 3) Preprocessing
# --------------------------
numeric_cols <- c("CA", "bmi", "PRS_ADRD", "PhenoAge", "FI", "TL", "ProtAge", "MetaboAge", PC_COLS)
numeric_cols <- numeric_cols[numeric_cols %in% names(df1)]
pp <- caret::preProcess(train_df[, numeric_cols, drop=FALSE], method = c("medianImpute", "center", "scale"))

train_df_pp <- train_df
test_df_pp  <- test_df
train_df_pp[, numeric_cols] <- predict(pp, train_df[, numeric_cols, drop=FALSE])
test_df_pp[, numeric_cols]  <- predict(pp, test_df[, numeric_cols, drop=FALSE])

mode_train <- function(x) {
  x <- as.factor(x)
  tab <- table(x, useNA = "no")
  if (length(tab) == 0) return(NA_character_)
  names(tab)[which.max(tab)]
}
cat_cols <- c("sex", "smoking", "alcohol_intake_frequency", "education")

for (v in cat_cols) {
  train_df_pp[[v]] <- as.factor(train_df_pp[[v]])
  m <- mode_train(train_df_pp[[v]])
  train_df_pp[[v]][is.na(train_df_pp[[v]])] <- m
  train_df_pp[[v]] <- droplevels(train_df_pp[[v]])
  test_df_pp[[v]] <- as.factor(test_df_pp[[v]])
  test_df_pp[[v]][is.na(test_df_pp[[v]])] <- m
  test_df_pp[[v]] <- factor(test_df_pp[[v]], levels=levels(train_df_pp[[v]]))
}

# --------------------------
# 4) Fit Fine–Gray model (TRAIN)
# --------------------------
rhs_terms <- c("CA", "sex", "smoking", "alcohol_intake_frequency", "bmi", "education",
               "PRS_ADRD", "PhenoAge", "FI", "TL", "ProtAge", "MetaboAge", PC_COLS)
rhs_terms <- rhs_terms[rhs_terms %in% names(train_df_pp)]

fg_formula <- as.formula(paste0("Hist(time, event) ~ ", paste(rhs_terms, collapse=" + ")))
fg_fit <- riskRegression::FGR(fg_formula, data=train_df_pp, cause=1)
saveRDS(fg_fit, file.path(OUT_DIR, "finegray_model_train.rds"))

# --------------------------
# 5) Predict absolute ADRD risk at 5y and 9y (TEST)
# --------------------------
risk_5y <- as.numeric(riskRegression::predictRisk(fg_fit, newdata=test_df_pp, times=TIME_5Y))
risk_9y <- as.numeric(riskRegression::predictRisk(fg_fit, newdata=test_df_pp, times=TIME_9Y))

pred_df <- test_df_pp %>%
  transmute(
    id, event, time, y_class,
    risk_5y=risk_5y,
    risk_9y=risk_9y
  )

write_csv(pred_df, file.path(OUT_DIR, "test_predicted_risks_5y_9y.csv"))

# --------------------------
# 6) Risk group stratification
# --------------------------
q75_5y <- as.numeric(stats::quantile(pred_df$risk_5y, probs=HIGH_RISK_Q, na.rm=TRUE))
pred_df <- pred_df %>%
  mutate(
    risk_group_5y = factor(ifelse(risk_5y > q75_5y, "High", "Low"), levels=c("Low", "High"))
  )
write_csv(
  tibble(q75_5y_threshold=q75_5y),
  file.path(OUT_DIR, "riskgroup_thresholds.csv")
)

# --------------------------
# 7) CIF curves in TEST group
# --------------------------
ci_obj <- cmprsk::cuminc(pred_df$time, pred_df$event, group=pred_df$risk_group_5y)
cause1_names <- names(ci_obj)[grepl(" 1$", names(ci_obj))]
cif_df <- bind_rows(lapply(cause1_names, function(nm) {
  e <- ci_obj[[nm]]
  grp <- sub(" 1$", "", nm)
  tibble(time=e$time, est=as.numeric(e$est), var=as.numeric(e$var), group=grp)
})) %>%
  mutate(
    se = sqrt(var),
    lower = pmax(0, est - 1.96 * se),
    upper = pmin(1, est + 1.96 * se),
    group = factor(group, levels=c("Low", "High"))
  )
write_csv(cif_df, file.path(OUT_DIR, "cif_curve_data_5y_group.csv"))

# --------------------------
# 8) sHR (High vs Low in TEST)
# --------------------------
mm <- model.matrix(~ risk_group_5y, data=pred_df)[, -1, drop=FALSE]
crr_fit <- cmprsk::crr(
  ftime=pred_df$time,
  fstatus=pred_df$event,
  cov1=mm,
  failcode=1,
  cencode=0
)
beta <- as.numeric(crr_fit$coef)
se_b <- sqrt(diag(crr_fit$var))
sHR <- exp(beta)
lcl <- exp(beta - 1.96 * se_b)
ucl <- exp(beta + 1.96 * se_b)
pval <- 2 * pnorm(-abs(beta/se_b))
shr_df <- tibble(
  Comparison = "High vs Low (top 25% vs bottom 75% by predicted 5-year risk, TEST set)",
  sHR = sHR, CI_low = lcl, CI_high = ucl, p_value = pval
)
write_csv(shr_df, file.path(OUT_DIR, "riskgroup_shr_high_vs_low_5y.csv"))
shr_text <- paste0("sHR (High vs Low) = ", sprintf("%.2f", sHR),
                   " (95% CI ", sprintf("%.2f", lcl), "–", sprintf("%.2f", ucl),
                   "), P ", ifelse(pval < 0.001, "< 0.001", sprintf("= %.3f", pval)))

# --------------------------
# 9) Plot CIF curves (Figure 5)
# --------------------------
p_cif <- ggplot(cif_df, aes(x=time, y=est, color=group, fill=group)) +
  geom_ribbon(aes(ymin=lower, ymax=upper), alpha=0.18, color=NA) +
  geom_step(linewidth=1) +
  scale_y_continuous(labels=scales::percent_format(accuracy=1)) +
  labs(
    title = "Cumulative Incidence of ADRD by Predicted Risk Group",
    subtitle = shr_text,
    x = "Time (days)", y = "Cumulative incidence",
    color = "Risk group", fill = "Risk group"
  ) +
  theme_classic(base_size=14) +
  theme(
    legend.position="top",
    plot.title = element_text(face="bold", hjust=0.5),
    plot.subtitle = element_text(hjust=0.5)
  )

ggsave(filename=file.path(OUT_DIR, "Figure5_CIF_5yRiskGroup_TEST.png"), plot=p_cif, width=8, height=6, dpi=500)

cat("\n✅ DONE: Competing-risk analysis completed.\n")
cat("Fine–Gray fit on TRAIN; risks, CIF, sHR computed on TEST.\n")
cat("Output folder:", OUT_DIR, "\n\n")
