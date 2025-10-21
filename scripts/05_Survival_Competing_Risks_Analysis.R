# ==================================================================================
# 05_Survival_Competing_Risks_Analysis.R
# Time-to-event analyses for ADRD with death as a competing risk, 
# including cumulative incidence curves and Fine–Gray model results.
# ==================================================================================

# ----------------------------------------------------------------------------------
# 0. Load required packages
# ----------------------------------------------------------------------------------
pkgs <- c("cmprsk", "riskRegression", "survival", "survminer",
          "dplyr", "prodlim", "scales", "ggplot2", "tidyr", "cowplot", "grid", "gridExtra", "forestplot")
to_install <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if(length(to_install)) install.packages(to_install)
lapply(pkgs, library, character.only = TRUE)

stopifnot(exists("Biomarkers_complete"))

# ----------------------------------------------------------------------------------
# 1. Prepare data
# ----------------------------------------------------------------------------------
predictor_vars <- c("CA", "bmi", "DDML_PRS_With_APOE", "PhenoAge", "FI", "TL", "MetaboAge", "ProtAge")
categorical_vars <- c("sex", "smoking", "alcohol", "education")
event_vars <- c("Dementia_status", "Time_to_Dementia", "death_status", "time_to_death")
all_cols <- c(predictor_vars, categorical_vars, event_vars)
missing_cols <- setdiff(all_cols, colnames(Biomarkers_complete))
if(length(missing_cols)) stop("Missing columns: ", paste(missing_cols, collapse = ", "))

df <- Biomarkers_complete[, all_cols]

# Impute numeric predictors with median
for(v in predictor_vars){
  if(!is.numeric(df[[v]])) df[[v]] <- as.numeric(as.character(df[[v]]))
  df[[v]][is.na(df[[v]])] <- median(df[[v]], na.rm = TRUE)
}

# Impute categorical predictors with mode
for(v in categorical_vars){
  if(!is.factor(df[[v]])) df[[v]] <- as.factor(df[[v]])
  mode_val <- names(sort(table(df[[v]]), decreasing = TRUE))[1]
  df[[v]][is.na(df[[v]])] <- mode_val
  df[[v]] <- droplevels(df[[v]])
}

# Standardize continuous predictors
df[, predictor_vars] <- scale(df[, predictor_vars])

# ----------------------------------------------------------------------------------
# 2. Define competing-risk events
# ----------------------------------------------------------------------------------
df$event <- dplyr::case_when(
  !is.na(df$Dementia_status) & df$Dementia_status == 1 ~ 1,
  !is.na(df$death_status) & df$death_status == 1     ~ 2,
  TRUE ~ 0
)
df$time <- pmin(df$Time_to_Dementia, df$time_to_death, na.rm = TRUE)
df <- df[!is.na(df$time), ]
df$event <- as.integer(df$event)

# ----------------------------------------------------------------------------------
# 3. Fit Fine–Gray model (cause = ADRD)
# ----------------------------------------------------------------------------------
fg_formula <- as.formula(paste("Hist(time, event) ~", paste(c(predictor_vars, categorical_vars), collapse = " + ")))
fg_fit <- tryCatch(
  riskRegression::FGR(fg_formula, data = df, cause = 1),
  error = function(e) stop("FGR failed: ", e$message)
)
print(summary(fg_fit))

# ----------------------------------------------------------------------------------
# 4. Predict 5-year ADRD risk (time = 1826 days)
# ----------------------------------------------------------------------------------
time_5y <- 1826
risk_pred <- tryCatch(
  riskRegression::predictRisk(fg_fit, newdata = df, times = time_5y),
  error = function(e) stop("predictRisk failed: ", e$message)
)
df$risk_5y <- if(is.matrix(risk_pred)) as.numeric(risk_pred[,1]) else as.numeric(risk_pred)

# ----------------------------------------------------------------------------------
# 5. Stratify High (top 25%) vs Low (bottom 75%)
# ----------------------------------------------------------------------------------
q3 <- quantile(df$risk_5y, 0.75, na.rm = TRUE)
df$risk_group <- factor(ifelse(df$risk_5y > q3, "High", "Low"), levels = c("Low", "High"))

# ----------------------------------------------------------------------------------
# 6. Compute cumulative incidence functions (CIF)
# ----------------------------------------------------------------------------------
ci_obj <- cmprsk::cuminc(ftime = df$time, fstatus = df$event, group = df$risk_group)
cause1_names <- names(ci_obj)[grepl(" 1$", names(ci_obj))]
cif_list <- lapply(cause1_names, function(nm){
  e <- ci_obj[[nm]]
  grp <- sub(" 1$", "", nm)
  data.frame(time = e$time,
             est  = as.numeric(e$est),
             var  = as.numeric(e$var),
             group = grp,
             stringsAsFactors = FALSE)
})
cif_df <- bind_rows(cif_list) %>%
  mutate(se = sqrt(var),
         lower = pmax(0, est - 1.96*se),
         upper = pmin(1, est + 1.96*se),
         group = factor(group, levels = c("Low", "High")))

# ----------------------------------------------------------------------------------
# 7. Compute subdistribution hazard ratio (sHR) for High vs Low
# ----------------------------------------------------------------------------------
mm <- model.matrix(~ risk_group, data = df)[, -1, drop = FALSE]
crr_fit <- cmprsk::crr(ftime = df$time, fstatus = df$event, cov1 = mm, failcode = 1, cencode = 0)
coef <- crr_fit$coef
se_coef <- sqrt(diag(crr_fit$var))
sHR <- exp(coef)
lower <- exp(coef - 1.96 * se_coef)
upper <- exp(coef + 1.96 * se_coef)
pval <- 2 * pnorm(-abs(coef / se_coef))
sHR_text <- paste0(
  "sHR (High vs Low) = ", sprintf("%.2f", sHR),
  " (95% CI ", sprintf("%.2f", lower), "–", sprintf("%.2f", upper),
  "), P ", ifelse(pval < 0.001, "< 0.001", sprintf("= %.3f", pval))
)
print(sHR_text)

# ----------------------------------------------------------------------------------
# 8. Plot CIF curves
# ----------------------------------------------------------------------------------
colors <- c("Low" = "#1f78b4", "High" = "#e31a1c")
ggplot(cif_df, aes(x = time, y = est, color = group, fill = group)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.18, color = NA) +
  geom_step(size = 1) +
  scale_color_manual(values = colors, labels = c("Lower Risk Group", "Highest Risk Group")) +
  scale_fill_manual(values = colors, labels = c("Lower Risk Group", "Highest Risk Group")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Cumulative Incidence of ADRD by Predicted Risk Group",
    subtitle = sHR_text,
    x = "Time (days)",
    y = "Cumulative incidence",
    color = "Strata",
    fill = "Strata"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", plot.title = element_text(face = "bold"))

# ----------------------------------------------------------------------------------
# 9. Forest plot of Fine–Gray sHR
# ----------------------------------------------------------------------------------
fg_summary <- summary(fg_fit)
fg_df <- data.frame(
  Variable  = rownames(fg_summary$coef),
  sHR       = fg_summary$coef[, "exp(coef)"],
  CI_lower  = fg_summary$conf.int[, "2.5%"],
  CI_upper  = fg_summary$conf.int[, "97.5%"],
  pval      = fg_summary$coef[, "p-value"]
)
forest_vars <- c("CA", "DDML_PRS_With_APOE", "PhenoAge", "FI", "TL", "MetaboAge", "ProtAge")
fg_df <- fg_df %>% filter(Variable %in% forest_vars)
fg_df$Variable <- ifelse(fg_df$Variable == "DDML_PRS_With_APOE", "PRS", fg_df$Variable)

variable_types <- data.frame(
  Variable = c("PRS", "CA", "PhenoAge", "FI", "TL", "MetaboAge", "ProtAge"),
  Type     = c("PRS", "ChronAge", "BioAge", "BioAge", "BioAge", "BioAge", "BioAge")
)
fg_df <- merge(fg_df, variable_types, by = "Variable")
fg_df$FDR <- p.adjust(fg_df$pval, method = "fdr")
type_colors <- c("PRS" = "#e31a1c", "ChronAge" = "#1f78b4", "BioAge" = "#33a02c")
fg_df$Color <- type_colors[fg_df$Type]
fg_df <- fg_df %>%
  mutate(
    sHR_label = sprintf("%.2f", sHR),
    CI_label  = paste0("(", sprintf("%.2f", CI_lower), "–", sprintf("%.2f", CI_upper), ")"),
    FDR_label = formatC(FDR, format = "e", digits = 2)
  ) %>%
  arrange(desc(sHR))

tabletext <- cbind(
  c("Variable", fg_df$Variable),
  c("sHR", fg_df$sHR_label),
  c("95% CI", fg_df$CI_label),
  c("FDR", fg_df$FDR_label)
)

forestplot(
  labeltext = tabletext,
  mean  = c(NA, fg_df$sHR),
  lower = c(NA, fg_df$CI_lower),
  upper = c(NA, fg_df$CI_upper),
  zero = 1,
  boxsize = 0.2,
  lineheight = "auto",
  col = fpColors(box = fg_df$Color, line = fg_df$Color, zero = "black"),
  xlab = "Sub-distribution Hazard Ratio (sHR per SD, 95% CI)",
  title = "Fine–Gray Model: sHR for ADRD Predictors",
  xticks = seq(0.5, 3, 0.5),
  txt_gp = fpTxtGp(
    xlab  = gpar(cex = 1.4),
    ticks = gpar(cex = 1.3),
    label = gpar(cex = 1.2),
    title = gpar(cex = 1.5, fontface = "bold")
  )
)
