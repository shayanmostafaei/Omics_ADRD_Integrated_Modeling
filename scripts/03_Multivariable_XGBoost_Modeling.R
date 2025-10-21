# ==================================================================================
# 03_Multivariable_XGBoost_Modeling.R
# Stepwise multivariable modeling strategy using XGBoost

library(xgboost)
library(caret)
library(pROC)
library(dplyr)

# Ensure Dementia_status is factor (binary outcome)
Biomarkers_complete$Dementia_status <- as.factor(Biomarkers_complete$Dementia_status)

# ----------------------------------------------------------------------------------
# 1. Split train/test (70% / 30%)
# ----------------------------------------------------------------------------------
set.seed(123)
trainIndex <- createDataPartition(Biomarkers_complete$Dementia_status, p = 0.7, list = FALSE)
trainData <- Biomarkers_complete[trainIndex, ]
testData  <- Biomarkers_complete[-trainIndex, ]

# ----------------------------------------------------------------------------------
# 2. Helper function to train + evaluate XGBoost model
# ----------------------------------------------------------------------------------
fit_xgb <- function(formula, trainData, testData, nrounds = 200) {
  # Model matrices
  train_matrix <- model.matrix(formula, data = trainData)[, -1] # drop intercept
  test_matrix  <- model.matrix(formula, data = testData)[, -1]
  
  train_label <- as.numeric(trainData$Dementia_status) - 1
  test_label  <- as.numeric(testData$Dementia_status) - 1
  
  dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
  dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)
  
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    watchlist = list(train = dtrain, test = dtest),
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  pred <- predict(model, newdata = dtest)
  roc_obj <- roc(test_label, pred)
  
  return(list(model = model, auc = auc(roc_obj), roc = roc_obj))
}

# ----------------------------------------------------------------------------------
# 3. Stepwise models
# ----------------------------------------------------------------------------------

# 1. Crude model: Age + Sex
m1 <- fit_xgb(Dementia_status ~ CA + sex, trainData, testData)
print(paste("Model 1 AUC:", round(m1$auc, 3)))
ci.auc(m1$roc, conf.level = 0.95)

# 2. + Lifestyle (smoking, alcohol, bmi)
preProc <- preProcess(trainData, method = c("medianImpute"))
trainData_imp <- predict(preProc, trainData)
testData_imp  <- predict(preProc, testData)
m2 <- fit_xgb(Dementia_status ~ CA + sex + smoking + alcohol + bmi,
              trainData_imp, testData_imp)
print(paste("Model 2 AUC:", round(m2$auc, 3)))
ci.auc(m2$roc, conf.level = 0.95)
roc.test(m1$roc, m2$roc, method = "delong")

# 3. + PRS (polygenic risk score)
m3 <- fit_xgb(Dementia_status ~ CA + sex + smoking + alcohol + bmi + DDML_PRS_With_APOE,
              trainData_imp, testData_imp)
print(paste("Model 3 AUC:", round(m3$auc, 3)))
ci.auc(m3$roc, conf.level = 0.95)
roc.test(m1$roc, m3$roc, method = "delong")

# 4. + Biological Age (PhenoAge, FI, TL)
m4 <- fit_xgb(Dementia_status ~ CA + sex + smoking + alcohol + bmi + FI + TL +
                DDML_PRS_With_APOE + PhenoAge,
              trainData_imp, testData_imp)
print(paste("Model 4 AUC:", round(m4$auc, 3)))
ci.auc(m4$roc, conf.level = 0.95)
roc.test(m1$roc, m4$roc, method = "delong")

# 5. + ProtAge
m5 <- fit_xgb(Dementia_status ~ CA + sex + smoking + alcohol + bmi + FI + TL +
                DDML_PRS_With_APOE + PhenoAge + ProtAge,
              trainData_imp, testData_imp)
print(paste("Model 5 AUC:", round(m5$auc, 3)))
ci.auc(m5$roc, conf.level = 0.95)

# 6. + MetaboAge
m6 <- fit_xgb(Dementia_status ~ CA + sex + smoking + alcohol + bmi + FI + TL +
                DDML_PRS_With_APOE + PhenoAge + ProtAge + MetaboAge,
              trainData_imp, testData_imp)
print(paste("Model 6 AUC:", round(m6$auc, 3)))
ci.auc(m6$roc, conf.level = 0.95)
