# ==================================================================================
# 04_SHAP_Analysis.py
# SHAP plot derived from XGBoost model for Model 5 (includes all predictors plus covariates)
# ==================================================================================

import pyreadr
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------
# 1. Load RData
# ----------------------------------------------------------------------------------
data = result["Biomarkers_complete"]

# ----------------------------------------------------------------------------------
# 2. Select predictors & target
# ----------------------------------------------------------------------------------
features = [
    "CA", "sex", "smoking", "alcohol", "bmi", "education",
    "DDML_PRS_With_APOE", "PhenoAge", "FI", "TL",
    "ProtAge", "MetaboAge"
]
X = data[features].copy()
y = data["Dementia_status"]

# ----------------------------------------------------------------------------------
# 2a. Standardize categorical variables
# ----------------------------------------------------------------------------------
X['sex'] = X['sex'].str.strip().str.lower()          # expected: 'men'/'women'
X['education'] = X['education'].str.strip().str.lower()  # expected: 'low', 'medium', 'high'

# 2b. Create dummy variables
X = pd.get_dummies(X, columns=["sex", "education"], drop_first=True)

# 2c. Ensure expected dummy columns exist
expected_dummies = ['sex_men', 'education_low']
for col in expected_dummies:
    if col not in X.columns:
        X[col] = 0

# 2d. Keep only desired features
keep_features = [
    "CA", "smoking", "alcohol", "bmi", "DDML_PRS_With_APOE",
    "PhenoAge", "FI", "TL", "ProtAge", "MetaboAge",
    "sex_men", "education_low"
]
X = X[keep_features]

# Ensure numeric type
X = X.apply(pd.to_numeric, errors="coerce").astype(float)

# Drop missing values
X = X.dropna()
y = y.loc[X.index]

# ----------------------------------------------------------------------------------
# 3. Rename columns for plotting
# ----------------------------------------------------------------------------------
rename_dict = {
    "DDML_PRS_With_APOE": "PRS",
    "bmi": "BMI",
    "alcohol": "Alcohol consumption",
    "sex_men": "Men",
    "education_low": "Low education"
}
X = X.rename(columns=rename_dict)

# ----------------------------------------------------------------------------------
# 4. Train/test split (stratified)
# ----------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------------------
# 5. Train XGBoost classifier
# ----------------------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------------------
# 6. Compute SHAP values
# ----------------------------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_train)

# ----------------------------------------------------------------------------------
# 7. SHAP beeswarm plot
# ----------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values.values,
    X_train,
    plot_type="dot",
    sort=True,
    color_bar=True,
    show=True
)
plt.title("SHAP Feature Importance - ADRD Risk", fontsize=14)
plt.tight_layout()
plt.show()

