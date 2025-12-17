# ==================================================================================
# 04_SHAP_Analysis.py
# SHAP analysis for FINAL integrated ADRD model (Model5)
#
#
# Requires outputs from Script 03:
#   results/multivariable_xgboost/model5_shap_input.rds
#   results/multivariable_xgboost/Model5_Add_MetaboAge_xgb_model.json
#
# Outputs:
#   results/shap/
#     - shap_summary_beeswarm.png
#     - shap_summary_bar.png
#     - shap_values_train.parquet
# ==================================================================================

import os
import numpy as np
import pandas as pd

import xgboost as xgb
import shap
import matplotlib.pyplot as plt

import pyreadr  # pip install pyreadr

# --------------------------
# USER SETTINGS
# --------------------------
RANDOM_STATE = 20250101

# Paths must match Script 03 outputs
IN_DIR = "results/multivariable_xgboost"
SHAP_INPUT_RDS = os.path.join(IN_DIR, "model5_shap_input.rds")
MODEL_JSON = os.path.join(IN_DIR, "Model5_Add_MetaboAge_xgb_model.json")

OUT_DIR = "results/shap"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# 1) LOAD EXACT DESIGN MATRICES FROM R
# --------------------------
if not os.path.exists(SHAP_INPUT_RDS):
    raise FileNotFoundError(
        f"Missing required file: {SHAP_INPUT_RDS}\n"
        "Create it by adding the 'Export EXACT Model 5 design matrices' block "
        "to 03_Multivariable_XGBoost_Modeling.R and re-running Script 03."
    )

r_obj = pyreadr.read_r(SHAP_INPUT_RDS)
# pyreadr returns a dict-like; our RDS contains a list -> keys may vary.
# We fetch the first object, then access list elements as columns if needed.
shap_input = list(r_obj.values())[0]

# If shap_input is a dict-like already:
if isinstance(shap_input, dict):
    X_train = shap_input["X_train"]
    X_test = shap_input["X_test"]
    y_train = shap_input["y_train"]
    y_test = shap_input["y_test"]
else:
    # Some pyreadr versions flatten lists into separate keys; handle that case:
    if all(k in r_obj for k in ["X_train", "X_test", "y_train", "y_test"]):
        X_train = r_obj["X_train"]
        X_test = r_obj["X_test"]
        y_train = r_obj["y_train"]
        y_test = r_obj["y_test"]
    else:
        raise RuntimeError(
            "Could not parse model5_shap_input.rds. "
            "Ensure it is saved as a list with elements: X_train, X_test, y_train, y_test."
        )

# Ensure pandas DataFrames / Series
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(np.array(y_train).astype(int).reshape(-1))
y_test = pd.Series(np.array(y_test).astype(int).reshape(-1))

# Sanity checks
if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Train/Test feature dimension mismatch. This should not happen with R model.matrix export.")
if not set(np.unique(y_train)).issubset({0, 1}):
    raise ValueError("y_train must be 0/1.")
if not set(np.unique(y_test)).issubset({0, 1}):
    raise ValueError("y_test must be 0/1.")

# --------------------------
# 2) LOAD THE EXACT TRAINED XGBOOST MODEL 
# --------------------------
if not os.path.exists(MODEL_JSON):
    raise FileNotFoundError(
        f"Missing required file: {MODEL_JSON}\n"
        "Export it from R using xgboost::xgb.save(model, '...xgb_model.json') "
        "in Script 03 and re-run Script 03."
    )

booster = xgb.Booster()
booster.load_model(MODEL_JSON)

# --------------------------
# 3) COMPUTE SHAP VALUES (ON TRAINING MATRIX)
# --------------------------
# Using Booster directly keeps the model exactly the one trained in R.
explainer = shap.TreeExplainer(booster)
shap_values = explainer(X_train)

# Save SHAP values for reproducibility
try:
    shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns, index=X_train.index)
    shap_df.to_parquet(os.path.join(OUT_DIR, "shap_values_train.parquet"))
except Exception as e:
    print(f"Note: could not save parquet shap values ({e}).")

# --------------------------
# 4) PLOTS
# --------------------------
# Beeswarm
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values.values,
    X_train,
    plot_type="dot",
    show=False
)
plt.title("SHAP Feature Importance (Model 5: Final Integrated ADRD Model)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary_beeswarm.png"), dpi=300)
plt.close()

# Bar (mean |SHAP|)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values.values,
    X_train,
    plot_type="bar",
    show=False
)
plt.title("SHAP Mean(|value|) Importance (Model 5: Final Integrated ADRD Model)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary_bar.png"), dpi=300)
plt.close()

print("\nDONE ✅ SHAP analysis completed (fully consistent with Script 03).")
print(f"Outputs written to: {OUT_DIR}")
print("- shap_summary_beeswarm.png")
print("- shap_summary_bar.png")
print("- shap_values_train.parquet (if saved successfully)")
