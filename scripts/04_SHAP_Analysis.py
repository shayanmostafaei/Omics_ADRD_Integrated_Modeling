# ==============================================================================
# 04_SHAP_Analysis.py
# SHAP and Permutation Analysis for FINAL ADRD XGBoost model ("Model 5")
# - Produces SHAP: train/test set, bar and beeswarm, .parquet/.csv, importances table
# - Computes permutation feature importance (test set)
# AUTHOR: Shayan Mostafaei
# DATE CREATED: 2026-03-18
# ==============================================================================

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pyreadr  # pip install pyreadr
from sklearn.metrics import roc_auc_score

# --------------------------
# USER SETTINGS
# --------------------------

RANDOM_STATE = 20250101
TOP_N_FEATURES = 30

IN_DIR = os.path.join("results", "04_stepwise_xgboost")
SHAP_INPUT_RDS = os.path.join(IN_DIR, "model5_shap_input.rds")
MODEL_JSON = os.path.join(IN_DIR, "Model5_Add_MetaboAge_xgb_model.json")
OUT_DIR = os.path.join(IN_DIR, "shap")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

# --------------------------
# 1) LOAD .RDS EXACT DESIGN MATRICES
# --------------------------

if not os.path.exists(SHAP_INPUT_RDS):
    raise FileNotFoundError(
        f"Missing: {SHAP_INPUT_RDS}\n"
        "Run 03_Multivariable_XGBoost_Modeling.R first to produce this file."
    )
r_obj = pyreadr.read_r(SHAP_INPUT_RDS)

def extract_shap_rds(obj):
    # Case: keys directly
    if all(k in obj.keys() for k in ["X_train", "X_test", "y_train", "y_test"]):
        return obj["X_train"], obj["X_test"], obj["y_train"], obj["y_test"]
    # Case: first value is dict
    first_val = list(obj.values())[0]
    if isinstance(first_val, dict) and all(k in first_val for k in ["X_train", "X_test", "y_train", "y_test"]):
        return first_val["X_train"], first_val["X_test"], first_val["y_train"], first_val["y_test"]
    raise RuntimeError(
        "Cannot parse .rds: expected elements X_train/X_test/y_train/y_test. pyreadr keys: "
        f"{list(obj.keys())}"
    )

X_train, X_test, y_train, y_test = extract_shap_rds(r_obj)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(np.array(y_train).astype(int).reshape(-1))
y_test = pd.Series(np.array(y_test).astype(int).reshape(-1))

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Train/test feature dimension mismatch.")
if not set(np.unique(y_train)).issubset({0, 1}):
    raise ValueError("y_train must be 0/1 only.")
if not set(np.unique(y_test)).issubset({0, 1}):
    raise ValueError("y_test must be 0/1 only.")

# --------------------------
# 2) LOAD XGBOOST MODEL
# --------------------------

if not os.path.exists(MODEL_JSON):
    raise FileNotFoundError(
        f"Missing: {MODEL_JSON}\n"
        "Export with xgboost::xgb.save(model, ...) from R 03_Multivariable_XGBoost_Modeling.R."
    )
booster = xgb.Booster()
booster.load_model(MODEL_JSON)

# --------------------------
# 3) SHAP COMPUTATION (TRAIN & TEST)
# --------------------------

explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
# SHAP for train set
shap_train = explainer(X_train)
shap_train_df = pd.DataFrame(shap_train.values, columns=X_train.columns, index=X_train.index)

# SHAP for test set
shap_test = explainer(X_test)
shap_test_df = pd.DataFrame(shap_test.values, columns=X_test.columns, index=X_test.index)

# Save SHAP values for reproducibility
def save_shap_df(df, filename_prefix):
    parquet_path = os.path.join(OUT_DIR, f"{filename_prefix}.parquet")
    csv_path = os.path.join(OUT_DIR, f"{filename_prefix}.csv")
    try:
        df.to_parquet(parquet_path)
    except Exception as e:
        print(f"[WARN] Could not save as parquet ({e}); saving CSV instead.")
        df.to_csv(csv_path, index=False)

save_shap_df(shap_train_df, "shap_values_train")
save_shap_df(shap_test_df, "shap_values_test")

# Compute and save mean(|SHAP|) importances for both sets
def calc_mean_abs_shap(shap_values, X_df, out_name):
    imp_df = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(os.path.join(OUT_DIR, out_name), index=False)
    return imp_df

train_imp_df = calc_mean_abs_shap(shap_train, X_train, "shap_mean_abs_importance_train.csv")
test_imp_df = calc_mean_abs_shap(shap_test, X_test, "shap_mean_abs_importance_test.csv")

# --------------------------
# 4) SHAP PLOTS (train & test)
# --------------------------
def summary_plot_lim(shap_values, X_df, top_features, figname, plot_type, title):
    X_sub = X_df[top_features]
    shap_sub = shap_values.values[:, [X_df.columns.get_loc(f) for f in top_features]]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_sub,
        X_sub,
        plot_type=plot_type,
        show=False
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, figname), dpi=300)
    plt.close()

# TRAIN
top_features_train = train_imp_df["feature"].head(TOP_N_FEATURES).tolist()
summary_plot_lim(
    shap_train, X_train, top_features_train, "Figure_A7_shap_beeswarm_train.png", "dot",
    "SHAP Summary (Train, Model 5: Final Integrated ADRD Model)"
)
summary_plot_lim(
    shap_train, X_train, top_features_train, "Figure_A7_shap_bar_train.png", "bar",
    "SHAP Mean(|SHAP|) Importance (Train, Model 5: Final Integrated ADRD Model)"
)

# TEST
top_features_test = test_imp_df["feature"].head(TOP_N_FEATURES).tolist()
summary_plot_lim(
    shap_test, X_test, top_features_test, "Figure_A7_shap_beeswarm_test.png", "dot",
    "SHAP Summary (Test, Model 5: Final Integrated ADRD Model)"
)
summary_plot_lim(
    shap_test, X_test, top_features_test, "Figure_A7_shap_bar_test.png", "bar",
    "SHAP Mean(|SHAP|) Importance (Test, Model 5: Final Integrated ADRD Model)"
)

# --------------------------
# 5) PERMUTATION FEATURE IMPORTANCE (TEST SET)
# --------------------------
def permutation_importance_xgb(booster, X, y, metric_fn=roc_auc_score, n_repeats=10, random_state=20250101):
    np.random.seed(random_state)
    orig_pred = booster.predict(xgb.DMatrix(X))
    base_score = metric_fn(y, orig_pred)
    scores = []
    for col in X.columns:
        permuted_scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_pred = booster.predict(xgb.DMatrix(X_perm))
            permuted_score = metric_fn(y, perm_pred)
            permuted_scores.append(base_score - permuted_score)
        scores.append({"feature": col, "mean_loss": np.mean(permuted_scores), "std_loss": np.std(permuted_scores)})
    pfi_df = pd.DataFrame(scores).sort_values("mean_loss", ascending=False)
    return base_score, pfi_df

print("Running permutation importance (test set)...")
test_auc, pfi_df = permutation_importance_xgb(booster, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
pfi_df.to_csv(os.path.join(OUT_DIR, "permutation_importance_test.csv"), index=False)

print(f"\n[INFO] Test AUC (unpermuted): {test_auc:.4f}")
print("\n[INFO] Top 10 permutation importance features:\n", pfi_df.head(10))

# --------------------------
# COMPLETION MESSAGE
# --------------------------
print("\n✅ DONE: SHAP (train & test) and permutation importance analysis completed.")
print("Outputs written to:", OUT_DIR)
print("- Figure_A7_shap_beeswarm_train.png, Figure_A7_shap_beeswarm_test.png")
print("- Figure_A7_shap_bar_train.png, Figure_A7_shap_bar_test.png")
print("- shap_values_train.parquet/.csv, shap_values_test.parquet/.csv")
print("- shap_mean_abs_importance_train.csv, shap_mean_abs_importance_test.csv")
print("- permutation_importance_test.csv")
