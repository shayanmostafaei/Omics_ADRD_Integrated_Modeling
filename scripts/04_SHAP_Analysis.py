# ==============================================================================
# 04_SHAP_Analysis.py
#
# Integrative Prediction of Alzheimer’s Disease and Related Dementias Using
# Multi-Omics Aging Clocks and Genetic Data
#
# Purpose:
# - Compute SHAP values for the final XGBoost model, Model 5.
# - Use the exact design matrices exported by 03_Multivariable_XGBoost_Modeling.R.
# - Generate revised Supplementary Figure A.8.
# - Export SHAP values, mean absolute SHAP importance, and permutation importance.
#
# Main supplementary output:
# - Figure A.8: SHAP summary plot for final integrated XGBoost model
#
# Required inputs from script 03:
# - results/04_stepwise_xgboost/model5_shap_input.rds
# - results/04_stepwise_xgboost/Model5_Add_MetaboAge_xgb_model.json
#
# Notes:
# - SHAP is used for model interpretation, not causal attribution.
# - Individual-level SHAP values should not be committed to the public repository.
#
# Author: Shayan Mostafaei
# Updated for revision: 2026-07-01
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pyreadr
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------

RANDOM_STATE = 20250101
TOP_N_FEATURES = 30
N_PERMUTATION_REPEATS = 10

IN_DIR = os.path.join("results", "04_stepwise_xgboost")
SHAP_INPUT_RDS = os.path.join(IN_DIR, "model5_shap_input.rds")
MODEL_JSON = os.path.join(IN_DIR, "Model5_Add_MetaboAge_xgb_model.json")
OUT_DIR = os.path.join(IN_DIR, "shap")

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def require_file(path: str, message: str = "") -> None:
    if not os.path.exists(path):
        extra = f"\n{message}" if message else ""
        raise FileNotFoundError(f"Missing required file: {path}{extra}")


def extract_shap_rds(obj):
    """
    Extract X_train, X_test, y_train, y_test from an RDS object written by R.
    pyreadr may return either direct keys or one nested object.
    """
    expected = {"X_train", "X_test", "y_train", "y_test"}

    if expected.issubset(set(obj.keys())):
        return obj["X_train"], obj["X_test"], obj["y_train"], obj["y_test"]

    if len(obj) == 1:
        first_val = list(obj.values())[0]
        if isinstance(first_val, dict) and expected.issubset(set(first_val.keys())):
            return (
                first_val["X_train"],
                first_val["X_test"],
                first_val["y_train"],
                first_val["y_test"],
            )

    raise RuntimeError(
        "Cannot parse model5_shap_input.rds. Expected elements: "
        "X_train, X_test, y_train, y_test. pyreadr keys: "
        f"{list(obj.keys())}"
    )


def as_dataframe(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.DataFrame(x)


def as_binary_series(y):
    y_arr = np.array(y).reshape(-1)
    y_ser = pd.Series(y_arr).astype(int)
    if not set(np.unique(y_ser)).issubset({0, 1}):
        raise ValueError("Outcome vector must contain only 0/1 values.")
    return y_ser


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep feature names readable but safe for output.
    """
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


def save_dataframe(df: pd.DataFrame, basename: str) -> None:
    """
    Prefer parquet when available; always save CSV for reviewer readability.
    """
    csv_path = os.path.join(OUT_DIR, f"{basename}.csv")
    df.to_csv(csv_path, index=False)

    parquet_path = os.path.join(OUT_DIR, f"{basename}.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        print(f"[WARN] Could not save {basename}.parquet: {exc}")


def calc_mean_abs_shap(shap_values, X_df: pd.DataFrame, out_name: str) -> pd.DataFrame:
    imp_df = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    imp_df.to_csv(os.path.join(OUT_DIR, out_name), index=False)
    return imp_df


def save_shap_summary_plot(shap_values, X_df: pd.DataFrame, top_features, filename: str,
                           plot_type: str, title: str) -> None:
    """
    Save SHAP summary plot for selected features.
    """
    feature_index = [X_df.columns.get_loc(f) for f in top_features]
    X_sub = X_df[top_features]
    shap_sub_values = shap_values.values[:, feature_index]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_sub_values,
        X_sub,
        plot_type=plot_type,
        show=False,
        max_display=len(top_features)
    )
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=500, bbox_inches="tight")
    plt.close()


def permutation_importance_xgb(booster, X, y, n_repeats=10, random_state=20250101):
    """
    Permutation feature importance using AUC loss in the held-out test set.
    """
    rng = np.random.default_rng(random_state)

    base_pred = booster.predict(xgb.DMatrix(X))
    base_auc = roc_auc_score(y, base_pred)

    rows = []

    for col in X.columns:
        losses = []

        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)

            perm_pred = booster.predict(xgb.DMatrix(X_perm))
            perm_auc = roc_auc_score(y, perm_pred)
            losses.append(base_auc - perm_auc)

        rows.append({
            "feature": col,
            "mean_auc_loss": float(np.mean(losses)),
            "sd_auc_loss": float(np.std(losses)),
            "n_repeats": n_repeats
        })

    pfi_df = pd.DataFrame(rows).sort_values("mean_auc_loss", ascending=False)
    return base_auc, pfi_df


# ------------------------------------------------------------------------------
# 1. LOAD INPUTS FROM SCRIPT 03
# ------------------------------------------------------------------------------

require_file(
    SHAP_INPUT_RDS,
    "Run scripts/03_Multivariable_XGBoost_Modeling.R before this script."
)

require_file(
    MODEL_JSON,
    "Run scripts/03_Multivariable_XGBoost_Modeling.R before this script."
)

print("[INFO] Loading SHAP input RDS...")
r_obj = pyreadr.read_r(SHAP_INPUT_RDS)

X_train, X_test, y_train, y_test = extract_shap_rds(r_obj)

X_train = sanitize_feature_names(as_dataframe(X_train))
X_test = sanitize_feature_names(as_dataframe(X_test))
y_train = as_binary_series(y_train)
y_test = as_binary_series(y_test)

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Train/test feature dimension mismatch.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train/test feature names do not match.")

print(f"[INFO] X_train shape: {X_train.shape}")
print(f"[INFO] X_test shape:  {X_test.shape}")
print(f"[INFO] Test cases:    {int(y_test.sum())}")
print(f"[INFO] Test controls: {int((1 - y_test).sum())}")

# ------------------------------------------------------------------------------
# 2. LOAD XGBOOST MODEL
# ------------------------------------------------------------------------------

print("[INFO] Loading final Model 5 XGBoost booster...")
booster = xgb.Booster()
booster.load_model(MODEL_JSON)

# ------------------------------------------------------------------------------
# 3. COMPUTE SHAP VALUES
# ------------------------------------------------------------------------------

print("[INFO] Computing SHAP values...")
explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")

shap_train = explainer(X_train)
shap_test = explainer(X_test)

shap_train_df = pd.DataFrame(shap_train.values, columns=X_train.columns)
shap_test_df = pd.DataFrame(shap_test.values, columns=X_test.columns)

save_dataframe(shap_train_df, "shap_values_train")
save_dataframe(shap_test_df, "shap_values_test")

train_imp_df = calc_mean_abs_shap(
    shap_train,
    X_train,
    "shap_mean_abs_importance_train.csv"
)

test_imp_df = calc_mean_abs_shap(
    shap_test,
    X_test,
    "shap_mean_abs_importance_test.csv"
)

# ------------------------------------------------------------------------------
# 4. GENERATE FIGURE A.8
# ------------------------------------------------------------------------------

top_features_test = test_imp_df["feature"].head(TOP_N_FEATURES).tolist()
top_features_train = train_imp_df["feature"].head(TOP_N_FEATURES).tolist()

save_shap_summary_plot(
    shap_test,
    X_test,
    top_features_test,
    "FigureA8_SHAP_beeswarm_test.png",
    "dot",
    "SHAP summary: final integrated ADRD model, held-out test set"
)

save_shap_summary_plot(
    shap_test,
    X_test,
    top_features_test,
    "FigureA8_SHAP_bar_test.png",
    "bar",
    "SHAP mean absolute importance: held-out test set"
)

save_shap_summary_plot(
    shap_train,
    X_train,
    top_features_train,
    "FigureA8_SHAP_beeswarm_train.png",
    "dot",
    "SHAP summary: final integrated ADRD model, training set"
)

save_shap_summary_plot(
    shap_train,
    X_train,
    top_features_train,
    "FigureA8_SHAP_bar_train.png",
    "bar",
    "SHAP mean absolute importance: training set"
)

# ------------------------------------------------------------------------------
# 5. PERMUTATION FEATURE IMPORTANCE IN HELD-OUT TEST SET
# ------------------------------------------------------------------------------

print("[INFO] Running permutation feature importance in held-out test set...")

test_auc, pfi_df = permutation_importance_xgb(
    booster=booster,
    X=X_test,
    y=y_test,
    n_repeats=N_PERMUTATION_REPEATS,
    random_state=RANDOM_STATE
)

pfi_df.to_csv(
    os.path.join(OUT_DIR, "permutation_importance_test_auc_loss.csv"),
    index=False
)

summary_df = pd.DataFrame({
    "metric": ["heldout_test_auc_unpermuted", "n_test", "n_test_cases", "n_test_controls"],
    "value": [
        test_auc,
        X_test.shape[0],
        int(y_test.sum()),
        int((1 - y_test).sum())
    ]
})

summary_df.to_csv(os.path.join(OUT_DIR, "shap_analysis_summary.csv"), index=False)

# ------------------------------------------------------------------------------
# COMPLETION MESSAGE
# ------------------------------------------------------------------------------

print("\nDONE: SHAP and permutation importance analysis completed.")
print("Outputs written to:", OUT_DIR)
print("Main revised output:")
print("- FigureA8_SHAP_beeswarm_test.png")
print("- FigureA8_SHAP_bar_test.png")
print("Supporting outputs:")
print("- shap_mean_abs_importance_test.csv")
print("- shap_values_test.csv/parquet")
print("- permutation_importance_test_auc_loss.csv")
print(f"Held-out test AUC from loaded booster: {test_auc:.4f}")
