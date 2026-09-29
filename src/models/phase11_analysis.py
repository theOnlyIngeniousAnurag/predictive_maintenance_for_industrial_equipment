
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
from src.features.build_features import get_feature_columns

RESULTS_SAVE_DIR = "reports/results"
MODEL_PATH = "models/final_model.joblib"
PREDICTIONS_PATH = "reports/results/final_predictions.csv"
FEATURE_COLS_PATH = "data/processed/feature_columns.txt"
DATASET_ARRAYS_PATH = "data/processed/dataset_arrays.npz"

def categorize_feature(name):
    if "_lag_" in name: return "lag"
    if "_roll_" in name: return "rolling"
    if "_diff_" in name: return "diff"
    if "_exp_" in name: return "exp"
    if "_dev_" in name: return "rolling" # deviation is part of rolling analysis
    return "raw"

def run_phase11_analysis():
    print("=== Phase 11: Error Analysis and Explainability ===")
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    
    # Load model, predictions, and training data for permutation importance
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PREDICTIONS_PATH)
    
    # Load training data for permutation importance
    data = np.load(DATASET_ARRAYS_PATH)
    X_train_dev = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_train_dev = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    
    # 1. FP/FN Analysis
    fp_df = df[(df['true_label'] == 0) & (df['pred_label'] == 1)]
    fn_df = df[(df['true_label'] == 1) & (df['pred_label'] == 0)]
    
    # FP Analysis
    fp_analysis = {
        "total_fp": len(fp_df),
        "unique_engines_fp": int(fp_df['unit_number'].nunique()),
        "fp_per_engine": fp_df.groupby('unit_number').size().describe().to_dict(),
        "prob_stats": fp_df['pred_prob'].describe().to_dict()
    }
    
    # FN Analysis
    fn_analysis = {
        "total_fn": len(fn_df),
        "unique_engines_fn": int(fn_df['unit_number'].nunique()),
        "fn_per_engine": fn_df.groupby('unit_number').size().describe().to_dict(),
        "prob_stats": fn_df['pred_prob'].describe().to_dict()
    }
    
    # 2. Feature Importance (Permutation Importance - Reduced n_repeats to save time)
    print("Calculating permutation importance...")
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
        
    perm_importance = permutation_importance(model, X_train_dev, y_train_dev, n_repeats=2, random_state=42, n_jobs=-1)
    
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": perm_importance.importances_mean
    }).sort_values(by="importance", ascending=False)
    
    importances["family"] = importances["feature"].apply(categorize_feature)
    
    # Family Importance
    family_importance = importances.groupby("family")["importance"].sum().sort_values(ascending=False)
    
    # 3. Save Artifacts
    importances.to_csv(os.path.join(RESULTS_SAVE_DIR, "feature_importance.csv"), index=False)
    family_importance.to_csv(os.path.join(RESULTS_SAVE_DIR, "feature_family_importance.csv"))
    
    with open(os.path.join(RESULTS_SAVE_DIR, "error_analysis_summary.json"), "w") as f:
        json.dump({
            "fp_analysis": fp_analysis,
            "fn_analysis": fn_analysis
        }, f, indent=2)
    
    fp_df.to_csv(os.path.join(RESULTS_SAVE_DIR, "false_positive_analysis.csv"), index=False)
    fn_df.to_csv(os.path.join(RESULTS_SAVE_DIR, "false_negative_analysis.csv"), index=False)
    
    print("Phase 11 analysis completed.")

if __name__ == "__main__":
    run_phase11_analysis()
