import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score
)
from src.data.loader import PROCESSED_DATA_DIR

MODEL_SAVE_DIR = "models"
RESULTS_SAVE_DIR = "reports/results"

# Phase 8 best parameters
BEST_PARAMS = {
    'min_samples_leaf': 10,
    'max_leaf_nodes': 31,
    'max_iter': 100,
    'learning_rate': 0.1,
    'l2_regularization': 1.0
}

def compute_threshold_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "threshold": threshold,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "tp": int(tp),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred))
    }

def run_threshold_optimization():
    print("=== Phase 9: Threshold Optimization ===")
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    
    # 1. Load data
    data = np.load(os.path.join(PROCESSED_DATA_DIR, "dataset_arrays.npz"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet"))
    train_groups = train_df["unit_number"].values

    # 2. Generate OOF probabilities
    print("Generating OOF probabilities...")
    oof_probs = np.zeros(len(y_train))
    gkf = GroupKFold(n_splits=3)
    
    for train_idx, val_idx in gkf.split(X_train, y_train, groups=train_groups):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model = HistGradientBoostingClassifier(**BEST_PARAMS, random_state=42)
        model.fit(X_fold_train, y_fold_train)
        oof_probs[val_idx] = model.predict_proba(X_fold_val)[:, 1]

    # 3. Analyze threshold grid
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    oof_analysis = [compute_threshold_metrics(y_train, oof_probs, t) for t in thresholds]
    
    oof_df = pd.DataFrame(oof_analysis)
    oof_df.to_csv(os.path.join(RESULTS_SAVE_DIR, "threshold_oof_analysis.csv"), index=False)
    with open(os.path.join(RESULTS_SAVE_DIR, "threshold_oof_analysis.json"), "w") as f:
        json.dump(oof_analysis, f, indent=2)

    # 4. Select threshold: Rule - max F1
    best_row = oof_df.loc[oof_df['f1'].idxmax()]
    selected_threshold = float(best_row['threshold'])
    print(f"Selected threshold: {selected_threshold} (based on max F1)")

    with open(os.path.join(RESULTS_SAVE_DIR, "phase9_threshold_selection.json"), "w") as f:
        json.dump({"selected_threshold": selected_threshold, "rule": "max_f1"}, f, indent=2)

    # 5. Final fit and validation evaluation
    print(f"Fitting final model with threshold {selected_threshold}...")
    final_model = HistGradientBoostingClassifier(**BEST_PARAMS, random_state=42)
    final_model.fit(X_train, y_train)
    
    y_val_prob = final_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_threshold_metrics(y_val, y_val_prob, selected_threshold)
    
    # Add PR/ROC AUC
    val_metrics["pr_auc"] = float(average_precision_score(y_val, y_val_prob))
    val_metrics["roc_auc"] = float(roc_auc_score(y_val, y_val_prob))
    
    with open(os.path.join(RESULTS_SAVE_DIR, "phase9_validation_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)

    print("Phase 9 completed.")

if __name__ == "__main__":
    run_threshold_optimization()
