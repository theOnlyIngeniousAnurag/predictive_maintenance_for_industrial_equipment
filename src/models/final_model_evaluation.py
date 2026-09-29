
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score
)
from src.data.loader import PROCESSED_DATA_DIR, load_raw_test_data, compute_test_rul_and_target
from src.features.build_features import engineer_all_features, get_feature_columns

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
FROZEN_THRESHOLD = 0.35

def compute_threshold_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "tp": int(tp),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }

def run_final_pipeline():
    print("=== Phase 10: Final Model Training and Evaluation ===")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    
    # 1. Train final model on ALL development data
    data = np.load(os.path.join(PROCESSED_DATA_DIR, "dataset_arrays.npz"))
    X_train_dev = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_train_dev = np.concatenate([data["y_train"], data["y_val"]], axis=0)
    
    print("Training final model on all development engines...")
    final_model = HistGradientBoostingClassifier(**BEST_PARAMS, random_state=42)
    final_model.fit(X_train_dev, y_train_dev)
    joblib.dump(final_model, os.path.join(MODEL_SAVE_DIR, "final_model.joblib"))
    
    # 2. Evaluate on official test set (LOCKED UNTIL NOW)
    print("Evaluating on official test set...")
    
    # Load and process test set
    test_df, rul_df = load_raw_test_data()
    test_df = compute_test_rul_and_target(test_df, rul_df)
    
    # Preprocessing: Need feature_scaler.joblib
    scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, "feature_scaler.joblib"))
    
    # Engineer features and select columns
    test_feat_df = engineer_all_features(test_df)
    feature_cols = get_feature_columns(test_feat_df)
    
    X_test = test_feat_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    y_test = test_feat_df["target"].values
    
    y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
    
    # Final evaluation
    test_metrics = compute_threshold_metrics(y_test, y_test_prob, FROZEN_THRESHOLD)
    
    # Save artifacts
    with open(os.path.join(RESULTS_SAVE_DIR, "final_evaluation_summary.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    predictions_df = test_feat_df[["unit_number", "time_cycles"]].copy()
    predictions_df["true_label"] = y_test
    predictions_df["pred_prob"] = y_test_prob
    predictions_df["pred_label"] = (y_test_prob >= FROZEN_THRESHOLD).astype(int)
    predictions_df.to_csv(os.path.join(RESULTS_SAVE_DIR, "final_predictions.csv"), index=False)
    
    print("Phase 10 completed.")

if __name__ == "__main__":
    run_final_pipeline()
