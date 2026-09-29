"""
Phase 8: Hyperparameter Tuning module.
Tunes HistGradientBoostingClassifier using RandomizedSearchCV and GroupKFold cross-validation (3 splits, grouped by unit_number)
strictly on the 80 training engines. Freezes best parameters, retrains on all training engines,
and evaluates once on the 20-engine validation set at threshold 0.50.
Keeps the official 100-engine test set completely locked.
"""

import os
import json
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
)
import joblib

from src.data.loader import PROCESSED_DATA_DIR

MODEL_SAVE_DIR = "models"
RESULTS_SAVE_DIR = "reports/results"


def compute_full_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    """
    Computes primary, secondary, and confusion matrix metrics at a given threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    accuracy = float(accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    false_alarm_rate = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "specificity": specificity,
        "false_alarm_rate": false_alarm_rate,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_alerts": int(fp + tp),
    }


def run_tuning_pipeline() -> dict:
    """
    Main Phase 8 hyperparameter tuning pipeline using RandomizedSearchCV.
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    print("=== Phase 8: Hyperparameter Tuning (HistGradientBoosting) ===")

    # 1. Load preprocessed arrays and training dataframe (for unit_number grouping)
    npz_path = os.path.join(PROCESSED_DATA_DIR, "dataset_arrays.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Preprocessed arrays not found at '{npz_path}'.")

    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    train_parquet_path = os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet")
    if not os.path.exists(train_parquet_path):
        raise FileNotFoundError(f"Training dataframe not found at '{train_parquet_path}'.")
    train_df = pd.read_parquet(train_parquet_path)
    train_groups = train_df["unit_number"].values

    assert len(train_groups) == len(X_train), "Training groups length must match X_train rows."

    # 2. Define Controlled Search Space for HistGradientBoostingClassifier
    estimator = HistGradientBoostingClassifier(random_state=42)
    
    param_distributions = {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_iter": [100, 200],
        "max_leaf_nodes": [31, 63],
        "min_samples_leaf": [10, 20],
        "l2_regularization": [0.0, 1.0],
    }

    # GroupKFold with 3 splits to ensure engine-level separation in inner cross-validation efficiently
    gkf = GroupKFold(n_splits=3)

    print(f"Starting RandomizedSearchCV with GroupKFold (3 splits, n_iter=6):")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Unique training engines: {len(np.unique(train_groups))}")
    print(f"  - Scoring metric: average_precision (PR-AUC)")

    t_start_search = time.time()
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=6,
        scoring="average_precision",
        cv=gkf,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    
    search.fit(X_train, y_train, groups=train_groups)
    search_duration = time.time() - t_start_search

    best_params = search.best_params_
    best_cv_score = float(search.best_score_)

    print(f"\nRandomizedSearch completed in {search_duration:.2f} seconds.")
    print(f"  - Best CV PR-AUC Score: {best_cv_score:.6f}")
    print(f"  - Best Hyperparameters: {best_params}")

    # 3. Retrain Best Model on All Training Engines
    print("\nRetraining tuned model on all 80 training engines...")
    tuned_model = HistGradientBoostingClassifier(**best_params, random_state=42)
    t_start_train = time.time()
    tuned_model.fit(X_train, y_train)
    retrain_duration = time.time() - t_start_train

    # Save tuned model artifact
    model_path = os.path.join(MODEL_SAVE_DIR, "tuned_gradient_boosting.joblib")
    joblib.dump(tuned_model, model_path)
    print(f"Saved tuned model artifact to: {model_path}")

    # 4. Evaluate Tuned Model on Training and Validation Splits (Threshold = 0.50)
    y_train_prob = tuned_model.predict_proba(X_train)[:, 1]
    train_metrics = compute_full_metrics(y_train, y_train_prob, threshold=0.50)

    y_val_prob = tuned_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_full_metrics(y_val, y_val_prob, threshold=0.50)

    print("\nTuned Model Performance on Validation Set (Threshold = 0.50):")
    print(f"  Recall:      {val_metrics['recall']:.6f}")
    print(f"  Precision:   {val_metrics['precision']:.6f}")
    print(f"  PR-AUC:      {val_metrics['pr_auc']:.6f}")
    print(f"  F1-Score:    {val_metrics['f1_score']:.6f}")
    print(f"  ROC-AUC:     {val_metrics['roc_auc']:.6f}")
    print(f"  Confusion Matrix: TN={val_metrics['true_negatives']}, FP={val_metrics['false_positives']}, FN={val_metrics['false_negatives']}, TP={val_metrics['true_positives']}")

    # 5. Load Phase 7 Untuned HistGradientBoosting Metrics for Comparison
    comparison_json_path = os.path.join(RESULTS_SAVE_DIR, "model_benchmark_comparison.json")
    untuned_val_metrics = {}
    if os.path.exists(comparison_json_path):
        with open(comparison_json_path, "r") as f:
            comp_data = json.load(f)
            if "HistGradientBoosting" in comp_data:
                untuned_val_metrics = comp_data["HistGradientBoosting"]["val_metrics"]

    deltas = {}
    if untuned_val_metrics:
        deltas = {
            "delta_recall": val_metrics["recall"] - untuned_val_metrics["recall"],
            "delta_precision": val_metrics["precision"] - untuned_val_metrics["precision"],
            "delta_pr_auc": val_metrics["pr_auc"] - untuned_val_metrics["pr_auc"],
            "delta_f1": val_metrics["f1_score"] - untuned_val_metrics["f1_score"],
            "delta_roc_auc": val_metrics["roc_auc"] - untuned_val_metrics["roc_auc"],
            "delta_fp": val_metrics["false_positives"] - untuned_val_metrics["false_positives"],
            "delta_fn": val_metrics["false_negatives"] - untuned_val_metrics["false_negatives"],
        }

    # 6. Save Artifacts
    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_csv_path = os.path.join(RESULTS_SAVE_DIR, "hyperparameter_search_results.csv")
    cv_results_df.to_csv(cv_results_csv_path, index=False)

    search_summary = {
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "search_duration_sec": search_duration,
        "n_candidates_evaluated": len(cv_results_df),
    }
    with open(os.path.join(RESULTS_SAVE_DIR, "hyperparameter_search_results.json"), "w") as f:
        json.dump(search_summary, f, indent=2)

    with open(os.path.join(RESULTS_SAVE_DIR, "tuned_model_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)

    comparison_payload = {
        "experiment_id": "EXP-006-gradient-boosting-tuned",
        "timestamp": datetime.datetime.now().isoformat(),
        "untuned_validation_metrics": untuned_val_metrics,
        "tuned_validation_metrics": val_metrics,
        "deltas_tuned_minus_untuned": deltas,
        "overfitting_gap": {
            "train_pr_auc": train_metrics["pr_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "gap_pr_auc": train_metrics["pr_auc"] - val_metrics["pr_auc"],
            "train_f1": train_metrics["f1_score"],
            "val_f1": val_metrics["f1_score"],
            "gap_f1": train_metrics["f1_score"] - val_metrics["f1_score"],
        },
    }
    with open(os.path.join(RESULTS_SAVE_DIR, "tuned_vs_untuned_comparison.json"), "w") as f:
        json.dump(comparison_payload, f, indent=2)

    print("Phase 8 Hyperparameter Tuning completed successfully.")
    return comparison_payload


if __name__ == "__main__":
    run_tuning_pipeline()
