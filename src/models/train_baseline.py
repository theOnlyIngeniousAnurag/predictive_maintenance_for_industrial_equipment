"""
Phase 6: Baseline Model Training.
Trains a scikit-learn Logistic Regression model on the C-MAPSS FD001 dataset
using the 80-engine training partition and evaluates it on the 20-engine validation partition.
Keeps the official 100-engine test partition completely locked.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def train_and_evaluate_baseline(
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
) -> None:
    """
    Main pipeline for training and evaluating the Logistic Regression baseline.
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    print("=== Phase 6: Baseline Logistic Regression ===")

    # 1. Load preprocessed and scaled data arrays
    npz_path = os.path.join(PROCESSED_DATA_DIR, "dataset_arrays.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Preprocessed arrays not found at '{npz_path}'. "
            "Please run 'src/data/split.py' first."
        )

    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Load validation dataframe to align predictions with unit_number and time_cycles
    val_parquet_path = os.path.join(PROCESSED_DATA_DIR, "val_engineered.parquet")
    if not os.path.exists(val_parquet_path):
        raise FileNotFoundError(
            f"Validation dataframe not found at '{val_parquet_path}'."
        )
    val_df = pd.read_parquet(val_parquet_path)

    # 2. Instantiate and train baseline model
    # Conservative standard configuration (default L2 regularization, standard max_iter)
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        C=C,
        class_weight=None,  # Standard baseline without imbalance adjustments
    )
    
    print(f"Fitting Logistic Regression model:")
    print(f"  - Features shape: {X_train.shape}")
    print(f"  - Parameters: random_state={random_state}, max_iter={max_iter}, C={C}")
    model.fit(X_train, y_train)

    # Save trained baseline model
    model_path = os.path.join(MODEL_SAVE_DIR, "baseline_logistic.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")

    # 3. Model Prediction on Validation Split
    # Evaluate strictly on validation split
    probs_val = model.predict_proba(X_val)[:, 1]
    preds_val = (probs_val >= 0.50).astype(int)  # Standard 0.50 threshold

    # 4. Metric Computation
    precision = precision_score(y_val, preds_val)
    recall = recall_score(y_val, preds_val)
    f1 = f1_score(y_val, preds_val)
    roc_auc = roc_auc_score(y_val, probs_val)
    pr_auc = average_precision_score(y_val, probs_val)
    accuracy = accuracy_score(y_val, preds_val)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_val, preds_val).ravel()

    print("\nBaseline Model Performance on Validation Set:")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall:    {recall:.6f}")
    print(f"  F1-Score:  {f1:.6f}")
    print(f"  ROC-AUC:   {roc_auc:.6f}")
    print(f"  PR-AUC:    {pr_auc:.6f}")
    print(f"  Accuracy:  {accuracy:.6f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Probability distribution summary
    prob_min = float(probs_val.min())
    prob_max = float(probs_val.max())
    prob_mean = float(probs_val.mean())
    prob_std = float(probs_val.std())

    # 5. Save Artifacts
    # A. Save metrics JSON
    metrics_dict = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(accuracy),
    }
    with open(os.path.join(RESULTS_SAVE_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # B. Save confusion matrix JSON
    cm_dict = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_positive_predictions": int(fp + tp),
        "total_negative_predictions": int(tn + fn),
        "actual_positive_count": int(fn + tp),
        "actual_negative_count": int(tn + fp),
    }
    with open(os.path.join(RESULTS_SAVE_DIR, "baseline_confusion_matrix.json"), "w") as f:
        json.dump(cm_dict, f, indent=2)

    # C. Save baseline predictions CSV
    predictions_df = pd.DataFrame({
        "unit_number": val_df["unit_number"].values,
        "time_cycles": val_df["time_cycles"].values,
        "true_target": y_val,
        "predicted_probability": probs_val,
        "predicted_label": preds_val,
    })
    predictions_csv_path = os.path.join(RESULTS_SAVE_DIR, "baseline_predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Saved baseline predictions to: {predictions_csv_path}")

    # D. Save run metadata JSON
    metadata = {
        "experiment_id": "EXP-001-baseline-logistic",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": "LogisticRegression",
        "parameters": {
            "random_state": random_state,
            "max_iter": max_iter,
            "C": C,
            "class_weight": None,
        },
        "feature_count": int(X_train.shape[1]),
        "training_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "probability_distribution": {
            "min": prob_min,
            "max": prob_max,
            "mean": prob_mean,
            "std": prob_std,
        },
    }
    with open(os.path.join(RESULTS_SAVE_DIR, "baseline_run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Phase 6 Baseline execution finished successfully.")


if __name__ == "__main__":
    train_and_evaluate_baseline()
