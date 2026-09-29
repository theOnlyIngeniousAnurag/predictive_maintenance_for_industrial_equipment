"""
Phase 7: Model Benchmarking module.
Trains and compares controlled Decision Tree, Random Forest, and Gradient Boosting models
against the established Logistic Regression baseline on the C-MAPSS FD001 dataset.
Evaluates strictly on the 20-engine validation set. The official 100-engine test set remains LOCKED.
"""

import os
import json
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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
    
    # Specificity = TN / (TN + FP)
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


def run_benchmark_pipeline() -> dict:
    """
    Main Phase 7 benchmarking execution pipeline.
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    print("=== Phase 7: Model Benchmarking ===")

    # 1. Load preprocessed dataset arrays
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

    val_parquet_path = os.path.join(PROCESSED_DATA_DIR, "val_engineered.parquet")
    val_df = pd.read_parquet(val_parquet_path)

    # 2. Define Controlled Benchmark Models
    # Conservative standard configurations to prevent uncontrolled overfitting
    models_dict = {
        "Logistic Regression (Baseline)": {
            "model": LogisticRegression(random_state=42, max_iter=1000, C=1.0, class_weight=None),
            "exp_id": "EXP-001-baseline-logistic",
            "artifact_filename": "baseline_logistic.joblib",
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=None,
            ),
            "exp_id": "EXP-003-decision-tree",
            "artifact_filename": "benchmark_decision_tree.joblib",
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=None,
                n_jobs=-1,
            ),
            "exp_id": "EXP-004-random-forest",
            "artifact_filename": "benchmark_random_forest.joblib",
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(
                random_state=42,
                max_iter=100,
                max_depth=8,
                min_samples_leaf=10,
            ),
            "exp_id": "EXP-005-gradient-boosting",
            "artifact_filename": "benchmark_gradient_boosting.joblib",
        },
    }

    benchmark_results = {}
    overfitting_analysis = {}
    confusion_matrices = {}
    all_predictions = {
        "unit_number": val_df["unit_number"].values,
        "time_cycles": val_df["time_cycles"].values,
        "true_target": y_val,
    }

    baseline_val_metrics = None

    # 3. Train and Evaluate each candidate model
    for name, spec in models_dict.items():
        clf = spec["model"]
        exp_id = spec["exp_id"]
        artifact_path = os.path.join(MODEL_SAVE_DIR, spec["artifact_filename"])

        print(f"\nEvaluating: {name} [{exp_id}]")

        # Fit model & measure training time
        t_start_train = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t_start_train

        # Save model artifact
        joblib.dump(clf, artifact_path)

        # Inference on training set for overfitting evaluation
        y_train_prob = clf.predict_proba(X_train)[:, 1]
        train_metrics = compute_full_metrics(y_train, y_train_prob, threshold=0.50)

        # Inference on validation set & measure inference time
        t_start_inf = time.time()
        y_val_prob = clf.predict_proba(X_val)[:, 1]
        val_time = time.time() - t_start_inf

        val_metrics = compute_full_metrics(y_val, y_val_prob, threshold=0.50)
        y_val_pred = (y_val_prob >= 0.50).astype(int)

        # Record predicted probabilities in prediction table
        col_prefix = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        all_predictions[f"{col_prefix}_prob"] = y_val_prob
        all_predictions[f"{col_prefix}_pred"] = y_val_pred

        if name == "Logistic Regression (Baseline)":
            baseline_val_metrics = val_metrics

        # Compute delta relative to Logistic Regression
        delta_val = {}
        if baseline_val_metrics is not None:
            delta_val = {
                "delta_recall": val_metrics["recall"] - baseline_val_metrics["recall"],
                "delta_precision": val_metrics["precision"] - baseline_val_metrics["precision"],
                "delta_pr_auc": val_metrics["pr_auc"] - baseline_val_metrics["pr_auc"],
                "delta_f1": val_metrics["f1_score"] - baseline_val_metrics["f1_score"],
                "delta_roc_auc": val_metrics["roc_auc"] - baseline_val_metrics["roc_auc"],
                "delta_fp": val_metrics["false_positives"] - baseline_val_metrics["false_positives"],
                "delta_fn": val_metrics["false_negatives"] - baseline_val_metrics["false_negatives"],
            }

        # Save model benchmark record
        benchmark_results[name] = {
            "experiment_id": exp_id,
            "model_class": clf.__class__.__name__,
            "parameters": clf.get_params(),
            "train_time_sec": round(train_time, 4),
            "inference_time_sec": round(val_time, 4),
            "val_metrics": val_metrics,
            "deltas_vs_baseline": delta_val,
        }

        # Save overfitting comparison
        overfitting_analysis[name] = {
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],
            "train_f1": train_metrics["f1_score"],
            "val_f1": val_metrics["f1_score"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "gap_pr_auc": train_metrics["pr_auc"] - val_metrics["pr_auc"],
            "gap_f1": train_metrics["f1_score"] - val_metrics["f1_score"],
        }

        confusion_matrices[name] = {
            "TN": val_metrics["true_negatives"],
            "FP": val_metrics["false_positives"],
            "FN": val_metrics["false_negatives"],
            "TP": val_metrics["true_positives"],
            "false_alarm_rate": val_metrics["false_alarm_rate"],
        }

        print(f"  Validation Recall:    {val_metrics['recall']:.6f} (Δ: {delta_val.get('delta_recall', 0.0):+.6f})")
        print(f"  Validation Precision: {val_metrics['precision']:.6f} (Δ: {delta_val.get('delta_precision', 0.0):+.6f})")
        print(f"  Validation PR-AUC:    {val_metrics['pr_auc']:.6f} (Δ: {delta_val.get('delta_pr_auc', 0.0):+.6f})")
        print(f"  Validation F1-Score:  {val_metrics['f1_score']:.6f} (Δ: {delta_val.get('delta_f1', 0.0):+.6f})")
        print(f"  Confusion Matrix: TN={val_metrics['true_negatives']}, FP={val_metrics['false_positives']}, FN={val_metrics['false_negatives']}, TP={val_metrics['true_positives']}")

    # 4. Save Artifacts
    # A. JSON summary of model benchmarks
    with open(os.path.join(RESULTS_SAVE_DIR, "model_benchmark_comparison.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # B. CSV comparison table
    rows = []
    for m_name, res in benchmark_results.items():
        v_m = res["val_metrics"]
        rows.append({
            "Model": m_name,
            "Recall": v_m["recall"],
            "Precision": v_m["precision"],
            "PR-AUC": v_m["pr_auc"],
            "FP": v_m["false_positives"],
            "FN": v_m["false_negatives"],
            "F1": v_m["f1_score"],
            "ROC-AUC": v_m["roc_auc"],
            "Specificity": v_m["specificity"],
            "Accuracy": v_m["accuracy"],
            "Train_Time_s": res["train_time_sec"],
            "Inference_Time_s": res["inference_time_sec"],
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(RESULTS_SAVE_DIR, "model_benchmark_comparison.csv"), index=False)

    # C. Save overfitting analysis JSON
    with open(os.path.join(RESULTS_SAVE_DIR, "overfitting_analysis.json"), "w") as f:
        json.dump(overfitting_analysis, f, indent=2)

    # D. Save confusion matrices JSON
    with open(os.path.join(RESULTS_SAVE_DIR, "model_confusion_matrices.json"), "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    # E. Save validation predictions CSV
    preds_df = pd.DataFrame(all_predictions)
    preds_df.to_csv(os.path.join(RESULTS_SAVE_DIR, "benchmark_predictions.csv"), index=False)

    # F. Save metadata
    metadata = {
        "phase": "Phase 7 — Model Benchmarking",
        "timestamp": datetime.datetime.now().isoformat(),
        "threshold": 0.50,
        "n_features": X_train.shape[1],
        "n_train_samples": X_train.shape[0],
        "n_val_samples": X_val.shape[0],
        "models_evaluated": list(models_dict.keys()),
        "test_set_isolation": "VERIFIED_LOCKED",
    }
    with open(os.path.join(RESULTS_SAVE_DIR, "model_benchmark_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nPhase 7 Model Benchmarking completed successfully.")
    return benchmark_results


if __name__ == "__main__":
    run_benchmark_pipeline()
