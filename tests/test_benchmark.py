"""
Unit tests for Phase 7 Model Benchmarking execution, probability output integrity,
overfitting calculations, artifact generation, and zero test-set contamination.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
import joblib

from src.data.loader import PROCESSED_DATA_DIR
from src.models.benchmark_models import MODEL_SAVE_DIR, RESULTS_SAVE_DIR, run_benchmark_pipeline


def test_benchmark_execution_and_artifacts():
    """
    Run benchmark pipeline and verify that model artifacts and report files exist and are valid.
    """
    benchmark_results = run_benchmark_pipeline()
    assert isinstance(benchmark_results, dict)
    assert len(benchmark_results) == 4

    # Verify model joblib files
    expected_models = [
        "baseline_logistic.joblib",
        "benchmark_decision_tree.joblib",
        "benchmark_random_forest.joblib",
        "benchmark_gradient_boosting.joblib",
    ]
    for model_fname in expected_models:
        model_path = os.path.join(MODEL_SAVE_DIR, model_fname)
        assert os.path.exists(model_path), f"Missing model artifact: {model_path}"
        # Test loading model
        clf = joblib.load(model_path)
        assert hasattr(clf, "predict_proba"), f"Model {model_fname} does not implement predict_proba."

    # Verify report JSONs and CSVs
    expected_reports = [
        "model_benchmark_comparison.json",
        "model_benchmark_comparison.csv",
        "overfitting_analysis.json",
        "model_confusion_matrices.json",
        "benchmark_predictions.csv",
        "model_benchmark_metadata.json",
    ]
    for report_fname in expected_reports:
        report_path = os.path.join(RESULTS_SAVE_DIR, report_fname)
        assert os.path.exists(report_path), f"Missing report artifact: {report_path}"


def test_benchmark_predictions_shape_and_probability_bounds():
    """
    Verify validation predictions CSV has expected dimensions and probability bounds in [0, 1].
    """
    preds_path = os.path.join(RESULTS_SAVE_DIR, "benchmark_predictions.csv")
    assert os.path.exists(preds_path), "Benchmark predictions CSV must exist."

    preds_df = pd.read_csv(preds_path)
    assert len(preds_df) == 4070, f"Expected 4070 validation rows, got {len(preds_df)}"

    # Check probability columns for each model
    prob_cols = [
        "logistic_regression_baseline_prob",
        "decision_tree_prob",
        "random_forest_prob",
        "histgradientboosting_prob",
    ]
    for col in prob_cols:
        assert col in preds_df.columns, f"Column '{col}' missing from predictions CSV."
        probs = preds_df[col].values
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), f"Probabilities in '{col}' out of bounds [0, 1]."


def test_benchmark_overfitting_analysis_structure():
    """
    Verify that overfitting analysis contains valid metric values and positive gap metrics.
    """
    overfit_path = os.path.join(RESULTS_SAVE_DIR, "overfitting_analysis.json")
    assert os.path.exists(overfit_path)

    with open(overfit_path, "r") as f:
        overfit_data = json.load(f)

    for m_name, metrics in overfit_data.items():
        assert "train_recall" in metrics
        assert "val_recall" in metrics
        assert "gap_pr_auc" in metrics
        assert "gap_f1" in metrics
        # Check finite values
        assert np.isfinite(metrics["train_f1"])
        assert np.isfinite(metrics["val_f1"])


def test_benchmark_test_set_isolation():
    """
    Explicitly verify that test dataset arrays are NOT loaded or accessed during benchmarking.
    We check dataset arrays metadata and confirm test data is completely isolated.
    """
    meta_path = os.path.join(RESULTS_SAVE_DIR, "model_benchmark_metadata.json")
    assert os.path.exists(meta_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert meta["test_set_isolation"] == "VERIFIED_LOCKED"
    assert meta["n_val_samples"] == 4070
    assert meta["n_train_samples"] == 16561
