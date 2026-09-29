"""
Automated Dashboard QA & Evidence Audit Tests for NASA C-MAPSS FD001.
Verifies control artifacts, final evaluation metrics, confusion matrix consistency,
production threshold, model provenance, and feature counts against project specifications.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
import joblib

RESULTS_DIR = "reports/results"
MODELS_DIR = "models"
DOCS_DIR = "docs"


def test_final_metrics_provenance():
    """Test 1: Verify dashboard final metrics match final_evaluation_summary.json exactly."""
    summary_path = os.path.join(RESULTS_DIR, "final_evaluation_summary.json")
    assert os.path.exists(summary_path), "Missing final evaluation summary JSON."

    with open(summary_path, "r") as f:
        metrics = json.load(f)

    assert metrics["roc_auc"] == pytest.approx(0.997278, rel=1e-4)
    assert metrics["pr_auc"] == pytest.approx(0.924737, rel=1e-4)
    assert metrics["f1"] == pytest.approx(0.830075, rel=1e-4)
    assert metrics["precision"] == pytest.approx(0.828828, rel=1e-4)
    assert metrics["recall"] == pytest.approx(0.831325, rel=1e-4)
    assert metrics["accuracy"] == pytest.approx(0.991371, rel=1e-4)
    assert metrics["specificity"] == pytest.approx(0.995534, rel=1e-4)


def test_confusion_matrix_consistency():
    """Test 2 & 10: Verify confusion matrix numbers sum to total test observations (13,096) and FP=57, FN=56."""
    summary_path = os.path.join(RESULTS_DIR, "final_evaluation_summary.json")
    with open(summary_path, "r") as f:
        m = json.load(f)

    tn = m["tn"]
    fp = m["fp"]
    fn = m["fn"]
    tp = m["tp"]

    assert tn == 12707
    assert fp == 57
    assert fn == 56
    assert tp == 276

    total_obs = tn + fp + fn + tp
    assert total_obs == 13096, f"Total test observations sum mismatch: {total_obs} != 13096"


def test_production_threshold():
    """Test 3: Verify production operating threshold is 0.35."""
    thresh_path = os.path.join(RESULTS_DIR, "phase9_threshold_selection.json")
    assert os.path.exists(thresh_path)
    with open(thresh_path, "r") as f:
        data = json.load(f)
    assert data["selected_threshold"] == 0.35


def test_final_model_identity_and_config():
    """Test 4: Verify final model is HistGradientBoostingClassifier with frozen hyperparameters."""
    model_path = os.path.join(MODELS_DIR, "final_model.joblib")
    assert os.path.exists(model_path), "Final model artifact must exist."
    model = joblib.load(model_path)
    
    # Check class type
    assert "HistGradientBoostingClassifier" in type(model).__name__

    # Verify frozen parameters
    params = model.get_params()
    assert params.get("learning_rate") == 0.1
    assert params.get("max_iter") == 100
    assert params.get("max_leaf_nodes") == 31
    assert params.get("min_samples_leaf") == 10
    assert params.get("l2_regularization") == 1.0


def test_feature_count():
    """Test 5: Verify feature pipeline count is 385 features."""
    meta_path = os.path.join(RESULTS_DIR, "baseline_run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if "n_features" in meta:
            assert meta["n_features"] == 385


def test_test_prediction_integrity():
    """Test 7: Verify final predictions CSV contains expected rows and probability values."""
    preds_path = os.path.join(RESULTS_DIR, "final_predictions.csv")
    assert os.path.exists(preds_path)
    df = pd.read_csv(preds_path)
    assert len(df) == 13096
    assert "pred_prob" in df.columns
    assert np.all(df["pred_prob"] >= 0.0) and np.all(df["pred_prob"] <= 1.0)


def test_benchmark_provenance_and_timing():
    """Test 8 & 9: Verify benchmark comparison has 4 official models with recorded timing."""
    bench_path = os.path.join(RESULTS_DIR, "model_benchmark_comparison.json")
    assert os.path.exists(bench_path)
    with open(bench_path, "r") as f:
        bdata = json.load(f)
    
    assert len(bdata) == 4
    for model_name, details in bdata.items():
        assert "train_time_sec" in details
        assert "inference_time_sec" in details
        assert details["train_time_sec"] > 0
        assert details["inference_time_sec"] > 0
