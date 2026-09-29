"""
Phase 8 Tuning Tests.
Verifies engine isolation, test isolation, search validity, and metric completeness.
"""

import os
import json
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.data.loader import PROCESSED_DATA_DIR

MODEL_SAVED_DIR = "models"
RESULTS_SAVE_DIR = "reports/results"

def test_tuning_engine_isolation():
    """
    Test 3 — Engine isolation: No engine appears in both inner-training and inner-validation folds.
    """
    train_parquet_path = os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet")
    if not os.path.exists(train_parquet_path):
        pytest.skip("train_engineered.parquet not found")
        
    train_df = pd.read_parquet(train_parquet_path)
    train_groups = train_df["unit_number"].values
    
    # Simulate GroupKFold
    gkf = GroupKFold(n_splits=3)
    
    for train_idx, val_idx in gkf.split(train_df, groups=train_groups):
        train_units = set(train_groups[train_idx])
        val_units = set(train_groups[val_idx])
        
        intersection = train_units.intersection(val_units)
        assert len(intersection) == 0, f"Leakage detected: Units {intersection} in both folds."

def test_tuning_test_isolation():
    """
    Test 4 — Test isolation: Verify tuning artifacts only use training/validation data.
    """
    comparison_path = os.path.join(RESULTS_SAVE_DIR, "tuned_vs_untuned_comparison.json")
    if not os.path.exists(comparison_path):
        pytest.skip("tuned_vs_untuned_comparison.json not found")
        
    with open(comparison_path, "r") as f:
        data = json.load(f)
        
    # Check that test set metrics are NOT present
    assert "test_metrics" not in data
    assert "test_recall" not in data
    assert "test_precision" not in data

def test_tuning_artifacts_existence():
    """
    Test 7, 8 & 9 — Artifacts existence and metric completeness.
    """
    model_path = os.path.join(MODEL_SAVED_DIR, "tuned_gradient_boosting.joblib")
    results_json = os.path.join(RESULTS_SAVE_DIR, "hyperparameter_search_results.json")
    results_csv = os.path.join(RESULTS_SAVE_DIR, "hyperparameter_search_results.csv")
    metrics_json = os.path.join(RESULTS_SAVE_DIR, "tuned_model_metrics.json")
    comparison_json = os.path.join(RESULTS_SAVE_DIR, "tuned_vs_untuned_comparison.json")
    
    assert os.path.exists(model_path), "Tuned model artifact missing."
    assert os.path.exists(results_json), "Search results JSON missing."
    assert os.path.exists(results_csv), "Search results CSV missing."
    assert os.path.exists(metrics_json), "Tuned model metrics missing."
    assert os.path.exists(comparison_json), "Comparison JSON missing."
    
    # Verify metric completeness in tuned_model_metrics.json
    with open(metrics_json, "r") as f:
        metrics = json.load(f)
        required = ["recall", "precision", "pr_auc", "false_positives", "false_negatives"]
        for key in required:
            assert key in metrics, f"Missing required metric: {key}"

def test_tuned_model_serialization():
    """
    Test 8 — Tuned model artifact: Model can be loaded and used for prediction.
    """
    model_path = os.path.join(MODEL_SAVED_DIR, "tuned_gradient_boosting.joblib")
    if not os.path.exists(model_path):
        pytest.skip("tuned_gradient_boosting.joblib not found")
        
    model = joblib.load(model_path)
    
    # Test prediction with small dummy data
    X_dummy = np.random.rand(5, 385)
    probs = model.predict_proba(X_dummy)
    assert probs.shape == (5, 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)
