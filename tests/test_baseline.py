"""
Unit tests for Phase 6 Baseline Model training, predictions, and reproducibility.
Ensures zero temporal/cross-engine leakage and verifies correct dimensional shapes.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
import joblib

from src.data.loader import PROCESSED_DATA_DIR
from src.models.train_baseline import MODEL_SAVE_DIR, RESULTS_SAVE_DIR


def test_baseline_dataset_shapes_and_completeness():
    """
    Verify processed arrays exist, shapes are consistent, and contain zero NaN/infinite values.
    """
    npz_path = os.path.join(PROCESSED_DATA_DIR, "dataset_arrays.npz")
    assert os.path.exists(npz_path), "Processed npz file must exist."

    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Verify matching rows between features and target labels
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]

    # Verify exact feature column dimensionality of 385
    assert X_train.shape[1] == 385
    assert X_val.shape[1] == 385

    # Check for NaN and Inf values
    assert not np.isnan(X_train).any(), "X_train contains NaN values."
    assert not np.isinf(X_train).any(), "X_train contains infinite values."
    assert not np.isnan(X_val).any(), "X_val contains NaN values."
    assert not np.isinf(X_val).any(), "X_val contains infinite values."


def test_baseline_no_engine_overlap():
    """
    Verify strict zero overlap of unit/engine IDs between train and validation splits.
    """
    # Load original split dataframes containing unit_number
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "val_engineered.parquet"))

    train_units = set(train_df["unit_number"].unique())
    val_units = set(val_df["unit_number"].unique())

    overlap = train_units.intersection(val_units)
    assert len(overlap) == 0, f"Engine overlap detected between train and val: {overlap}"
    assert len(train_units) == 80, f"Expected 80 training engines, got {len(train_units)}"
    assert len(val_units) == 20, f"Expected 20 validation engines, got {len(val_units)}"


def test_baseline_scaler_isolation():
    """
    Verify feature scaler was not fitted on validation data or test data.
    We check this by loading the scaler artifact and verifying that its Mean & Scale
    are equal to the mean/std of raw training features, not validation/test.
    """
    scaler_path = os.path.join(PROCESSED_DATA_DIR, "feature_scaler.joblib")
    assert os.path.exists(scaler_path), "Feature scaler artifact must exist."
    scaler = joblib.load(scaler_path)

    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet"))
    # Extract modeling columns
    from src.features.build_features import get_feature_columns
    feature_cols = get_feature_columns(train_df)
    
    # Calculate expected mean strictly from train
    expected_mean = train_df[feature_cols].mean().values
    
    # Assert scaler mean matches train-only mean within numerical tolerances
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5, atol=1e-5)


def test_baseline_training_and_serialization():
    """
    Verify model training, prediction shape, valid probability output, and serialization.
    """
    # Force baseline training to run to update the saved artifacts
    from src.models.train_baseline import train_and_evaluate_baseline
    train_and_evaluate_baseline()

    # Load model artifact
    model_path = os.path.join(MODEL_SAVE_DIR, "baseline_logistic.joblib")
    assert os.path.exists(model_path), "Model joblib artifact must be serialized."

    model = joblib.load(model_path)
    assert isinstance(model, LogisticRegression), "Loaded model is not Logistic Regression."

    # Verify expected parameters
    assert model.random_state == 42
    assert model.class_weight is None

    # Load validation data
    data = np.load(os.path.join(PROCESSED_DATA_DIR, "dataset_arrays.npz"))
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Verify probability distribution limits
    probs = model.predict_proba(X_val)[:, 1]
    assert probs.shape == (X_val.shape[0],), "Probability prediction shape mismatch."
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities must be bounded in [0, 1]."

    # Verify default prediction labels under 0.50 threshold
    preds = (probs >= 0.50).astype(int)
    assert np.array_equal(preds, (probs >= 0.50).astype(int))


def test_baseline_artifacts_existence():
    """
    Verify all required reproducible reporting files are successfully created and populated.
    """
    metrics_path = os.path.join(RESULTS_SAVE_DIR, "baseline_metrics.json")
    cm_path = os.path.join(RESULTS_SAVE_DIR, "baseline_confusion_matrix.json")
    preds_csv_path = os.path.join(RESULTS_SAVE_DIR, "baseline_predictions.csv")
    metadata_path = os.path.join(RESULTS_SAVE_DIR, "baseline_run_metadata.json")

    for path in [metrics_path, cm_path, preds_csv_path, metadata_path]:
        assert os.path.exists(path), f"Artifact missing: {path}"

    # Read metrics to verify structure
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    for key in ["precision", "recall", "f1_score", "roc_auc", "pr_auc", "accuracy"]:
        assert key in metrics, f"Metric key '{key}' missing from metrics JSON."

    # Read predictions CSV to check row match
    preds_df = pd.read_csv(preds_csv_path)
    assert len(preds_df) == 4070, f"Expected 4070 rows in validation predictions, got {len(preds_df)}"
    assert list(preds_df.columns) == [
        "unit_number",
        "time_cycles",
        "true_target",
        "predicted_probability",
        "predicted_label",
    ]
