"""
Unit tests for target construction, leakage prevention, and engine-level dataset splitting.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.loader import (
    load_raw_train_data,
    load_raw_test_data,
    compute_train_rul_and_target,
    compute_test_rul_and_target,
    audit_target_distribution,
    COLUMN_NAMES,
    INFORMATIVE_SENSORS,
)
from src.features.build_features import (
    engineer_all_features,
    get_feature_columns,
    create_lag_features,
    create_rolling_features,
)
from src.data.split import split_train_val_by_engine, fit_and_scale_features, prepare_full_pipeline_splits


def test_target_boundary_conditions():
    """
    Verify exact target definition:
    y(t) = 1 if 0 < RUL(t) <= 30
    y(t) = 0 if RUL(t) == 0 (failure point itself) or RUL(t) > 30
    """
    raw_train = load_raw_train_data()
    train_df = compute_train_rul_and_target(raw_train, horizon=30)

    # 1. Check RUL = 30
    rul_30 = train_df[train_df["rul"] == 30]
    assert len(rul_30) > 0
    assert (rul_30["target"] == 1).all(), "RUL=30 must have target=1"

    # 2. Check RUL = 1
    rul_1 = train_df[train_df["rul"] == 1]
    assert len(rul_1) > 0
    assert (rul_1["target"] == 1).all(), "RUL=1 must have target=1"

    # 3. Check RUL = 0 (failure cycle itself)
    rul_0 = train_df[train_df["rul"] == 0]
    assert len(rul_0) > 0
    assert (rul_0["target"] == 0).all(), "RUL=0 must have target=0 (imminent future failure is strictly 0 < RUL <= 30)"

    # 4. Check RUL = 31
    rul_31 = train_df[train_df["rul"] == 31]
    assert len(rul_31) > 0
    assert (rul_31["target"] == 0).all(), "RUL=31 must have target=0"

    # 5. Check positive count per engine: each of the 100 engines runs to failure with RUL from Max to 0.
    # Therefore, each engine must have exactly 30 positive cycles (RUL in [1, 30]).
    pos_per_engine = train_df.groupby("unit_number")["target"].sum()
    assert (pos_per_engine == 30).all(), "Every engine in train_FD001 must have exactly 30 positive cycles"
    assert len(pos_per_engine) == 100, "There must be 100 engines in training set"
    assert train_df["target"].sum() == 3000, "Total positive samples across 100 engines must be exactly 3,000"


def test_temporal_leakage_absence():
    """
    Leakage Test:
    Verify that altering sensor values in future cycles (t + k) does NOT affect engineered features at cycle t.
    """
    raw_train = load_raw_train_data().head(300).copy()
    
    # Baseline features
    feat_orig = engineer_all_features(raw_train.copy())
    
    # Perturb future data at cycle >= 50 for unit 1
    raw_perturbed = raw_train.copy()
    mask_future = (raw_perturbed["unit_number"] == 1) & (raw_perturbed["time_cycles"] >= 50)
    raw_perturbed.loc[mask_future, INFORMATIVE_SENSORS] += 1000.0
    
    feat_perturbed = engineer_all_features(raw_perturbed.copy())
    
    # Features at cycles 1 to 49 for unit 1 MUST BE EXACTLY IDENTICAL
    feat_cols = get_feature_columns(feat_orig)
    mask_past = (feat_orig["unit_number"] == 1) & (feat_orig["time_cycles"] < 50)
    
    orig_past = feat_orig.loc[mask_past, feat_cols].values
    pert_past = feat_perturbed.loc[mask_past, feat_cols].values
    
    np.testing.assert_allclose(
        orig_past,
        pert_past,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Temporal leakage detected! Past features changed when future raw sensor data was perturbed."
    )


def test_engine_cross_leakage_absence():
    """
    Cross-engine Leakage Test:
    Verify that modifying data for Unit 2 does NOT affect any features for Unit 1.
    """
    raw_train = load_raw_train_data().head(500).copy()
    feat_orig = engineer_all_features(raw_train.copy())
    
    # Perturb unit 2 completely
    raw_perturbed = raw_train.copy()
    mask_unit2 = raw_perturbed["unit_number"] == 2
    raw_perturbed.loc[mask_unit2, INFORMATIVE_SENSORS] *= 100.0
    
    feat_perturbed = engineer_all_features(raw_perturbed.copy())
    
    feat_cols = get_feature_columns(feat_orig)
    mask_unit1 = feat_orig["unit_number"] == 1
    
    orig_unit1 = feat_orig.loc[mask_unit1, feat_cols].values
    pert_unit1 = feat_perturbed.loc[mask_unit1, feat_cols].values
    
    np.testing.assert_allclose(
        orig_unit1,
        pert_unit1,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Cross-engine leakage detected! Unit 1 features changed when Unit 2 was perturbed."
    )


def test_grouped_train_val_split():
    """
    Split Purity Test:
    Verify that no engine exists in both train and validation sets.
    """
    raw_train = load_raw_train_data()
    train_with_target = compute_train_rul_and_target(raw_train, horizon=30)
    
    train_df, val_df, train_units, val_units = split_train_val_by_engine(
        train_with_target, val_ratio=0.20, random_state=42
    )
    
    # 1. Disjoint units
    overlap = set(train_units).intersection(set(val_units))
    assert len(overlap) == 0, f"Found overlapping engine units in train and val: {overlap}"
    assert len(train_units) == 80, "Train split must have 80 engines"
    assert len(val_units) == 20, "Validation split must have 20 engines"
    
    # 2. Complete coverage
    assert len(train_units) + len(val_units) == 100


def test_full_pipeline_execution():
    """
    Full pipeline run and persistence check.
    """
    results = prepare_full_pipeline_splits(horizon=30, val_ratio=0.20, random_state=42, save_to_disk=True)
    summary = results["summary"]
    
    assert summary["n_features"] > 50, "Should engineer > 50 informative features"
    assert summary["n_train_engines"] == 80
    assert summary["n_val_engines"] == 20
    assert summary["n_test_engines"] == 100
    assert summary["train_positives"] == 80 * 30  # 2400
    assert summary["val_positives"] == 20 * 30    # 600
    assert summary["test_positives"] > 0
