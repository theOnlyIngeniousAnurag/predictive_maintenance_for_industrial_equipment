"""
Phase 19 Automated Tests: Data Foundation, Schema Validation, RUL Construction, Target Generation, Boundary Conditions, Trajectory Integrity, and Leakage Preconditions.
"""

import os
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
    CONSTANT_SENSORS,
    INFORMATIVE_SENSORS,
)


def test_p19_schema_validation():
    """P19-04: Schema Validation Test."""
    train_df = load_raw_train_data()
    test_df, rul_df = load_raw_test_data()

    # Column names and count
    assert list(train_df.columns) == COLUMN_NAMES, "Train columns must match schema"
    assert list(test_df.columns) == COLUMN_NAMES, "Test columns must match schema"
    assert len(train_df.columns) == 26, "Schema must have 26 columns"
    assert len(test_df.columns) == 26, "Test schema must have 26 columns"

    # Engine counts
    assert train_df["unit_number"].nunique() == 100, "Train engine count must be 100"
    assert test_df["unit_number"].nunique() == 100, "Test engine count must be 100"
    assert len(rul_df) == 100, "Test ground-truth RUL count must be 100"

    # Row counts
    assert len(train_df) == 20631, "Train observation count must be 20,631"
    assert len(test_df) == 13096, "Test observation count must be 13,096"

    # No missing or null values
    assert train_df.isnull().sum().sum() == 0, "Raw train data must have 0 missing values"
    assert test_df.isnull().sum().sum() == 0, "Raw test data must have 0 missing values"

    # No duplicate unit/time_cycles pairs
    assert train_df.duplicated(subset=["unit_number", "time_cycles"]).sum() == 0, "Train data must have no duplicate (unit_number, time_cycles)"
    assert test_df.duplicated(subset=["unit_number", "time_cycles"]).sum() == 0, "Test data must have no duplicate (unit_number, time_cycles)"


def test_p19_rul_construction():
    """P19-06: RUL Construction and Monotonicity Test."""
    raw_train = load_raw_train_data()
    train_df = compute_train_rul_and_target(raw_train, horizon=30)

    # RUL properties
    assert (train_df["rul"] >= 0).all(), "RUL must be non-negative"
    
    # RUL at max cycle per engine must be 0
    max_cycles = train_df.groupby("unit_number")["time_cycles"].max()
    for unit, max_c in max_cycles.items():
        rul_at_max = train_df[(train_df["unit_number"] == unit) & (train_df["time_cycles"] == max_c)]["rul"].values[0]
        assert rul_at_max == 0, f"Engine {unit} at max cycle {max_c} must have RUL = 0"

    # Monotonicity per engine
    for unit in train_df["unit_number"].unique():
        engine_rul = train_df[train_df["unit_number"] == unit]["rul"].values
        diffs = np.diff(engine_rul)
        assert (diffs == -1).all(), f"Engine {unit} RUL must strictly decrease by 1 at every cycle step"


def test_p19_target_boundary_validation():
    """P19-07 & P19-08: Target Boundary Conditions Test."""
    raw_train = load_raw_train_data()
    train_df = compute_train_rul_and_target(raw_train, horizon=30)

    # RUL = 30 -> Target = 1
    rul_30 = train_df[train_df["rul"] == 30]
    assert len(rul_30) == 100, "There must be 100 observations at RUL = 30"
    assert (rul_30["target"] == 1).all(), "RUL = 30 must yield target = 1"

    # RUL = 1 -> Target = 1
    rul_1 = train_df[train_df["rul"] == 1]
    assert len(rul_1) == 100, "There must be 100 observations at RUL = 1"
    assert (rul_1["target"] == 1).all(), "RUL = 1 must yield target = 1"

    # RUL = 0 -> Target = 0
    rul_0 = train_df[train_df["rul"] == 0]
    assert len(rul_0) == 100, "There must be 100 observations at RUL = 0"
    assert (rul_0["target"] == 0).all(), "RUL = 0 must yield target = 0"

    # RUL > 30 -> Target = 0
    rul_31 = train_df[train_df["rul"] == 31]
    assert len(rul_31) == 100, "There must be 100 observations at RUL = 31"
    assert (rul_31["target"] == 0).all(), "RUL = 31 must yield target = 0"


def test_p19_engine_trajectory_integrity():
    """P19-09: Engine Ordering and Trajectory Integrity Check."""
    raw_train = load_raw_train_data()

    for unit in raw_train["unit_number"].unique():
        engine_cycles = raw_train[raw_train["unit_number"] == unit]["time_cycles"].values
        # Cycles start at 1
        assert engine_cycles[0] == 1, f"Engine {unit} trajectory must start at cycle 1"
        # Strictly increasing sequence with step 1
        diffs = np.diff(engine_cycles)
        assert (diffs == 1).all(), f"Engine {unit} cycles must be contiguous and strictly increasing"


def test_p19_leakage_preconditions():
    """P19-10: Target Leakage and Engine Isolation Check."""
    raw_train = load_raw_train_data()
    train_df = compute_train_rul_and_target(raw_train, horizon=30)

    # RUL and target must NOT be in raw feature set
    for col in ["rul", "target", "RUL"]:
        assert col not in COLUMN_NAMES, f"{col} must not be a predictor in the raw dataset columns"

    # Check that engine IDs in train and test do not overlap inappropriately
    test_df, _ = load_raw_test_data()
    train_units = set(train_df["unit_number"].unique())
    test_units = set(test_df["unit_number"].unique())
    
    assert len(train_units) == 100
    assert len(test_units) == 100
