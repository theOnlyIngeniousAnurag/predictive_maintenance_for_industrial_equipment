"""
Leakage-safe feature engineering pipeline for C-MAPSS FD001.
Strictly respects chronological causality within each unit/engine.
Constructs features with modular helpers and vectorized batch assembly.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.data.loader import INFORMATIVE_SENSORS, CONSTANT_SENSORS


def create_lag_features(
    df: pd.DataFrame,
    sensor_cols: List[str] = INFORMATIVE_SENSORS,
    lags: List[int] = [1, 2, 3, 5],
) -> pd.DataFrame:
    """
    Create past lag features strictly within each engine (unit_number).
    For cycle t, lag k provides value at t - k.
    Initial missing values at beginning of engine trajectory are backward-filled within that unit
    (clamped to initial baseline of that unit, never looking ahead).
    """
    df = df.copy()
    grouped = df.groupby("unit_number")
    new_cols: Dict[str, pd.Series] = {}
    for col in sensor_cols:
        for lag in lags:
            col_name = f"{col}_lag_{lag}"
            new_cols[col_name] = grouped[col].shift(lag)
            new_cols[col_name] = df.groupby("unit_number")[col].transform(
                lambda s: s.shift(lag).bfill()
            )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def create_rolling_features(
    df: pd.DataFrame,
    sensor_cols: List[str] = INFORMATIVE_SENSORS,
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Create rolling window statistics strictly within each engine.
    Uses closed='right' with min_periods=1, guaranteeing only observations up to current cycle t are included.
    """
    df = df.copy()
    grouped = df.groupby("unit_number")
    new_cols: Dict[str, pd.Series] = {}
    for col in sensor_cols:
        for w in windows:
            roll_mean = grouped[col].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            roll_std = grouped[col].rolling(w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
            roll_min = grouped[col].rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            roll_max = grouped[col].rolling(w, min_periods=1).max().reset_index(level=0, drop=True)

            new_cols[f"{col}_roll_mean_{w}"] = roll_mean
            new_cols[f"{col}_roll_std_{w}"] = roll_std
            new_cols[f"{col}_roll_min_{w}"] = roll_min
            new_cols[f"{col}_roll_max_{w}"] = roll_max

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def engineer_all_features(
    df: pd.DataFrame,
    sensor_cols: List[str] = INFORMATIVE_SENSORS,
    lags: List[int] = [1, 2, 3, 5],
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for C-MAPSS.
    Strictly causal: all features at cycle t use only observations at or before t for that unit.
    """
    # 1. Strict chronological sorting by engine and cycle
    df = df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    grouped = df.groupby("unit_number")
    new_cols: Dict[str, pd.Series] = {}

    for col in sensor_cols:
        # Lags
        for lag in lags:
            col_name = f"{col}_lag_{lag}"
            new_cols[col_name] = df.groupby("unit_number")[col].transform(
                lambda s: s.shift(lag).bfill()
            )

        # Rolling statistics (min_periods=1, closed='right' by default ensures no lookahead)
        for w in windows:
            roll_mean = grouped[col].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            new_cols[f"{col}_roll_mean_{w}"] = roll_mean

            roll_std = grouped[col].rolling(w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
            new_cols[f"{col}_roll_std_{w}"] = roll_std

            roll_min = grouped[col].rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            new_cols[f"{col}_roll_min_{w}"] = roll_min

            roll_max = grouped[col].rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            new_cols[f"{col}_roll_max_{w}"] = roll_max

            # Deviation from rolling mean
            new_cols[f"{col}_dev_mean_{w}"] = df[col] - roll_mean

        # Differences / rate of change
        lag1 = new_cols[f"{col}_lag_1"]
        lag5 = new_cols[f"{col}_lag_5"]
        new_cols[f"{col}_diff_1"] = df[col] - lag1
        new_cols[f"{col}_diff_5"] = df[col] - lag5

        # Expanding lifetime baseline
        exp_mean = grouped[col].expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        new_cols[f"{col}_exp_mean"] = exp_mean
        new_cols[f"{col}_exp_dev"] = df[col] - exp_mean

    # Concatenate all generated features at once
    engineered_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return engineered_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract only the engineered feature column names, excluding metadata and targets.
    """
    excluded = {
        "unit_number",
        "time_cycles",
        "rul",
        "target",
    } | set(CONSTANT_SENSORS)
    
    feature_cols = [c for c in df.columns if c not in excluded]
    if "time_cycles" not in feature_cols:
        feature_cols = ["time_cycles"] + feature_cols
    return feature_cols
