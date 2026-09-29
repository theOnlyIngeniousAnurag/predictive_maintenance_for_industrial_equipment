"""
Data loading, validation, and target construction for NASA C-MAPSS FD001 dataset.
"""

import os
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")

COLUMN_NAMES = [
    "unit_number",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Constant / near-zero variance sensors in FD001 (sea-level single condition)
CONSTANT_SENSORS = [
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

INFORMATIVE_SENSORS = [
    "op_setting_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]


def load_raw_train_data(data_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load raw train_FD001.txt with column validation.
    """
    file_path = os.path.join(data_dir, "train_FD001.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw training file not found at {file_path}")

    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_raw_test_data(data_dir: str = RAW_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw test_FD001.txt and RUL_FD001.txt ground-truth.
    """
    test_file = os.path.join(data_dir, "test_FD001.txt")
    rul_file = os.path.join(data_dir, "RUL_FD001.txt")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Raw test file not found at {test_file}")
    if not os.path.exists(rul_file):
        raise FileNotFoundError(f"Raw test RUL file not found at {rul_file}")

    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["true_rul"])
    rul_df["unit_number"] = np.arange(1, len(rul_df) + 1)

    return test_df, rul_df


def compute_train_rul_and_target(
    df: pd.DataFrame,
    horizon: int = 30,
) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) and the binary future-failure target for training trajectories.
    
    Target definition (Mentor Approved):
        y(t) = 1 if 0 < RUL(t) <= horizon (imminent failure within next H cycles)
        y(t) = 0 otherwise (healthy / distant from failure or at RUL=0 failure point)
        
    Formula:
        max_cycle(unit) = max(time_cycles for unit)
        RUL(unit, t) = max_cycle(unit) - t
    """
    df = df.copy()
    
    # Sort strictly by unit_number and time_cycles
    df = df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    # Compute max cycle per engine
    max_cycles = df.groupby("unit_number")["time_cycles"].transform("max")
    df["rul"] = max_cycles - df["time_cycles"]

    # Target calculation: binary future failure in (0, H]
    df["target"] = ((df["rul"] > 0) & (df["rul"] <= horizon)).astype(int)

    return df


def compute_test_rul_and_target(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    horizon: int = 30,
) -> pd.DataFrame:
    """
    Compute RUL and binary future-failure target for truncated test trajectories.
    In test_FD001, for each unit, max_observed_cycle + true_rul = total lifetime.
    RUL(unit, t) = (max_observed_cycle + true_rul) - t
    """
    test_df = test_df.copy()
    test_df = test_df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    max_observed = test_df.groupby("unit_number")["time_cycles"].transform("max")
    
    # Map true_rul to test_df
    rul_map = rul_df.set_index("unit_number")["true_rul"].to_dict()
    unit_true_rul = test_df["unit_number"].map(rul_map)

    total_lifetime = max_observed + unit_true_rul
    test_df["rul"] = total_lifetime - test_df["time_cycles"]
    test_df["target"] = ((test_df["rul"] > 0) & (test_df["rul"] <= horizon)).astype(int)

    return test_df


def audit_target_distribution(df: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
    """
    Perform a target-label verification audit.
    """
    total_samples = len(df)
    positives = int(df["target"].sum())
    negatives = total_samples - positives
    pos_pct = (positives / total_samples) * 100.0
    neg_pct = (negatives / total_samples) * 100.0
    imbalance_ratio = negatives / positives if positives > 0 else np.nan
    pos_per_engine = df.groupby("unit_number")["target"].sum()

    # Verify boundary cases
    rul_30_target = df[df["rul"] == horizon]["target"].tolist() if (df["rul"] == horizon).any() else []
    rul_1_target = df[df["rul"] == 1]["target"].tolist() if (df["rul"] == 1).any() else []
    rul_0_target = df[df["rul"] == 0]["target"].tolist() if (df["rul"] == 0).any() else []

    audit_res = {
        "total_observations": total_samples,
        "positive_count": positives,
        "negative_count": negatives,
        "positive_percentage": round(pos_pct, 4),
        "negative_percentage": round(neg_pct, 4),
        "imbalance_ratio": f"1:{imbalance_ratio:.2f}",
        "min_positives_per_engine": int(pos_per_engine.min()),
        "max_positives_per_engine": int(pos_per_engine.max()),
        "mean_positives_per_engine": float(pos_per_engine.mean()),
        "rul_30_labels": set(rul_30_target),
        "rul_1_labels": set(rul_1_target),
        "rul_0_labels": set(rul_0_target),
    }
    return audit_res
