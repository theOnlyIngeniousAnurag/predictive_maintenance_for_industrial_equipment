"""
Dataset splitting, grouped partitioning, and leakage-safe scaling for C-MAPSS FD001.
"""

import os
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

from src.data.loader import (
    load_raw_train_data,
    load_raw_test_data,
    compute_train_rul_and_target,
    compute_test_rul_and_target,
    PROCESSED_DATA_DIR,
)
from src.features.build_features import engineer_all_features, get_feature_columns


def split_train_val_by_engine(
    df: pd.DataFrame,
    val_ratio: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    Split the run-to-failure training dataset into train and validation sets strictly by engine unit_number.
    Ensures zero temporal or trajectory leakage across engine partitions.
    """
    unique_units = df["unit_number"].unique()
    np.random.seed(random_state)
    shuffled_units = np.random.permutation(unique_units)

    n_val = int(np.round(len(unique_units) * val_ratio))
    val_units = sorted(list(shuffled_units[:n_val]))
    train_units = sorted(list(shuffled_units[n_val:]))

    train_df = df[df["unit_number"].isin(train_units)].copy().reset_index(drop=True)
    val_df = df[df["unit_number"].isin(val_units)].copy().reset_index(drop=True)

    return train_df, val_df, train_units, val_units


def fit_and_scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Fit feature scaler ONLY on the training partition X_train.
    Transform validation and test feature partitions without leakage.
    """
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    # Fit strictly on train
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform validation and test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_full_pipeline_splits(
    horizon: int = 30,
    val_ratio: float = 0.20,
    random_state: int = 42,
    save_to_disk: bool = True,
) -> Dict[str, Any]:
    """
    Executes the end-to-end Phase 1-5 pipeline:
    1. Ingestion & Validation
    2. Target computation (0 < RUL <= 30)
    3. Leakage-safe feature engineering
    4. Grouped Engine-Level Train/Validation split
    5. Test set feature engineering & target assignment
    6. Training-only scaler fitting
    7. Disk persistence for reproducibility
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Load raw data
    raw_train = load_raw_train_data()
    raw_test, rul_df = load_raw_test_data()

    # 2. Compute RUL and binary targets
    train_with_target = compute_train_rul_and_target(raw_train, horizon=horizon)
    test_with_target = compute_test_rul_and_target(raw_test, rul_df, horizon=horizon)

    # 3. Engineer features (causally per engine)
    train_feat_df = engineer_all_features(train_with_target)
    test_feat_df = engineer_all_features(test_with_target)

    feature_cols = get_feature_columns(train_feat_df)

    # 4. Engine-level Train / Validation split
    train_split_df, val_split_df, train_units, val_units = split_train_val_by_engine(
        train_feat_df, val_ratio=val_ratio, random_state=random_state
    )

    X_train = train_split_df[feature_cols]
    y_train = train_split_df["target"].values
    rul_train = train_split_df["rul"].values

    X_val = val_split_df[feature_cols]
    y_val = val_split_df["target"].values
    rul_val = val_split_df["rul"].values

    X_test = test_feat_df[feature_cols]
    y_test = test_feat_df["target"].values
    rul_test = test_feat_df["rul"].values

    # 5. Fit scaler strictly on X_train
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = fit_and_scale_features(
        X_train, X_val, X_test, scaler_type="standard"
    )

    # Summary statistics
    summary = {
        "horizon": horizon,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "train_units": train_units,
        "val_units": val_units,
        "n_train_engines": len(train_units),
        "n_val_engines": len(val_units),
        "n_test_engines": len(raw_test["unit_number"].unique()),
        "train_samples": len(train_split_df),
        "val_samples": len(val_split_df),
        "test_samples": len(test_feat_df),
        "train_positives": int(y_train.sum()),
        "train_negatives": int(len(y_train) - y_train.sum()),
        "train_pos_pct": float(np.round((y_train.sum() / len(y_train)) * 100, 4)),
        "val_positives": int(y_val.sum()),
        "val_negatives": int(len(y_val) - y_val.sum()),
        "val_pos_pct": float(np.round((y_val.sum() / len(y_val)) * 100, 4)),
        "test_positives": int(y_test.sum()),
        "test_negatives": int(len(y_test) - y_test.sum()),
        "test_pos_pct": float(np.round((y_test.sum() / len(y_test)) * 100, 4)),
    }

    if save_to_disk:
        # Save parquet / csv
        train_split_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet"), index=False)
        val_split_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "val_engineered.parquet"), index=False)
        test_feat_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "test_engineered.parquet"), index=False)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(PROCESSED_DATA_DIR, "feature_scaler.joblib"))
        
        # Save numpy scaled arrays for immediate model training
        np.savez_compressed(
            os.path.join(PROCESSED_DATA_DIR, "dataset_arrays.npz"),
            X_train=X_train_scaled,
            y_train=y_train,
            rul_train=rul_train,
            X_val=X_val_scaled,
            y_val=y_val,
            rul_val=rul_val,
            X_test=X_test_scaled,
            y_test=y_test,
            rul_test=rul_test,
        )
        
        # Save feature names
        with open(os.path.join(PROCESSED_DATA_DIR, "feature_columns.txt"), "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")

    return {
        "summary": summary,
        "feature_cols": feature_cols,
        "X_train_scaled": X_train_scaled,
        "y_train": y_train,
        "X_val_scaled": X_val_scaled,
        "y_val": y_val,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
    }
