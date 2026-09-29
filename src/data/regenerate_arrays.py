"""
Regenerates the scaled numpy arrays (dataset_arrays.npz) from the Parquet data
if they are missing.
"""
import os
import numpy as np
import pandas as pd
import joblib
from src.data.loader import PROCESSED_DATA_DIR

def regenerate_arrays():
    print("Regenerating dataset_arrays.npz from existing Parquet files...")
    
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train_engineered.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "val_engineered.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "test_engineered.parquet"))
    
    with open(os.path.join(PROCESSED_DATA_DIR, "feature_columns.txt"), "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
        
    scaler = joblib.load(os.path.join(PROCESSED_DATA_DIR, "feature_scaler.joblib"))
    
    X_train_scaled = scaler.transform(train_df[feature_cols])
    y_train = train_df["target"].values
    rul_train = train_df["rul"].values
    
    X_val_scaled = scaler.transform(val_df[feature_cols])
    y_val = val_df["target"].values
    rul_val = val_df["rul"].values
    
    X_test_scaled = scaler.transform(test_df[feature_cols])
    y_test = test_df["target"].values
    rul_test = test_df["rul"].values
    
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
    print(f"Successfully saved {os.path.join(PROCESSED_DATA_DIR, 'dataset_arrays.npz')}")

if __name__ == "__main__":
    regenerate_arrays()
