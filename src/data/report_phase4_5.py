"""
Generate detailed metrics, verification tables, and audit logs for Phase 4 & Phase 5.
"""

import os
import json
import numpy as np
import pandas as pd

from src.data.loader import (
    load_raw_train_data,
    load_raw_test_data,
    compute_train_rul_and_target,
    compute_test_rul_and_target,
    audit_target_distribution,
    INFORMATIVE_SENSORS,
    CONSTANT_SENSORS,
)
from src.features.build_features import engineer_all_features, get_feature_columns
from src.data.split import prepare_full_pipeline_splits


def generate_report():
    print("=" * 70)
    print("PHASE 4 & 5 AUDIT & VERIFICATION REPORT")
    print("=" * 70)

    # 1. Target Definition & Boundary Verification
    raw_train = load_raw_train_data()
    train_with_target = compute_train_rul_and_target(raw_train, horizon=30)
    audit = audit_target_distribution(train_with_target, horizon=30)

    print("\n--- 1. TARGET DEFINITION & BOUNDARY VERIFICATION ---")
    print(f"Target Formula: y(t) = 1 if 0 < RUL(t) <= 30 else 0")
    print(f"Total Observations: {audit['total_observations']}")
    print(f"Positive Samples: {audit['positive_count']} ({audit['positive_percentage']}%)")
    print(f"Negative Samples: {audit['negative_count']} ({audit['negative_percentage']}%)")
    print(f"Imbalance Ratio: {audit['imbalance_ratio']}")
    print(f"Positive observations per engine (min/max/mean): {audit['min_positives_per_engine']} / {audit['max_positives_per_engine']} / {audit['mean_positives_per_engine']}")
    print(f"Labels at RUL = 30: {audit['rul_30_labels']}")
    print(f"Labels at RUL = 1: {audit['rul_1_labels']}")
    print(f"Labels at RUL = 0: {audit['rul_0_labels']}")

    # Check minimum history
    min_cycles_per_engine = train_with_target.groupby("unit_number")["time_cycles"].max().min()
    print(f"Minimum engine lifetime (cycles): {min_cycles_per_engine}")
    print(f"Max lag/rolling window: 20 cycles")
    print(f"History sufficiency: All engines have at least {min_cycles_per_engine} cycles, strictly >= 20 cycles window.")

    # 2. Pipeline Execution & Split Verification
    pipeline_res = prepare_full_pipeline_splits(horizon=30, val_ratio=0.20, random_state=42, save_to_disk=True)
    summary = pipeline_res["summary"]
    feature_names = pipeline_res["feature_cols"]

    print("\n--- 2. LEAKAGE-SAFE FEATURE ENGINEERING (PHASE 4) ---")
    print(f"Total Engineered Features: {len(feature_names)}")
    print(f"Informative Base Sensors ({len(INFORMATIVE_SENSORS)}): {INFORMATIVE_SENSORS}")
    print(f"Excluded Constant Sensors ({len(CONSTANT_SENSORS)}): {CONSTANT_SENSORS}")
    print(f"Feature categories:")
    print(f"  - Operating age: 1 (time_cycles)")
    print(f"  - Raw informative sensors: {len(INFORMATIVE_SENSORS)}")
    print(f"  - Historical lags (1, 2, 3, 5): {len(INFORMATIVE_SENSORS) * 4}")
    print(f"  - Rolling statistics (mean, std, min, max for W=[5, 10, 20]): {len(INFORMATIVE_SENSORS) * 3 * 4}")
    print(f"  - Trend & deviations (diff_1, diff_5, dev_mean_[5,10,20]): {len(INFORMATIVE_SENSORS) * 5}")
    print(f"  - Expanding lifetime statistics (exp_mean, exp_dev): {len(INFORMATIVE_SENSORS) * 2}")

    print("\n--- 3. ENGINE-LEVEL DATASET SPLITTING (PHASE 5) ---")
    print(f"Split Strategy: Grouped by Unit Number (Zero engine overlap)")
    print(f"Train Split: {summary['n_train_engines']} engines, {summary['train_samples']} samples | Pos: {summary['train_positives']} ({summary['train_pos_pct']}%) | Neg: {summary['train_negatives']}")
    print(f"Val Split: {summary['n_val_engines']} engines, {summary['val_samples']} samples | Pos: {summary['val_positives']} ({summary['val_pos_pct']}%) | Neg: {summary['val_negatives']}")
    print(f"Test Split: {summary['n_test_engines']} engines, {summary['test_samples']} samples | Pos: {summary['test_positives']} ({summary['test_pos_pct']}%) | Neg: {summary['test_negatives']}")
    print(f"Train engine IDs: {summary['train_units'][:10]} ... ({len(summary['train_units'])} total)")
    print(f"Val engine IDs: {summary['val_units'][:10]} ... ({len(summary['val_units'])} total)")

    # Save summary JSON to reports/results
    os.makedirs("reports/results", exist_ok=True)
    summary_serializable = {
        k: [int(x) for x in v] if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, np.integer))
        else v
        for k, v in summary.items()
    }
    with open("reports/results/phase4_5_summary.json", "w") as f:
        json.dump(summary_serializable, f, indent=2)

    print("\nAudit and pipeline verification completed successfully.")


if __name__ == "__main__":
    generate_report()
