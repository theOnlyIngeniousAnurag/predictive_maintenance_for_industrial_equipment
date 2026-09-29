"""
NASA C-MAPSS FD001 Dataset Ingestion and Profiling Module.
Analyzes run-to-failure engine trajectories, cycle distributions,
sensor characteristics, and candidate prediction horizons for binary failure classification.
"""

import os
import json
import pandas as pd
import numpy as np

COLUMNS = [
    'unit_number', 'time_cycles',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

SENSOR_DESCRIPTIONS = {
    'op_setting_1': 'Operational Setting 1 (Altitude / Mach number)',
    'op_setting_2': 'Operational Setting 2 (Throttle Resolver Angle)',
    'op_setting_3': 'Operational Setting 3 (Sea Level Setting)',
    'sensor_1': 'T2 - Total temperature at fan inlet (°R)',
    'sensor_2': 'T24 - Total temperature at LPC outlet (°R)',
    'sensor_3': 'T30 - Total temperature at HPC outlet (°R)',
    'sensor_4': 'T50 - Total temperature at LPT outlet (°R)',
    'sensor_5': 'P2 - Pressure at fan inlet (psia)',
    'sensor_6': 'P15 - Total pressure in bypass-duct (psia)',
    'sensor_7': 'P30 - Total pressure at HPC outlet (psia)',
    'sensor_8': 'Nf - Physical fan speed (rpm)',
    'sensor_9': 'Nc - Physical core speed (rpm)',
    'sensor_10': 'epr - Engine pressure ratio (P50/P2)',
    'sensor_11': 'Ps30 - Static pressure at HPC outlet (psia)',
    'sensor_12': 'phi - Ratio of fuel flow to Ps30 (pps/psia)',
    'sensor_13': 'NRf - Corrected fan speed (rpm)',
    'sensor_14': 'NRc - Corrected core speed (rpm)',
    'sensor_15': 'BPR - Bypass Ratio',
    'sensor_16': 'farB - Burner fuel-air ratio',
    'sensor_17': 'htBleed - Bleed Enthalpy',
    'sensor_18': 'Nf_dmd - Demanded fan speed (rpm)',
    'sensor_19': 'PCNfR_dmd - Demanded corrected fan speed (rpm)',
    'sensor_20': 'W31 - HPT coolant bleed (lbm/s)',
    'sensor_21': 'W32 - LPT coolant bleed (lbm/s)'
}

def load_cmapss_raw(filepath):
    """Load space-delimited C-MAPSS text files into pandas DataFrame."""
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMNS)
    return df

def profile_dataset():
    """Run full Phase 1 profiling on C-MAPSS FD001."""
    train_path = 'data/raw/train_FD001.txt'
    test_path = 'data/raw/test_FD001.txt'
    rul_path = 'data/raw/RUL_FD001.txt'
    
    train_df = load_cmapss_raw(train_path)
    test_df = load_cmapss_raw(test_path)
    rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL_ground_truth'])
    
    # 1. Dataset Dimensions & Basic Quality
    train_rows, train_cols = train_df.shape
    test_rows, test_cols = test_df.shape
    num_train_engines = train_df['unit_number'].nunique()
    num_test_engines = test_df['unit_number'].nunique()
    
    missing_train = train_df.isnull().sum().to_dict()
    total_missing_train = int(train_df.isnull().sum().sum())
    duplicates_train = int(train_df.duplicated(subset=['unit_number', 'time_cycles']).sum())
    
    # 2. Trajectory & Cycle Analysis
    max_cycles_per_engine = train_df.groupby('unit_number')['time_cycles'].max()
    cycle_stats = {
        'min_cycles_to_failure': int(max_cycles_per_engine.min()),
        'max_cycles_to_failure': int(max_cycles_per_engine.max()),
        'mean_cycles_to_failure': float(max_cycles_per_engine.mean()),
        'median_cycles_to_failure': float(max_cycles_per_engine.median()),
        'std_cycles_to_failure': float(max_cycles_per_engine.std()),
        '25th_percentile': float(max_cycles_per_engine.quantile(0.25)),
        '75th_percentile': float(max_cycles_per_engine.quantile(0.75))
    }
    
    # Calculate ground-truth Remaining Useful Life (RUL) for each training observation
    train_df['max_cycle'] = train_df.groupby('unit_number')['time_cycles'].transform('max')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycles']
    
    # 3. Sensor Variance Analysis (Detect constant sensors with zero variance)
    sensor_cols = [c for c in COLUMNS if c.startswith('sensor_') or c.startswith('op_setting_')]
    sensor_variances = train_df[sensor_cols].var().to_dict()
    sensor_stds = train_df[sensor_cols].std().to_dict()
    
    constant_sensors = [c for c, v in sensor_variances.items() if v < 1e-6]
    active_sensors = [c for c, v in sensor_variances.items() if v >= 1e-6]
    
    # 4. Candidate Prediction Horizon Analysis
    candidate_horizons = [10, 15, 20, 25, 30, 35, 40, 50]
    horizon_analysis = []
    
    for H in candidate_horizons:
        # Binary failure target: 1 if RUL <= H (failure within next H cycles), 0 otherwise
        target_col = f'failure_in_{H}_cycles'
        binary_target = (train_df['RUL'] <= H).astype(int)
        
        pos_count = int(binary_target.sum())
        neg_count = int((binary_target == 0).sum())
        pos_pct = float(pos_count / len(train_df) * 100)
        neg_pct = float(neg_count / len(train_df) * 100)
        imbalance_ratio = f'1:{neg_count / pos_count:.2f}' if pos_count > 0 else 'N/A'
        
        horizon_analysis.append({
            'horizon_H': H,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'positive_pct': round(pos_pct, 2),
            'negative_pct': round(neg_pct, 2),
            'imbalance_ratio': imbalance_ratio,
            'avg_positive_cycles_per_engine': round(pos_count / num_train_engines, 1)
        })
    
    # 5. Output Report Data Structure
    profile_summary = {
        'dataset_name': 'NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)',
        'source': 'NASA Ames Prognostics Data Repository & NASA Open Data Portal',
        'files': {
            'train_file': 'train_FD001.txt',
            'test_file': 'test_FD001.txt',
            'rul_file': 'RUL_FD001.txt'
        },
        'train_records': train_rows,
        'train_engines': num_train_engines,
        'test_records': test_rows,
        'test_engines': num_test_engines,
        'features_count': train_cols,
        'missing_values': total_missing_train,
        'duplicates': duplicates_train,
        'cycle_statistics': cycle_stats,
        'constant_sensors': constant_sensors,
        'active_sensors': active_sensors,
        'sensor_descriptions': SENSOR_DESCRIPTIONS,
        'candidate_horizons': horizon_analysis
    }
    
    os.makedirs('reports/results', exist_ok=True)
    with open('reports/results/cmapss_fd001_profile.json', 'w') as f:
        json.dump(profile_summary, f, indent=2)
        
    print("=== NASA C-MAPSS FD001 PROFILING COMPLETE ===")
    print(f"Train records: {train_rows} across {num_train_engines} engines")
    print(f"Test records: {test_rows} across {num_test_engines} engines")
    print(f"Missing values: {total_missing_train}, Duplicates: {duplicates_train}")
    print(f"Cycle range to failure: min={cycle_stats['min_cycles_to_failure']}, max={cycle_stats['max_cycles_to_failure']}, median={cycle_stats['median_cycles_to_failure']}")
    print(f"Constant/near-zero variance channels ({len(constant_sensors)}): {constant_sensors}")
    print(f"Active informative sensors ({len(active_sensors)}): {active_sensors}")
    print("\nCandidate Horizons Analysis:")
    for h in horizon_analysis:
        print(f"  H={h['horizon_H']:2d} cycles -> Positives: {h['positive_count']:5d} ({h['positive_pct']:5.2f}%) | Negatives: {h['negative_count']:5d} ({h['negative_pct']:5.2f}%) | Imbalance: {h['imbalance_ratio']}")

if __name__ == '__main__':
    profile_dataset()
