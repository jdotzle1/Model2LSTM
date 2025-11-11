#!/usr/bin/env python3
"""
XGBoost Readiness Validation

Validates that processed data meets all XGBoost training requirements:
1. Binary labels (0/1)
2. Positive weights
3. Numeric features
4. Reasonable data quality
5. Sufficient samples per mode

Usage:
    python tests/validation/validate_xgboost_readiness.py <input_file.parquet>
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.weighted_labeling import TRADING_MODES


def validate_xgboost_readiness(df: pd.DataFrame) -> dict:
    """
    Comprehensive XGBoost readiness validation
    
    Returns dict with validation results
    """
    print("=" * 80)
    print("XGBOOST READINESS VALIDATION")
    print("=" * 80)
    print(f"Dataset: {len(df):,} rows × {len(df.columns)} columns\n")
    
    results = {
        'passed': True,
        'critical_issues': [],
        'warnings': [],
        'mode_stats': {}
    }
    
    # Check 1: All label columns exist and are binary
    print("✓ Check 1: Label columns (binary 0/1)")
    for mode_name, mode in TRADING_MODES.items():
        label_col = mode.label_column
        
        if label_col not in df.columns:
            results['passed'] = False
            results['critical_issues'].append(f"Missing label column: {label_col}")
            print(f"  ❌ {label_col}: MISSING")
            continue
        
        unique_vals = set(df[label_col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            results['passed'] = False
            results['critical_issues'].append(f"{label_col} has non-binary values: {unique_vals}")
            print(f"  ❌ {label_col}: Non-binary values {unique_vals}")
        else:
            print(f"  ✅ {label_col}: Binary (0/1)")
    
    # Check 2: All weight columns exist and are positive
    print("\n✓ Check 2: Weight columns (positive values)")
    for mode_name, mode in TRADING_MODES.items():
        weight_col = mode.weight_column
        
        if weight_col not in df.columns:
            results['passed'] = False
            results['critical_issues'].append(f"Missing weight column: {weight_col}")
            print(f"  ❌ {weight_col}: MISSING")
            continue
        
        min_weight = df[weight_col].min()
        max_weight = df[weight_col].max()
        
        if min_weight <= 0:
            results['passed'] = False
            results['critical_issues'].append(f"{weight_col} has non-positive values (min={min_weight})")
            print(f"  ❌ {weight_col}: Non-positive (min={min_weight:.3f})")
        elif not np.isfinite(min_weight) or not np.isfinite(max_weight):
            results['passed'] = False
            results['critical_issues'].append(f"{weight_col} has infinite/NaN values")
            print(f"  ❌ {weight_col}: Infinite/NaN values")
        else:
            print(f"  ✅ {weight_col}: Positive [{min_weight:.3f}, {max_weight:.3f}]")
    
    # Check 3: Sufficient samples per mode
    print("\n✓ Check 3: Sample counts per mode")
    min_samples = 1000  # Minimum for meaningful training
    for mode_name, mode in TRADING_MODES.items():
        label_col = mode.label_column
        
        if label_col not in df.columns:
            continue
        
        total = len(df)
        winners = (df[label_col] == 1).sum()
        losers = (df[label_col] == 0).sum()
        win_rate = winners / total if total > 0 else 0
        
        results['mode_stats'][mode_name] = {
            'total': int(total),
            'winners': int(winners),
            'losers': int(losers),
            'win_rate': float(win_rate)
        }
        
        if winners < min_samples:
            results['warnings'].append(f"{mode_name}: Only {winners} winners (min recommended: {min_samples})")
            print(f"  ⚠️  {mode_name}: {winners:,} winners ({win_rate:.1%}) - LOW SAMPLE COUNT")
        else:
            print(f"  ✅ {mode_name}: {winners:,} winners ({win_rate:.1%})")
    
    # Check 4: Feature columns are numeric
    print("\n✓ Check 4: Feature columns (numeric)")
    feature_cols = [c for c in df.columns if not c.startswith(('label_', 'weight_', 'timestamp', 'session', 'symbol', 'rtype', 'publisher', 'instrument'))]
    
    non_numeric = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    
    if non_numeric:
        results['passed'] = False
        results['critical_issues'].append(f"Non-numeric feature columns: {non_numeric[:5]}")
        print(f"  ❌ {len(non_numeric)} non-numeric columns: {non_numeric[:5]}")
    else:
        print(f"  ✅ All {len(feature_cols)} feature columns are numeric")
    
    # Check 5: Handle infinite/NaN values
    print("\n✓ Check 5: Infinite/NaN values in features")
    inf_counts = {}
    nan_counts = {}
    
    for col in feature_cols:
        inf_count = np.isinf(df[col]).sum()
        nan_count = df[col].isna().sum()
        
        if inf_count > 0:
            inf_counts[col] = inf_count
        if nan_count > 0:
            nan_counts[col] = nan_count
    
    total_cells = len(df) * len(feature_cols)
    total_inf = sum(inf_counts.values())
    total_nan = sum(nan_counts.values())
    
    inf_pct = (total_inf / total_cells) * 100
    nan_pct = (total_nan / total_cells) * 100
    
    print(f"  Infinite values: {total_inf:,} ({inf_pct:.2f}% of feature cells)")
    print(f"  NaN values: {total_nan:,} ({nan_pct:.2f}% of feature cells)")
    
    if inf_pct > 10:
        results['warnings'].append(f"High percentage of infinite values: {inf_pct:.1f}%")
        print(f"  ⚠️  High inf percentage - XGBoost will treat as missing")
    
    if nan_pct > 10:
        results['warnings'].append(f"High percentage of NaN values: {nan_pct:.1f}%")
        print(f"  ⚠️  High NaN percentage - XGBoost will treat as missing")
    
    # Check 6: Data types suitable for XGBoost
    print("\n✓ Check 6: Data types")
    print(f"  Label columns: {df[[m.label_column for m in TRADING_MODES.values()]].dtypes.unique()}")
    print(f"  Weight columns: {df[[m.weight_column for m in TRADING_MODES.values()]].dtypes.unique()}")
    print(f"  Feature columns: {df[feature_cols].dtypes.value_counts().to_dict()}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if results['passed']:
        print("✅ XGBOOST READINESS: PASSED")
        print(f"\nDataset is ready for XGBoost training:")
        print(f"  • {len(df):,} total samples")
        print(f"  • 6 trading modes with binary labels")
        print(f"  • Positive sample weights")
        print(f"  • {len(feature_cols)} numeric features")
        
        if results['warnings']:
            print(f"\n⚠️  {len(results['warnings'])} warnings (non-critical):")
            for warning in results['warnings'][:5]:
                print(f"  • {warning}")
    else:
        print("❌ XGBOOST READINESS: FAILED")
        print(f"\n{len(results['critical_issues'])} critical issues found:")
        for issue in results['critical_issues']:
            print(f"  • {issue}")
    
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate XGBoost readiness')
    parser.add_argument('input_file', help='Path to processed Parquet file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"❌ File not found: {args.input_file}")
        sys.exit(1)
    
    print(f"Loading: {args.input_file}\n")
    df = pd.read_parquet(args.input_file)
    
    results = validate_xgboost_readiness(df)
    
    # Exit code
    sys.exit(0 if results['passed'] else 1)


if __name__ == "__main__":
    main()
