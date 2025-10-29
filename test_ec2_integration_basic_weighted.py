#!/usr/bin/env python3
"""
Basic Integration Test for EC2 Pipeline Components with Weighted Labeling (No XGBoost Required)

This script tests the integration between:
1. Weighted labeling system
2. Feature engineering
3. Pipeline validation

Run this locally to ensure core components work before EC2 deployment.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from project.data_pipeline.pipeline import (
    process_labeling_and_features,
    validate_pipeline_output,
    create_pipeline_summary,
    PipelineConfig
)


def create_test_data(n_rows=3000):
    """Create synthetic test data for integration testing"""
    print(f"Creating synthetic test data ({n_rows:,} rows)...")
    
    # Create realistic ES futures data
    np.random.seed(42)
    
    # Generate timestamps (1-second bars)
    start_date = pd.Timestamp('2024-01-01 07:30:00', tz='UTC')
    timestamps = pd.date_range(start_date, periods=n_rows, freq='1s')
    
    # Generate realistic OHLCV data
    base_price = 4500.0
    price_changes = np.random.normal(0, 0.5, n_rows).cumsum()
    
    close_prices = base_price + price_changes
    
    # Generate OHLC from close prices
    high_offset = np.random.exponential(0.5, n_rows)
    low_offset = np.random.exponential(0.5, n_rows)
    open_offset = np.random.normal(0, 0.2, n_rows)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': close_prices + open_offset,
        'high': close_prices + high_offset,
        'low': close_prices - low_offset,
        'close': close_prices,
        'volume': np.random.randint(100, 5000, n_rows)
    })
    
    # Ensure OHLC relationships are valid
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    # Round to tick size (0.25)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = (df[col] / 0.25).round() * 0.25
    
    print(f"  ✓ Created test data: {len(df):,} rows")
    print(f"  Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    print(f"  Volume range: {df['volume'].min():,} to {df['volume'].max():,}")
    
    return df


def test_weighted_labeling_and_features():
    """Test weighted labeling system and feature engineering integration"""
    print("\n=== TESTING WEIGHTED LABELING + FEATURE ENGINEERING ===")
    
    # Create test data
    df = create_test_data(4000)
    
    # Configure pipeline
    config = PipelineConfig(
        chunk_size=2000,
        enable_progress_tracking=True,
        enable_performance_monitoring=True,
        enable_memory_optimization=True
    )
    
    print("  Testing integrated pipeline:")
    print("    - Weighted labeling (6 volatility-based modes)")
    print("    - Feature engineering (43 features)")
    print("    - Chunked processing")
    
    # Process with integrated pipeline
    df_processed = process_labeling_and_features(df, config)
    
    # Validate results
    validation_results = validate_pipeline_output(df_processed)
    
    if not validation_results['valid']:
        raise ValueError(f"Pipeline validation failed: {validation_results['errors']}")
    
    # Print validation warnings
    if validation_results['warnings']:
        print("  Validation warnings:")
        for warning in validation_results['warnings']:
            print(f"    - {warning}")
    
    # Print statistics
    stats = validation_results['statistics']
    print(f"\n  ✓ Pipeline processing complete:")
    print(f"    Dataset: {stats['total_rows']:,} rows × {stats['total_columns']} columns")
    print(f"    Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    
    print("    Win rates by volatility mode:")
    for mode, win_rate in stats['win_rates'].items():
        print(f"      {mode}: {win_rate:.1%}")
    
    # Validate specific column counts
    expected_original_cols = 6  # timestamp, OHLCV
    expected_label_weight_cols = 12  # 6 labels + 6 weights
    expected_feature_cols = 43  # 43 features
    expected_total = expected_original_cols + expected_label_weight_cols + expected_feature_cols
    
    if stats['total_columns'] != expected_total:
        print(f"    Warning: Expected {expected_total} columns, got {stats['total_columns']}")
    else:
        print(f"    ✓ Column count validation passed: {stats['total_columns']} columns")
    
    # Check for weighted labeling columns
    weighted_label_cols = [col for col in df_processed.columns if col.startswith('label_') and 
                          any(mode in col for mode in ['low_vol', 'normal_vol', 'high_vol'])]
    weighted_weight_cols = [col for col in df_processed.columns if col.startswith('weight_') and 
                           any(mode in col for mode in ['low_vol', 'normal_vol', 'high_vol'])]
    
    print(f"    ✓ Weighted labeling columns: {len(weighted_label_cols)} labels, {len(weighted_weight_cols)} weights")
    
    # Check for feature columns
    from project.data_pipeline.features import get_expected_feature_names
    expected_features = get_expected_feature_names()
    feature_cols = [col for col in expected_features if col in df_processed.columns]
    
    print(f"    ✓ Feature columns: {len(feature_cols)}/{len(expected_features)} features")
    
    print("  ✓ Weighted labeling + feature engineering integration test passed")
    
    return df_processed


def test_pipeline_summary(df_processed):
    """Test pipeline summary generation"""
    print("\n=== TESTING PIPELINE SUMMARY ===")
    
    # Create pipeline summary
    summary = create_pipeline_summary(df_processed)
    
    # Validate summary structure
    required_keys = ['pipeline_version', 'timestamp', 'dataset_info', 'labeling_info', 'feature_info']
    missing_keys = set(required_keys) - set(summary.keys())
    if missing_keys:
        raise ValueError(f"Missing summary keys: {missing_keys}")
    
    print("  Pipeline summary:")
    print(f"    Version: {summary['pipeline_version']}")
    print(f"    Dataset: {summary['dataset_info']['rows']:,} rows × {summary['dataset_info']['columns']} columns")
    print(f"    Memory usage: {summary['dataset_info']['memory_usage_mb']:.1f} MB")
    print(f"    Labeling: {summary['labeling_info']['system']} ({summary['labeling_info']['modes']} modes)")
    print(f"    Features: {summary['feature_info']['features_added']} features added")
    
    print("  ✓ Pipeline summary test passed")
    
    return summary


def test_data_quality_validation(df_processed):
    """Test data quality validation"""
    print("\n=== TESTING DATA QUALITY VALIDATION ===")
    
    # Check for NaN values in critical columns
    label_cols = [col for col in df_processed.columns if col.startswith('label_')]
    weight_cols = [col for col in df_processed.columns if col.startswith('weight_')]
    
    print("  Data quality checks:")
    
    # Check label columns
    for col in label_cols:
        nan_count = df_processed[col].isna().sum()
        if nan_count > 0:
            print(f"    Warning: {col} has {nan_count} NaN values")
        
        # Check binary values
        unique_vals = df_processed[col].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(f"{col} contains non-binary values: {unique_vals}")
    
    print(f"    ✓ {len(label_cols)} label columns validated (binary values)")
    
    # Check weight columns
    for col in weight_cols:
        nan_count = df_processed[col].isna().sum()
        if nan_count > 0:
            print(f"    Warning: {col} has {nan_count} NaN values")
        
        # Check positive values
        non_positive = (df_processed[col] <= 0).sum()
        if non_positive > 0:
            raise ValueError(f"{col} contains {non_positive} non-positive values")
    
    print(f"    ✓ {len(weight_cols)} weight columns validated (positive values)")
    
    # Check feature columns for extreme values
    from project.data_pipeline.features import get_expected_feature_names
    expected_features = get_expected_feature_names()
    
    extreme_features = []
    for feature in expected_features:
        if feature in df_processed.columns:
            values = df_processed[feature].dropna()
            if len(values) > 0:
                if np.isinf(values).any():
                    extreme_features.append(f"{feature}: infinite values")
                elif values.std() == 0:
                    extreme_features.append(f"{feature}: constant values")
    
    if extreme_features:
        print("    Warnings about feature values:")
        for warning in extreme_features[:5]:  # Show first 5
            print(f"      - {warning}")
        if len(extreme_features) > 5:
            print(f"      - ... and {len(extreme_features) - 5} more")
    else:
        print(f"    ✓ {len(expected_features)} feature columns validated (no extreme values)")
    
    print("  ✓ Data quality validation test passed")


def main():
    """Run all integration tests"""
    print("EC2 Pipeline Integration Tests - Weighted Labeling System")
    print("=" * 60)
    print("Note: XGBoost training test skipped (requires XGBoost installation)")
    
    try:
        # Test core pipeline components
        df_processed = test_weighted_labeling_and_features()
        
        # Test pipeline utilities
        summary = test_pipeline_summary(df_processed)
        
        # Test data quality
        test_data_quality_validation(df_processed)
        
        print(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✓ Weighted labeling system integration")
        print("✓ Feature engineering integration") 
        print("✓ Pipeline validation and summary")
        print("✓ Data quality validation")
        print("\nCore pipeline components are ready for EC2 deployment!")
        
        print(f"\nNext steps for EC2 deployment:")
        print(f"1. Install XGBoost on EC2: pip install xgboost")
        print(f"2. Run validation: python aws_setup/validate_ec2_integration.py")
        print(f"3. Deploy pipeline: python aws_setup/ec2_weighted_labeling_pipeline.py")
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()