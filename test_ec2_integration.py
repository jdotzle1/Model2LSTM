#!/usr/bin/env python3
"""
Integration Test for EC2 Pipeline Components

This script tests the integration between:
1. Weighted labeling system
2. Feature engineering
3. XGBoost training
4. Complete pipeline

Run this before deploying to EC2 to ensure all components work together.
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
    process_complete_pipeline, 
    process_labeling_and_features,
    train_xgboost_models,
    validate_pipeline_output,
    create_pipeline_summary,
    PipelineConfig
)


def create_test_data(n_rows=5000):
    """Create synthetic test data for integration testing"""
    print(f"Creating synthetic test data ({n_rows:,} rows)...")
    
    # Create realistic ES futures data
    np.random.seed(42)
    
    # Generate timestamps (1-second bars)
    start_date = pd.Timestamp('2024-01-01 07:30:00', tz='UTC')
    timestamps = pd.date_range(start_date, periods=n_rows, freq='1S')
    
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


def test_weighted_labeling_integration():
    """Test weighted labeling system integration"""
    print("\n=== TESTING WEIGHTED LABELING INTEGRATION ===")
    
    # Create test data
    df = create_test_data(2000)
    
    # Test weighted labeling
    from project.data_pipeline.weighted_labeling import process_weighted_labeling, LabelingConfig
    
    config = LabelingConfig(
        chunk_size=1000,
        enable_progress_tracking=True,
        enable_performance_monitoring=False  # Disable for test
    )
    
    print("  Testing new weighted labeling system...")
    print("    - 6 volatility-based trading modes")
    print("    - Quality, velocity, and time decay weights")
    
    df_labeled = process_weighted_labeling(df, config)
    
    # Validate results
    expected_columns = []
    for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                 'low_vol_short', 'normal_vol_short', 'high_vol_short']:
        expected_columns.extend([f'label_{mode}', f'weight_{mode}'])
    
    missing_cols = set(expected_columns) - set(df_labeled.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    
    print(f"  ✓ Added {len(expected_columns)} columns (6 labels + 6 weights)")
    
    # Check data quality
    total_winners = 0
    total_samples = len(df_labeled)
    
    for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                 'low_vol_short', 'normal_vol_short', 'high_vol_short']:
        label_col = f'label_{mode}'
        weight_col = f'weight_{mode}'
        
        # Check label values
        if not df_labeled[label_col].isin([0, 1]).all():
            raise ValueError(f"{label_col} contains non-binary values")
        
        # Check weight values
        if not (df_labeled[weight_col] > 0).all():
            raise ValueError(f"{weight_col} contains non-positive values")
        
        win_rate = df_labeled[label_col].mean()
        avg_weight = df_labeled[weight_col].mean()
        winners = df_labeled[label_col].sum()
        total_winners += winners
        
        print(f"    {mode}: {win_rate:.1%} win rate ({winners:,} winners), avg weight: {avg_weight:.3f}")
    
    print(f"  ✓ Total winners across all modes: {total_winners:,} / {total_samples * 6:,} ({total_winners/(total_samples * 6):.1%})")
    print("  ✓ Weighted labeling integration test passed")
    return df_labeled


def test_feature_engineering_integration(df_labeled):
    """Test feature engineering integration with labeled data"""
    print("\n=== TESTING FEATURE ENGINEERING INTEGRATION ===")
    
    from project.data_pipeline.features import (
        create_all_features, 
        get_expected_feature_names,
        validate_weighted_labeling_compatibility
    )
    
    # Validate compatibility with weighted labeling system
    labeling_info = validate_weighted_labeling_compatibility(df_labeled)
    print(f"  Labeling system: {labeling_info['labeling_system']}")
    
    if not labeling_info['compatible']:
        raise ValueError(f"Dataset not compatible: {labeling_info['warnings']}")
    
    # Apply feature engineering
    print("  Adding 43 engineered features...")
    df_featured = create_all_features(df_labeled)
    
    # Validate results
    expected_features = get_expected_feature_names()
    missing_features = set(expected_features) - set(df_featured.columns)
    if missing_features:
        raise ValueError(f"Missing expected features: {missing_features}")
    
    # Check that original columns are preserved (including weighted labeling columns)
    original_cols = set(df_labeled.columns)
    preserved_cols = set(df_featured.columns) & original_cols
    if len(preserved_cols) != len(original_cols):
        missing_original = original_cols - preserved_cols
        raise ValueError(f"Missing original columns: {missing_original}")
    
    # Specifically check that weighted labeling columns are preserved
    label_weight_cols = labeling_info['label_columns'] + labeling_info['weight_columns']
    preserved_lw_cols = [col for col in label_weight_cols if col in df_featured.columns]
    if len(preserved_lw_cols) != len(label_weight_cols):
        missing_lw = set(label_weight_cols) - set(preserved_lw_cols)
        raise ValueError(f"Missing label/weight columns: {missing_lw}")
    
    print(f"  ✓ Added {len(expected_features)} features")
    print(f"  ✓ Preserved {len(preserved_cols)} original columns")
    print(f"  ✓ Preserved {len(preserved_lw_cols)} label/weight columns")
    print(f"  Final dataset: {len(df_featured):,} rows × {len(df_featured.columns)} columns")
    
    print("  ✓ Feature engineering integration test passed")
    return df_featured


def test_xgboost_training_integration(df_featured):
    """Test XGBoost training integration with featured data"""
    print("\n=== TESTING XGBOOST TRAINING INTEGRATION ===")
    
    config = PipelineConfig(
        enable_progress_tracking=True,
        output_dir='./test_models'
    )
    
    # Train models
    models = train_xgboost_models(df_featured, config, save_models=False)
    
    # Validate results
    expected_modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                     'low_vol_short', 'normal_vol_short', 'high_vol_short']
    
    if len(models) != len(expected_modes):
        raise ValueError(f"Expected {len(expected_modes)} models, got {len(models)}")
    
    for mode in expected_modes:
        if mode not in models:
            raise ValueError(f"Missing model for {mode}")
        
        model_info = models[mode]
        required_keys = ['model', 'test_auc', 'win_rate', 'feature_importance']
        missing_keys = set(required_keys) - set(model_info.keys())
        if missing_keys:
            raise ValueError(f"Missing model info keys for {mode}: {missing_keys}")
        
        # Check AUC is reasonable
        auc = model_info['test_auc']
        if auc < 0.4 or auc > 1.0:
            print(f"  Warning: {mode} has unusual AUC: {auc:.4f}")
    
    print("  ✓ XGBoost training integration test passed")
    return models


def test_complete_pipeline_integration():
    """Test complete end-to-end pipeline"""
    print("\n=== TESTING COMPLETE PIPELINE INTEGRATION ===")
    
    # Create test data and save to file
    df = create_test_data(3000)
    test_input_path = './test_input.parquet'
    test_output_path = './test_output.parquet'
    
    df.to_parquet(test_input_path)
    
    try:
        # Test complete pipeline
        config = PipelineConfig(
            chunk_size=1500,
            enable_progress_tracking=True,
            enable_performance_monitoring=False,
            output_dir='./test_pipeline_output'
        )
        
        df_result = process_complete_pipeline(test_input_path, test_output_path, config)
        
        # Validate pipeline output
        validation_results = validate_pipeline_output(df_result)
        
        if not validation_results['valid']:
            raise ValueError(f"Pipeline validation failed: {validation_results['errors']}")
        
        if validation_results['warnings']:
            print("  Warnings:")
            for warning in validation_results['warnings']:
                print(f"    - {warning}")
        
        # Create pipeline summary
        summary = create_pipeline_summary(df_result)
        print(f"  Pipeline summary:")
        print(f"    - Dataset: {summary['dataset_info']['rows']:,} rows × {summary['dataset_info']['columns']} columns")
        print(f"    - Memory usage: {summary['dataset_info']['memory_usage_mb']:.1f} MB")
        print(f"    - Labeling: {summary['labeling_info']['modes']} modes, {summary['labeling_info']['columns_added']} columns")
        print(f"    - Features: {summary['feature_info']['features_added']} features")
        
        print("  ✓ Complete pipeline integration test passed")
        
    finally:
        # Cleanup test files
        for path in [test_input_path, test_output_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # Cleanup test directories
        import shutil
        for dir_path in ['./test_models', './test_pipeline_output']:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)


def main():
    """Run all integration tests"""
    print("EC2 Pipeline Integration Tests")
    print("=" * 50)
    
    try:
        # Test individual components
        df_labeled = test_weighted_labeling_integration()
        df_featured = test_feature_engineering_integration(df_labeled)
        models = test_xgboost_training_integration(df_featured)
        
        # Test complete pipeline
        test_complete_pipeline_integration()
        
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✓ Weighted labeling system integration")
        print("✓ Feature engineering integration") 
        print("✓ XGBoost training integration")
        print("✓ Complete pipeline integration")
        print("\nEC2 pipeline is ready for deployment!")
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()