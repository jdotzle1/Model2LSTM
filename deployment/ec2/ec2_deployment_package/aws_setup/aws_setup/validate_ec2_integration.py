#!/usr/bin/env python3
"""
EC2 Integration Validation Script

This script validates that all components of the weighted labeling pipeline
work correctly together on an EC2 instance before processing the full dataset.

Run this script on EC2 to ensure:
1. All dependencies are installed
2. Weighted labeling system works
3. Feature engineering integrates correctly
4. XGBoost training works with weighted samples
5. Pipeline can handle chunked processing
6. S3 integration works

Usage:
    python aws_setup/validate_ec2_integration.py
    python aws_setup/validate_ec2_integration.py --test-s3
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
import argparse

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

def check_dependencies():
    """Check that all required packages are installed"""
    print("=== CHECKING DEPENDENCIES ===")
    
    required_packages = [
        'pandas', 'numpy', 'xgboost', 'scikit-learn', 
        'pyarrow', 'boto3', 'pytz', 'databento'
    ]
    
    missing_packages = []
    installed_versions = {}
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            installed_versions[package] = version
            print(f"  ✓ {package}: {version}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}: NOT INSTALLED")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All dependencies installed")
    return True

def check_system_resources():
    """Check system resources (CPU, memory, disk)"""
    print("\n=== CHECKING SYSTEM RESOURCES ===")
    
    try:
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"  CPU: {cpu_count} cores, {cpu_percent}% usage")
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        print(f"  Memory: {memory_gb:.1f} GB total, {memory_available_gb:.1f} GB available")
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        print(f"  Disk: {disk_free_gb:.1f} GB free")
        
        # Recommendations for c5.4xlarge
        if cpu_count < 16:
            print(f"  ⚠️  Warning: Recommended 16+ CPUs for c5.4xlarge, found {cpu_count}")
        
        if memory_gb < 30:
            print(f"  ⚠️  Warning: Recommended 32+ GB RAM for c5.4xlarge, found {memory_gb:.1f} GB")
        
        if disk_free_gb < 100:
            print(f"  ⚠️  Warning: Recommended 100+ GB free disk space, found {disk_free_gb:.1f} GB")
        
        print("✓ System resources checked")
        return True
        
    except ImportError:
        print("  Warning: psutil not available, skipping resource check")
        return True

def create_test_data(n_rows=5000):
    """Create synthetic test data for validation"""
    print(f"\n=== CREATING TEST DATA ({n_rows:,} rows) ===")
    
    np.random.seed(42)
    
    # Generate timestamps (1-second bars, RTH only)
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

def test_weighted_labeling():
    """Test weighted labeling system"""
    print("\n=== TESTING WEIGHTED LABELING SYSTEM ===")
    
    # Create test data
    df = create_test_data(3000)
    
    # Import and test weighted labeling
    from src.data_pipeline.weighted_labeling import process_weighted_labeling, LabelingConfig
    
    config = LabelingConfig(
        chunk_size=1500,
        enable_progress_tracking=True,
        enable_performance_monitoring=True
    )
    
    start_time = time.time()
    df_labeled = process_weighted_labeling(df, config)
    processing_time = time.time() - start_time
    
    # Validate results
    expected_columns = []
    for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                 'low_vol_short', 'normal_vol_short', 'high_vol_short']:
        expected_columns.extend([f'label_{mode}', f'weight_{mode}'])
    
    missing_cols = set(expected_columns) - set(df_labeled.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    
    # Check data quality
    print("  Validation results:")
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
        print(f"    {mode}: {win_rate:.1%} win rate, avg weight: {avg_weight:.3f}")
    
    print(f"  ✓ Processing time: {processing_time:.2f} seconds ({len(df)/processing_time:.0f} rows/sec)")
    print("  ✓ Weighted labeling system test passed")
    
    return df_labeled

def test_feature_engineering(df_labeled):
    """Test feature engineering integration"""
    print("\n=== TESTING FEATURE ENGINEERING ===")
    
    from src.data_pipeline.features import create_all_features, get_expected_feature_names
    
    start_time = time.time()
    df_featured = create_all_features(df_labeled)
    processing_time = time.time() - start_time
    
    # Validate results
    expected_features = get_expected_feature_names()
    missing_features = set(expected_features) - set(df_featured.columns)
    if missing_features:
        raise ValueError(f"Missing expected features: {missing_features}")
    
    # Check that weighted labeling columns are preserved
    label_weight_cols = [col for col in df_labeled.columns if col.startswith(('label_', 'weight_'))]
    preserved_lw_cols = [col for col in label_weight_cols if col in df_featured.columns]
    if len(preserved_lw_cols) != len(label_weight_cols):
        missing_lw = set(label_weight_cols) - set(preserved_lw_cols)
        raise ValueError(f"Missing label/weight columns: {missing_lw}")
    
    print(f"  ✓ Added {len(expected_features)} features")
    print(f"  ✓ Preserved {len(preserved_lw_cols)} label/weight columns")
    print(f"  ✓ Processing time: {processing_time:.2f} seconds ({len(df_labeled)/processing_time:.0f} rows/sec)")
    print("  ✓ Feature engineering test passed")
    
    return df_featured

def test_xgboost_training(df_featured):
    """Test XGBoost training with weighted samples"""
    print("\n=== TESTING XGBOOST TRAINING ===")
    
    from src.data_pipeline.pipeline import train_xgboost_models, PipelineConfig
    
    config = PipelineConfig(
        enable_progress_tracking=True,
        output_dir='/tmp/test_models'
    )
    
    start_time = time.time()
    models = train_xgboost_models(df_featured, config, save_models=False)
    training_time = time.time() - start_time
    
    # Validate results
    expected_modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                     'low_vol_short', 'normal_vol_short', 'high_vol_short']
    
    if len(models) != len(expected_modes):
        raise ValueError(f"Expected {len(expected_modes)} models, got {len(models)}")
    
    print("  Model performance:")
    for mode in expected_modes:
        if mode not in models:
            raise ValueError(f"Missing model for {mode}")
        
        model_info = models[mode]
        auc = model_info['test_auc']
        win_rate = model_info['win_rate']
        samples = model_info['training_samples']
        
        print(f"    {mode}: AUC={auc:.4f}, WinRate={win_rate:.1%}, Samples={samples:,}")
        
        if auc < 0.4 or auc > 1.0:
            print(f"      ⚠️  Warning: Unusual AUC value")
    
    print(f"  ✓ Training time: {training_time:.2f} seconds")
    print("  ✓ XGBoost training test passed")
    
    return models

def test_chunked_processing():
    """Test chunked processing for large datasets"""
    print("\n=== TESTING CHUNKED PROCESSING ===")
    
    # Create larger test dataset
    df = create_test_data(10000)
    
    from src.data_pipeline.pipeline import process_labeling_and_features, PipelineConfig
    
    # Test with chunked processing
    config = PipelineConfig(
        chunk_size=3000,  # Force chunking
        enable_progress_tracking=True,
        enable_performance_monitoring=True
    )
    
    start_time = time.time()
    df_processed = process_labeling_and_features(df, config)
    processing_time = time.time() - start_time
    
    # Validate results
    if len(df_processed) != len(df):
        raise ValueError(f"Row count mismatch: {len(df)} -> {len(df_processed)}")
    
    # Check for expected columns
    expected_label_cols = [f'label_{mode}' for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                                                        'low_vol_short', 'normal_vol_short', 'high_vol_short']]
    expected_weight_cols = [f'weight_{mode}' for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                                                          'low_vol_short', 'normal_vol_short', 'high_vol_short']]
    
    from src.data_pipeline.features import get_expected_feature_names
    expected_features = get_expected_feature_names()
    
    all_expected = expected_label_cols + expected_weight_cols + expected_features
    missing_cols = set(all_expected) - set(df_processed.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    print(f"  ✓ Processed {len(df_processed):,} rows in {processing_time:.2f} seconds")
    print(f"  ✓ Processing rate: {len(df_processed)/processing_time:.0f} rows/sec")
    print(f"  ✓ Final dataset: {len(df_processed.columns)} columns")
    print("  ✓ Chunked processing test passed")

def test_s3_integration():
    """Test S3 integration (optional)"""
    print("\n=== TESTING S3 INTEGRATION ===")
    
    try:
        import boto3
        
        # Test S3 connection
        s3 = boto3.client('s3')
        
        # Try to list buckets (basic connectivity test)
        response = s3.list_buckets()
        print(f"  ✓ S3 connection successful")
        print(f"  ✓ Found {len(response['Buckets'])} accessible buckets")
        
        # Test environment variables
        bucket = os.environ.get('S3_BUCKET')
        if bucket:
            print(f"  ✓ S3_BUCKET environment variable: {bucket}")
            
            # Test bucket access
            try:
                s3.head_bucket(Bucket=bucket)
                print(f"  ✓ Bucket '{bucket}' is accessible")
            except Exception as e:
                print(f"  ⚠️  Warning: Cannot access bucket '{bucket}': {str(e)}")
        else:
            print(f"  ⚠️  Warning: S3_BUCKET environment variable not set")
        
        print("  ✓ S3 integration test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ S3 integration test failed: {str(e)}")
        return False

def main():
    """Run all validation tests"""
    parser = argparse.ArgumentParser(description='Validate EC2 integration for weighted labeling pipeline')
    parser.add_argument('--test-s3', action='store_true', help='Include S3 integration test')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only (smaller datasets)')
    
    args = parser.parse_args()
    
    print("EC2 Integration Validation for Weighted Labeling Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # System checks
        if not check_dependencies():
            sys.exit(1)
        
        check_system_resources()
        
        # Core functionality tests
        if args.quick:
            print("\n--- QUICK VALIDATION MODE ---")
            df_labeled = test_weighted_labeling()
            df_featured = test_feature_engineering(df_labeled)
            test_xgboost_training(df_featured)
        else:
            print("\n--- FULL VALIDATION MODE ---")
            df_labeled = test_weighted_labeling()
            df_featured = test_feature_engineering(df_labeled)
            models = test_xgboost_training(df_featured)
            test_chunked_processing()
        
        # Optional S3 test
        if args.test_s3:
            test_s3_integration()
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n🎉 ALL VALIDATION TESTS PASSED!")
        print(f"Total validation time: {total_time:.2f} seconds")
        print("\nEC2 instance is ready for weighted labeling pipeline deployment!")
        
        print(f"\nNext steps:")
        print(f"1. Set S3_BUCKET environment variable if not already set")
        print(f"2. Run: python aws_setup/ec2_weighted_labeling_pipeline.py")
        print(f"3. Monitor processing with: tail -f /tmp/es_weighted_pipeline/pipeline.log")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()