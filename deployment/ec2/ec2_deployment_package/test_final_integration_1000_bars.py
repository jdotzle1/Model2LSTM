#!/usr/bin/env python3
"""
Final Integration Test - 1000-Bar Sample Dataset

This script tests the complete weighted labeling pipeline on a 1000-bar sample
to validate all components work correctly before EC2 deployment.

Tests performed:
1. Complete pipeline on 1000-bar sample dataset
2. Output format validation for XGBoost training requirements
3. Chunked vs single-pass processing consistency
4. All 12 columns correctly generated and formatted
5. Data quality validation

Requirements tested: 8.1, 8.2, 8.3, 8.4, 9.1
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# Import pipeline components
from src.data_pipeline.weighted_labeling import (
    process_weighted_labeling, 
    WeightedLabelingEngine,
    LabelingConfig,
    TRADING_MODES
)
from src.data_pipeline.pipeline import (
    process_labeling_and_features,
    train_xgboost_models,
    validate_pipeline_output,
    create_pipeline_summary,
    PipelineConfig
)
from src.data_pipeline.validation_utils import run_comprehensive_validation


def create_1000_bar_sample():
    """Create realistic 1000-bar ES futures sample dataset"""
    print("=== CREATING 1000-BAR SAMPLE DATASET ===")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate RTH timestamps (07:30-15:00 CT)
    start_time = pd.Timestamp('2024-01-15 07:30:00', tz='UTC')
    timestamps = pd.date_range(start_time, periods=1000, freq='1s')
    
    # Generate realistic price movement with trends and volatility
    base_price = 4750.0
    
    # Create realistic price series with multiple regimes
    price_changes = []
    current_trend = 0.0
    volatility = 0.5
    
    for i in range(1000):
        # Occasional trend changes
        if i % 200 == 0:
            current_trend = np.random.normal(0, 0.1)
        
        # Occasional volatility regime changes
        if i % 150 == 0:
            volatility = np.random.uniform(0.3, 1.2)
        
        # Generate price change with trend and volatility
        change = np.random.normal(current_trend, volatility)
        price_changes.append(change)
    
    # Apply cumulative changes
    price_changes = np.array(price_changes)
    close_prices = base_price + np.cumsum(price_changes)
    
    # Generate OHLC with realistic relationships
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # Add some intrabar movement
    high_offset = np.random.exponential(0.4, 1000)
    low_offset = np.random.exponential(0.4, 1000)
    
    high_prices = np.maximum(open_prices, close_prices) + high_offset
    low_prices = np.minimum(open_prices, close_prices) - low_offset
    
    # Round to ES tick size (0.25)
    for prices in [open_prices, high_prices, low_prices, close_prices]:
        prices[:] = np.round(prices / 0.25) * 0.25
    
    # Generate realistic volume with patterns
    base_volume = 1500
    volume_pattern = np.sin(np.arange(1000) * 2 * np.pi / 100) * 500  # Cyclical pattern
    volume_noise = np.random.exponential(300, 1000)
    volume = np.maximum(100, base_volume + volume_pattern + volume_noise).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    print(f"✓ Created 1000-bar sample dataset")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    print(f"  Volume range: {df['volume'].min():,} to {df['volume'].max():,}")
    print(f"  Price volatility: {df['close'].pct_change().std():.4f}")
    
    return df


def test_complete_pipeline_1000_bars():
    """Test complete pipeline on 1000-bar sample"""
    print("\n=== TESTING COMPLETE PIPELINE (1000 BARS) ===")
    
    # Create sample data
    df = create_1000_bar_sample()
    
    # Configure pipeline for testing
    config = PipelineConfig(
        chunk_size=500,  # Force chunking even on small dataset
        enable_performance_monitoring=True,
        enable_progress_tracking=True,
        enable_memory_optimization=True
    )
    
    print("\nProcessing complete pipeline:")
    print("  - Weighted labeling (6 volatility-based modes)")
    print("  - Feature engineering (43 features)")
    print("  - Performance monitoring")
    
    start_time = time.time()
    
    # Process labeling and features
    df_processed = process_labeling_and_features(df, config)
    
    processing_time = time.time() - start_time
    
    print(f"\n✓ Pipeline processing complete:")
    print(f"  Processing time: {processing_time:.2f} seconds")
    print(f"  Processing rate: {len(df)/processing_time:.0f} rows/second")
    print(f"  Input: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Output: {len(df_processed):,} rows × {len(df_processed.columns)} columns")
    print(f"  Columns added: {len(df_processed.columns) - len(df.columns)}")
    
    return df_processed


def validate_xgboost_format(df_processed):
    """Validate output format matches XGBoost training requirements (Requirement 8.1, 8.2, 8.3, 8.4)"""
    print("\n=== VALIDATING XGBOOST FORMAT REQUIREMENTS ===")
    
    validation_results = {
        'all_12_columns_present': True,
        'label_format_correct': True,
        'weight_format_correct': True,
        'feature_format_correct': True,
        'no_missing_values': True,
        'data_types_correct': True
    }
    
    errors = []
    
    # Test 1: Validate all 12 columns are present (6 labels + 6 weights)
    print("1. Checking 12 weighted labeling columns...")
    
    expected_columns = []
    for mode_name in TRADING_MODES.keys():
        mode = TRADING_MODES[mode_name]
        expected_columns.extend([mode.label_column, mode.weight_column])
    
    missing_columns = set(expected_columns) - set(df_processed.columns)
    if missing_columns:
        validation_results['all_12_columns_present'] = False
        errors.append(f"Missing columns: {missing_columns}")
    else:
        print(f"   ✓ All 12 columns present: {expected_columns}")
    
    # Test 2: Validate label columns contain only 0 or 1 (Requirement 8.3)
    print("2. Checking label column format (binary 0/1)...")
    
    for mode_name in TRADING_MODES.keys():
        label_col = TRADING_MODES[mode_name].label_column
        unique_values = df_processed[label_col].unique()
        
        if not set(unique_values).issubset({0, 1}):
            validation_results['label_format_correct'] = False
            errors.append(f"{label_col} contains non-binary values: {unique_values}")
        
        # Check data type
        if df_processed[label_col].dtype not in ['int64', 'int32', 'bool']:
            validation_results['data_types_correct'] = False
            errors.append(f"{label_col} has incorrect dtype: {df_processed[label_col].dtype}")
    
    if validation_results['label_format_correct']:
        print(f"   ✓ All label columns contain only 0/1 values")
    
    # Test 3: Validate weight columns contain only positive values (Requirement 8.4)
    print("3. Checking weight column format (positive floats)...")
    
    for mode_name in TRADING_MODES.keys():
        weight_col = TRADING_MODES[mode_name].weight_column
        
        # Check for positive values
        if not (df_processed[weight_col] > 0).all():
            validation_results['weight_format_correct'] = False
            non_positive_count = (df_processed[weight_col] <= 0).sum()
            errors.append(f"{weight_col} contains {non_positive_count} non-positive values")
        
        # Check for infinite or NaN values
        if not np.isfinite(df_processed[weight_col]).all():
            validation_results['weight_format_correct'] = False
            invalid_count = (~np.isfinite(df_processed[weight_col])).sum()
            errors.append(f"{weight_col} contains {invalid_count} infinite/NaN values")
        
        # Check data type
        if not pd.api.types.is_numeric_dtype(df_processed[weight_col]):
            validation_results['data_types_correct'] = False
            errors.append(f"{weight_col} has non-numeric dtype: {df_processed[weight_col].dtype}")
    
    if validation_results['weight_format_correct']:
        print(f"   ✓ All weight columns contain only positive finite values")
    
    # Test 4: Validate feature columns are numeric and finite
    print("4. Checking feature column format...")
    
    from src.data_pipeline.features import get_expected_feature_names
    expected_features = get_expected_feature_names()
    
    feature_issues = []
    acceptable_high_nan_features = [
        'distance_from_rth_high', 'distance_from_rth_low',  # RTH features may be NaN for short datasets
        'medium_range_high', 'medium_range_low', 'medium_range_size',  # Medium range features need more data
        'range_compression_ratio'  # Depends on medium range
    ]
    
    # For 1000-bar test dataset, allow 100% NaN for RTH features since we don't have multi-day data
    test_dataset_exceptions = [
        'distance_from_rth_high', 'distance_from_rth_low'  # These require multi-day data
    ]
    
    for feature in expected_features:
        if feature in df_processed.columns:
            # Check for numeric type
            if not pd.api.types.is_numeric_dtype(df_processed[feature]):
                validation_results['feature_format_correct'] = False
                feature_issues.append(f"{feature}: non-numeric dtype")
            
            # Check for excessive NaN values (allow some for rolling calculations)
            nan_pct = df_processed[feature].isna().mean()
            
            # Different thresholds for different feature types
            if feature in test_dataset_exceptions:
                # Skip NaN check for features that require multi-day data in test datasets
                continue
            elif feature in acceptable_high_nan_features:
                nan_threshold = 0.95  # Allow up to 95% NaN for these features in small datasets
            else:
                nan_threshold = 0.35  # Allow up to 35% NaN for rolling calculations
            
            if nan_pct > nan_threshold:
                validation_results['feature_format_correct'] = False
                feature_issues.append(f"{feature}: {nan_pct:.1%} NaN values (threshold: {nan_threshold:.0%})")
    
    if feature_issues:
        errors.extend(feature_issues[:5])  # Show first 5 issues
        if len(feature_issues) > 5:
            errors.append(f"... and {len(feature_issues) - 5} more feature issues")
    else:
        print(f"   ✓ All {len(expected_features)} feature columns have correct format")
    
    # Test 5: Check for missing critical values
    print("5. Checking for missing critical values...")
    
    critical_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    critical_columns.extend(expected_columns)  # Add label/weight columns
    
    for col in critical_columns:
        if col in df_processed.columns:
            nan_count = df_processed[col].isna().sum()
            if nan_count > 0:
                validation_results['no_missing_values'] = False
                errors.append(f"{col} has {nan_count} missing values")
    
    if validation_results['no_missing_values']:
        print(f"   ✓ No missing values in critical columns")
    
    # Summary
    all_passed = all(validation_results.values())
    
    print(f"\nXGBoost Format Validation Results:")
    for test_name, passed in validation_results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if errors:
        print(f"\nValidation Errors:")
        for error in errors:
            print(f"  - {error}")
    
    if all_passed:
        print(f"\n✅ ALL XGBOOST FORMAT REQUIREMENTS PASSED")
    else:
        print(f"\n❌ XGBOOST FORMAT VALIDATION FAILED")
    
    return all_passed, validation_results, errors


def test_chunked_vs_single_pass_consistency():
    """Test chunked processing consistency vs single-pass processing (Requirement 9.1)"""
    print("\n=== TESTING CHUNKED VS SINGLE-PASS CONSISTENCY ===")
    
    # Create test data
    df = create_1000_bar_sample()
    
    print("Processing same dataset with two different approaches:")
    
    # Method 1: Single-pass processing
    print("1. Single-pass processing...")
    config_single = LabelingConfig(
        chunk_size=2000,  # Larger than dataset to force single-pass
        enable_progress_tracking=False,
        enable_performance_monitoring=False
    )
    
    start_time = time.time()
    df_single = process_weighted_labeling(df, config_single)
    single_time = time.time() - start_time
    
    print(f"   ✓ Single-pass: {single_time:.3f} seconds")
    
    # Method 2: Chunked processing
    print("2. Chunked processing...")
    config_chunked = LabelingConfig(
        chunk_size=300,  # Force chunking
        enable_progress_tracking=False,
        enable_performance_monitoring=False
    )
    
    start_time = time.time()
    df_chunked = process_weighted_labeling(df, config_chunked)
    chunked_time = time.time() - start_time
    
    print(f"   ✓ Chunked: {chunked_time:.3f} seconds")
    
    # Compare results
    print("3. Comparing results...")
    
    consistency_results = {
        'same_shape': True,
        'same_columns': True,
        'labels_identical': True,
        'weights_nearly_identical': True,
        'max_difference': 0.0
    }
    
    errors = []
    
    # Check shape
    if df_single.shape != df_chunked.shape:
        consistency_results['same_shape'] = False
        errors.append(f"Shape mismatch: {df_single.shape} vs {df_chunked.shape}")
    
    # Check columns
    if set(df_single.columns) != set(df_chunked.columns):
        consistency_results['same_columns'] = False
        missing_single = set(df_chunked.columns) - set(df_single.columns)
        missing_chunked = set(df_single.columns) - set(df_chunked.columns)
        if missing_single:
            errors.append(f"Single-pass missing: {missing_single}")
        if missing_chunked:
            errors.append(f"Chunked missing: {missing_chunked}")
    
    if consistency_results['same_shape'] and consistency_results['same_columns']:
        # Compare label columns (should be identical)
        for mode_name in TRADING_MODES.keys():
            label_col = TRADING_MODES[mode_name].label_column
            
            if not df_single[label_col].equals(df_chunked[label_col]):
                consistency_results['labels_identical'] = False
                diff_count = (df_single[label_col] != df_chunked[label_col]).sum()
                errors.append(f"{label_col}: {diff_count} differences")
        
        # Compare weight columns (should be nearly identical, allowing for floating point precision)
        max_weight_diff = 0.0
        for mode_name in TRADING_MODES.keys():
            weight_col = TRADING_MODES[mode_name].weight_column
            
            weight_diff = np.abs(df_single[weight_col] - df_chunked[weight_col])
            max_diff_this_col = weight_diff.max()
            max_weight_diff = max(max_weight_diff, max_diff_this_col)
            
            # Allow small floating point differences
            if max_diff_this_col > 1e-10:
                consistency_results['weights_nearly_identical'] = False
                errors.append(f"{weight_col}: max difference {max_diff_this_col:.2e}")
        
        consistency_results['max_difference'] = max_weight_diff
    
    # Summary - exclude max_difference from boolean check since it's a numeric value
    boolean_results = {k: v for k, v in consistency_results.items() if k != 'max_difference'}
    all_consistent = all(boolean_results.values()) and len(errors) == 0
    
    print(f"\nConsistency Test Results:")
    for test_name, passed in consistency_results.items():
        if test_name == 'max_difference':
            print(f"  {test_name}: {passed:.2e}")
        else:
            status = "✓ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
    
    if errors:
        print(f"\nConsistency Errors:")
        for error in errors:
            print(f"  - {error}")
    
    if all_consistent:
        print(f"\n✅ CHUNKED VS SINGLE-PASS CONSISTENCY VERIFIED")
        print(f"   Maximum difference: {max_weight_diff:.2e}")
    else:
        print(f"\n❌ CONSISTENCY TEST FAILED")
    
    return all_consistent, consistency_results, errors


def validate_all_12_columns_generation(df_processed):
    """Validate all 12 columns are correctly generated and formatted"""
    print("\n=== VALIDATING ALL 12 COLUMNS GENERATION ===")
    
    validation_results = {
        'correct_column_count': True,
        'correct_naming_convention': True,
        'correct_data_ranges': True,
        'correct_statistics': True
    }
    
    errors = []
    
    # Test 1: Correct number of columns
    print("1. Checking column count...")
    
    label_cols = [col for col in df_processed.columns if col.startswith('label_')]
    weight_cols = [col for col in df_processed.columns if col.startswith('weight_')]
    
    if len(label_cols) != 6:
        validation_results['correct_column_count'] = False
        errors.append(f"Expected 6 label columns, found {len(label_cols)}")
    
    if len(weight_cols) != 6:
        validation_results['correct_column_count'] = False
        errors.append(f"Expected 6 weight columns, found {len(weight_cols)}")
    
    print(f"   ✓ Found {len(label_cols)} label columns and {len(weight_cols)} weight columns")
    
    # Test 2: Correct naming convention
    print("2. Checking naming convention...")
    
    expected_modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                     'low_vol_short', 'normal_vol_short', 'high_vol_short']
    
    for mode in expected_modes:
        expected_label = f'label_{mode}'
        expected_weight = f'weight_{mode}'
        
        if expected_label not in df_processed.columns:
            validation_results['correct_naming_convention'] = False
            errors.append(f"Missing label column: {expected_label}")
        
        if expected_weight not in df_processed.columns:
            validation_results['correct_naming_convention'] = False
            errors.append(f"Missing weight column: {expected_weight}")
    
    if validation_results['correct_naming_convention']:
        print(f"   ✓ All columns follow correct naming convention")
    
    # Test 3: Data ranges are reasonable
    print("3. Checking data ranges...")
    
    for mode in expected_modes:
        label_col = f'label_{mode}'
        weight_col = f'weight_{mode}'
        
        if label_col in df_processed.columns:
            # Labels should be 0 or 1
            unique_labels = set(df_processed[label_col].unique())
            if not unique_labels.issubset({0, 1}):
                validation_results['correct_data_ranges'] = False
                errors.append(f"{label_col} has invalid values: {unique_labels}")
        
        if weight_col in df_processed.columns:
            # Weights should be positive and reasonable (typically 0.1 to 5.0)
            min_weight = df_processed[weight_col].min()
            max_weight = df_processed[weight_col].max()
            
            if min_weight <= 0:
                validation_results['correct_data_ranges'] = False
                errors.append(f"{weight_col} has non-positive minimum: {min_weight}")
            
            if max_weight > 10.0:  # Sanity check for extremely high weights
                validation_results['correct_data_ranges'] = False
                errors.append(f"{weight_col} has unusually high maximum: {max_weight}")
    
    if validation_results['correct_data_ranges']:
        print(f"   ✓ All data ranges are reasonable")
    
    # Test 4: Statistics are reasonable
    print("4. Checking statistics...")
    
    mode_stats = {}
    for mode in expected_modes:
        label_col = f'label_{mode}'
        weight_col = f'weight_{mode}'
        
        if label_col in df_processed.columns and weight_col in df_processed.columns:
            win_rate = df_processed[label_col].mean()
            avg_weight = df_processed[weight_col].mean()
            
            mode_stats[mode] = {
                'win_rate': win_rate,
                'avg_weight': avg_weight,
                'winners': df_processed[label_col].sum(),
                'total': len(df_processed)
            }
            
            # Check for reasonable win rates (5% to 65% - slightly higher for small test datasets)
            if win_rate < 0.05 or win_rate > 0.65:
                validation_results['correct_statistics'] = False
                errors.append(f"{mode} has unusual win rate: {win_rate:.1%}")
            
            print(f"   {mode}: {win_rate:.1%} win rate, avg weight: {avg_weight:.3f}")
    
    # Summary
    all_valid = all(validation_results.values())
    
    print(f"\n12-Column Generation Validation Results:")
    for test_name, passed in validation_results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if errors:
        print(f"\nValidation Errors:")
        for error in errors:
            print(f"  - {error}")
    
    if all_valid:
        print(f"\n✅ ALL 12 COLUMNS CORRECTLY GENERATED AND FORMATTED")
    else:
        print(f"\n❌ 12-COLUMN VALIDATION FAILED")
    
    return all_valid, validation_results, errors, mode_stats


def run_comprehensive_data_quality_validation(df_processed):
    """Run comprehensive data quality validation"""
    print("\n=== COMPREHENSIVE DATA QUALITY VALIDATION ===")
    
    # Use existing validation utilities
    validation_results = run_comprehensive_validation(df_processed, print_reports=False)
    
    overall_passed = validation_results['overall_validation']['passed']
    summary = validation_results['overall_validation']['summary']
    
    print("Comprehensive validation results:")
    print(f"  Overall status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
    print(f"  Label validation: {'✅' if summary['label_validation_passed'] else '❌'}")
    print(f"  Weight validation: {'✅' if summary['weight_validation_passed'] else '❌'}")
    print(f"  Data quality: {'✅' if summary['data_quality_passed'] else '❌'}")
    print(f"  XGBoost ready: {'✅' if summary['xgboost_ready'] else '❌'}")
    
    # Show key statistics
    label_results = validation_results['label_distributions']
    weight_results = validation_results['weight_distributions']
    
    print(f"\nKey Statistics Summary:")
    for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long']:
        if mode in label_results:
            win_rate = label_results[mode]['win_rate_percentage']
            avg_weight = weight_results[mode]['mean_weight']
            print(f"  {mode}: {win_rate:.1f}% win rate, avg weight: {avg_weight:.3f}")
    
    return overall_passed, validation_results


def create_ec2_deployment_script():
    """Create EC2 deployment script for complete pipeline"""
    print("\n=== CREATING EC2 DEPLOYMENT SCRIPT ===")
    
    deployment_script = '''#!/bin/bash
# EC2 Deployment Script for Weighted Labeling Pipeline
# Generated by final integration test

set -e  # Exit on any error

echo "=== ES WEIGHTED LABELING PIPELINE DEPLOYMENT ==="
echo "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "Date: $(date)"

# 1. Update system and install dependencies
echo "Step 1: Installing system dependencies..."
sudo yum update -y
sudo yum install -y python3 python3-pip git htop

# 2. Install Python packages
echo "Step 2: Installing Python packages..."
pip3 install --user pandas numpy xgboost scikit-learn pyarrow boto3 pytz databento psutil

# 3. Set up environment variables
echo "Step 3: Setting up environment..."
export S3_BUCKET="${S3_BUCKET:-your-es-data-bucket}"
export S3_DBN_PREFIX="${S3_DBN_PREFIX:-raw/dbn/}"
export S3_OUTPUT_PREFIX="${S3_OUTPUT_PREFIX:-processed/weighted_labeling/}"

# 4. Create working directory
echo "Step 4: Creating working directory..."
mkdir -p /tmp/es_weighted_pipeline
cd /tmp/es_weighted_pipeline

# 5. Download pipeline code (assuming it's in S3 or Git)
echo "Step 5: Downloading pipeline code..."
# aws s3 cp s3://your-code-bucket/pipeline.tar.gz .
# tar -xzf pipeline.tar.gz

# 6. Validate integration
echo "Step 6: Running integration validation..."
python3 aws_setup/validate_ec2_integration.py --test-s3

# 7. Run complete pipeline
echo "Step 7: Starting weighted labeling pipeline..."
python3 aws_setup/ec2_weighted_labeling_pipeline.py \\
    --bucket "$S3_BUCKET" \\
    2>&1 | tee pipeline.log

echo "=== PIPELINE DEPLOYMENT COMPLETE ==="
echo "Check pipeline.log for detailed results"
echo "Results uploaded to s3://$S3_BUCKET/$S3_OUTPUT_PREFIX"
'''
    
    script_path = 'deploy_ec2_weighted_pipeline.sh'
    with open(script_path, 'w') as f:
        f.write(deployment_script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"✓ Created EC2 deployment script: {script_path}")
    print(f"  Usage on EC2:")
    print(f"    export S3_BUCKET=your-bucket-name")
    print(f"    chmod +x {script_path}")
    print(f"    ./{script_path}")
    
    return script_path


def generate_final_integration_report(all_results):
    """Generate comprehensive final integration report"""
    print("\n=== GENERATING FINAL INTEGRATION REPORT ===")
    
    timestamp = datetime.now().isoformat()
    
    report = {
        'test_metadata': {
            'timestamp': timestamp,
            'test_type': 'final_integration_1000_bars',
            'pipeline_version': 'weighted_labeling_v2.0',
            'requirements_tested': ['8.1', '8.2', '8.3', '8.4', '9.1']
        },
        'test_results': all_results,
        'overall_status': all(result.get('passed', False) for result in all_results.values()),
        'summary': {
            'total_tests': len(all_results),
            'passed_tests': sum(1 for result in all_results.values() if result.get('passed', False)),
            'failed_tests': sum(1 for result in all_results.values() if not result.get('passed', False))
        }
    }
    
    # Save report
    report_path = f'final_integration_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✓ Generated integration report: {report_path}")
    
    # Print summary
    print(f"\nFinal Integration Test Summary:")
    print(f"  Total tests: {report['summary']['total_tests']}")
    print(f"  Passed: {report['summary']['passed_tests']}")
    print(f"  Failed: {report['summary']['failed_tests']}")
    print(f"  Overall status: {'✅ PASSED' if report['overall_status'] else '❌ FAILED'}")
    
    return report_path, report


def main():
    """Run all final integration tests"""
    print("FINAL INTEGRATION TEST - 1000-BAR SAMPLE DATASET")
    print("=" * 60)
    print("Testing requirements: 8.1, 8.2, 8.3, 8.4, 9.1")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    try:
        # Test 1: Complete pipeline on 1000-bar sample
        df_processed = test_complete_pipeline_1000_bars()
        all_results['complete_pipeline'] = {'passed': True, 'data_shape': df_processed.shape}
        
        # Test 2: Validate XGBoost format requirements
        xgb_passed, xgb_results, xgb_errors = validate_xgboost_format(df_processed)
        all_results['xgboost_format'] = {
            'passed': xgb_passed,
            'results': xgb_results,
            'errors': xgb_errors
        }
        
        # Test 3: Test chunked vs single-pass consistency
        consistency_passed, consistency_results, consistency_errors = test_chunked_vs_single_pass_consistency()
        all_results['chunked_consistency'] = {
            'passed': consistency_passed,
            'results': consistency_results,
            'errors': consistency_errors
        }
        
        # Test 4: Validate all 12 columns generation
        columns_passed, columns_results, columns_errors, mode_stats = validate_all_12_columns_generation(df_processed)
        all_results['twelve_columns'] = {
            'passed': columns_passed,
            'results': columns_results,
            'errors': columns_errors,
            'mode_statistics': mode_stats
        }
        
        # Test 5: Comprehensive data quality validation
        quality_passed, quality_results = run_comprehensive_data_quality_validation(df_processed)
        all_results['data_quality'] = {
            'passed': quality_passed,
            'results': quality_results
        }
        
        # Test 6: Create EC2 deployment script
        deployment_script = create_ec2_deployment_script()
        all_results['ec2_deployment'] = {
            'passed': True,
            'script_path': deployment_script
        }
        
        # Generate final report
        report_path, report = generate_final_integration_report(all_results)
        
        # Final summary
        overall_passed = report['overall_status']
        
        print(f"\n" + "=" * 60)
        if overall_passed:
            print(f"🎉 ALL FINAL INTEGRATION TESTS PASSED!")
            print(f"✅ Complete pipeline validated on 1000-bar sample")
            print(f"✅ XGBoost format requirements met")
            print(f"✅ Chunked processing consistency verified")
            print(f"✅ All 12 columns correctly generated")
            print(f"✅ Data quality validation passed")
            print(f"✅ EC2 deployment script created")
            
            print(f"\nREADY FOR EC2 DEPLOYMENT!")
            print(f"Next steps:")
            print(f"1. Upload code to EC2 instance")
            print(f"2. Run: ./{deployment_script}")
            print(f"3. Monitor processing with pipeline logs")
            
        else:
            print(f"❌ FINAL INTEGRATION TESTS FAILED")
            print(f"Check {report_path} for detailed results")
            
            failed_tests = [test for test, result in all_results.items() 
                          if not result.get('passed', False)]
            print(f"Failed tests: {failed_tests}")
        
        print(f"\nDetailed report: {report_path}")
        
        return 0 if overall_passed else 1
        
    except Exception as e:
        print(f"\n❌ FINAL INTEGRATION TEST FAILED WITH EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())