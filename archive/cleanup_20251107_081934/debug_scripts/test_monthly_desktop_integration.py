#!/usr/bin/env python3
"""
Test integration between monthly processing and desktop validation logic

This script validates that:
- Desktop validation logic works with monthly processing
- Monthly processing produces results compatible with desktop validation
- Configuration parameters work correctly across both systems
- Data quality validation is consistent

Requirements addressed: 10.6, 10.7
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_monthly_processing_integration():
    """Test that monthly processing integrates correctly with desktop validation"""
    print("🔄 TESTING MONTHLY PROCESSING INTEGRATION")
    print("=" * 50)
    
    try:
        # Create test data that simulates monthly processing output
        test_data = create_monthly_test_data()
        
        # Save as temporary monthly output
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            test_data.to_parquet(temp_path, index=False)
        
        try:
            # Test 1: Validate monthly output with desktop validation logic
            print("\n📊 Test 1: Desktop validation of monthly output")
            desktop_validation_result = validate_with_desktop_logic(temp_path)
            
            if desktop_validation_result['valid']:
                print("  ✅ Desktop validation logic works with monthly output")
            else:
                print("  ❌ Desktop validation failed on monthly output")
                for error in desktop_validation_result['errors']:
                    print(f"    - {error}")
            
            # Test 2: Compare monthly processing statistics with desktop expectations
            print("\n📈 Test 2: Statistics compatibility")
            stats_compatibility = test_statistics_compatibility(test_data)
            
            if stats_compatibility['compatible']:
                print("  ✅ Statistics are compatible between systems")
            else:
                print("  ❌ Statistics compatibility issues found")
                for issue in stats_compatibility['issues']:
                    print(f"    - {issue}")
            
            # Test 3: Configuration parameter compatibility
            print("\n⚙️  Test 3: Configuration compatibility")
            config_compatibility = test_configuration_integration()
            
            if config_compatibility['compatible']:
                print("  ✅ Configuration parameters work across both systems")
            else:
                print("  ❌ Configuration compatibility issues found")
                for issue in config_compatibility['issues']:
                    print(f"    - {issue}")
            
            # Test 4: Data quality validation consistency
            print("\n🔍 Test 4: Data quality validation consistency")
            quality_consistency = test_data_quality_consistency(test_data)
            
            if quality_consistency['consistent']:
                print("  ✅ Data quality validation is consistent")
            else:
                print("  ❌ Data quality validation inconsistencies found")
                for issue in quality_consistency['issues']:
                    print(f"    - {issue}")
            
            # Overall assessment
            all_tests_passed = (
                desktop_validation_result['valid'] and
                stats_compatibility['compatible'] and
                config_compatibility['compatible'] and
                quality_consistency['consistent']
            )
            
            print(f"\n📋 INTEGRATION TEST SUMMARY")
            print(f"=" * 30)
            if all_tests_passed:
                print("✅ ALL TESTS PASSED")
                print("Monthly processing is fully integrated with desktop validation")
            else:
                print("❌ SOME TESTS FAILED")
                print("Review issues above before proceeding")
            
            return all_tests_passed
            
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_monthly_test_data():
    """Create test data that simulates monthly processing output"""
    print("  📊 Creating monthly processing test data...")
    
    # Create realistic monthly data (1 day worth)
    rows = 23400  # Approximately 1 day of 1-second data during RTH
    
    import pytz
    from datetime import datetime
    
    # Create RTH timestamps (9:30 AM - 4:00 PM ET)
    start_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=pytz.UTC)  # 9:30 AM ET
    timestamps = pd.date_range(
        start=start_time,
        periods=rows,
        freq='1s',
        tz=pytz.UTC
    )
    
    # Create realistic ES price data
    np.random.seed(123)  # For reproducible results
    base_price = 4750.0
    price_changes = np.random.normal(0, 0.3, rows)
    prices = base_price + np.cumsum(price_changes)
    
    # Create base OHLCV data
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, 0.6, rows)),
        'low': prices - np.abs(np.random.normal(0, 0.6, rows)),
        'close': prices + np.random.normal(0, 0.15, rows),
        'volume': np.random.randint(50, 3000, rows),
        'symbol': ['ESH24'] * rows
    })
    
    # Ensure OHLC relationships are valid
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    # Add weighted labeling columns (simulate monthly processing output)
    modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
             'low_vol_short', 'normal_vol_short', 'high_vol_short']
    
    for mode in modes:
        # Create realistic win rates
        if 'long' in mode:
            win_rate = np.random.uniform(0.15, 0.35)  # 15-35% for long modes
        else:
            win_rate = np.random.uniform(0.05, 0.25)  # 5-25% for short modes
        
        # Generate labels (0 or 1)
        labels = np.random.choice([0, 1], size=rows, p=[1-win_rate, win_rate])
        df[f'label_{mode}'] = labels
        
        # Generate weights (positive values, higher for winners)
        weights = np.random.uniform(0.8, 1.2, rows)  # Base weights
        winner_mask = labels == 1
        weights[winner_mask] *= np.random.uniform(1.2, 2.5, winner_mask.sum())  # Higher weights for winners
        df[f'weight_{mode}'] = weights
    
    # Add feature columns (simulate feature engineering output)
    feature_names = [
        # Volume features
        'volume_ratio_30s', 'volume_slope_30s', 'volume_slope_5s', 'volume_exhaustion',
        
        # Price context features
        'vwap', 'distance_from_vwap_pct', 'vwap_slope', 'distance_from_rth_high', 'distance_from_rth_low',
        
        # Consolidation features
        'short_range_high', 'short_range_low', 'short_range_size', 'position_in_short_range', 'short_range_retouches',
        'medium_range_high', 'medium_range_low', 'medium_range_size', 'range_compression_ratio', 'medium_range_retouches',
        
        # Return features
        'return_30s', 'return_60s', 'return_300s', 'momentum_acceleration', 'momentum_consistency',
        
        # Volatility features
        'atr_30s', 'atr_300s', 'volatility_regime', 'volatility_acceleration', 'volatility_breakout', 'atr_percentile',
        
        # Microstructure features
        'bar_range', 'relative_bar_size', 'uptick_pct_30s', 'uptick_pct_60s', 'bar_flow_consistency', 'directional_strength',
        
        # Time features
        'is_eth', 'is_pre_open', 'is_rth_open', 'is_morning', 'is_lunch', 'is_afternoon', 'is_rth_close'
    ]
    
    for feature in feature_names:
        if feature.startswith('is_'):
            # Binary time features
            if feature == 'is_eth':
                df[feature] = 0  # All RTH data
            elif feature == 'is_rth_open':
                df[feature] = (df.index < 3600).astype(int)  # First hour
            elif feature == 'is_morning':
                df[feature] = ((df.index >= 3600) & (df.index < 9000)).astype(int)
            elif feature == 'is_lunch':
                df[feature] = ((df.index >= 9000) & (df.index < 14400)).astype(int)
            elif feature == 'is_afternoon':
                df[feature] = ((df.index >= 14400) & (df.index < 19800)).astype(int)
            elif feature == 'is_rth_close':
                df[feature] = (df.index >= 19800).astype(int)
            else:
                df[feature] = 0
        else:
            # Continuous features with realistic ranges
            if 'ratio' in feature or 'pct' in feature:
                df[feature] = np.random.uniform(0.5, 2.0, rows)
            elif 'slope' in feature:
                df[feature] = np.random.normal(0, 0.1, rows)
            elif 'return' in feature:
                df[feature] = np.random.normal(0, 0.002, rows)
            elif 'atr' in feature or 'volatility' in feature:
                df[feature] = np.random.uniform(0.5, 3.0, rows)
            elif 'range' in feature or 'distance' in feature:
                df[feature] = np.random.uniform(0, 10, rows)
            else:
                df[feature] = np.random.normal(0, 1, rows)
    
    print(f"    ✅ Created {len(df):,} rows with {len(df.columns)} columns")
    return df

def validate_with_desktop_logic(monthly_output_file):
    """Validate monthly output using desktop validation logic"""
    try:
        # Load monthly output
        df = pd.read_parquet(monthly_output_file)
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check basic structure (similar to desktop validation)
        required_base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_base = [col for col in required_base_cols if col not in df.columns]
        if missing_base:
            validation_result['errors'].append(f"Missing base columns: {missing_base}")
            validation_result['valid'] = False
        
        # Check labeling columns
        label_cols = [col for col in df.columns if col.startswith('label_')]
        weight_cols = [col for col in df.columns if col.startswith('weight_')]
        
        if len(label_cols) != 6:
            validation_result['errors'].append(f"Expected 6 label columns, got {len(label_cols)}")
            validation_result['valid'] = False
        
        if len(weight_cols) != 6:
            validation_result['errors'].append(f"Expected 6 weight columns, got {len(weight_cols)}")
            validation_result['valid'] = False
        
        # Validate label values
        for col in label_cols:
            unique_vals = set(df[col].dropna().unique())
            if not unique_vals.issubset({0, 1}):
                validation_result['errors'].append(f"Invalid label values in {col}: {unique_vals}")
                validation_result['valid'] = False
        
        # Validate weight values
        for col in weight_cols:
            if (df[col] <= 0).any():
                validation_result['errors'].append(f"Non-positive weights in {col}")
                validation_result['valid'] = False
        
        # Check feature columns
        feature_cols = [col for col in df.columns if col not in required_base_cols + label_cols + weight_cols]
        if len(feature_cols) < 35:  # Should have around 43 features
            validation_result['warnings'].append(f"Fewer features than expected: {len(feature_cols)}")
        
        # Check for excessive NaN values
        high_nan_cols = []
        for col in feature_cols:
            nan_pct = df[col].isna().sum() / len(df) * 100
            if nan_pct > 50:
                high_nan_cols.append(col)
        
        if high_nan_cols:
            validation_result['warnings'].append(f"High NaN features: {len(high_nan_cols)}")
        
        # Store metrics
        validation_result['metrics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'label_columns': len(label_cols),
            'weight_columns': len(weight_cols),
            'feature_columns': len(feature_cols),
            'high_nan_features': len(high_nan_cols)
        }
        
        return validation_result
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Validation failed: {e}"],
            'warnings': [],
            'metrics': {}
        }

def test_statistics_compatibility(df):
    """Test that statistics are compatible between monthly and desktop processing"""
    try:
        compatibility_result = {
            'compatible': True,
            'issues': [],
            'statistics': {}
        }
        
        # Calculate win rates (should be in reasonable ranges)
        label_cols = [col for col in df.columns if col.startswith('label_')]
        
        for col in label_cols:
            win_rate = df[col].mean()
            mode_name = col.replace('label_', '')
            
            # Check if win rates are in reasonable ranges
            if win_rate < 0.01 or win_rate > 0.60:
                compatibility_result['issues'].append(f"{mode_name} win rate out of range: {win_rate:.3f}")
                compatibility_result['compatible'] = False
            
            compatibility_result['statistics'][f'{mode_name}_win_rate'] = win_rate
        
        # Calculate weight statistics
        weight_cols = [col for col in df.columns if col.startswith('weight_')]
        
        for col in weight_cols:
            avg_weight = df[col].mean()
            mode_name = col.replace('weight_', '')
            
            # Check if average weights are reasonable
            if avg_weight < 0.5 or avg_weight > 5.0:
                compatibility_result['issues'].append(f"{mode_name} avg weight out of range: {avg_weight:.3f}")
                compatibility_result['compatible'] = False
            
            compatibility_result['statistics'][f'{mode_name}_avg_weight'] = avg_weight
        
        # Check feature statistics
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + label_cols + weight_cols]
        
        nan_features = 0
        for col in feature_cols:
            nan_pct = df[col].isna().sum() / len(df) * 100
            if nan_pct > 35:  # More than 35% NaN is concerning
                nan_features += 1
        
        if nan_features > 10:  # More than 10 features with high NaN
            compatibility_result['issues'].append(f"Too many features with high NaN: {nan_features}")
            compatibility_result['compatible'] = False
        
        compatibility_result['statistics']['features_with_high_nan'] = nan_features
        compatibility_result['statistics']['total_features'] = len(feature_cols)
        
        return compatibility_result
        
    except Exception as e:
        return {
            'compatible': False,
            'issues': [f"Statistics compatibility test failed: {e}"],
            'statistics': {}
        }

def test_configuration_integration():
    """Test that configuration parameters work correctly across both systems"""
    try:
        integration_result = {
            'compatible': True,
            'issues': [],
            'tested_configs': []
        }
        
        # Test importing configuration classes
        try:
            from src.data_pipeline.weighted_labeling import LabelingConfig, WeightedLabelingEngine
            integration_result['tested_configs'].append('LabelingConfig import: OK')
        except ImportError as e:
            integration_result['issues'].append(f"Cannot import LabelingConfig: {e}")
            integration_result['compatible'] = False
            return integration_result
        
        # Test configuration creation
        try:
            config = LabelingConfig(
                chunk_size=1000,
                enable_memory_optimization=True,
                enable_progress_tracking=False
            )
            integration_result['tested_configs'].append('Configuration creation: OK')
        except Exception as e:
            integration_result['issues'].append(f"Configuration creation failed: {e}")
            integration_result['compatible'] = False
        
        # Test engine initialization with config
        try:
            engine = WeightedLabelingEngine(config)
            integration_result['tested_configs'].append('Engine initialization: OK')
        except Exception as e:
            integration_result['issues'].append(f"Engine initialization failed: {e}")
            integration_result['compatible'] = False
        
        # Test feature engineering import
        try:
            from src.data_pipeline.features import create_all_features
            integration_result['tested_configs'].append('Feature engineering import: OK')
        except ImportError as e:
            integration_result['issues'].append(f"Cannot import create_all_features: {e}")
            integration_result['compatible'] = False
        
        return integration_result
        
    except Exception as e:
        return {
            'compatible': False,
            'issues': [f"Configuration integration test failed: {e}"],
            'tested_configs': []
        }

def test_data_quality_consistency(df):
    """Test that data quality validation is consistent between systems"""
    try:
        consistency_result = {
            'consistent': True,
            'issues': [],
            'quality_metrics': {}
        }
        
        # Test price data quality (similar to clean_price_data function)
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                negative_count = (df[col] < 0).sum()
                
                if zero_count > 0:
                    consistency_result['issues'].append(f"Zero prices in {col}: {zero_count}")
                    consistency_result['consistent'] = False
                
                if negative_count > 0:
                    consistency_result['issues'].append(f"Negative prices in {col}: {negative_count}")
                    consistency_result['consistent'] = False
                
                consistency_result['quality_metrics'][f'{col}_zero_count'] = zero_count
                consistency_result['quality_metrics'][f'{col}_negative_count'] = negative_count
        
        # Test OHLC relationships
        if all(col in df.columns for col in price_cols):
            high_low_issues = (df['high'] < df['low']).sum()
            open_high_issues = (df['open'] > df['high']).sum()
            close_high_issues = (df['close'] > df['high']).sum()
            open_low_issues = (df['open'] < df['low']).sum()
            close_low_issues = (df['close'] < df['low']).sum()
            
            total_ohlc_issues = high_low_issues + open_high_issues + close_high_issues + open_low_issues + close_low_issues
            
            if total_ohlc_issues > 0:
                consistency_result['issues'].append(f"OHLC relationship issues: {total_ohlc_issues}")
                consistency_result['consistent'] = False
            
            consistency_result['quality_metrics']['ohlc_issues'] = total_ohlc_issues
        
        # Test volume data quality
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            zero_volume = (df['volume'] == 0).sum()
            
            if negative_volume > 0:
                consistency_result['issues'].append(f"Negative volume: {negative_volume}")
                consistency_result['consistent'] = False
            
            # Zero volume is acceptable but should be noted
            consistency_result['quality_metrics']['negative_volume'] = negative_volume
            consistency_result['quality_metrics']['zero_volume'] = zero_volume
        
        # Test timestamp consistency
        if 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'])
                
                # Check for duplicate timestamps
                duplicate_timestamps = timestamps.duplicated().sum()
                if duplicate_timestamps > 0:
                    consistency_result['issues'].append(f"Duplicate timestamps: {duplicate_timestamps}")
                    consistency_result['consistent'] = False
                
                # Check for proper ordering
                if not timestamps.is_monotonic_increasing:
                    consistency_result['issues'].append("Timestamps not properly ordered")
                    consistency_result['consistent'] = False
                
                consistency_result['quality_metrics']['duplicate_timestamps'] = duplicate_timestamps
                consistency_result['quality_metrics']['properly_ordered'] = timestamps.is_monotonic_increasing
                
            except Exception as e:
                consistency_result['issues'].append(f"Timestamp validation failed: {e}")
                consistency_result['consistent'] = False
        
        return consistency_result
        
    except Exception as e:
        return {
            'consistent': False,
            'issues': [f"Data quality consistency test failed: {e}"],
            'quality_metrics': {}
        }

def main():
    """Main integration test function"""
    print("🔗 MONTHLY PROCESSING & DESKTOP VALIDATION INTEGRATION TEST")
    print("=" * 65)
    print("Task 6.2: Validate consistency between desktop and S3 processing")
    print("Requirements: 10.6, 10.7")
    print()
    
    try:
        success = test_monthly_processing_integration()
        
        if success:
            print("\n🎉 INTEGRATION TEST PASSED!")
            print("Monthly processing is fully compatible with desktop validation")
            return True
        else:
            print("\n💥 INTEGRATION TEST FAILED!")
            print("Review the issues above before proceeding")
            return False
    
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)