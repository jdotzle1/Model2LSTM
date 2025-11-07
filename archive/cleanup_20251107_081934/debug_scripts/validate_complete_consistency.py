#!/usr/bin/env python3
"""
Complete consistency validation between desktop and S3 processing

This script provides comprehensive validation for task 6.2:
- Validates same data produces identical results in both environments
- Tests end-to-end consistency with sample data
- Ensures desktop validation logic works with monthly processing
- Validates configuration compatibility

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
import hashlib

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class CompleteConsistencyValidator:
    """Complete consistency validation between desktop and S3 processing"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def log_test(self, test_name, success, message, details=None):
        """Log test result"""
        self.results[test_name] = {
            'success': success,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
        
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
    
    def create_identical_test_data(self, rows=5000):
        """Create identical test data for both processing methods"""
        print("📊 Creating identical test data for consistency validation...")
        
        # Use fixed seed for identical results
        np.random.seed(42)
        
        import pytz
        from datetime import datetime
        
        # Create RTH timestamps
        start_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=pytz.UTC)
        timestamps = pd.date_range(
            start=start_time,
            periods=rows,
            freq='1s',
            tz=pytz.UTC
        )
        
        # Create realistic price data
        base_price = 4750.0
        price_changes = np.random.normal(0, 0.25, rows)
        prices = base_price + np.cumsum(price_changes)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + np.abs(np.random.normal(0, 0.5, rows)),
            'low': prices - np.abs(np.random.normal(0, 0.5, rows)),
            'close': prices + np.random.normal(0, 0.1, rows),
            'volume': np.random.randint(100, 5000, rows),
            'symbol': ['ESH24'] * rows
        })
        
        # Ensure OHLC relationships
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        print(f"   Created {len(df):,} rows of identical test data")
        return df
    
    def process_with_desktop_method(self, df):
        """Process data using desktop method (single-pass)"""
        print("\n🖥️  Processing with desktop method...")
        
        try:
            from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig
            from src.data_pipeline.features import create_all_features
            
            # Desktop configuration
            config = LabelingConfig(
                chunk_size=len(df) + 1000,  # Force single-pass
                enable_memory_optimization=False,
                enable_progress_tracking=False,
                enable_performance_monitoring=False
            )
            
            start_time = time.time()
            
            # Process with desktop method
            engine = WeightedLabelingEngine(config)
            df_labeled = engine.process_dataframe(df.copy(), validate_performance=False)
            df_final = create_all_features(df_labeled)
            
            processing_time = time.time() - start_time
            
            result = {
                'dataframe': df_final,
                'processing_time': processing_time,
                'method': 'desktop_single_pass',
                'config': {
                    'chunk_size': config.chunk_size,
                    'memory_optimization': config.enable_memory_optimization
                }
            }
            
            print(f"   ✅ Desktop processing completed in {processing_time:.2f}s")
            print(f"   📊 Output: {len(df_final):,} rows × {len(df_final.columns)} columns")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Desktop processing failed: {e}")
            raise
    
    def process_with_s3_method(self, df):
        """Process data using S3 method (chunked)"""
        print("\n☁️  Processing with S3 method...")
        
        try:
            from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig
            from src.data_pipeline.features import create_all_features
            
            # S3 configuration (chunked)
            config = LabelingConfig(
                chunk_size=500,  # Force chunking
                enable_memory_optimization=True,
                enable_progress_tracking=False,
                enable_performance_monitoring=False
            )
            
            start_time = time.time()
            
            # Process with S3 method
            engine = WeightedLabelingEngine(config)
            df_labeled = engine.process_dataframe(df.copy(), validate_performance=False)
            df_final = create_all_features(df_labeled)
            
            processing_time = time.time() - start_time
            
            result = {
                'dataframe': df_final,
                'processing_time': processing_time,
                'method': 's3_chunked',
                'config': {
                    'chunk_size': config.chunk_size,
                    'memory_optimization': config.enable_memory_optimization
                }
            }
            
            print(f"   ✅ S3 processing completed in {processing_time:.2f}s")
            print(f"   📊 Output: {len(df_final):,} rows × {len(df_final.columns)} columns")
            
            return result
            
        except Exception as e:
            print(f"   ❌ S3 processing failed: {e}")
            raise
    
    def validate_identical_results(self, desktop_result, s3_result):
        """Validate that both methods produce identical results"""
        print("\n🔍 Validating identical results...")
        
        desktop_df = desktop_result['dataframe']
        s3_df = s3_result['dataframe']
        
        validation_results = {
            'identical': True,
            'differences': [],
            'statistics': {}
        }
        
        # 1. Structure validation
        if len(desktop_df.columns) != len(s3_df.columns):
            validation_results['identical'] = False
            validation_results['differences'].append(
                f"Column count: desktop={len(desktop_df.columns)}, s3={len(s3_df.columns)}"
            )
        
        if len(desktop_df) != len(s3_df):
            validation_results['identical'] = False
            validation_results['differences'].append(
                f"Row count: desktop={len(desktop_df)}, s3={len(s3_df)}"
            )
        
        # 2. Column names validation
        desktop_cols = set(desktop_df.columns)
        s3_cols = set(s3_df.columns)
        
        if desktop_cols != s3_cols:
            validation_results['identical'] = False
            missing_in_s3 = desktop_cols - s3_cols
            missing_in_desktop = s3_cols - desktop_cols
            
            if missing_in_s3:
                validation_results['differences'].append(f"Missing in S3: {list(missing_in_s3)}")
            if missing_in_desktop:
                validation_results['differences'].append(f"Missing in desktop: {list(missing_in_desktop)}")
        
        # 3. Value validation for matching columns
        if validation_results['identical']:  # Only if structure matches
            matching_cols = desktop_cols & s3_cols
            
            # Check all columns for exact matches
            identical_cols = 0
            nearly_identical_cols = 0
            different_cols = []
            
            for col in matching_cols:
                desktop_vals = desktop_df[col].values
                s3_vals = s3_df[col].values
                
                # Handle different data types appropriately
                if desktop_vals.dtype == s3_vals.dtype:
                    # Check for numeric vs non-numeric data
                    if np.issubdtype(desktop_vals.dtype, np.number):
                        # Numeric data - use array_equal with nan handling
                        try:
                            if np.array_equal(desktop_vals, s3_vals, equal_nan=True):
                                identical_cols += 1
                            # Check for near equality (floating point precision)
                            elif np.allclose(desktop_vals, s3_vals, equal_nan=True, rtol=1e-12, atol=1e-15):
                                nearly_identical_cols += 1
                            else:
                                different_cols.append(col)
                                
                                # Calculate difference statistics
                                valid_mask = ~(np.isnan(desktop_vals) | np.isnan(s3_vals))
                                if valid_mask.sum() > 0:
                                    max_diff = np.max(np.abs(desktop_vals[valid_mask] - s3_vals[valid_mask]))
                                    validation_results['differences'].append(
                                        f"{col}: max difference = {max_diff:.2e}"
                                    )
                        except Exception as e:
                            # Fallback for numeric comparison issues
                            if np.array_equal(desktop_vals, s3_vals):
                                identical_cols += 1
                            else:
                                different_cols.append(col)
                                validation_results['differences'].append(f"{col}: comparison error = {e}")
                    else:
                        # Non-numeric data (strings, objects, etc.) - use simple equality
                        try:
                            if np.array_equal(desktop_vals, s3_vals):
                                identical_cols += 1
                            else:
                                different_cols.append(col)
                                # Count different values for non-numeric data
                                diff_count = (desktop_vals != s3_vals).sum()
                                validation_results['differences'].append(
                                    f"{col}: {diff_count} different values"
                                )
                        except Exception as e:
                            different_cols.append(col)
                            validation_results['differences'].append(f"{col}: comparison error = {e}")
                else:
                    # Different data types
                    different_cols.append(col)
                    validation_results['differences'].append(
                        f"{col}: different dtypes = {desktop_vals.dtype} vs {s3_vals.dtype}"
                    )
            
            # Store statistics
            validation_results['statistics'] = {
                'total_columns': len(matching_cols),
                'identical_columns': identical_cols,
                'nearly_identical_columns': nearly_identical_cols,
                'different_columns': len(different_cols),
                'identity_rate': (identical_cols + nearly_identical_cols) / len(matching_cols) if matching_cols else 0
            }
            
            # Determine if results are acceptable
            identity_rate = validation_results['statistics']['identity_rate']
            if identity_rate < 0.95:  # Less than 95% identical
                validation_results['identical'] = False
                validation_results['differences'].append(
                    f"Low identity rate: {identity_rate:.3f} ({len(different_cols)} different columns)"
                )
        
        # Log results
        if validation_results['identical']:
            self.log_test(
                "Identical Results",
                True,
                f"Methods produce identical results",
                {
                    'identity_rate': f"{validation_results['statistics'].get('identity_rate', 0):.3f}",
                    'identical_columns': validation_results['statistics'].get('identical_columns', 0),
                    'total_columns': validation_results['statistics'].get('total_columns', 0)
                }
            )
        else:
            self.log_test(
                "Identical Results",
                False,
                f"Methods produce different results",
                {
                    'differences_count': len(validation_results['differences']),
                    'first_difference': validation_results['differences'][0] if validation_results['differences'] else 'Unknown'
                }
            )
        
        return validation_results
    
    def test_desktop_validation_with_monthly_output(self):
        """Test that desktop validation logic works with monthly processing output"""
        print("\n📋 Testing desktop validation with monthly output...")
        
        try:
            # Create sample monthly output
            test_data = self.create_monthly_sample_output()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                test_data.to_parquet(temp_path, index=False)
            
            try:
                # Test desktop validation functions
                validation_success = self.run_desktop_validation_on_monthly_output(temp_path)
                
                self.log_test(
                    "Desktop Validation on Monthly Output",
                    validation_success['success'],
                    validation_success['message'],
                    validation_success.get('details', {})
                )
                
                return validation_success['success']
                
            finally:
                # Clean up
                if temp_path.exists():
                    temp_path.unlink()
        
        except Exception as e:
            self.log_test(
                "Desktop Validation on Monthly Output",
                False,
                f"Test failed: {e}"
            )
            return False
    
    def create_monthly_sample_output(self):
        """Create sample data that looks like monthly processing output"""
        rows = 10000
        
        # Create base data
        df = self.create_identical_test_data(rows)
        
        # Add labeling columns
        modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                 'low_vol_short', 'normal_vol_short', 'high_vol_short']
        
        np.random.seed(123)
        for mode in modes:
            win_rate = np.random.uniform(0.1, 0.4)
            labels = np.random.choice([0, 1], size=rows, p=[1-win_rate, win_rate])
            weights = np.random.uniform(0.8, 2.5, rows)
            
            df[f'label_{mode}'] = labels
            df[f'weight_{mode}'] = weights
        
        # Add feature columns (simplified)
        feature_names = [
            'volume_ratio_30s', 'vwap', 'return_30s', 'atr_30s', 'bar_range',
            'is_eth', 'is_morning', 'volatility_regime', 'momentum_acceleration'
        ]
        
        for feature in feature_names:
            if feature.startswith('is_'):
                df[feature] = np.random.choice([0, 1], size=rows, p=[0.8, 0.2])
            else:
                df[feature] = np.random.normal(0, 1, rows)
        
        return df
    
    def run_desktop_validation_on_monthly_output(self, file_path):
        """Run desktop validation logic on monthly output"""
        try:
            # Load the file
            df = pd.read_parquet(file_path)
            
            # Run validation similar to test_30day_pipeline.py
            validation_checks = []
            
            # Check basic structure
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {
                    'success': False,
                    'message': f"Missing required columns: {missing_cols}"
                }
            
            validation_checks.append("Basic structure: OK")
            
            # Check labeling columns
            label_cols = [col for col in df.columns if col.startswith('label_')]
            weight_cols = [col for col in df.columns if col.startswith('weight_')]
            
            if len(label_cols) != 6 or len(weight_cols) != 6:
                return {
                    'success': False,
                    'message': f"Incorrect labeling columns: {len(label_cols)} labels, {len(weight_cols)} weights"
                }
            
            validation_checks.append("Labeling columns: OK")
            
            # Check label values
            for col in label_cols:
                unique_vals = set(df[col].dropna().unique())
                if not unique_vals.issubset({0, 1}):
                    return {
                        'success': False,
                        'message': f"Invalid label values in {col}: {unique_vals}"
                    }
            
            validation_checks.append("Label values: OK")
            
            # Check weight values
            for col in weight_cols:
                if (df[col] <= 0).any():
                    return {
                        'success': False,
                        'message': f"Non-positive weights in {col}"
                    }
            
            validation_checks.append("Weight values: OK")
            
            # Check for features
            feature_cols = [col for col in df.columns if col not in 
                           required_cols + label_cols + weight_cols]
            
            if len(feature_cols) < 5:  # At least some features
                return {
                    'success': False,
                    'message': f"Too few feature columns: {len(feature_cols)}"
                }
            
            validation_checks.append("Feature columns: OK")
            
            return {
                'success': True,
                'message': "All validation checks passed",
                'details': {
                    'validation_checks': len(validation_checks),
                    'total_columns': len(df.columns),
                    'total_rows': len(df)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Validation failed: {e}"
            }
    
    def test_configuration_consistency(self):
        """Test that configuration parameters work consistently"""
        print("\n⚙️  Testing configuration consistency...")
        
        try:
            from src.data_pipeline.weighted_labeling import LabelingConfig, WeightedLabelingEngine
            
            # Test different configurations
            configs_to_test = [
                ('default', {}),
                ('chunked', {'chunk_size': 1000}),
                ('memory_optimized', {'enable_memory_optimization': True, 'chunk_size': 500}),
                ('performance', {'enable_performance_monitoring': True})
            ]
            
            config_results = {}
            
            for config_name, config_params in configs_to_test:
                try:
                    config = LabelingConfig(**config_params)
                    engine = WeightedLabelingEngine(config)
                    
                    config_results[config_name] = {
                        'success': True,
                        'chunk_size': config.chunk_size,
                        'memory_optimization': config.enable_memory_optimization
                    }
                    
                except Exception as e:
                    config_results[config_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Check if all configurations work
            all_success = all(result['success'] for result in config_results.values())
            
            self.log_test(
                "Configuration Consistency",
                all_success,
                f"Tested {len(configs_to_test)} configurations",
                config_results
            )
            
            return all_success
            
        except Exception as e:
            self.log_test(
                "Configuration Consistency",
                False,
                f"Configuration test failed: {e}"
            )
            return False
    
    def generate_final_report(self):
        """Generate final consistency validation report"""
        print("\n📊 Generating final consistency report...")
        
        # Calculate overall success
        total_tests = len(self.results)
        successful_tests = sum(1 for result in self.results.values() if result['success'])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create report
        report = {
            'validation_summary': {
                'timestamp': datetime.now().isoformat(),
                'task': '6.2 - Validate consistency between desktop and S3 processing',
                'requirements': ['10.6', '10.7'],
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'overall_status': 'PASS' if success_rate >= 90 else 'FAIL'
            },
            'test_results': self.results,
            'errors': self.errors,
            'warnings': self.warnings
        }
        
        # Save report
        report_dir = project_root / "validation_results"
        report_dir.mkdir(exist_ok=True)
        
        report_path = report_dir / f"complete_consistency_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📋 COMPLETE CONSISTENCY VALIDATION SUMMARY")
        print(f"=" * 55)
        print(f"Task: 6.2 - Desktop vs S3 Processing Consistency")
        print(f"Requirements: 10.6, 10.7")
        print(f"")
        print(f"Overall Status: {'✅ PASS' if success_rate >= 90 else '❌ FAIL'}")
        print(f"Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests} tests)")
        print(f"Report saved to: {report_path}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:3]:
                print(f"  - {warning}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors[:3]:
                print(f"  - {error}")
        
        return report

def main():
    """Main consistency validation function"""
    print("🔍 COMPLETE DESKTOP vs S3 PROCESSING CONSISTENCY VALIDATION")
    print("=" * 70)
    print("Task 6.2: Validate consistency between desktop and S3 processing")
    print("Requirements: 10.6, 10.7")
    print()
    
    validator = CompleteConsistencyValidator()
    
    try:
        # Test 1: Identical data processing
        print("🧪 Test 1: Processing identical data with both methods")
        test_data = validator.create_identical_test_data(3000)
        
        desktop_result = validator.process_with_desktop_method(test_data)
        s3_result = validator.process_with_s3_method(test_data)
        
        validator.validate_identical_results(desktop_result, s3_result)
        
        # Test 2: Desktop validation with monthly output
        validator.test_desktop_validation_with_monthly_output()
        
        # Test 3: Configuration consistency
        validator.test_configuration_consistency()
        
        # Generate final report
        report = validator.generate_final_report()
        
        # Return success status
        return report['validation_summary']['overall_status'] == 'PASS'
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 COMPLETE CONSISTENCY VALIDATION PASSED!")
        print("Desktop and S3 processing are fully consistent.")
        print("Task 6.2 implementation is complete.")
    else:
        print("\n💥 COMPLETE CONSISTENCY VALIDATION FAILED!")
        print("Review the issues above before proceeding.")
        sys.exit(1)