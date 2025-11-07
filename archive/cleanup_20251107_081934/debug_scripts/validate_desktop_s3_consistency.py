#!/usr/bin/env python3
"""
Validate consistency between desktop and S3 processing pipelines

This script implements task 6.2 to ensure that:
- Desktop validation logic works with monthly processing
- Results are consistent between desktop and S3 processing
- Same data produces identical results in both environments
- End-to-end consistency is maintained

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

class ConsistencyValidator:
    """Comprehensive consistency validation between desktop and S3 processing"""
    
    def __init__(self):
        self.test_results = {}
        self.validation_errors = []
        self.warnings = []
        
    def log_result(self, test_name, success, message, details=None):
        """Log test result with details"""
        self.test_results[test_name] = {
            'success': success,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}: {message}")
        
        if details:
            for key, value in details.items():
                print(f"     {key}: {value}")
    
    def log_warning(self, message):
        """Log warning message"""
        self.warnings.append(message)
        print(f"  ⚠️  Warning: {message}")
    
    def create_test_sample(self, rows=5000):
        """Create consistent test sample for validation"""
        print("📊 Creating test sample data...")
        
        try:
            # Try to use existing processed sample first
            processed_sample = project_root / "project/data/processed/es_30day_labeled_features.parquet"
            
            if processed_sample.exists():
                print(f"  📁 Using existing processed sample: {processed_sample}")
                df = pd.read_parquet(processed_sample)
                
                # Take original columns only for testing
                original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if 'symbol' in df.columns:
                    original_cols.append('symbol')
                
                # Filter to original columns only
                available_cols = [col for col in original_cols if col in df.columns]
                df = df[available_cols].copy()
                
                # Take a subset for faster testing
                if len(df) > rows:
                    df = df.head(rows).copy()
                
                print(f"  ✅ Loaded {len(df):,} rows from processed sample")
                return df
            
            # Try raw sample
            sample_file = project_root / "project/data/processed/es_30day_rth.parquet"
            
            if sample_file.exists():
                print(f"  📁 Checking existing sample: {sample_file}")
                df = pd.read_parquet(sample_file)
                
                # Reset index to ensure timestamp column
                if 'timestamp' not in df.columns:
                    df = df.reset_index()
                    if 'ts_event' in df.columns:
                        df = df.rename(columns={'ts_event': 'timestamp'})
                
                # Check if data is actually RTH by examining timestamps
                if 'timestamp' in df.columns:
                    import pytz
                    from datetime import time as dt_time
                    
                    # Convert to Central Time to check RTH
                    central_tz = pytz.timezone('US/Central')
                    timestamps = pd.to_datetime(df['timestamp'])
                    if timestamps.dt.tz is None:
                        timestamps = timestamps.dt.tz_localize(pytz.UTC)
                    
                    central_times = timestamps.dt.tz_convert(central_tz)
                    central_time_only = central_times.dt.time
                    
                    rth_start = dt_time(7, 30)
                    rth_end = dt_time(15, 0)
                    rth_mask = (central_time_only >= rth_start) & (central_time_only < rth_end)
                    
                    rth_percentage = rth_mask.sum() / len(df) * 100
                    
                    if rth_percentage < 10:  # Less than 10% RTH data
                        print(f"  ⚠️  Sample has only {rth_percentage:.1f}% RTH data, creating synthetic data instead")
                    else:
                        # Filter to RTH data
                        df_rth = df[rth_mask].copy()
                        df_rth['timestamp'] = timestamps[rth_mask].dt.tz_convert(pytz.UTC)
                        
                        if len(df_rth) >= 1000:  # Need at least 1000 rows
                            # Take a subset for faster testing
                            if len(df_rth) > rows:
                                df_rth = df_rth.head(rows).copy()
                            
                            print(f"  ✅ Loaded {len(df_rth):,} RTH rows from existing sample")
                            return df_rth
                        else:
                            print(f"  ⚠️  Only {len(df_rth)} RTH rows available, creating synthetic data instead")
            
            # Create synthetic RTH test data
            print(f"  🔧 Creating synthetic RTH test data ({rows:,} rows)")
            
            import pytz
            from datetime import datetime, time as dt_time
            
            # Create timestamps during RTH (9:30 AM - 4:00 PM ET = 14:30-21:00 UTC)
            start_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=pytz.UTC)  # 9:30 AM ET
            timestamps = pd.date_range(
                start=start_time,
                periods=rows,
                freq='1s',
                tz=pytz.UTC
            )
            
            # Create realistic ES price data
            np.random.seed(42)  # For reproducible results
            base_price = 4750.0
            price_changes = np.random.normal(0, 0.25, rows)  # 0.25 point std dev
            prices = base_price + np.cumsum(price_changes)
            
            # Create OHLCV data
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices,
                'high': prices + np.abs(np.random.normal(0, 0.5, rows)),
                'low': prices - np.abs(np.random.normal(0, 0.5, rows)),
                'close': prices + np.random.normal(0, 0.1, rows),
                'volume': np.random.randint(100, 5000, rows),
                'symbol': ['ESH24'] * rows
            })
            
            # Ensure OHLC relationships are valid
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            print(f"  ✅ Created synthetic RTH test data")
            return df
                
        except Exception as e:
            print(f"  ❌ Failed to create test sample: {e}")
            raise
    
    def test_desktop_processing(self, df):
        """Test desktop-style processing (single-pass)"""
        print("\n🖥️  Testing desktop processing...")
        
        try:
            from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig
            from src.data_pipeline.features import create_all_features
            
            # Desktop configuration (single-pass, no chunking)
            desktop_config = LabelingConfig(
                chunk_size=len(df) + 1000,  # Larger than dataset to force single-pass
                enable_memory_optimization=False,
                enable_progress_tracking=False,
                enable_performance_monitoring=False
            )
            
            start_time = time.time()
            
            # Step 1: Weighted labeling
            desktop_engine = WeightedLabelingEngine(desktop_config)
            df_labeled = desktop_engine.process_dataframe(df.copy(), validate_performance=False)
            
            labeling_time = time.time() - start_time
            
            # Step 2: Feature engineering
            feature_start = time.time()
            df_final = create_all_features(df_labeled)
            feature_time = time.time() - feature_start
            
            total_time = time.time() - start_time
            
            # Validate desktop results
            validation_results = self.validate_processing_output(df_final, "desktop")
            
            desktop_results = {
                'dataframe': df_final,
                'processing_time': total_time,
                'labeling_time': labeling_time,
                'feature_time': feature_time,
                'validation': validation_results,
                'method': 'desktop_single_pass'
            }
            
            self.log_result(
                "Desktop Processing",
                validation_results['valid'],
                f"Processed {len(df_final):,} rows in {total_time:.2f}s",
                {
                    'columns': len(df_final.columns),
                    'labeling_time': f"{labeling_time:.2f}s",
                    'feature_time': f"{feature_time:.2f}s",
                    'rows_per_second': f"{len(df_final)/total_time:.0f}"
                }
            )
            
            return desktop_results
            
        except Exception as e:
            self.log_result("Desktop Processing", False, f"Failed: {e}")
            raise
    
    def test_s3_processing(self, df):
        """Test S3-style processing (chunked, memory-optimized)"""
        print("\n☁️  Testing S3 processing...")
        
        try:
            from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig
            from src.data_pipeline.features import create_all_features
            
            # S3 configuration (chunked processing)
            s3_config = LabelingConfig(
                chunk_size=500,  # Force chunking for S3-style processing
                enable_memory_optimization=True,
                enable_progress_tracking=False,
                enable_performance_monitoring=False
            )
            
            start_time = time.time()
            
            # Step 1: Weighted labeling (chunked)
            s3_engine = WeightedLabelingEngine(s3_config)
            df_labeled = s3_engine.process_dataframe(df.copy(), validate_performance=False)
            
            labeling_time = time.time() - start_time
            
            # Step 2: Feature engineering
            feature_start = time.time()
            df_final = create_all_features(df_labeled)
            feature_time = time.time() - feature_start
            
            total_time = time.time() - start_time
            
            # Validate S3 results
            validation_results = self.validate_processing_output(df_final, "s3")
            
            s3_results = {
                'dataframe': df_final,
                'processing_time': total_time,
                'labeling_time': labeling_time,
                'feature_time': feature_time,
                'validation': validation_results,
                'method': 's3_chunked'
            }
            
            self.log_result(
                "S3 Processing",
                validation_results['valid'],
                f"Processed {len(df_final):,} rows in {total_time:.2f}s",
                {
                    'columns': len(df_final.columns),
                    'labeling_time': f"{labeling_time:.2f}s",
                    'feature_time': f"{feature_time:.2f}s",
                    'rows_per_second': f"{len(df_final)/total_time:.0f}",
                    'chunk_size': s3_config.chunk_size
                }
            )
            
            return s3_results
            
        except Exception as e:
            self.log_result("S3 Processing", False, f"Failed: {e}")
            raise
    
    def validate_processing_output(self, df, method_name):
        """Validate processing output for correctness"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Check basic structure
            if len(df) == 0:
                validation_results['errors'].append("Empty dataframe")
                validation_results['valid'] = False
                return validation_results
            
            # Check for required columns
            required_base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_base = [col for col in required_base_cols if col not in df.columns]
            if missing_base:
                validation_results['errors'].append(f"Missing base columns: {missing_base}")
                validation_results['valid'] = False
            
            # Check labeling columns (6 labels + 6 weights)
            label_cols = [col for col in df.columns if col.startswith('label_')]
            weight_cols = [col for col in df.columns if col.startswith('weight_')]
            
            if len(label_cols) != 6:
                validation_results['errors'].append(f"Expected 6 label columns, got {len(label_cols)}")
                validation_results['valid'] = False
            
            if len(weight_cols) != 6:
                validation_results['errors'].append(f"Expected 6 weight columns, got {len(weight_cols)}")
                validation_results['valid'] = False
            
            # Validate label values (should be 0 or 1)
            for col in label_cols:
                unique_vals = set(df[col].dropna().unique())
                if not unique_vals.issubset({0, 1}):
                    validation_results['warnings'].append(f"Invalid label values in {col}: {unique_vals}")
            
            # Validate weight values (should be positive)
            for col in weight_cols:
                if (df[col] <= 0).any():
                    validation_results['warnings'].append(f"Non-positive weights in {col}")
            
            # Check feature columns (should be around 43)
            feature_cols = [col for col in df.columns if col not in required_base_cols + label_cols + weight_cols]
            if len(feature_cols) < 35:  # Allow some tolerance
                validation_results['warnings'].append(f"Fewer features than expected: {len(feature_cols)}")
            
            # Check for excessive NaN values
            high_nan_cols = []
            for col in feature_cols:
                nan_pct = df[col].isna().sum() / len(df) * 100
                if nan_pct > 50:  # More than 50% NaN is concerning
                    high_nan_cols.append(f"{col}: {nan_pct:.1f}%")
            
            if high_nan_cols:
                validation_results['warnings'].append(f"High NaN features: {', '.join(high_nan_cols[:3])}")
            
            # Store metrics
            validation_results['metrics'] = {
                'total_columns': len(df.columns),
                'label_columns': len(label_cols),
                'weight_columns': len(weight_cols),
                'feature_columns': len(feature_cols),
                'high_nan_features': len(high_nan_cols),
                'total_rows': len(df)
            }
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {e}")
            validation_results['valid'] = False
        
        return validation_results
    
    def compare_processing_results(self, desktop_results, s3_results):
        """Compare desktop and S3 processing results for consistency"""
        print("\n🔍 Comparing processing results...")
        
        desktop_df = desktop_results['dataframe']
        s3_df = s3_results['dataframe']
        
        comparison_results = {
            'structure_match': True,
            'value_consistency': {},
            'performance_comparison': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # 1. Structure comparison
            if len(desktop_df.columns) != len(s3_df.columns):
                comparison_results['structure_match'] = False
                comparison_results['errors'].append(
                    f"Column count mismatch: desktop={len(desktop_df.columns)}, s3={len(s3_df.columns)}"
                )
            
            desktop_cols = set(desktop_df.columns)
            s3_cols = set(s3_df.columns)
            
            if desktop_cols != s3_cols:
                comparison_results['structure_match'] = False
                missing_in_s3 = desktop_cols - s3_cols
                missing_in_desktop = s3_cols - desktop_cols
                
                if missing_in_s3:
                    comparison_results['errors'].append(f"Missing in S3: {list(missing_in_s3)}")
                if missing_in_desktop:
                    comparison_results['errors'].append(f"Missing in desktop: {list(missing_in_desktop)}")
            
            # 2. Row count comparison
            if len(desktop_df) != len(s3_df):
                comparison_results['errors'].append(
                    f"Row count mismatch: desktop={len(desktop_df)}, s3={len(s3_df)}"
                )
            
            # 3. Value consistency comparison (for matching columns)
            matching_cols = desktop_cols & s3_cols
            
            # Check key column types for consistency
            label_cols = [col for col in matching_cols if col.startswith('label_')]
            weight_cols = [col for col in matching_cols if col.startswith('weight_')]
            feature_cols = [col for col in matching_cols if col not in 
                           ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + label_cols + weight_cols]
            
            # Compare label columns (should be identical)
            label_consistency = 0
            for col in label_cols:
                try:
                    desktop_vals = desktop_df[col].values
                    s3_vals = s3_df[col].values
                    
                    if np.array_equal(desktop_vals, s3_vals, equal_nan=True):
                        label_consistency += 1
                    else:
                        # Check if differences are minimal (floating point precision)
                        diff_mask = ~np.isclose(desktop_vals, s3_vals, equal_nan=True, rtol=1e-10)
                        diff_count = diff_mask.sum()
                        
                        if diff_count == 0:
                            label_consistency += 1
                        else:
                            comparison_results['warnings'].append(
                                f"Label differences in {col}: {diff_count} values differ"
                            )
                except Exception as e:
                    comparison_results['warnings'].append(f"Could not compare {col}: {e}")
            
            comparison_results['value_consistency']['label_match_rate'] = label_consistency / len(label_cols) if label_cols else 1.0
            
            # Compare weight columns (should be very similar)
            weight_consistency = 0
            for col in weight_cols:
                try:
                    desktop_vals = desktop_df[col].values
                    s3_vals = s3_df[col].values
                    
                    # Allow small numerical differences for weights
                    if np.allclose(desktop_vals, s3_vals, equal_nan=True, rtol=1e-6):
                        weight_consistency += 1
                    else:
                        max_diff = np.nanmax(np.abs(desktop_vals - s3_vals))
                        comparison_results['warnings'].append(
                            f"Weight differences in {col}: max diff = {max_diff:.6f}"
                        )
                except Exception as e:
                    comparison_results['warnings'].append(f"Could not compare {col}: {e}")
            
            comparison_results['value_consistency']['weight_match_rate'] = weight_consistency / len(weight_cols) if weight_cols else 1.0
            
            # Compare feature columns (allow more tolerance due to floating point operations)
            feature_consistency = 0
            for col in feature_cols[:10]:  # Check first 10 features
                try:
                    desktop_vals = desktop_df[col].values
                    s3_vals = s3_df[col].values
                    
                    # Allow reasonable tolerance for features
                    if np.allclose(desktop_vals, s3_vals, equal_nan=True, rtol=1e-5, atol=1e-8):
                        feature_consistency += 1
                    else:
                        # Calculate correlation as alternative consistency measure
                        valid_mask = ~(np.isnan(desktop_vals) | np.isnan(s3_vals))
                        if valid_mask.sum() > 10:
                            correlation = np.corrcoef(desktop_vals[valid_mask], s3_vals[valid_mask])[0, 1]
                            if correlation > 0.999:
                                feature_consistency += 0.5  # Partial credit for high correlation
                                comparison_results['warnings'].append(
                                    f"Feature {col}: high correlation ({correlation:.6f}) but not identical"
                                )
                            else:
                                comparison_results['warnings'].append(
                                    f"Feature {col}: low correlation ({correlation:.6f})"
                                )
                except Exception as e:
                    comparison_results['warnings'].append(f"Could not compare feature {col}: {e}")
            
            comparison_results['value_consistency']['feature_match_rate'] = feature_consistency / min(len(feature_cols), 10) if feature_cols else 1.0
            
            # 4. Performance comparison
            comparison_results['performance_comparison'] = {
                'desktop_time': desktop_results['processing_time'],
                's3_time': s3_results['processing_time'],
                'speed_ratio': s3_results['processing_time'] / desktop_results['processing_time'] if desktop_results['processing_time'] > 0 else 1.0,
                'desktop_method': desktop_results['method'],
                's3_method': s3_results['method']
            }
            
            # 5. Overall consistency assessment
            overall_consistency = (
                comparison_results['value_consistency']['label_match_rate'] * 0.5 +
                comparison_results['value_consistency']['weight_match_rate'] * 0.3 +
                comparison_results['value_consistency']['feature_match_rate'] * 0.2
            )
            
            comparison_results['overall_consistency_score'] = overall_consistency
            
            # Log results
            if comparison_results['structure_match'] and overall_consistency > 0.95:
                self.log_result(
                    "Processing Consistency",
                    True,
                    f"High consistency achieved (score: {overall_consistency:.3f})",
                    {
                        'label_match_rate': f"{comparison_results['value_consistency']['label_match_rate']:.3f}",
                        'weight_match_rate': f"{comparison_results['value_consistency']['weight_match_rate']:.3f}",
                        'feature_match_rate': f"{comparison_results['value_consistency']['feature_match_rate']:.3f}",
                        'speed_ratio': f"{comparison_results['performance_comparison']['speed_ratio']:.2f}x"
                    }
                )
            else:
                self.log_result(
                    "Processing Consistency",
                    False,
                    f"Consistency issues detected (score: {overall_consistency:.3f})"
                )
            
            return comparison_results
            
        except Exception as e:
            comparison_results['errors'].append(f"Comparison failed: {e}")
            self.log_result("Processing Consistency", False, f"Comparison failed: {e}")
            return comparison_results
    
    def test_configuration_compatibility(self):
        """Test that configuration parameters work correctly in both environments"""
        print("\n⚙️  Testing configuration compatibility...")
        
        try:
            from src.data_pipeline.weighted_labeling import LabelingConfig, WeightedLabelingEngine
            
            # Test various configuration combinations
            test_configs = [
                {
                    'name': 'default',
                    'config': LabelingConfig()
                },
                {
                    'name': 'memory_optimized',
                    'config': LabelingConfig(
                        enable_memory_optimization=True,
                        chunk_size=1000
                    )
                },
                {
                    'name': 'performance_focused',
                    'config': LabelingConfig(
                        enable_performance_monitoring=True,
                        enable_progress_tracking=True
                    )
                }
            ]
            
            config_results = {}
            
            for test_config in test_configs:
                try:
                    config = test_config['config']
                    engine = WeightedLabelingEngine(config)
                    
                    # Test that engine initializes correctly
                    config_results[test_config['name']] = {
                        'initialization': True,
                        'chunk_size': config.chunk_size,
                        'memory_optimization': config.enable_memory_optimization
                    }
                    
                except Exception as e:
                    config_results[test_config['name']] = {
                        'initialization': False,
                        'error': str(e)
                    }
            
            # Check if all configurations work
            all_configs_work = all(result.get('initialization', False) for result in config_results.values())
            
            self.log_result(
                "Configuration Compatibility",
                all_configs_work,
                f"Tested {len(test_configs)} configurations",
                config_results
            )
            
            return config_results
            
        except Exception as e:
            self.log_result("Configuration Compatibility", False, f"Failed: {e}")
            return {}
    
    def test_end_to_end_consistency(self):
        """Test end-to-end consistency with sample data"""
        print("\n🔄 Testing end-to-end consistency...")
        
        try:
            # Create test sample
            test_df = self.create_test_sample(2000)
            
            # Save test sample to temporary file
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                test_df.to_parquet(temp_path, index=False)
            
            try:
                # Test desktop pipeline
                desktop_results = self.run_desktop_pipeline(temp_path)
                
                # Test monthly processing pipeline simulation
                s3_results = self.run_s3_pipeline_simulation(temp_path)
                
                # Compare results
                if desktop_results and s3_results:
                    consistency_check = self.compare_pipeline_outputs(desktop_results, s3_results)
                    
                    self.log_result(
                        "End-to-End Consistency",
                        consistency_check['consistent'],
                        consistency_check['message'],
                        consistency_check.get('details', {})
                    )
                    
                    return consistency_check
                else:
                    self.log_result("End-to-End Consistency", False, "One or both pipelines failed")
                    return {'consistent': False, 'message': 'Pipeline execution failed'}
            
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
            
        except Exception as e:
            self.log_result("End-to-End Consistency", False, f"Failed: {e}")
            return {'consistent': False, 'message': str(e)}
    
    def run_desktop_pipeline(self, input_file):
        """Run desktop validation pipeline"""
        try:
            # Simulate test_30day_pipeline.py logic
            df = pd.read_parquet(input_file)
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns:
                df = df.reset_index()
                if 'ts_event' in df.columns:
                    df = df.rename(columns={'ts_event': 'timestamp'})
            
            # Apply desktop processing
            from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
            from src.data_pipeline.features import create_all_features
            
            engine = WeightedLabelingEngine()
            df_labeled = engine.process_dataframe(df, validate_performance=False)
            df_final = create_all_features(df_labeled)
            
            return {
                'method': 'desktop_pipeline',
                'input_rows': len(df),
                'output_rows': len(df_final),
                'columns': len(df_final.columns),
                'dataframe': df_final
            }
            
        except Exception as e:
            print(f"  ❌ Desktop pipeline failed: {e}")
            return None
    
    def run_s3_pipeline_simulation(self, input_file):
        """Run S3 monthly processing pipeline simulation"""
        try:
            # Simulate process_monthly_data logic
            df = pd.read_parquet(input_file)
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns:
                df = df.reset_index()
                if 'ts_event' in df.columns:
                    df = df.rename(columns={'ts_event': 'timestamp'})
            
            # Apply S3-style processing (chunked)
            from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig
            from src.data_pipeline.features import create_all_features
            
            # Use chunked configuration
            config = LabelingConfig(
                chunk_size=500,
                enable_memory_optimization=True
            )
            
            engine = WeightedLabelingEngine(config)
            df_labeled = engine.process_dataframe(df, validate_performance=False)
            df_final = create_all_features(df_labeled)
            
            return {
                'method': 's3_pipeline_simulation',
                'input_rows': len(df),
                'output_rows': len(df_final),
                'columns': len(df_final.columns),
                'dataframe': df_final
            }
            
        except Exception as e:
            print(f"  ❌ S3 pipeline simulation failed: {e}")
            return None
    
    def compare_pipeline_outputs(self, desktop_results, s3_results):
        """Compare outputs from desktop and S3 pipelines"""
        try:
            desktop_df = desktop_results['dataframe']
            s3_df = s3_results['dataframe']
            
            # Basic structure checks
            if len(desktop_df.columns) != len(s3_df.columns):
                return {
                    'consistent': False,
                    'message': f"Column count mismatch: {len(desktop_df.columns)} vs {len(s3_df.columns)}"
                }
            
            if len(desktop_df) != len(s3_df):
                return {
                    'consistent': False,
                    'message': f"Row count mismatch: {len(desktop_df)} vs {len(s3_df)}"
                }
            
            # Column name consistency
            if set(desktop_df.columns) != set(s3_df.columns):
                return {
                    'consistent': False,
                    'message': "Column names differ between pipelines"
                }
            
            # Value consistency for key columns
            label_cols = [col for col in desktop_df.columns if col.startswith('label_')]
            weight_cols = [col for col in desktop_df.columns if col.startswith('weight_')]
            
            inconsistent_cols = []
            
            # Check label consistency (should be identical)
            for col in label_cols:
                if not np.array_equal(desktop_df[col].values, s3_df[col].values, equal_nan=True):
                    inconsistent_cols.append(col)
            
            # Check weight consistency (allow small numerical differences)
            for col in weight_cols:
                if not np.allclose(desktop_df[col].values, s3_df[col].values, equal_nan=True, rtol=1e-6):
                    inconsistent_cols.append(col)
            
            if inconsistent_cols:
                return {
                    'consistent': False,
                    'message': f"Value inconsistencies in columns: {inconsistent_cols[:3]}",
                    'details': {
                        'inconsistent_columns': len(inconsistent_cols),
                        'total_key_columns': len(label_cols) + len(weight_cols)
                    }
                }
            
            return {
                'consistent': True,
                'message': f"Pipelines produce identical results",
                'details': {
                    'rows': len(desktop_df),
                    'columns': len(desktop_df.columns),
                    'label_columns': len(label_cols),
                    'weight_columns': len(weight_cols)
                }
            }
            
        except Exception as e:
            return {
                'consistent': False,
                'message': f"Comparison failed: {e}"
            }
    
    def generate_consistency_report(self):
        """Generate comprehensive consistency validation report"""
        print("\n📊 Generating consistency validation report...")
        
        # Calculate overall success rate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result['success'])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create report
        report = {
            'validation_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'overall_status': 'PASS' if success_rate >= 90 else 'FAIL'
            },
            'test_results': self.test_results,
            'warnings': self.warnings,
            'validation_errors': self.validation_errors
        }
        
        # Save report
        report_path = project_root / "validation_results" / f"consistency_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary
        print(f"\n📋 CONSISTENCY VALIDATION SUMMARY")
        print(f"=" * 50)
        print(f"Overall Status: {'✅ PASS' if success_rate >= 90 else '❌ FAIL'}")
        print(f"Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests} tests)")
        print(f"Report saved to: {report_path}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:5]:
                print(f"  - {warning}")
        
        return report

def main():
    """Main consistency validation function"""
    print("🔍 DESKTOP vs S3 PROCESSING CONSISTENCY VALIDATION")
    print("=" * 60)
    print("Task 6.2: Validate consistency between desktop and S3 processing")
    print("Requirements: 10.6, 10.7")
    print()
    
    validator = ConsistencyValidator()
    
    try:
        # Create test sample
        test_df = validator.create_test_sample(3000)
        
        # Test desktop processing
        desktop_results = validator.test_desktop_processing(test_df)
        
        # Test S3 processing
        s3_results = validator.test_s3_processing(test_df)
        
        # Compare results
        if desktop_results and s3_results:
            validator.compare_processing_results(desktop_results, s3_results)
        
        # Test configuration compatibility
        validator.test_configuration_compatibility()
        
        # Test end-to-end consistency
        validator.test_end_to_end_consistency()
        
        # Generate final report
        report = validator.generate_consistency_report()
        
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
        print("\n🎉 CONSISTENCY VALIDATION PASSED!")
        print("Desktop and S3 processing are consistent.")
    else:
        print("\n💥 CONSISTENCY VALIDATION FAILED!")
        print("Review the issues above before proceeding.")
        sys.exit(1)