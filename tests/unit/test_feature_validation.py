"""
Unit tests for feature validation and outlier detection

Tests the comprehensive feature validation system including:
- Feature count validation
- Range validation
- NaN percentage validation
- Outlier detection

Requirements: 5.4 - Add feature validation and outlier detection
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.data_pipeline.feature_validation import FeatureValidator, validate_features_comprehensive
from src.data_pipeline.features import get_expected_feature_names


class TestFeatureValidator:
    """Test the FeatureValidator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = FeatureValidator(nan_threshold=0.35)
        
        # Create sample data with all expected features
        self.sample_size = 1000
        np.random.seed(42)  # For reproducible tests
        
        # Create base OHLCV data
        base_price = 5000
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=self.sample_size, freq='1s'),
            'open': base_price + np.random.normal(0, 5, self.sample_size),
            'high': base_price + np.random.normal(2, 5, self.sample_size),
            'low': base_price + np.random.normal(-2, 5, self.sample_size),
            'close': base_price + np.random.normal(0, 5, self.sample_size),
            'volume': np.random.exponential(1000, self.sample_size)
        })
        
        # Add all expected features with realistic values
        self._add_sample_features()
    
    def _add_sample_features(self):
        """Add sample feature data for testing"""
        n = self.sample_size
        
        # Volume Features (4)
        self.df['volume_ratio_30s'] = np.random.lognormal(0, 0.5, n)  # Positive values
        self.df['volume_slope_30s'] = np.random.normal(0, 50, n)
        self.df['volume_slope_5s'] = np.random.normal(0, 100, n)
        self.df['volume_exhaustion'] = self.df['volume_ratio_30s'] * self.df['volume_slope_5s']
        
        # Price Context Features (5)
        self.df['vwap'] = 5000 + np.random.normal(0, 10, n)
        self.df['distance_from_vwap_pct'] = np.random.normal(0, 0.5, n)
        self.df['vwap_slope'] = np.random.normal(0, 2, n)
        self.df['distance_from_rth_high'] = -np.abs(np.random.exponential(5, n))  # Always negative
        self.df['distance_from_rth_low'] = np.abs(np.random.exponential(5, n))    # Always positive
        
        # Consolidation Features (10)
        self.df['short_range_high'] = 5000 + np.random.uniform(0, 20, n)
        self.df['short_range_low'] = 5000 - np.random.uniform(0, 20, n)
        self.df['short_range_size'] = self.df['short_range_high'] - self.df['short_range_low']
        self.df['position_in_short_range'] = np.random.uniform(0, 1, n)  # [0,1] range
        self.df['medium_range_high'] = 5000 + np.random.uniform(0, 30, n)
        self.df['medium_range_low'] = 5000 - np.random.uniform(0, 30, n)
        self.df['medium_range_size'] = self.df['medium_range_high'] - self.df['medium_range_low']
        self.df['range_compression_ratio'] = np.random.uniform(0, 1, n)  # [0,1] range
        self.df['short_range_retouches'] = np.random.choice([0, 1], n)
        self.df['medium_range_retouches'] = np.random.choice([0, 1], n)
        
        # Return Features (5)
        self.df['return_30s'] = np.random.normal(0, 0.01, n)  # Small returns
        self.df['return_60s'] = np.random.normal(0, 0.015, n)
        self.df['return_300s'] = np.random.normal(0, 0.03, n)
        self.df['momentum_acceleration'] = self.df['return_30s'] - self.df['return_60s']
        self.df['momentum_consistency'] = np.abs(np.random.normal(0, 0.005, n))
        
        # Volatility Features (6)
        self.df['atr_30s'] = np.random.exponential(2, n)  # Positive values
        self.df['atr_300s'] = np.random.exponential(3, n)
        self.df['volatility_regime'] = self.df['atr_30s'] / self.df['atr_300s']
        self.df['volatility_acceleration'] = np.random.normal(0, 0.2, n)
        self.df['volatility_breakout'] = np.random.normal(0, 1, n)  # Z-score
        self.df['atr_percentile'] = np.random.uniform(0, 100, n)
        
        # Microstructure Features (6)
        self.df['bar_range'] = np.abs(self.df['high'] - self.df['low'])
        self.df['relative_bar_size'] = self.df['bar_range'] / self.df['atr_30s']
        self.df['uptick_pct_30s'] = np.random.uniform(0, 100, n)
        self.df['uptick_pct_60s'] = np.random.uniform(0, 100, n)
        self.df['bar_flow_consistency'] = np.abs(self.df['uptick_pct_30s'] - self.df['uptick_pct_60s'])
        self.df['directional_strength'] = np.abs(self.df['uptick_pct_30s'] - 50) * 2
        
        # Time Features (7) - Binary features
        self.df['is_eth'] = 0  # Always 0 for RTH data
        self.df['is_pre_open'] = np.random.choice([0, 1], n, p=[0.8, 0.2])
        self.df['is_rth_open'] = np.random.choice([0, 1], n, p=[0.9, 0.1])
        self.df['is_morning'] = np.random.choice([0, 1], n, p=[0.7, 0.3])
        self.df['is_lunch'] = np.random.choice([0, 1], n, p=[0.8, 0.2])
        self.df['is_afternoon'] = np.random.choice([0, 1], n, p=[0.7, 0.3])
        self.df['is_rth_close'] = np.random.choice([0, 1], n, p=[0.9, 0.1])
    
    def test_feature_count_validation_all_present(self):
        """Test feature count validation when all features are present"""
        result = self.validator._validate_feature_count(self.df)
        
        assert result['validation_passed'] is True
        assert result['expected_count'] == 43
        assert result['present_count'] == 43
        assert result['missing_count'] == 0
        assert len(result['missing_features']) == 0
    
    def test_feature_count_validation_missing_features(self):
        """Test feature count validation when features are missing"""
        # Remove some features
        df_missing = self.df.drop(['volume_ratio_30s', 'vwap', 'is_morning'], axis=1)
        
        result = self.validator._validate_feature_count(df_missing)
        
        assert result['validation_passed'] is False
        assert result['expected_count'] == 43
        assert result['present_count'] == 40
        assert result['missing_count'] == 3
        assert 'volume_ratio_30s' in result['missing_features']
        assert 'vwap' in result['missing_features']
        assert 'is_morning' in result['missing_features']
    
    def test_range_validation_normal_values(self):
        """Test range validation with normal values"""
        result = self.validator._validate_feature_ranges(self.df)
        
        # Most features should pass with our realistic sample data
        passed_features = [f for f, r in result.items() if r.get('status') == 'passed']
        failed_features = [f for f, r in result.items() if r.get('status') == 'failed']
        
        assert len(passed_features) > len(failed_features)
        
        # Check specific features that should pass
        assert result['position_in_short_range']['status'] == 'passed'
        assert result['range_compression_ratio']['status'] == 'passed'
        assert result['is_eth']['status'] == 'passed'
    
    def test_range_validation_extreme_values(self):
        """Test range validation with extreme values"""
        # Create data with extreme values
        df_extreme = self.df.copy()
        df_extreme.loc[0, 'volume_ratio_30s'] = 1000  # Extremely high
        df_extreme.loc[1, 'position_in_short_range'] = 2.0  # Outside [0,1] range
        df_extreme.loc[2, 'uptick_pct_30s'] = 150  # Outside [0,100] range
        
        result = self.validator._validate_feature_ranges(df_extreme)
        
        # These features should fail validation
        assert result['volume_ratio_30s']['status'] == 'failed'
        assert result['position_in_short_range']['status'] == 'failed'
        assert result['uptick_pct_30s']['status'] == 'failed'
    
    def test_range_validation_infinite_values(self):
        """Test range validation with infinite values"""
        df_inf = self.df.copy()
        df_inf.loc[0, 'volume_ratio_30s'] = np.inf
        df_inf.loc[1, 'volatility_regime'] = -np.inf
        
        result = self.validator._validate_feature_ranges(df_inf)
        
        assert result['volume_ratio_30s']['status'] == 'failed'
        assert 'infinite' in result['volume_ratio_30s']['message']
        assert result['volatility_regime']['status'] == 'failed'
        assert 'infinite' in result['volatility_regime']['message']
    
    def test_nan_validation_low_nan(self):
        """Test NaN validation with acceptable NaN levels"""
        # Add some NaN values but keep below threshold
        df_nan = self.df.copy()
        nan_indices = np.random.choice(len(df_nan), size=int(len(df_nan) * 0.1), replace=False)
        df_nan.loc[nan_indices, 'volume_ratio_30s'] = np.nan
        
        result = self.validator._validate_nan_percentages(df_nan)
        
        # Should pass because 10% < 35% threshold for rolling features
        assert result['volume_ratio_30s']['status'] == 'passed'
        assert result['volume_ratio_30s']['nan_percentage'] == 10.0
    
    def test_nan_validation_high_nan(self):
        """Test NaN validation with excessive NaN levels"""
        # Add many NaN values above threshold
        df_nan = self.df.copy()
        nan_indices = np.random.choice(len(df_nan), size=int(len(df_nan) * 0.5), replace=False)
        df_nan.loc[nan_indices, 'volume_ratio_30s'] = np.nan
        
        result = self.validator._validate_nan_percentages(df_nan)
        
        # Should fail because 50% > 35% threshold
        assert result['volume_ratio_30s']['status'] == 'failed'
        assert result['volume_ratio_30s']['nan_percentage'] == 50.0
    
    def test_nan_validation_binary_features(self):
        """Test NaN validation for binary features (stricter threshold)"""
        # Add NaN to binary feature
        df_nan = self.df.copy()
        nan_indices = np.random.choice(len(df_nan), size=int(len(df_nan) * 0.05), replace=False)
        df_nan.loc[nan_indices, 'is_morning'] = np.nan
        
        result = self.validator._validate_nan_percentages(df_nan)
        
        # Should fail because 5% > 1% threshold for binary features
        assert result['is_morning']['status'] == 'failed'
        assert result['is_morning']['nan_percentage'] == 5.0
    
    def test_outlier_detection_normal_distribution(self):
        """Test outlier detection with normal distribution"""
        result = self.validator._detect_outliers(self.df)
        
        # Check that outlier detection ran for non-binary features
        assert 'volume_ratio_30s' in result
        assert 'method' in result['volume_ratio_30s']
        assert result['volume_ratio_30s']['method'] == 'IQR_and_zscore'
        
        # Should have some basic statistics
        assert 'Q1' in result['volume_ratio_30s']
        assert 'Q3' in result['volume_ratio_30s']
        assert 'IQR' in result['volume_ratio_30s']
    
    def test_outlier_detection_with_outliers(self):
        """Test outlier detection with actual outliers"""
        df_outliers = self.df.copy()
        
        # Add extreme outliers
        df_outliers.loc[0, 'volume_ratio_30s'] = 1000  # Extreme outlier
        df_outliers.loc[1, 'volume_ratio_30s'] = -100   # Extreme outlier
        
        result = self.validator._detect_outliers(df_outliers)
        
        # Should detect outliers
        assert result['volume_ratio_30s']['outlier_count'] >= 2
        assert result['volume_ratio_30s']['outlier_percentage'] > 0
    
    def test_outlier_detection_constant_values(self):
        """Test outlier detection with constant values"""
        df_constant = self.df.copy()
        df_constant['volume_ratio_30s'] = 1.0  # All same value
        
        result = self.validator._detect_outliers(df_constant)
        
        # Should handle constant values gracefully
        assert result['volume_ratio_30s']['method'] == 'constant_values'
        assert result['volume_ratio_30s']['outlier_count'] == 0
    
    def test_comprehensive_validation_success(self):
        """Test comprehensive validation with good data"""
        results = self.validator.validate_all_features(self.df)
        
        # Check structure
        assert 'feature_count_validation' in results
        assert 'range_validation' in results
        assert 'nan_validation' in results
        assert 'outlier_detection' in results
        assert 'overall_summary' in results
        
        # Should pass feature count validation
        assert results['feature_count_validation']['validation_passed'] is True
        
        # Should have reasonable results
        summary = results['overall_summary']
        assert 'overall_validation_passed' in summary
        assert 'feature_presence' in summary
        assert 'range_validation' in summary
        assert 'nan_validation' in summary
        assert 'outlier_detection' in summary
    
    def test_comprehensive_validation_with_issues(self):
        """Test comprehensive validation with data issues"""
        # Create problematic data
        df_bad = self.df.copy()
        
        # Remove some features
        df_bad = df_bad.drop(['volume_ratio_30s', 'vwap'], axis=1)
        
        # Add extreme values
        df_bad.loc[0, 'position_in_short_range'] = 5.0  # Outside range
        
        # Add excessive NaN
        nan_indices = np.random.choice(len(df_bad), size=int(len(df_bad) * 0.6), replace=False)
        df_bad.loc[nan_indices, 'return_30s'] = np.nan
        
        results = self.validator.validate_all_features(df_bad)
        
        # Should detect issues
        assert results['feature_count_validation']['validation_passed'] is False
        assert results['overall_summary']['overall_validation_passed'] is False


class TestValidationIntegration:
    """Test integration with existing feature engineering"""
    
    def test_validate_features_comprehensive_function(self):
        """Test the convenience function"""
        # Create minimal test data
        df = pd.DataFrame({
            'volume_ratio_30s': [1.0, 1.5, 0.8],
            'position_in_short_range': [0.2, 0.8, 0.5],
            'is_morning': [0, 1, 0],
            'return_30s': [0.001, -0.002, 0.003]
        })
        
        # Should run without error
        results = validate_features_comprehensive(df, print_report=False)
        
        assert 'overall_summary' in results
        assert 'feature_count_validation' in results
    
    def test_integration_with_features_module(self):
        """Test integration with features.py validation"""
        # Create test data with some features
        df = pd.DataFrame({
            'volume_ratio_30s': np.random.lognormal(0, 0.5, 100),
            'position_in_short_range': np.random.uniform(0, 1, 100),
            'is_morning': np.random.choice([0, 1], 100),
            'uptick_pct_30s': np.random.uniform(0, 100, 100)
        })
        
        # Import and test the enhanced validation function
        from src.data_pipeline.features import validate_feature_ranges
        
        # Should run without error and return results
        results = validate_feature_ranges(df)
        
        # If feature_validation is available, should return results
        if results is not None:
            assert 'overall_summary' in results


if __name__ == '__main__':
    # Run basic tests
    validator = FeatureValidator()
    
    # Create simple test data
    test_df = pd.DataFrame({
        'volume_ratio_30s': [1.0, 2.0, 0.5],
        'position_in_short_range': [0.2, 0.8, 0.5],
        'is_morning': [0, 1, 0]
    })
    
    print("Running basic feature validation test...")
    results = validator.validate_all_features(test_df)
    validator.print_validation_report(results)
    
    print("\nBasic test completed successfully!")