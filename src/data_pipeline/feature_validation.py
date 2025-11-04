"""
Feature Validation and Outlier Detection Module

This module provides comprehensive validation for the 43 engineered features,
including range validation, outlier detection, and NaN percentage monitoring.

Requirements: 5.4 - Add feature validation and outlier detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime


class FeatureValidator:
    """Comprehensive feature validation and outlier detection"""
    
    def __init__(self, nan_threshold: float = 0.35):
        """
        Initialize feature validator
        
        Args:
            nan_threshold: Maximum allowed NaN percentage for rolling features (default 35%)
        """
        self.nan_threshold = nan_threshold
        self.validation_results = {}
        
        # Expected feature ranges based on actual market data analysis
        self.expected_ranges = {
            # Volume Features (4)
            'volume_ratio_30s': (0.001, 30.0),
            'volume_slope_30s': (-300, 300),
            'volume_slope_5s': (-6000, 4000),
            'volume_exhaustion': (-5000, 35000),
            
            # Price Context Features (5)
            'vwap': (1000, 8000),  # ES price range over time
            'distance_from_vwap_pct': (-5.0, 5.0),
            'vwap_slope': (-12, 12),
            'distance_from_rth_high': (-200, 0.1),
            'distance_from_rth_low': (-0.1, 200),
            
            # Consolidation Features (10)
            'short_range_high': (1000, 8000),
            'short_range_low': (1000, 8000),
            'short_range_size': (0.5, 200),
            'position_in_short_range': (0.0, 1.0),
            'medium_range_high': (1000, 8000),
            'medium_range_low': (1000, 8000),
            'medium_range_size': (1.0, 200),
            'range_compression_ratio': (0.0, 1.0),
            'short_range_retouches': (0, 1),
            'medium_range_retouches': (0, 1),
            
            # Return Features (5)
            'return_30s': (-0.1, 0.1),  # 10% max return in 30 seconds
            'return_60s': (-0.1, 0.1),
            'return_300s': (-0.2, 0.2),  # 20% max return in 5 minutes
            'momentum_acceleration': (-0.2, 0.2),
            'momentum_consistency': (0.0, 0.05),
            
            # Volatility Features (6)
            'atr_30s': (0.01, 50.0),
            'atr_300s': (0.07, 50.0),
            'volatility_regime': (0.0, 5.0),
            'volatility_acceleration': (-1.0, 1.0),
            'volatility_breakout': (-5.0, 5.0),
            'atr_percentile': (0, 100),
            
            # Microstructure Features (6)
            'bar_range': (0.0, 25.0),
            'relative_bar_size': (0.0, 10.0),
            'uptick_pct_30s': (0, 100),
            'uptick_pct_60s': (0, 100),
            'bar_flow_consistency': (0, 50),
            'directional_strength': (0, 100),
            
            # Time Features (7) - Binary features
            'is_eth': (0, 1),
            'is_pre_open': (0, 1),
            'is_rth_open': (0, 1),
            'is_morning': (0, 1),
            'is_lunch': (0, 1),
            'is_afternoon': (0, 1),
            'is_rth_close': (0, 1),
        }
        
        # Features that are expected to have higher NaN percentages due to rolling calculations
        self.rolling_features = [
            'volume_ratio_30s', 'volume_slope_30s', 'volume_slope_5s',
            'vwap', 'distance_from_vwap_pct', 'vwap_slope',
            'short_range_high', 'short_range_low', 'medium_range_high', 'medium_range_low',
            'return_30s', 'return_60s', 'return_300s', 'momentum_acceleration', 'momentum_consistency',
            'atr_30s', 'atr_300s', 'volatility_regime', 'volatility_acceleration', 'volatility_breakout',
            'relative_bar_size', 'uptick_pct_30s', 'uptick_pct_60s'
        ]
    
    def validate_all_features(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Comprehensive validation of all 43 features
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary with validation results
        """
        print("=" * 80)
        print("COMPREHENSIVE FEATURE VALIDATION")
        print("=" * 80)
        print(f"Validating {len(df):,} rows with {len(df.columns)} columns")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(df),
            'feature_count_validation': {},
            'range_validation': {},
            'nan_validation': {},
            'outlier_detection': {},
            'overall_summary': {}
        }
        
        # 1. Validate feature count and presence
        print("\n1. Validating feature count and presence...")
        results['feature_count_validation'] = self._validate_feature_count(df)
        
        # 2. Validate feature ranges
        print("\n2. Validating feature value ranges...")
        results['range_validation'] = self._validate_feature_ranges(df)
        
        # 3. Validate NaN percentages
        print("\n3. Validating NaN percentages...")
        results['nan_validation'] = self._validate_nan_percentages(df)
        
        # 4. Detect outliers
        print("\n4. Detecting outliers...")
        results['outlier_detection'] = self._detect_outliers(df)
        
        # 5. Generate overall summary
        results['overall_summary'] = self._generate_summary(results)
        
        self.validation_results = results
        return results
    
    def _validate_feature_count(self, df: pd.DataFrame) -> Dict:
        """Validate that all 43 expected features are present"""
        from .features import get_expected_feature_names
        
        expected_features = get_expected_feature_names()
        present_features = [f for f in expected_features if f in df.columns]
        missing_features = [f for f in expected_features if f not in df.columns]
        
        result = {
            'expected_count': len(expected_features),
            'present_count': len(present_features),
            'missing_count': len(missing_features),
            'missing_features': missing_features,
            'validation_passed': len(missing_features) == 0
        }
        
        if result['validation_passed']:
            print(f"   ✓ All {len(expected_features)} expected features are present")
        else:
            print(f"   ❌ Missing {len(missing_features)} features: {missing_features}")
        
        return result
    
    def _validate_feature_ranges(self, df: pd.DataFrame) -> Dict:
        """Validate that feature values are within expected ranges"""
        results = {}
        passed_count = 0
        
        for feature, (min_expected, max_expected) in self.expected_ranges.items():
            if feature not in df.columns:
                results[feature] = {
                    'status': 'missing',
                    'message': 'Feature not found in dataset'
                }
                continue
            
            values = df[feature].dropna()
            if len(values) == 0:
                results[feature] = {
                    'status': 'no_data',
                    'message': 'All values are NaN'
                }
                continue
            
            min_actual = values.min()
            max_actual = values.max()
            
            # Check for infinite values
            if np.isinf(values).any():
                results[feature] = {
                    'status': 'failed',
                    'message': 'Contains infinite values',
                    'min_actual': min_actual,
                    'max_actual': max_actual,
                    'min_expected': min_expected,
                    'max_expected': max_expected
                }
                print(f"   ❌ {feature}: Contains infinite values")
                continue
            
            # Check range with tolerance for edge cases
            tolerance_factor = 0.1  # 10% tolerance
            min_tolerance = abs(min_expected) * tolerance_factor if min_expected != 0 else 0.1
            max_tolerance = abs(max_expected) * tolerance_factor if max_expected != 0 else 0.1
            
            range_ok = (min_actual >= min_expected - min_tolerance and 
                       max_actual <= max_expected + max_tolerance)
            
            results[feature] = {
                'status': 'passed' if range_ok else 'failed',
                'min_actual': float(min_actual),
                'max_actual': float(max_actual),
                'min_expected': min_expected,
                'max_expected': max_expected,
                'range_ok': range_ok
            }
            
            if range_ok:
                print(f"   ✓ {feature}: [{min_actual:.3f}, {max_actual:.3f}] within expected [{min_expected}, {max_expected}]")
                passed_count += 1
            else:
                print(f"   ❌ {feature}: [{min_actual:.3f}, {max_actual:.3f}] outside expected [{min_expected}, {max_expected}]")
        
        print(f"   Range validation: {passed_count}/{len(self.expected_ranges)} features passed")
        return results
    
    def _validate_nan_percentages(self, df: pd.DataFrame) -> Dict:
        """Validate NaN percentages, especially for rolling features"""
        results = {}
        passed_count = 0
        high_nan_features = []
        
        from .features import get_expected_feature_names
        expected_features = get_expected_feature_names()
        
        for feature in expected_features:
            if feature not in df.columns:
                results[feature] = {
                    'status': 'missing',
                    'nan_percentage': 100.0
                }
                continue
            
            nan_count = df[feature].isnull().sum()
            nan_percentage = (nan_count / len(df)) * 100
            
            # Different thresholds for different feature types
            if feature in self.rolling_features:
                # Rolling features can have higher NaN percentages
                threshold = self.nan_threshold * 100  # 35%
            elif feature.startswith('is_'):
                # Time features should have very low NaN
                threshold = 1.0  # 1%
            else:
                # Other features should have low NaN
                threshold = 10.0  # 10%
            
            passed = nan_percentage <= threshold
            
            results[feature] = {
                'status': 'passed' if passed else 'failed',
                'nan_count': int(nan_count),
                'nan_percentage': float(nan_percentage),
                'threshold': threshold,
                'is_rolling_feature': feature in self.rolling_features
            }
            
            if passed:
                passed_count += 1
                if nan_percentage > 0:
                    print(f"   ✓ {feature}: {nan_percentage:.1f}% NaN (≤{threshold:.1f}%)")
            else:
                high_nan_features.append((feature, nan_percentage))
                print(f"   ❌ {feature}: {nan_percentage:.1f}% NaN (>{threshold:.1f}%)")
        
        print(f"   NaN validation: {passed_count}/{len(expected_features)} features passed")
        if high_nan_features:
            print(f"   High NaN features: {len(high_nan_features)}")
            for feature, pct in sorted(high_nan_features, key=lambda x: x[1], reverse=True)[:5]:
                print(f"     {feature}: {pct:.1f}% NaN")
        
        return results
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using IQR method and z-score"""
        results = {}
        
        from .features import get_expected_feature_names
        expected_features = get_expected_feature_names()
        
        # Skip binary features for outlier detection
        binary_features = [f for f in expected_features if f.startswith('is_') or 
                          f in ['short_range_retouches', 'medium_range_retouches']]
        
        for feature in expected_features:
            if feature not in df.columns or feature in binary_features:
                continue
            
            values = df[feature].dropna()
            if len(values) < 10:  # Need sufficient data
                results[feature] = {
                    'method': 'insufficient_data',
                    'outlier_count': 0,
                    'outlier_percentage': 0.0
                }
                continue
            
            # IQR method for outlier detection
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # Constant values
                results[feature] = {
                    'method': 'constant_values',
                    'outlier_count': 0,
                    'outlier_percentage': 0.0
                }
                continue
            
            # Use conservative outlier bounds (3 * IQR instead of 1.5)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers_mask = (values < lower_bound) | (values > upper_bound)
            outlier_count = outliers_mask.sum()
            outlier_percentage = (outlier_count / len(values)) * 100
            
            # Z-score method for additional validation
            z_scores = np.abs((values - values.mean()) / values.std())
            extreme_outliers = (z_scores > 4).sum()  # Very extreme outliers
            
            results[feature] = {
                'method': 'IQR_and_zscore',
                'outlier_count': int(outlier_count),
                'outlier_percentage': float(outlier_percentage),
                'extreme_outliers': int(extreme_outliers),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'mean': float(values.mean()),
                'std': float(values.std())
            }
            
            if outlier_percentage > 5:  # More than 5% outliers
                print(f"   ⚠️  {feature}: {outlier_count:,} outliers ({outlier_percentage:.1f}%)")
                if extreme_outliers > 0:
                    print(f"       {extreme_outliers:,} extreme outliers (z-score > 4)")
            elif outlier_count > 0:
                print(f"   ✓ {feature}: {outlier_count:,} outliers ({outlier_percentage:.1f}%) - normal")
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate overall validation summary"""
        feature_count = results['feature_count_validation']
        range_validation = results['range_validation']
        nan_validation = results['nan_validation']
        outlier_detection = results['outlier_detection']
        
        # Count passed validations
        range_passed = sum(1 for r in range_validation.values() if r.get('status') == 'passed')
        nan_passed = sum(1 for r in nan_validation.values() if r.get('status') == 'passed')
        
        # Count high outlier features
        high_outlier_features = sum(1 for r in outlier_detection.values() 
                                  if r.get('outlier_percentage', 0) > 10)
        
        # Overall validation status
        all_features_present = feature_count['validation_passed']
        range_validation_good = range_passed >= len(range_validation) * 0.9  # 90% pass rate
        nan_validation_good = nan_passed >= len(nan_validation) * 0.9  # 90% pass rate
        outliers_acceptable = high_outlier_features <= len(outlier_detection) * 0.1  # ≤10% high outlier features
        
        overall_passed = (all_features_present and range_validation_good and 
                         nan_validation_good and outliers_acceptable)
        
        summary = {
            'overall_validation_passed': overall_passed,
            'feature_presence': {
                'passed': all_features_present,
                'present_count': feature_count['present_count'],
                'expected_count': feature_count['expected_count']
            },
            'range_validation': {
                'passed': range_validation_good,
                'passed_count': range_passed,
                'total_count': len(range_validation)
            },
            'nan_validation': {
                'passed': nan_validation_good,
                'passed_count': nan_passed,
                'total_count': len(nan_validation)
            },
            'outlier_detection': {
                'acceptable': outliers_acceptable,
                'high_outlier_features': high_outlier_features,
                'total_features_checked': len(outlier_detection)
            }
        }
        
        return summary
    
    def print_validation_report(self, results: Optional[Dict] = None) -> None:
        """Print comprehensive validation report"""
        if results is None:
            results = self.validation_results
        
        if not results:
            print("No validation results available")
            return
        
        print("\n" + "=" * 80)
        print("FEATURE VALIDATION SUMMARY REPORT")
        print("=" * 80)
        
        summary = results['overall_summary']
        
        # Overall status
        status = "✅ PASSED" if summary['overall_validation_passed'] else "❌ FAILED"
        print(f"Overall Validation: {status}")
        print(f"Dataset: {results['dataset_size']:,} rows")
        print(f"Timestamp: {results['timestamp']}")
        
        # Feature presence
        fp = summary['feature_presence']
        fp_status = "✅" if fp['passed'] else "❌"
        print(f"\n{fp_status} Feature Presence: {fp['present_count']}/{fp['expected_count']} features present")
        
        # Range validation
        rv = summary['range_validation']
        rv_status = "✅" if rv['passed'] else "❌"
        print(f"{rv_status} Range Validation: {rv['passed_count']}/{rv['total_count']} features within expected ranges")
        
        # NaN validation
        nv = summary['nan_validation']
        nv_status = "✅" if nv['passed'] else "❌"
        print(f"{nv_status} NaN Validation: {nv['passed_count']}/{nv['total_count']} features below NaN threshold")
        
        # Outlier detection
        od = summary['outlier_detection']
        od_status = "✅" if od['acceptable'] else "❌"
        print(f"{od_status} Outlier Detection: {od['high_outlier_features']}/{od['total_features_checked']} features with high outlier rates")
        
        # Recommendations
        print(f"\nRecommendations:")
        if summary['overall_validation_passed']:
            print("  - Features are ready for model training")
            print("  - All validation checks passed")
        else:
            if not fp['passed']:
                print("  - Fix missing features before proceeding")
            if not rv['passed']:
                print("  - Investigate features with values outside expected ranges")
            if not nv['passed']:
                print("  - Check features with high NaN percentages")
            if not od['acceptable']:
                print("  - Review features with excessive outliers")
        
        print("=" * 80)


def validate_features_comprehensive(df: pd.DataFrame, 
                                  nan_threshold: float = 0.35,
                                  print_report: bool = True) -> Dict:
    """
    Convenience function for comprehensive feature validation
    
    Args:
        df: DataFrame with engineered features
        nan_threshold: Maximum allowed NaN percentage for rolling features
        print_report: Whether to print detailed report
        
    Returns:
        Dictionary with validation results
    """
    validator = FeatureValidator(nan_threshold=nan_threshold)
    results = validator.validate_all_features(df)
    
    if print_report:
        validator.print_validation_report(results)
    
    return results


def check_feature_distributions(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze feature distributions for quality assessment
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Dictionary with distribution statistics
    """
    from .features import get_expected_feature_names
    
    expected_features = get_expected_feature_names()
    distributions = {}
    
    print("Analyzing feature distributions...")
    
    for feature in expected_features:
        if feature not in df.columns:
            continue
        
        values = df[feature].dropna()
        if len(values) == 0:
            continue
        
        # Basic statistics
        stats = {
            'count': len(values),
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'median': float(values.median()),
            'skewness': float(values.skew()),
            'kurtosis': float(values.kurtosis()),
            'nan_count': int(df[feature].isnull().sum()),
            'nan_percentage': float((df[feature].isnull().sum() / len(df)) * 100)
        }
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'p{p}'] = float(values.quantile(p/100))
        
        distributions[feature] = stats
    
    print(f"Analyzed distributions for {len(distributions)} features")
    return distributions