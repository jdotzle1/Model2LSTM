"""
Validation and Quality Assurance Utilities for Weighted Labeling System

This module provides utilities to validate the output of the weighted labeling system
including label distributions, weight distributions, data quality checks, and 
comparison with the original labeling system.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime

from .weighted_labeling import TRADING_MODES, TradingMode, OutputDataFrame


class LabelDistributionValidator:
    """Validates label distributions per trading mode"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_label_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Validate label distributions for each trading mode
        
        Args:
            df: DataFrame with weighted labeling results
            
        Returns:
            Dictionary with distribution statistics per mode
            
        Requirements: 10.3, 10.5
        """
        results = {}
        
        for mode_name, mode in TRADING_MODES.items():
            label_col = mode.label_column
            
            if label_col not in df.columns:
                results[mode_name] = {'error': f'Missing column {label_col}'}
                continue
            
            labels = df[label_col]
            
            # Basic distribution statistics
            total_samples = len(labels)
            winners = (labels == 1).sum()
            losers = (labels == 0).sum()
            win_rate = winners / total_samples if total_samples > 0 else 0.0
            
            # Validation checks
            valid_values = labels.isin([0, 1]).all()
            has_nan = labels.isnull().any()
            reasonable_win_rate = 0.05 <= win_rate <= 0.50  # 5-50% range per requirement 10.5
            
            results[mode_name] = {
                'total_samples': total_samples,
                'winners': int(winners),
                'losers': int(losers),
                'win_rate': win_rate,
                'win_rate_percentage': win_rate * 100,
                'valid_values': valid_values,
                'has_nan': has_nan,
                'reasonable_win_rate': reasonable_win_rate,
                'validation_passed': valid_values and not has_nan and reasonable_win_rate
            }
        
        self.validation_results['label_distributions'] = results
        return results
    
    def print_label_distribution_report(self, results: Dict[str, Dict[str, float]]) -> None:
        """Print formatted report of label distribution validation"""
        print("=" * 80)
        print("LABEL DISTRIBUTION VALIDATION REPORT")
        print("=" * 80)
        
        all_passed = True
        
        for mode_name, stats in results.items():
            if 'error' in stats:
                print(f"\n❌ {mode_name}: {stats['error']}")
                all_passed = False
                continue
            
            status = "✅" if stats['validation_passed'] else "❌"
            print(f"\n{status} {mode_name.upper()}")
            print(f"   Total samples: {stats['total_samples']:,}")
            print(f"   Winners: {stats['winners']:,} ({stats['win_rate_percentage']:.1f}%)")
            print(f"   Losers: {stats['losers']:,} ({100-stats['win_rate_percentage']:.1f}%)")
            
            # Validation details
            if not stats['valid_values']:
                print(f"   ⚠ Invalid label values found (not 0 or 1)")
                all_passed = False
            
            if stats['has_nan']:
                print(f"   ⚠ NaN values found in labels")
                all_passed = False
            
            if not stats['reasonable_win_rate']:
                print(f"   ⚠ Win rate {stats['win_rate_percentage']:.1f}% outside reasonable range (5-50%)")
                all_passed = False
        
        print("\n" + "=" * 80)
        if all_passed:
            print("✅ ALL LABEL DISTRIBUTION VALIDATIONS PASSED")
        else:
            print("❌ LABEL DISTRIBUTION VALIDATION ISSUES FOUND")
        print("=" * 80)


class WeightDistributionValidator:
    """Validates weight distributions per trading mode"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_weight_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Validate weight distributions for each trading mode
        
        Args:
            df: DataFrame with weighted labeling results
            
        Returns:
            Dictionary with weight statistics per mode
            
        Requirements: 10.2, 10.4
        """
        results = {}
        
        for mode_name, mode in TRADING_MODES.items():
            weight_col = mode.weight_column
            label_col = mode.label_column
            
            if weight_col not in df.columns:
                results[mode_name] = {'error': f'Missing column {weight_col}'}
                continue
            
            weights = df[weight_col]
            labels = df[label_col] if label_col in df.columns else None
            
            # Basic weight statistics
            total_samples = len(weights)
            mean_weight = weights.mean()
            std_weight = weights.std()
            min_weight = weights.min()
            max_weight = weights.max()
            median_weight = weights.median()
            
            # Percentile analysis
            percentiles = [5, 25, 75, 95]
            weight_percentiles = {f'p{p}': weights.quantile(p/100) for p in percentiles}
            
            # Validation checks
            all_positive = (weights > 0).all()
            has_nan = weights.isnull().any()
            has_infinite = (~np.isfinite(weights)).any()
            
            # Weight range analysis for winners vs losers
            winner_stats = {}
            loser_stats = {}
            
            if labels is not None:
                winners_mask = labels == 1
                losers_mask = labels == 0
                
                if winners_mask.any():
                    winner_weights = weights[winners_mask]
                    winner_stats = {
                        'count': len(winner_weights),
                        'mean': winner_weights.mean(),
                        'std': winner_weights.std(),
                        'min': winner_weights.min(),
                        'max': winner_weights.max(),
                        'median': winner_weights.median()
                    }
                
                if losers_mask.any():
                    loser_weights = weights[losers_mask]
                    loser_stats = {
                        'count': len(loser_weights),
                        'mean': loser_weights.mean(),
                        'std': loser_weights.std(),
                        'min': loser_weights.min(),
                        'max': loser_weights.max(),
                        'median': loser_weights.median()
                    }
            
            results[mode_name] = {
                'total_samples': total_samples,
                'mean_weight': mean_weight,
                'std_weight': std_weight,
                'min_weight': min_weight,
                'max_weight': max_weight,
                'median_weight': median_weight,
                'percentiles': weight_percentiles,
                'winner_stats': winner_stats,
                'loser_stats': loser_stats,
                'all_positive': all_positive,
                'has_nan': has_nan,
                'has_infinite': has_infinite,
                'validation_passed': all_positive and not has_nan and not has_infinite
            }
        
        self.validation_results['weight_distributions'] = results
        return results
    
    def print_weight_distribution_report(self, results: Dict[str, Dict[str, float]]) -> None:
        """Print formatted report of weight distribution validation"""
        print("=" * 80)
        print("WEIGHT DISTRIBUTION VALIDATION REPORT")
        print("=" * 80)
        
        all_passed = True
        
        for mode_name, stats in results.items():
            if 'error' in stats:
                print(f"\n❌ {mode_name}: {stats['error']}")
                all_passed = False
                continue
            
            status = "✅" if stats['validation_passed'] else "❌"
            print(f"\n{status} {mode_name.upper()}")
            print(f"   Total samples: {stats['total_samples']:,}")
            print(f"   Weight range: {stats['min_weight']:.3f} - {stats['max_weight']:.3f}")
            print(f"   Mean weight: {stats['mean_weight']:.3f} ± {stats['std_weight']:.3f}")
            print(f"   Median weight: {stats['median_weight']:.3f}")
            
            # Percentile distribution
            p = stats['percentiles']
            print(f"   Percentiles: P5={p['p5']:.3f}, P25={p['p25']:.3f}, P75={p['p75']:.3f}, P95={p['p95']:.3f}")
            
            # Winner vs loser weight comparison
            if stats['winner_stats'] and stats['loser_stats']:
                w_stats = stats['winner_stats']
                l_stats = stats['loser_stats']
                print(f"   Winners: {w_stats['count']:,} samples, mean={w_stats['mean']:.3f}")
                print(f"   Losers: {l_stats['count']:,} samples, mean={l_stats['mean']:.3f}")
            
            # Validation issues
            if not stats['all_positive']:
                print(f"   ⚠ Non-positive weights found")
                all_passed = False
            
            if stats['has_nan']:
                print(f"   ⚠ NaN values found in weights")
                all_passed = False
            
            if stats['has_infinite']:
                print(f"   ⚠ Infinite values found in weights")
                all_passed = False
        
        print("\n" + "=" * 80)
        if all_passed:
            print("✅ ALL WEIGHT DISTRIBUTION VALIDATIONS PASSED")
        else:
            print("❌ WEIGHT DISTRIBUTION VALIDATION ISSUES FOUND")
        print("=" * 80)


class DataQualityChecker:
    """Performs comprehensive data quality checks"""
    
    def __init__(self):
        self.validation_results = {}
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Perform comprehensive data quality checks for NaN/infinite values
        
        Args:
            df: DataFrame with weighted labeling results
            
        Returns:
            Dictionary with quality check results
            
        Requirements: 10.6
        """
        results = {
            'overall_stats': {},
            'column_analysis': {},
            'quality_issues': []
        }
        
        # Overall DataFrame statistics
        total_rows = len(df)
        total_cols = len(df.columns)
        
        results['overall_stats'] = {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'total_cells': total_rows * total_cols
        }
        
        # Check each column for quality issues
        for col in df.columns:
            col_data = df[col]
            col_analysis = {
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100
            }
            
            # For numeric columns, check for infinite values
            if pd.api.types.is_numeric_dtype(col_data):
                finite_mask = np.isfinite(col_data)
                col_analysis.update({
                    'infinite_count': (~finite_mask).sum(),
                    'infinite_percentage': ((~finite_mask).sum() / len(col_data)) * 100,
                    'min_value': col_data[finite_mask].min() if finite_mask.any() else np.nan,
                    'max_value': col_data[finite_mask].max() if finite_mask.any() else np.nan,
                    'mean_value': col_data[finite_mask].mean() if finite_mask.any() else np.nan
                })
                
                # Check for infinite values
                if col_analysis['infinite_count'] > 0:
                    results['quality_issues'].append({
                        'column': col,
                        'issue': 'infinite_values',
                        'count': col_analysis['infinite_count'],
                        'severity': 'high'
                    })
            
            # Check for NaN values
            if col_analysis['null_count'] > 0:
                # Determine severity based on column type and percentage
                severity = 'low'
                if col.startswith('label_') or col.startswith('weight_'):
                    severity = 'high'  # Labels and weights should not have NaN
                elif col_analysis['null_percentage'] > 10:
                    severity = 'medium'
                
                results['quality_issues'].append({
                    'column': col,
                    'issue': 'null_values',
                    'count': col_analysis['null_count'],
                    'percentage': col_analysis['null_percentage'],
                    'severity': severity
                })
            
            results['column_analysis'][col] = col_analysis
        
        # Specific checks for weighted labeling columns
        label_weight_issues = self._check_label_weight_columns(df)
        results['quality_issues'].extend(label_weight_issues)
        
        # Overall quality assessment
        high_severity_issues = [issue for issue in results['quality_issues'] if issue['severity'] == 'high']
        results['overall_stats']['quality_passed'] = len(high_severity_issues) == 0
        results['overall_stats']['high_severity_issues'] = len(high_severity_issues)
        results['overall_stats']['total_issues'] = len(results['quality_issues'])
        
        self.validation_results['data_quality'] = results
        return results
    
    def _check_label_weight_columns(self, df: pd.DataFrame) -> List[Dict]:
        """Check specific quality issues for label and weight columns"""
        issues = []
        
        for mode_name, mode in TRADING_MODES.items():
            label_col = mode.label_column
            weight_col = mode.weight_column
            
            # Check label column
            if label_col in df.columns:
                labels = df[label_col]
                
                # Labels should only be 0 or 1
                invalid_labels = ~labels.isin([0, 1])
                if invalid_labels.any():
                    issues.append({
                        'column': label_col,
                        'issue': 'invalid_label_values',
                        'count': invalid_labels.sum(),
                        'severity': 'high',
                        'details': 'Labels must be 0 or 1'
                    })
            
            # Check weight column
            if weight_col in df.columns:
                weights = df[weight_col]
                
                # Weights should be positive
                non_positive_weights = weights <= 0
                if non_positive_weights.any():
                    issues.append({
                        'column': weight_col,
                        'issue': 'non_positive_weights',
                        'count': non_positive_weights.sum(),
                        'severity': 'high',
                        'details': 'Weights must be positive'
                    })
        
        return issues
    
    def print_data_quality_report(self, results: Dict[str, Dict]) -> None:
        """Print formatted data quality report"""
        print("=" * 80)
        print("DATA QUALITY VALIDATION REPORT")
        print("=" * 80)
        
        # Overall statistics
        stats = results['overall_stats']
        print(f"\nDataset Overview:")
        print(f"  Rows: {stats['total_rows']:,}")
        print(f"  Columns: {stats['total_columns']}")
        print(f"  Total cells: {stats['total_cells']:,}")
        
        # Quality issues summary
        issues = results['quality_issues']
        high_issues = [i for i in issues if i['severity'] == 'high']
        medium_issues = [i for i in issues if i['severity'] == 'medium']
        low_issues = [i for i in issues if i['severity'] == 'low']
        
        print(f"\nQuality Issues Summary:")
        print(f"  High severity: {len(high_issues)}")
        print(f"  Medium severity: {len(medium_issues)}")
        print(f"  Low severity: {len(low_issues)}")
        print(f"  Total issues: {len(issues)}")
        
        # Detailed issue reporting
        if issues:
            print(f"\nDetailed Issues:")
            for issue in sorted(issues, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['severity']]):
                severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[issue['severity']]
                print(f"  {severity_icon} {issue['column']}: {issue['issue']}")
                print(f"     Count: {issue['count']}")
                if 'percentage' in issue:
                    print(f"     Percentage: {issue['percentage']:.2f}%")
                if 'details' in issue:
                    print(f"     Details: {issue['details']}")
        
        # Column-by-column analysis for problematic columns
        problematic_columns = [issue['column'] for issue in high_issues + medium_issues]
        if problematic_columns:
            print(f"\nProblematic Columns Analysis:")
            for col in set(problematic_columns):
                col_analysis = results['column_analysis'][col]
                print(f"  {col}:")
                print(f"    Type: {col_analysis['dtype']}")
                print(f"    Non-null: {col_analysis['non_null_count']:,} / {stats['total_rows']:,}")
                if col_analysis['null_count'] > 0:
                    print(f"    Null: {col_analysis['null_count']:,} ({col_analysis['null_percentage']:.2f}%)")
                if 'infinite_count' in col_analysis and col_analysis['infinite_count'] > 0:
                    print(f"    Infinite: {col_analysis['infinite_count']:,} ({col_analysis['infinite_percentage']:.2f}%)")
        
        print("\n" + "=" * 80)
        if stats['quality_passed']:
            print("✅ DATA QUALITY VALIDATION PASSED")
        else:
            print("❌ DATA QUALITY VALIDATION FAILED")
            print(f"   {stats['high_severity_issues']} high-severity issues found")
        print("=" * 80)


class OriginalLabelingComparator:
    """Compares weighted labeling results with original labeling system"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_with_original_labeling(self, df_weighted: pd.DataFrame, 
                                     df_original: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
        """
        Compare weighted labeling results with original labeling system
        
        Args:
            df_weighted: DataFrame with weighted labeling results
            df_original: DataFrame with original labeling results (optional)
            
        Returns:
            Dictionary with comparison results
            
        Requirements: Task 9 - comparison utility vs original labeling system
        """
        results = {
            'comparison_available': df_original is not None,
            'weighted_system_stats': {},
            'original_system_stats': {},
            'differences': {},
            'compatibility_analysis': {}
        }
        
        # Analyze weighted system
        results['weighted_system_stats'] = self._analyze_labeling_system(df_weighted, 'weighted')
        
        if df_original is not None:
            # Analyze original system
            results['original_system_stats'] = self._analyze_labeling_system(df_original, 'original')
            
            # Compare systems
            results['differences'] = self._compare_systems(df_weighted, df_original)
        else:
            # Generate original labeling for comparison if not provided
            try:
                df_original_generated = self._generate_original_labeling(df_weighted)
                results['original_system_stats'] = self._analyze_labeling_system(df_original_generated, 'original')
                results['differences'] = self._compare_systems(df_weighted, df_original_generated)
                results['comparison_available'] = True
            except Exception as e:
                results['comparison_error'] = str(e)
                warnings.warn(f"Could not generate original labeling for comparison: {e}")
        
        # Compatibility analysis
        results['compatibility_analysis'] = self._analyze_compatibility(df_weighted)
        
        self.comparison_results = results
        return results
    
    def _analyze_labeling_system(self, df: pd.DataFrame, system_type: str) -> Dict:
        """Analyze statistics for a labeling system"""
        stats = {
            'system_type': system_type,
            'total_rows': len(df),
            'columns': list(df.columns),
            'mode_stats': {}
        }
        
        if system_type == 'weighted':
            # Analyze weighted system modes
            for mode_name, mode in TRADING_MODES.items():
                if mode.label_column in df.columns and mode.weight_column in df.columns:
                    labels = df[mode.label_column]
                    weights = df[mode.weight_column]
                    
                    stats['mode_stats'][mode_name] = {
                        'win_rate': labels.mean(),
                        'total_winners': labels.sum(),
                        'avg_weight': weights.mean(),
                        'weight_std': weights.std(),
                        'weight_range': (weights.min(), weights.max())
                    }
        else:
            # Analyze original system (look for profile columns)
            profile_columns = [col for col in df.columns if 'label' in col.lower()]
            for col in profile_columns:
                if df[col].dtype in ['int64', 'float64']:
                    labels = df[col]
                    # Original system uses -1, 0, 1 (loss, suboptimal, optimal)
                    win_rate = (labels == 1).mean() if (labels.isin([-1, 0, 1])).any() else labels.mean()
                    
                    stats['mode_stats'][col] = {
                        'win_rate': win_rate,
                        'total_winners': (labels == 1).sum() if (labels.isin([-1, 0, 1])).any() else labels.sum(),
                        'unique_values': sorted(labels.unique()),
                        'value_counts': labels.value_counts().to_dict()
                    }
        
        return stats
    
    def _compare_systems(self, df_weighted: pd.DataFrame, df_original: pd.DataFrame) -> Dict:
        """Compare two labeling systems"""
        differences = {
            'row_count_match': len(df_weighted) == len(df_original),
            'mode_comparisons': {},
            'overall_differences': {}
        }
        
        # Compare mode by mode if possible
        for mode_name, mode in TRADING_MODES.items():
            weighted_label_col = mode.label_column
            
            # Try to find corresponding original column
            original_col = self._find_corresponding_original_column(mode_name, df_original.columns)
            
            if weighted_label_col in df_weighted.columns and original_col:
                weighted_labels = df_weighted[weighted_label_col]
                original_labels = df_original[original_col]
                
                # Convert original labels to binary if needed (-1,0,1 -> 0,1)
                if original_labels.isin([-1, 0, 1]).any():
                    original_binary = (original_labels == 1).astype(int)
                else:
                    original_binary = original_labels
                
                # Compare win rates
                weighted_win_rate = weighted_labels.mean()
                original_win_rate = original_binary.mean()
                
                differences['mode_comparisons'][mode_name] = {
                    'weighted_win_rate': weighted_win_rate,
                    'original_win_rate': original_win_rate,
                    'win_rate_difference': weighted_win_rate - original_win_rate,
                    'correlation': np.corrcoef(weighted_labels, original_binary)[0, 1] if len(weighted_labels) > 1 else np.nan
                }
        
        return differences
    
    def _find_corresponding_original_column(self, mode_name: str, columns: List[str]) -> Optional[str]:
        """Find corresponding column in original labeling system"""
        # Map weighted mode names to original profile names
        mode_mapping = {
            'low_vol_long': ['small', 'long'],
            'normal_vol_long': ['medium', 'long'],
            'high_vol_long': ['large', 'long'],
            'low_vol_short': ['small', 'short'],
            'normal_vol_short': ['medium', 'short'],
            'high_vol_short': ['large', 'short']
        }
        
        if mode_name in mode_mapping:
            keywords = mode_mapping[mode_name]
            
            # Look for columns containing these keywords
            for col in columns:
                col_lower = col.lower()
                if all(keyword in col_lower for keyword in keywords) and 'label' in col_lower:
                    return col
        
        return None
    
    def _generate_original_labeling(self, df_weighted: pd.DataFrame) -> pd.DataFrame:
        """Generate original labeling for comparison (if original system is available)"""
        try:
            # Try to import and use original labeling system
            from project.data_pipeline.labeling import calculate_labels_for_all_profiles
            
            # Extract base OHLCV data
            base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_base = df_weighted[base_columns].copy()
            
            # Generate original labels
            df_original = calculate_labels_for_all_profiles(df_base)
            return df_original
            
        except ImportError:
            raise ImportError("Original labeling system not available for comparison")
    
    def _analyze_compatibility(self, df_weighted: pd.DataFrame) -> Dict:
        """Analyze compatibility with XGBoost training requirements"""
        compatibility = {
            'xgboost_ready': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check for required columns
        expected_columns = []
        for mode in TRADING_MODES.values():
            expected_columns.extend([mode.label_column, mode.weight_column])
        
        missing_columns = [col for col in expected_columns if col not in df_weighted.columns]
        if missing_columns:
            compatibility['xgboost_ready'] = False
            compatibility['issues'].append(f"Missing columns: {missing_columns}")
        
        # Check data types and ranges
        for mode in TRADING_MODES.values():
            if mode.label_column in df_weighted.columns:
                labels = df_weighted[mode.label_column]
                if not labels.isin([0, 1]).all():
                    compatibility['xgboost_ready'] = False
                    compatibility['issues'].append(f"Invalid label values in {mode.label_column}")
            
            if mode.weight_column in df_weighted.columns:
                weights = df_weighted[mode.weight_column]
                if not (weights > 0).all():
                    compatibility['xgboost_ready'] = False
                    compatibility['issues'].append(f"Non-positive weights in {mode.weight_column}")
        
        # Recommendations
        if compatibility['xgboost_ready']:
            compatibility['recommendations'].append("Data is ready for XGBoost training")
        else:
            compatibility['recommendations'].append("Fix data quality issues before training")
        
        return compatibility
    
    def print_comparison_report(self, results: Dict[str, Dict]) -> None:
        """Print formatted comparison report"""
        print("=" * 80)
        print("ORIGINAL VS WEIGHTED LABELING COMPARISON REPORT")
        print("=" * 80)
        
        if not results['comparison_available']:
            print("❌ Original labeling data not available for comparison")
            if 'comparison_error' in results:
                print(f"   Error: {results['comparison_error']}")
            print("\nWeighted System Analysis Only:")
        else:
            print("✅ Comparison with original labeling system available")
        
        # Weighted system stats
        weighted_stats = results['weighted_system_stats']
        print(f"\nWeighted Labeling System:")
        print(f"  Total rows: {weighted_stats['total_rows']:,}")
        print(f"  Modes analyzed: {len(weighted_stats['mode_stats'])}")
        
        for mode_name, stats in weighted_stats['mode_stats'].items():
            print(f"    {mode_name}: {stats['win_rate']:.1%} win rate, avg weight: {stats['avg_weight']:.3f}")
        
        # Original system stats (if available)
        if results['comparison_available'] and 'original_system_stats' in results:
            original_stats = results['original_system_stats']
            print(f"\nOriginal Labeling System:")
            print(f"  Total rows: {original_stats['total_rows']:,}")
            print(f"  Profiles analyzed: {len(original_stats['mode_stats'])}")
            
            for profile_name, stats in original_stats['mode_stats'].items():
                print(f"    {profile_name}: {stats['win_rate']:.1%} win rate")
        
        # Differences analysis
        if 'differences' in results and results['differences']['mode_comparisons']:
            print(f"\nMode-by-Mode Comparison:")
            for mode_name, comparison in results['differences']['mode_comparisons'].items():
                print(f"  {mode_name}:")
                print(f"    Weighted win rate: {comparison['weighted_win_rate']:.1%}")
                print(f"    Original win rate: {comparison['original_win_rate']:.1%}")
                print(f"    Difference: {comparison['win_rate_difference']:+.1%}")
                if not np.isnan(comparison['correlation']):
                    print(f"    Correlation: {comparison['correlation']:.3f}")
        
        # Compatibility analysis
        compatibility = results['compatibility_analysis']
        print(f"\nXGBoost Compatibility Analysis:")
        status = "✅" if compatibility['xgboost_ready'] else "❌"
        print(f"  {status} XGBoost Ready: {compatibility['xgboost_ready']}")
        
        if compatibility['issues']:
            print(f"  Issues found:")
            for issue in compatibility['issues']:
                print(f"    - {issue}")
        
        if compatibility['recommendations']:
            print(f"  Recommendations:")
            for rec in compatibility['recommendations']:
                print(f"    - {rec}")
        
        print("=" * 80)


def run_comprehensive_validation(df: pd.DataFrame, 
                               df_original: Optional[pd.DataFrame] = None,
                               print_reports: bool = True) -> Dict[str, Dict]:
    """
    Run comprehensive validation suite on weighted labeling results
    
    Args:
        df: DataFrame with weighted labeling results
        df_original: Optional DataFrame with original labeling results
        print_reports: Whether to print detailed reports
        
    Returns:
        Dictionary with all validation results
        
    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
    """
    print("=" * 80)
    print("COMPREHENSIVE WEIGHTED LABELING VALIDATION SUITE")
    print("=" * 80)
    print(f"Validating dataset with {len(df):,} rows and {len(df.columns)} columns")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # 1. Label Distribution Validation
    print("\n1. Running label distribution validation...")
    label_validator = LabelDistributionValidator()
    label_results = label_validator.validate_label_distributions(df)
    all_results['label_distributions'] = label_results
    
    if print_reports:
        label_validator.print_label_distribution_report(label_results)
    
    # 2. Weight Distribution Validation
    print("\n2. Running weight distribution validation...")
    weight_validator = WeightDistributionValidator()
    weight_results = weight_validator.validate_weight_distributions(df)
    all_results['weight_distributions'] = weight_results
    
    if print_reports:
        weight_validator.print_weight_distribution_report(weight_results)
    
    # 3. Data Quality Checks
    print("\n3. Running data quality checks...")
    quality_checker = DataQualityChecker()
    quality_results = quality_checker.check_data_quality(df)
    all_results['data_quality'] = quality_results
    
    if print_reports:
        quality_checker.print_data_quality_report(quality_results)
    
    # 4. Original System Comparison
    print("\n4. Running comparison with original labeling system...")
    comparator = OriginalLabelingComparator()
    comparison_results = comparator.compare_with_original_labeling(df, df_original)
    all_results['original_comparison'] = comparison_results
    
    if print_reports:
        comparator.print_comparison_report(comparison_results)
    
    # Overall validation summary
    overall_passed = (
        all(stats['validation_passed'] for stats in label_results.values() if 'validation_passed' in stats) and
        all(stats['validation_passed'] for stats in weight_results.values() if 'validation_passed' in stats) and
        quality_results['overall_stats']['quality_passed'] and
        comparison_results['compatibility_analysis']['xgboost_ready']
    )
    
    all_results['overall_validation'] = {
        'passed': overall_passed,
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(df),
        'summary': {
            'label_validation_passed': all(stats.get('validation_passed', False) for stats in label_results.values()),
            'weight_validation_passed': all(stats.get('validation_passed', False) for stats in weight_results.values()),
            'data_quality_passed': quality_results['overall_stats']['quality_passed'],
            'xgboost_ready': comparison_results['compatibility_analysis']['xgboost_ready']
        }
    }
    
    if print_reports:
        print("\n" + "=" * 80)
        print("OVERALL VALIDATION SUMMARY")
        print("=" * 80)
        status = "✅ PASSED" if overall_passed else "❌ FAILED"
        print(f"Overall Validation: {status}")
        
        summary = all_results['overall_validation']['summary']
        print(f"  Label validation: {'✅' if summary['label_validation_passed'] else '❌'}")
        print(f"  Weight validation: {'✅' if summary['weight_validation_passed'] else '❌'}")
        print(f"  Data quality: {'✅' if summary['data_quality_passed'] else '❌'}")
        print(f"  XGBoost ready: {'✅' if summary['xgboost_ready'] else '❌'}")
        print("=" * 80)
    
    return all_results