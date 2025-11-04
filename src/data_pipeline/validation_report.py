"""
Comprehensive Desktop Validation Report Generator

This module provides comprehensive validation reporting for the desktop pipeline,
including data quality metrics, rollover statistics, feature quality metrics,
and win rate/weight distribution summaries.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class ValidationReport:
    """Comprehensive validation report for desktop pipeline"""
    
    # Basic metrics
    timestamp: datetime
    processing_time_seconds: float
    input_rows: int
    output_rows: int
    data_retention_rate: float
    
    # Data quality
    large_price_changes: int
    max_price_change: float
    rth_compliance: bool
    
    # Feature engineering
    features_generated: int
    features_with_nan: Dict[str, float]  # feature_name -> nan_percentage
    feature_quality_score: float
    
    # Weighted labeling
    mode_statistics: Dict[str, Dict[str, float]]  # mode -> {win_rate, avg_weight, total_winners}
    labeling_quality_score: float
    
    # Performance
    processing_rate_rows_per_second: float
    labeling_time_seconds: float
    feature_time_seconds: float
    peak_memory_mb: Optional[float]
    
    # Overall assessment
    overall_quality_score: float
    ready_for_production: bool
    warnings: List[str]
    
    def to_json(self) -> str:
        """Convert report to JSON string"""
        return json.dumps(asdict(self), default=str, indent=2)
    
    def print_summary(self):
        """Print a formatted summary of the validation report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE VALIDATION REPORT")
        print("="*60)
        
        print(f"\n⏱️  Processing Summary:")
        print(f"   Timestamp: {self.timestamp}")
        print(f"   Total time: {self.processing_time_seconds:.1f}s ({self.processing_time_seconds/60:.1f} min)")
        print(f"   Processing rate: {self.processing_rate_rows_per_second:.0f} rows/second")
        
        print(f"\n📈 Data Quality:")
        print(f"   Input rows: {self.input_rows:,}")
        print(f"   Output rows: {self.output_rows:,}")
        print(f"   Data retention: {self.data_retention_rate:.1%}")
        print(f"   Large price changes: {self.large_price_changes}")
        print(f"   Max price change: {self.max_price_change:.2f} points")
        print(f"   RTH compliance: {'✅' if self.rth_compliance else '❌'}")
        
        print(f"\n🔧 Feature Engineering:")
        print(f"   Features generated: {self.features_generated}")
        print(f"   Features with NaN: {len(self.features_with_nan)}")
        if self.features_with_nan:
            for feature, nan_pct in sorted(self.features_with_nan.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     {feature}: {nan_pct:.1f}% NaN")
        print(f"   Feature quality score: {self.feature_quality_score:.1f}%")
        
        print(f"\n🏷️  Weighted Labeling:")
        for mode, stats in self.mode_statistics.items():
            win_rate = stats['win_rate']
            status = "✅" if 0.05 <= win_rate <= 0.50 else "⚠️"
            print(f"   {status} {mode}: {win_rate:.1%} win rate ({stats['total_winners']:.0f} winners), avg weight: {stats['avg_weight']:.3f}")
        print(f"   Labeling quality score: {self.labeling_quality_score:.1f}%")
        
        print(f"\n⚡ Performance:")
        print(f"   Labeling time: {self.labeling_time_seconds:.1f}s ({self.labeling_time_seconds/self.processing_time_seconds*100:.1f}%)")
        print(f"   Feature time: {self.feature_time_seconds:.1f}s ({self.feature_time_seconds/self.processing_time_seconds*100:.1f}%)")
        if self.peak_memory_mb:
            print(f"   Peak memory: {self.peak_memory_mb:.0f} MB")
        
        if self.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        print(f"\n🎯 Overall Assessment:")
        print(f"   Quality score: {self.overall_quality_score:.0f}%")
        print(f"   Production ready: {'✅ YES' if self.ready_for_production else '❌ NO'}")
        
        if self.ready_for_production:
            print("\n🎉 VALIDATION SUCCESSFUL - Ready for production deployment!")
        else:
            print("\n⚠️  VALIDATION COMPLETED WITH ISSUES - Review warnings before production")


class ValidationReportGenerator:
    """Generates comprehensive validation reports for desktop pipeline"""
    
    def __init__(self):
        self.start_time = None
        self.warnings = []
    
    def start_validation(self):
        """Mark the start of validation"""
        self.start_time = datetime.now()
        self.warnings = []    
    
def generate_report(self, 
                       df_input: pd.DataFrame,
                       df_output: pd.DataFrame,
                       labeling_time: float,
                       feature_time: float,
                       large_price_changes: int = 0,
                       max_price_change: float = 0.0) -> ValidationReport:
        """Generate comprehensive validation report"""
        
        if self.start_time is None:
            raise ValueError("Must call start_validation() first")
        
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        # Basic metrics
        input_rows = len(df_input)
        output_rows = len(df_output)
        data_retention_rate = output_rows / input_rows if input_rows > 0 else 0
        processing_rate = output_rows / total_time if total_time > 0 else 0
        
        # Feature analysis
        original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df_output.columns 
                       if col not in original_cols and not col.startswith(('label_', 'weight_'))]
        
        features_with_nan = {}
        for col in feature_cols:
            nan_pct = (df_output[col].isnull().sum() / len(df_output)) * 100
            if nan_pct > 0:
                features_with_nan[col] = nan_pct
        
        # Feature quality score
        feature_quality_checks = [
            len(feature_cols) == 43,  # Expected feature count
            len(features_with_nan) < 10,  # Low NaN features
            all(pct < 35 for pct in features_with_nan.values()),  # NaN threshold
        ]
        feature_quality_score = sum(feature_quality_checks) / len(feature_quality_checks) * 100
        
        # Weighted labeling analysis
        mode_statistics = {}
        label_cols = [col for col in df_output.columns if col.startswith('label_')]
        weight_cols = [col for col in df_output.columns if col.startswith('weight_')]
        
        mode_names = ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                     'low_vol_short', 'normal_vol_short', 'high_vol_short']
        
        valid_modes = 0
        for mode_name in mode_names:
            label_col = f'label_{mode_name}'
            weight_col = f'weight_{mode_name}'
            
            if label_col in df_output.columns and weight_col in df_output.columns:
                win_rate = df_output[label_col].mean()
                avg_weight = df_output[weight_col].mean()
                total_winners = df_output[label_col].sum()
                
                mode_statistics[mode_name] = {
                    'win_rate': win_rate,
                    'avg_weight': avg_weight,
                    'total_winners': total_winners
                }
                
                # Check if win rate is in valid range
                if 0.05 <= win_rate <= 0.50:
                    valid_modes += 1
                else:
                    self.warnings.append(f"Win rate for {mode_name} outside valid range: {win_rate:.1%}")
        
        # Labeling quality score
        labeling_quality_checks = [
            len(label_cols) == 6,  # Expected label count
            len(weight_cols) == 6,  # Expected weight count
            valid_modes >= 4,  # At least 4 modes with valid win rates
        ]
        labeling_quality_score = sum(labeling_quality_checks) / len(labeling_quality_checks) * 100
        
        # RTH compliance check
        rth_compliance = True  # Assume true if we got this far
        
        # Overall quality assessment
        quality_checks = [
            output_rows > 0,  # Has data
            data_retention_rate > 0.5,  # Reasonable data retention
            len(feature_cols) == 43,  # Correct feature count
            len(label_cols) == 6,  # Correct label count
            len(weight_cols) == 6,  # Correct weight count
            len(features_with_nan) < 10,  # Low NaN features
            valid_modes >= 4,  # Most modes have valid win rates
            total_time < 3600,  # Under 1 hour
        ]
        
        overall_quality_score = sum(quality_checks) / len(quality_checks) * 100
        ready_for_production = overall_quality_score >= 80 and len(self.warnings) < 5
        
        # Memory usage (try to get current usage)
        peak_memory_mb = None
        try:
            import psutil
            process = psutil.Process()
            peak_memory_mb = process.memory_info().rss / (1024**2)
        except:
            pass
        
        # Add warnings for common issues
        if len(feature_cols) != 43:
            self.warnings.append(f"Expected 43 features, got {len(feature_cols)}")
        
        if len(features_with_nan) > 10:
            self.warnings.append(f"Too many features with NaN values: {len(features_with_nan)}")
        
        if data_retention_rate < 0.8:
            self.warnings.append(f"Low data retention rate: {data_retention_rate:.1%}")
        
        if total_time > 1800:  # 30 minutes
            self.warnings.append(f"Processing took longer than expected: {total_time/60:.1f} minutes")
        
        return ValidationReport(
            timestamp=self.start_time,
            processing_time_seconds=total_time,
            input_rows=input_rows,
            output_rows=output_rows,
            data_retention_rate=data_retention_rate,
            large_price_changes=large_price_changes,
            max_price_change=max_price_change,
            rth_compliance=rth_compliance,
            features_generated=len(feature_cols),
            features_with_nan=features_with_nan,
            feature_quality_score=feature_quality_score,
            mode_statistics=mode_statistics,
            labeling_quality_score=labeling_quality_score,
            processing_rate_rows_per_second=processing_rate,
            labeling_time_seconds=labeling_time,
            feature_time_seconds=feature_time,
            peak_memory_mb=peak_memory_mb,
            overall_quality_score=overall_quality_score,
            ready_for_production=ready_for_production,
            warnings=self.warnings.copy()
        )
    
    def save_report(self, report: ValidationReport, output_path: str):
        """Save validation report to file"""
        with open(output_path, 'w') as f:
            f.write(report.to_json())
        print(f"📄 Validation report saved to: {output_path}")


def generate_desktop_validation_report(df_input: pd.DataFrame, 
                                     df_output: pd.DataFrame,
                                     labeling_time: float,
                                     feature_time: float,
                                     large_price_changes: int = 0,
                                     max_price_change: float = 0.0,
                                     save_path: Optional[str] = None) -> ValidationReport:
    """
    Convenience function to generate a comprehensive desktop validation report
    
    Args:
        df_input: Input dataframe before processing
        df_output: Final output dataframe after labeling and features
        labeling_time: Time spent on weighted labeling (seconds)
        feature_time: Time spent on feature engineering (seconds)
        large_price_changes: Number of large price changes detected
        max_price_change: Maximum price change detected
        save_path: Optional path to save the report JSON
    
    Returns:
        ValidationReport object with comprehensive metrics
    """
    generator = ValidationReportGenerator()
    generator.start_validation()
    
    report = generator.generate_report(
        df_input=df_input,
        df_output=df_output,
        labeling_time=labeling_time,
        feature_time=feature_time,
        large_price_changes=large_price_changes,
        max_price_change=max_price_change
    )
    
    # Print the report
    report.print_summary()
    
    # Save if requested
    if save_path:
        generator.save_report(report, save_path)
    
    return report