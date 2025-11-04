"""
Monthly Statistics Collection System

This module implements comprehensive statistics collection for monthly processing
as specified in requirement 3.1, 3.2, 3.4, and 3.5.

Provides detailed tracking of:
- Processing metrics (time, memory usage)
- Rollover event statistics
- Feature quality metrics
- Data quality flags
- Win rates and weight distributions
- Processing performance indicators
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class RolloverEvent:
    """Details of a contract rollover event"""
    timestamp: datetime
    price_gap: float
    bars_affected: int
    gap_direction: str  # 'up' or 'down'
    detection_method: str = "price_gap_threshold"


@dataclass
class ModeStatistics:
    """Statistics for a single trading mode"""
    win_rate: float
    total_winners: int
    total_samples: int
    loss_rate: float
    avg_weight: float
    weight_std: float
    min_weight: float
    max_weight: float
    weight_percentiles: Dict[str, float]  # p25, p50, p75, p90, p95
    
    # Data quality metrics
    nan_percentage_labels: float
    nan_percentage_weights: float
    has_nan_labels: bool
    has_nan_weights: bool
    has_infinite_labels: bool
    has_infinite_weights: bool
    
    # Validation flags
    labels_binary: bool
    weights_positive: bool
    win_rate_reasonable: bool  # 5-50% range
    validation_passed: bool
    
    # Quality flags for reprocessing recommendations
    quality_flags: List[str]


@dataclass
class FeatureStatistics:
    """Statistics for feature engineering quality"""
    features_generated: int
    expected_features: int
    feature_completeness: float  # percentage of expected features generated
    
    # NaN analysis
    nan_percentages: Dict[str, float]  # feature_name -> nan_percentage
    high_nan_features: List[str]  # features with >35% NaN
    avg_nan_percentage: float
    max_nan_percentage: float
    
    # Outlier detection
    outlier_counts: Dict[str, int]  # feature_name -> outlier_count
    features_with_outliers: List[str]
    
    # Value range validation
    value_ranges: Dict[str, Tuple[float, float]]  # feature_name -> (min, max)
    suspicious_ranges: List[str]  # features with suspicious value ranges
    
    # Overall quality score
    quality_score: float  # 0-1 score based on completeness, NaN levels, outliers


@dataclass
class DataQualityMetrics:
    """Data quality metrics throughout processing pipeline"""
    
    # Data flow statistics
    raw_bars: int
    cleaned_bars: int
    rth_bars: int
    final_bars: int
    
    # Data retention rates
    cleaning_retention_rate: float  # cleaned/raw
    rth_retention_rate: float  # rth/cleaned
    final_retention_rate: float  # final/rth
    overall_retention_rate: float  # final/raw
    
    # Data quality issues found and fixed
    price_issues_fixed: int
    negative_volume_fixed: int
    invalid_ohlc_fixed: int
    timezone_issues: int
    
    # Quality flags
    excessive_data_loss: bool  # >50% data loss in any stage
    unusual_rth_percentage: bool  # RTH percentage outside 20-60%
    data_quality_warnings: List[str]


@dataclass
class ProcessingPerformanceMetrics:
    """Processing performance and resource usage metrics"""
    
    # Timing metrics
    processing_start_time: datetime
    processing_end_time: datetime
    total_processing_time_minutes: float
    
    # Stage timing breakdown
    stage_times: Dict[str, float]  # stage_name -> time_in_seconds
    slowest_stage: str
    slowest_stage_time: float
    
    # Performance indicators
    rows_per_minute: float
    mb_per_minute: float
    processing_efficiency_score: float  # 0-1 based on target performance
    
    # Memory usage tracking
    peak_memory_mb: float
    final_memory_mb: float
    memory_reduction_mb: float
    memory_per_row_kb: float
    memory_at_stages: Dict[str, float]  # stage_name -> memory_mb
    
    # Component versions and methods used
    component_versions: Dict[str, str]
    processing_methods: Dict[str, str]  # stage -> method_used


@dataclass
class MonthlyProcessingStatistics:
    """
    Comprehensive statistics for monthly processing
    
    This is the main statistics container that includes all processing metrics,
    data quality indicators, and validation results for a single month.
    """
    
    # Basic identification
    month_str: str
    processing_date: datetime
    
    # Data quality and flow metrics
    data_quality: DataQualityMetrics
    
    # Rollover event tracking
    rollover_events: List[RolloverEvent]
    total_rollover_events: int
    bars_excluded_rollover: int
    rollover_affected_percentage: float
    
    # Mode-specific statistics (6 trading modes)
    mode_statistics: Dict[str, ModeStatistics]
    
    # Feature engineering statistics
    feature_statistics: FeatureStatistics
    
    # Processing performance metrics
    performance_metrics: ProcessingPerformanceMetrics
    
    # Overall quality assessment
    overall_quality_score: float  # 0-1 composite score
    requires_reprocessing: bool
    reprocessing_reasons: List[str]
    
    # Processing status
    processing_successful: bool
    processing_errors: List[str]
    processing_warnings: List[str]
    
    def to_json(self) -> str:
        """Serialize to JSON for S3 storage"""
        return json.dumps(asdict(self), default=str, indent=2)
    
    def save_to_file(self, file_path: Path) -> bool:
        """Save statistics to JSON file"""
        try:
            with open(file_path, 'w') as f:
                f.write(self.to_json())
            return True
        except Exception as e:
            print(f"Failed to save statistics to {file_path}: {e}")
            return False
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MonthlyProcessingStatistics':
        """Load statistics from JSON string"""
        data = json.loads(json_str)
        
        # Convert nested dictionaries back to dataclasses
        # This is a simplified version - full implementation would handle all nested types
        return cls(**data)


class QualityScorer:
    """
    Quality scoring algorithm for monthly processing results
    
    Implements comprehensive quality scoring based on multiple factors including
    win rates, data quality, feature engineering, rollover events, and processing performance.
    """
    
    def __init__(self):
        """Initialize quality scorer with default thresholds"""
        self.win_rate_min = 0.05  # 5% minimum win rate
        self.win_rate_max = 0.50  # 50% maximum win rate
        self.feature_quality_threshold = 0.6  # Minimum feature quality score
        self.data_retention_threshold = 0.3  # Minimum data retention rate
        self.rollover_threshold = 0.15  # Maximum rollover affected percentage
        self.processing_time_threshold = 45  # Maximum processing time in minutes
        self.memory_threshold = 8000  # Maximum memory usage in MB
    
    def calculate_quality_score(self, monthly_stats: 'MonthlyProcessingStatistics') -> float:
        """
        Calculate overall quality score based on multiple factors
        
        Args:
            monthly_stats: MonthlyProcessingStatistics object
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        scores = []
        weights = []
        
        # 1. Mode Statistics Quality (40% weight)
        mode_score = self._score_mode_statistics(monthly_stats.mode_statistics)
        scores.append(mode_score)
        weights.append(0.40)
        
        # 2. Data Quality Score (25% weight)
        data_score = self._score_data_quality(monthly_stats.data_quality)
        scores.append(data_score)
        weights.append(0.25)
        
        # 3. Feature Quality Score (20% weight)
        feature_score = self._score_feature_quality(monthly_stats.feature_statistics)
        scores.append(feature_score)
        weights.append(0.20)
        
        # 4. Processing Performance Score (10% weight)
        performance_score = self._score_processing_performance(monthly_stats.performance_metrics)
        scores.append(performance_score)
        weights.append(0.10)
        
        # 5. Rollover Impact Score (5% weight)
        rollover_score = self._score_rollover_impact(monthly_stats)
        scores.append(rollover_score)
        weights.append(0.05)
        
        # Calculate weighted average
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return max(0.0, min(1.0, weighted_score))  # Clamp to [0, 1]
    
    def _score_mode_statistics(self, mode_statistics: Dict[str, ModeStatistics]) -> float:
        """Score trading mode statistics quality"""
        if not mode_statistics:
            return 0.0
        
        mode_scores = []
        
        for mode_name, stats in mode_statistics.items():
            score = 1.0
            
            # Win rate validation (5-50% range)
            if not (self.win_rate_min <= stats.win_rate <= self.win_rate_max):
                score -= 0.4
            
            # Data quality checks
            if stats.has_nan_labels or stats.has_nan_weights:
                score -= 0.3
            
            if stats.has_infinite_labels or stats.has_infinite_weights:
                score -= 0.3
            
            # Validation checks
            if not stats.labels_binary:
                score -= 0.2
            
            if not stats.weights_positive:
                score -= 0.2
            
            # Quality flags penalty
            score -= len(stats.quality_flags) * 0.1
            
            mode_scores.append(max(0.0, score))
        
        return sum(mode_scores) / len(mode_scores)
    
    def _score_data_quality(self, data_quality: DataQualityMetrics) -> float:
        """Score data quality metrics"""
        score = 1.0
        
        # Data retention penalties
        if data_quality.overall_retention_rate < self.data_retention_threshold:
            score -= 0.5
        elif data_quality.overall_retention_rate < 0.5:
            score -= 0.3
        elif data_quality.overall_retention_rate < 0.7:
            score -= 0.1
        
        # Excessive data loss penalty
        if data_quality.excessive_data_loss:
            score -= 0.3
        
        # Unusual RTH percentage penalty
        if data_quality.unusual_rth_percentage:
            score -= 0.2
        
        # Data quality warnings penalty
        score -= len(data_quality.data_quality_warnings) * 0.05
        
        return max(0.0, score)
    
    def _score_feature_quality(self, feature_stats: FeatureStatistics) -> float:
        """Score feature engineering quality"""
        score = feature_stats.quality_score  # Base score from feature analysis
        
        # Feature completeness bonus/penalty
        if feature_stats.feature_completeness >= 100:
            score += 0.1
        elif feature_stats.feature_completeness < 90:
            score -= 0.2
        
        # High NaN features penalty
        if len(feature_stats.high_nan_features) > 5:
            score -= 0.2
        elif len(feature_stats.high_nan_features) > 2:
            score -= 0.1
        
        # Outlier features penalty
        if len(feature_stats.features_with_outliers) > 10:
            score -= 0.2
        elif len(feature_stats.features_with_outliers) > 5:
            score -= 0.1
        
        # Suspicious ranges penalty
        if len(feature_stats.suspicious_ranges) > 3:
            score -= 0.2
        elif len(feature_stats.suspicious_ranges) > 1:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_processing_performance(self, performance: ProcessingPerformanceMetrics) -> float:
        """Score processing performance"""
        score = 1.0
        
        # Processing time penalty
        if performance.total_processing_time_minutes > self.processing_time_threshold:
            score -= 0.4
        elif performance.total_processing_time_minutes > 30:
            score -= 0.2
        
        # Memory usage penalty
        if performance.peak_memory_mb > self.memory_threshold:
            score -= 0.3
        elif performance.peak_memory_mb > 6000:
            score -= 0.1
        
        # Processing efficiency bonus
        if performance.processing_efficiency_score > 0.8:
            score += 0.1
        elif performance.processing_efficiency_score < 0.5:
            score -= 0.2
        
        return max(0.0, score)
    
    def _score_rollover_impact(self, monthly_stats: 'MonthlyProcessingStatistics') -> float:
        """Score rollover event impact"""
        score = 1.0
        
        # Rollover affected percentage penalty
        if monthly_stats.rollover_affected_percentage > self.rollover_threshold:
            score -= 0.5
        elif monthly_stats.rollover_affected_percentage > 0.10:
            score -= 0.3
        elif monthly_stats.rollover_affected_percentage > 0.05:
            score -= 0.1
        
        # Excessive rollover events penalty
        if monthly_stats.total_rollover_events > 10:
            score -= 0.3
        elif monthly_stats.total_rollover_events > 5:
            score -= 0.1
        
        return max(0.0, score)
    
    def detect_reprocessing_requirements(self, monthly_stats: 'MonthlyProcessingStatistics') -> Tuple[bool, List[str]]:
        """
        Detect if month requires reprocessing and provide reasons
        
        Args:
            monthly_stats: MonthlyProcessingStatistics object
            
        Returns:
            Tuple of (requires_reprocessing, list_of_reasons)
        """
        reasons = []
        
        # Quality score threshold
        if monthly_stats.overall_quality_score < 0.7:
            reasons.append(f'low_overall_quality_score_{monthly_stats.overall_quality_score:.2f}')
        
        # Processing errors
        if len(monthly_stats.processing_errors) > 0:
            reasons.append('processing_errors_occurred')
        
        # Data quality issues
        if monthly_stats.data_quality.excessive_data_loss:
            reasons.append('excessive_data_loss')
        
        if monthly_stats.data_quality.overall_retention_rate < self.data_retention_threshold:
            reasons.append(f'low_data_retention_{monthly_stats.data_quality.overall_retention_rate:.2f}')
        
        # Feature quality issues
        if monthly_stats.feature_statistics.quality_score < self.feature_quality_threshold:
            reasons.append(f'poor_feature_quality_{monthly_stats.feature_statistics.quality_score:.2f}')
        
        # Mode validation issues
        invalid_modes = []
        for mode_name, mode_stats in monthly_stats.mode_statistics.items():
            if not mode_stats.validation_passed:
                invalid_modes.append(mode_name)
        
        if invalid_modes:
            reasons.append(f'invalid_modes_{",".join(invalid_modes)}')
        
        # Win rate issues
        problematic_win_rates = []
        for mode_name, mode_stats in monthly_stats.mode_statistics.items():
            if not (self.win_rate_min <= mode_stats.win_rate <= self.win_rate_max):
                problematic_win_rates.append(f'{mode_name}_{mode_stats.win_rate:.3f}')
        
        if problematic_win_rates:
            reasons.append(f'unreasonable_win_rates_{",".join(problematic_win_rates)}')
        
        # Excessive rollover events
        if monthly_stats.rollover_affected_percentage > self.rollover_threshold:
            reasons.append(f'excessive_rollover_events_{monthly_stats.rollover_affected_percentage:.2f}%')
        
        return len(reasons) > 0, reasons


class MonthlyStatisticsCollector:
    """
    Collector for comprehensive monthly statistics
    
    This class orchestrates the collection of all statistics during monthly processing
    and provides methods to analyze and validate the results.
    """
    
    def __init__(self, month_str: str):
        """
        Initialize statistics collector for a specific month
        
        Args:
            month_str: Month identifier (e.g., "2024-03")
        """
        self.month_str = month_str
        self.processing_start_time = datetime.now()
        self.stage_start_times = {}
        self.stage_times = {}
        self.memory_at_stages = {}
        self.rollover_events = []
        self.processing_errors = []
        self.processing_warnings = []
        self.component_versions = {}
        self.processing_methods = {}
        
        # Data flow tracking
        self.raw_bars = 0
        self.cleaned_bars = 0
        self.rth_bars = 0
        self.final_bars = 0
        
        # Data quality issue tracking
        self.price_issues_fixed = 0
        self.negative_volume_fixed = 0
        self.invalid_ohlc_fixed = 0
        self.timezone_issues = 0
        
        # Memory tracking
        self.peak_memory_mb = 0.0
        self.final_memory_mb = 0.0
        
        # Initialize quality scorer
        self.quality_scorer = QualityScorer()
    
    def start_stage(self, stage_name: str) -> None:
        """Start timing a processing stage"""
        self.stage_start_times[stage_name] = time.time()
    
    def end_stage(self, stage_name: str, memory_mb: float = 0.0) -> None:
        """End timing a processing stage and record memory usage"""
        if stage_name in self.stage_start_times:
            elapsed = time.time() - self.stage_start_times[stage_name]
            self.stage_times[stage_name] = elapsed
            
        if memory_mb > 0:
            self.memory_at_stages[stage_name] = memory_mb
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
    
    def record_rollover_event(self, timestamp: datetime, price_gap: float, 
                            bars_affected: int, gap_direction: str) -> None:
        """Record a rollover event"""
        event = RolloverEvent(
            timestamp=timestamp,
            price_gap=price_gap,
            bars_affected=bars_affected,
            gap_direction=gap_direction
        )
        self.rollover_events.append(event)
    
    def record_data_flow(self, stage: str, bar_count: int) -> None:
        """Record data flow at different processing stages"""
        if stage == "raw":
            self.raw_bars = bar_count
        elif stage == "cleaned":
            self.cleaned_bars = bar_count
        elif stage == "rth":
            self.rth_bars = bar_count
        elif stage == "final":
            self.final_bars = bar_count
    
    def record_data_quality_fix(self, issue_type: str, count: int) -> None:
        """Record data quality issues that were fixed"""
        if issue_type == "price_issues":
            self.price_issues_fixed = count
        elif issue_type == "negative_volume":
            self.negative_volume_fixed = count
        elif issue_type == "invalid_ohlc":
            self.invalid_ohlc_fixed = count
        elif issue_type == "timezone_issues":
            self.timezone_issues = count
    
    def record_component_version(self, component: str, version: str) -> None:
        """Record component version used in processing"""
        self.component_versions[component] = version
    
    def record_processing_method(self, stage: str, method: str) -> None:
        """Record processing method used for a stage"""
        self.processing_methods[stage] = method
    
    def add_error(self, error_msg: str) -> None:
        """Add processing error"""
        self.processing_errors.append(error_msg)
    
    def add_warning(self, warning_msg: str) -> None:
        """Add processing warning"""
        self.processing_warnings.append(warning_msg)
    
    def analyze_feature_quality(self, df: pd.DataFrame) -> FeatureStatistics:
        """
        Analyze feature engineering quality
        
        Args:
            df: DataFrame with features
            
        Returns:
            FeatureStatistics object with comprehensive feature analysis
        """
        # Identify feature columns (exclude original and labeling columns)
        original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        labeling_cols = [col for col in df.columns if col.startswith(('label_', 'weight_'))]
        feature_cols = [col for col in df.columns if col not in original_cols + labeling_cols]
        
        features_generated = len(feature_cols)
        expected_features = 43  # As per system specification
        feature_completeness = (features_generated / expected_features) * 100 if expected_features > 0 else 0
        
        # Analyze NaN percentages
        nan_percentages = {}
        high_nan_features = []
        nan_values = []
        
        for col in feature_cols:
            nan_pct = (df[col].isna().sum() / len(df)) * 100
            nan_percentages[col] = nan_pct
            nan_values.append(nan_pct)
            
            if nan_pct > 35:  # Threshold from requirement 5.4
                high_nan_features.append(col)
        
        avg_nan_percentage = np.mean(nan_values) if nan_values else 0
        max_nan_percentage = np.max(nan_values) if nan_values else 0
        
        # Analyze outliers (simplified - using IQR method)
        outlier_counts = {}
        features_with_outliers = []
        
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    outlier_counts[col] = outliers
                    
                    if outliers > len(df) * 0.05:  # More than 5% outliers
                        features_with_outliers.append(col)
        
        # Analyze value ranges
        value_ranges = {}
        suspicious_ranges = []
        
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                min_val = df[col].min()
                max_val = df[col].max()
                value_ranges[col] = (min_val, max_val)
                
                # Check for suspicious ranges (e.g., all zeros, extreme values)
                if min_val == max_val:  # No variation
                    suspicious_ranges.append(col)
                elif abs(min_val) > 1e6 or abs(max_val) > 1e6:  # Extreme values
                    suspicious_ranges.append(col)
        
        # Calculate overall quality score
        completeness_score = min(feature_completeness / 100, 1.0)
        nan_score = max(0, 1.0 - (avg_nan_percentage / 100))
        outlier_score = max(0, 1.0 - (len(features_with_outliers) / max(len(feature_cols), 1)))
        range_score = max(0, 1.0 - (len(suspicious_ranges) / max(len(feature_cols), 1)))
        
        quality_score = (completeness_score + nan_score + outlier_score + range_score) / 4
        
        return FeatureStatistics(
            features_generated=features_generated,
            expected_features=expected_features,
            feature_completeness=feature_completeness,
            nan_percentages=nan_percentages,
            high_nan_features=high_nan_features,
            avg_nan_percentage=avg_nan_percentage,
            max_nan_percentage=max_nan_percentage,
            outlier_counts=outlier_counts,
            features_with_outliers=features_with_outliers,
            value_ranges=value_ranges,
            suspicious_ranges=suspicious_ranges,
            quality_score=quality_score
        )
    
    def collect_comprehensive_statistics(self, df: pd.DataFrame) -> MonthlyProcessingStatistics:
        """
        Collect comprehensive statistics for the processed month
        
        Args:
            df: Final processed DataFrame with all columns
            
        Returns:
            MonthlyProcessingStatistics object with all collected metrics
        """
        processing_end_time = datetime.now()
        total_processing_time = (processing_end_time - self.processing_start_time).total_seconds() / 60
        
        # Create data quality metrics
        data_quality = DataQualityMetrics(
            raw_bars=self.raw_bars,
            cleaned_bars=self.cleaned_bars,
            rth_bars=self.rth_bars,
            final_bars=self.final_bars,
            cleaning_retention_rate=(self.cleaned_bars / self.raw_bars * 100) if self.raw_bars > 0 else 0,
            rth_retention_rate=(self.rth_bars / self.cleaned_bars * 100) if self.cleaned_bars > 0 else 0,
            final_retention_rate=(self.final_bars / self.rth_bars * 100) if self.rth_bars > 0 else 0,
            overall_retention_rate=(self.final_bars / self.raw_bars * 100) if self.raw_bars > 0 else 0,
            price_issues_fixed=self.price_issues_fixed,
            negative_volume_fixed=self.negative_volume_fixed,
            invalid_ohlc_fixed=self.invalid_ohlc_fixed,
            timezone_issues=self.timezone_issues,
            excessive_data_loss=((self.raw_bars - self.final_bars) / self.raw_bars > 0.5) if self.raw_bars > 0 else False,
            unusual_rth_percentage=(self.rth_bars / self.cleaned_bars < 0.2 or self.rth_bars / self.cleaned_bars > 0.6) if self.cleaned_bars > 0 else False,
            data_quality_warnings=self.processing_warnings.copy()
        )
        
        # Collect comprehensive mode statistics using enhanced OutputDataFrame.get_statistics()
        from .weighted_labeling import OutputDataFrame
        output_data = OutputDataFrame(df, ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Prepare enhanced statistics with processing metrics, rollover events, and feature quality
        processing_metrics = {
            'processing_time_minutes': total_processing_time,
            'memory_peak_mb': self.peak_memory_mb,
            'memory_final_mb': self.final_memory_mb,
            'rows_per_minute': (self.final_bars / total_processing_time) if total_processing_time > 0 else 0,
            'processing_efficiency_score': min(1.0, (self.final_bars / total_processing_time) / 10000) if total_processing_time > 0 else 0,
            'stage_times': self.stage_times.copy(),
            'slowest_stage': slowest_stage[0],
            'slowest_stage_time': slowest_stage[1]
        }
        
        # Prepare rollover events data
        rollover_events_data = []
        for event in self.rollover_events:
            rollover_events_data.append({
                'timestamp': event.timestamp,
                'price_gap': event.price_gap,
                'bars_affected': event.bars_affected,
                'gap_direction': event.gap_direction
            })
        
        # Get enhanced statistics with all metrics
        mode_stats_raw = output_data.get_statistics(
            processing_metrics=processing_metrics,
            rollover_events=rollover_events_data,
            feature_quality=feature_statistics.__dict__ if hasattr(feature_statistics, '__dict__') else None
        )
        
        # Convert to ModeStatistics objects
        mode_statistics = {}
        for mode_name, stats in mode_stats_raw.items():
            if mode_name != 'dataset_summary':  # Skip summary entry
                quality_flags = []
                
                # Add quality flags based on validation results
                if not stats.get('labels_binary', True):
                    quality_flags.append('invalid_label_values')
                if not stats.get('weights_positive', True):
                    quality_flags.append('non_positive_weights')
                if not stats.get('win_rate_reasonable', True):
                    quality_flags.append('unreasonable_win_rate')
                if stats.get('nan_percentage_labels', 0) > 0:
                    quality_flags.append('nan_in_labels')
                if stats.get('nan_percentage_weights', 0) > 0:
                    quality_flags.append('nan_in_weights')
                
                mode_statistics[mode_name] = ModeStatistics(
                    win_rate=stats.get('win_rate', 0.0),
                    total_winners=stats.get('total_winners', 0),
                    total_samples=stats.get('total_samples', 0),
                    loss_rate=stats.get('loss_rate', 0.0),
                    avg_weight=stats.get('avg_weight', 0.0),
                    weight_std=stats.get('weight_std', 0.0),
                    min_weight=stats.get('min_weight', 0.0),
                    max_weight=stats.get('max_weight', 0.0),
                    weight_percentiles=stats.get('weight_percentiles', {}),
                    nan_percentage_labels=stats.get('nan_percentage_labels', 0.0),
                    nan_percentage_weights=stats.get('nan_percentage_weights', 0.0),
                    has_nan_labels=stats.get('has_nan_labels', False),
                    has_nan_weights=stats.get('has_nan_weights', False),
                    has_infinite_labels=stats.get('has_infinite_labels', False),
                    has_infinite_weights=stats.get('has_infinite_weights', False),
                    labels_binary=stats.get('labels_binary', False),
                    weights_positive=stats.get('weights_positive', False),
                    win_rate_reasonable=stats.get('win_rate_reasonable', False),
                    validation_passed=stats.get('validation_passed', False),
                    quality_flags=quality_flags
                )
        
        # Analyze feature quality
        feature_statistics = self.analyze_feature_quality(df)
        
        # Create performance metrics
        slowest_stage = max(self.stage_times.items(), key=lambda x: x[1]) if self.stage_times else ('unknown', 0)
        
        performance_metrics = ProcessingPerformanceMetrics(
            processing_start_time=self.processing_start_time,
            processing_end_time=processing_end_time,
            total_processing_time_minutes=total_processing_time,
            stage_times=self.stage_times.copy(),
            slowest_stage=slowest_stage[0],
            slowest_stage_time=slowest_stage[1],
            rows_per_minute=(self.final_bars / total_processing_time) if total_processing_time > 0 else 0,
            mb_per_minute=0.0,  # Would need file size info
            processing_efficiency_score=min(1.0, (self.final_bars / total_processing_time) / 10000) if total_processing_time > 0 else 0,  # Target: 10K rows/min
            peak_memory_mb=self.peak_memory_mb,
            final_memory_mb=self.final_memory_mb,
            memory_reduction_mb=self.peak_memory_mb - self.final_memory_mb,
            memory_per_row_kb=(self.peak_memory_mb * 1024 / self.final_bars) if self.final_bars > 0 else 0,
            memory_at_stages=self.memory_at_stages.copy(),
            component_versions=self.component_versions.copy(),
            processing_methods=self.processing_methods.copy()
        )
        
        # Calculate rollover statistics
        total_rollover_events = len(self.rollover_events)
        bars_excluded_rollover = sum(event.bars_affected for event in self.rollover_events)
        rollover_affected_percentage = (bars_excluded_rollover / self.final_bars * 100) if self.final_bars > 0 else 0
        
        # Create preliminary statistics object for quality scoring
        preliminary_stats = MonthlyProcessingStatistics(
            month_str=self.month_str,
            processing_date=processing_end_time,
            data_quality=data_quality,
            rollover_events=self.rollover_events.copy(),
            total_rollover_events=total_rollover_events,
            bars_excluded_rollover=bars_excluded_rollover,
            rollover_affected_percentage=rollover_affected_percentage,
            mode_statistics=mode_statistics,
            feature_statistics=feature_statistics,
            performance_metrics=performance_metrics,
            overall_quality_score=0.0,  # Will be calculated below
            requires_reprocessing=False,  # Will be determined below
            reprocessing_reasons=[],  # Will be populated below
            processing_successful=len(self.processing_errors) == 0,
            processing_errors=self.processing_errors.copy(),
            processing_warnings=self.processing_warnings.copy()
        )
        
        # Calculate comprehensive quality score using the quality scorer
        overall_quality_score = self.quality_scorer.calculate_quality_score(preliminary_stats)
        
        # Determine if reprocessing is required using automated detection
        requires_reprocessing, reprocessing_reasons = self.quality_scorer.detect_reprocessing_requirements(preliminary_stats)
        
        # Update the preliminary statistics with final scores
        preliminary_stats.overall_quality_score = overall_quality_score
        preliminary_stats.requires_reprocessing = requires_reprocessing
        preliminary_stats.reprocessing_reasons = reprocessing_reasons
        
        return preliminary_stats


def validate_monthly_quality(stats: MonthlyProcessingStatistics) -> Dict[str, Any]:
    """
    Validate monthly processing quality against established thresholds
    
    Args:
        stats: MonthlyProcessingStatistics object
        
    Returns:
        Dictionary with validation results and recommendations
    """
    validation_results = {
        'overall_valid': True,
        'quality_score': stats.overall_quality_score,
        'validation_checks': {},
        'failed_checks': [],
        'recommendations': [],
        'reprocessing_required': stats.requires_reprocessing,
        'reprocessing_reasons': stats.reprocessing_reasons
    }
    
    # Win rate validation (5-50% range per mode)
    win_rate_issues = []
    for mode_name, mode_stats in stats.mode_statistics.items():
        if not (0.05 <= mode_stats.win_rate <= 0.50):
            win_rate_issues.append(f"{mode_name}: {mode_stats.win_rate:.1%}")
            validation_results['overall_valid'] = False
    
    validation_results['validation_checks']['win_rates_reasonable'] = len(win_rate_issues) == 0
    if win_rate_issues:
        validation_results['failed_checks'].append(f"Unreasonable win rates: {', '.join(win_rate_issues)}")
        validation_results['recommendations'].append("Review labeling logic and rollover detection for affected modes")
    
    # Data quality validation
    data_quality_valid = (
        stats.data_quality.overall_retention_rate >= 0.3 and
        not stats.data_quality.excessive_data_loss and
        len(stats.data_quality.data_quality_warnings) <= 5
    )
    validation_results['validation_checks']['data_quality_acceptable'] = data_quality_valid
    if not data_quality_valid:
        validation_results['failed_checks'].append("Data quality issues detected")
        validation_results['recommendations'].append("Review data cleaning and RTH filtering processes")
        validation_results['overall_valid'] = False
    
    # Feature quality validation
    feature_quality_valid = (
        stats.feature_statistics.quality_score >= 0.6 and
        len(stats.feature_statistics.high_nan_features) <= 5 and
        stats.feature_statistics.feature_completeness >= 90
    )
    validation_results['validation_checks']['feature_quality_acceptable'] = feature_quality_valid
    if not feature_quality_valid:
        validation_results['failed_checks'].append("Feature engineering quality issues")
        validation_results['recommendations'].append("Review feature engineering pipeline and NaN handling")
        validation_results['overall_valid'] = False
    
    # Rollover event validation
    rollover_acceptable = (
        stats.rollover_affected_percentage <= 15.0 and
        stats.total_rollover_events <= 10
    )
    validation_results['validation_checks']['rollover_events_acceptable'] = rollover_acceptable
    if not rollover_acceptable:
        validation_results['failed_checks'].append("Excessive rollover events detected")
        validation_results['recommendations'].append("Review rollover detection threshold and contract roll calendar")
        validation_results['overall_valid'] = False
    
    # Processing performance validation
    performance_acceptable = (
        stats.performance_metrics.total_processing_time_minutes <= 45 and
        stats.performance_metrics.peak_memory_mb <= 8000 and
        len(stats.processing_errors) == 0
    )
    validation_results['validation_checks']['processing_performance_acceptable'] = performance_acceptable
    if not performance_acceptable:
        validation_results['failed_checks'].append("Processing performance issues")
        validation_results['recommendations'].append("Optimize memory usage and processing algorithms")
        validation_results['overall_valid'] = False
    
    # Overall quality score validation
    quality_score_acceptable = stats.overall_quality_score >= 0.7
    validation_results['validation_checks']['quality_score_acceptable'] = quality_score_acceptable
    if not quality_score_acceptable:
        validation_results['failed_checks'].append(f"Low overall quality score: {stats.overall_quality_score:.2f}")
        validation_results['recommendations'].append("Address specific quality issues identified above")
        validation_results['overall_valid'] = False
    
    return validation_results


def create_monthly_quality_report(stats: MonthlyProcessingStatistics) -> str:
    """
    Create a comprehensive quality report for a processed month
    
    Args:
        stats: Monthly processing statistics
        
    Returns:
        Formatted quality report as string
    """
    report_lines = []
    
    # Header
    report_lines.append(f"# Monthly Processing Quality Report")
    report_lines.append(f"**Month:** {stats.month_str}")
    report_lines.append(f"**Processing Date:** {stats.processing_date}")
    report_lines.append(f"**Overall Quality Score:** {stats.overall_quality_score:.2f}/1.00")
    report_lines.append(f"**Requires Reprocessing:** {'Yes' if stats.requires_reprocessing else 'No'}")
    report_lines.append("")
    
    # Data Flow Summary
    report_lines.append("## Data Flow Summary")
    report_lines.append(f"- Raw bars: {stats.data_quality.raw_bars:,}")
    report_lines.append(f"- Cleaned bars: {stats.data_quality.cleaned_bars:,} ({stats.data_quality.cleaning_retention_rate:.1f}% retention)")
    report_lines.append(f"- RTH bars: {stats.data_quality.rth_bars:,} ({stats.data_quality.rth_retention_rate:.1f}% retention)")
    report_lines.append(f"- Final bars: {stats.data_quality.final_bars:,} ({stats.data_quality.final_retention_rate:.1f}% retention)")
    report_lines.append(f"- Overall retention: {stats.data_quality.overall_retention_rate:.1f}%")
    report_lines.append("")
    
    # Performance Summary
    report_lines.append("## Performance Summary")
    report_lines.append(f"- Total processing time: {stats.performance_metrics.total_processing_time_minutes:.1f} minutes")
    report_lines.append(f"- Processing rate: {stats.performance_metrics.rows_per_minute:.0f} rows/minute")
    report_lines.append(f"- Peak memory usage: {stats.performance_metrics.peak_memory_mb:.1f} MB")
    report_lines.append(f"- Memory per row: {stats.performance_metrics.memory_per_row_kb:.2f} KB")
    report_lines.append(f"- Slowest stage: {stats.performance_metrics.slowest_stage} ({stats.performance_metrics.slowest_stage_time:.1f}s)")
    report_lines.append("")
    
    # Mode Statistics Summary
    report_lines.append("## Trading Mode Statistics")
    for mode_name, mode_stats in stats.mode_statistics.items():
        status = "✅" if mode_stats.validation_passed else "❌"
        report_lines.append(f"- **{mode_name}** {status}: {mode_stats.win_rate:.1%} win rate, avg weight: {mode_stats.avg_weight:.3f}")
        if mode_stats.quality_flags:
            report_lines.append(f"  - Quality issues: {', '.join(mode_stats.quality_flags)}")
    report_lines.append("")
    
    # Feature Quality Summary
    report_lines.append("## Feature Quality Summary")
    report_lines.append(f"- Features generated: {stats.feature_statistics.features_generated}/{stats.feature_statistics.expected_features}")
    report_lines.append(f"- Feature completeness: {stats.feature_statistics.feature_completeness:.1f}%")
    report_lines.append(f"- Average NaN percentage: {stats.feature_statistics.avg_nan_percentage:.1f}%")
    report_lines.append(f"- High NaN features: {len(stats.feature_statistics.high_nan_features)}")
    report_lines.append(f"- Features with outliers: {len(stats.feature_statistics.features_with_outliers)}")
    report_lines.append(f"- Feature quality score: {stats.feature_statistics.quality_score:.2f}/1.00")
    report_lines.append("")
    
    # Rollover Events
    report_lines.append("## Rollover Events")
    report_lines.append(f"- Total rollover events: {stats.total_rollover_events}")
    report_lines.append(f"- Bars excluded: {stats.bars_excluded_rollover:,} ({stats.rollover_affected_percentage:.2f}%)")
    if stats.rollover_events:
        report_lines.append("- Recent events:")
        for event in stats.rollover_events[-3:]:  # Show last 3 events
            report_lines.append(f"  - {event.timestamp}: {event.price_gap:.1f} point gap ({event.gap_direction}), {event.bars_affected} bars affected")
    report_lines.append("")
    
    # Quality Validation Results
    validation_results = validate_monthly_quality(stats)
    
    if validation_results['overall_valid']:
        report_lines.append("## ✅ Quality Validation")
        report_lines.append("All quality checks passed successfully.")
    else:
        report_lines.append("## ⚠️ Quality Issues Detected")
        for failed_check in validation_results['failed_checks']:
            report_lines.append(f"- {failed_check}")
        
        if validation_results['recommendations']:
            report_lines.append("")
            report_lines.append("**Recommendations:**")
            for recommendation in validation_results['recommendations']:
                report_lines.append(f"- {recommendation}")
    
    report_lines.append("")
    
    # Issues and Recommendations
    if stats.requires_reprocessing:
        report_lines.append("## ⚠️ Reprocessing Required")
        report_lines.append("**Reasons:**")
        for reason in stats.reprocessing_reasons:
            report_lines.append(f"- {reason.replace('_', ' ').title()}")
        report_lines.append("")
    
    if stats.processing_warnings:
        report_lines.append("## Warnings")
        for warning in stats.processing_warnings[:5]:  # Show first 5 warnings
            report_lines.append(f"- {warning}")
        if len(stats.processing_warnings) > 5:
            report_lines.append(f"- ... and {len(stats.processing_warnings) - 5} more warnings")
        report_lines.append("")
    
    if stats.processing_errors:
        report_lines.append("## ❌ Errors")
        for error in stats.processing_errors:
            report_lines.append(f"- {error}")
        report_lines.append("")
    
    return "\n".join(report_lines)


class MultiMonthReportGenerator:
    """
    Generator for comprehensive reports across multiple months
    
    Provides trend analysis, summary statistics, and reprocessing recommendations
    across multiple processed months.
    """
    
    def __init__(self):
        """Initialize multi-month report generator"""
        self.monthly_stats = []
        self.quality_scorer = QualityScorer()
    
    def add_monthly_stats(self, stats: MonthlyProcessingStatistics) -> None:
        """Add monthly statistics to the collection"""
        self.monthly_stats.append(stats)
    
    def generate_summary_report(self) -> str:
        """
        Generate comprehensive summary report across all months
        
        Returns:
            Formatted summary report as string
        """
        if not self.monthly_stats:
            return "No monthly statistics available for summary report."
        
        report_lines = []
        
        # Header
        report_lines.append("# Multi-Month Processing Summary Report")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Months Analyzed:** {len(self.monthly_stats)}")
        
        # Date range
        months = [stats.month_str for stats in self.monthly_stats]
        report_lines.append(f"**Date Range:** {min(months)} to {max(months)}")
        report_lines.append("")
        
        # Overall Summary
        report_lines.append("## Overall Summary")
        
        successful_months = sum(1 for stats in self.monthly_stats if stats.processing_successful)
        requires_reprocessing = sum(1 for stats in self.monthly_stats if stats.requires_reprocessing)
        avg_quality_score = sum(stats.overall_quality_score for stats in self.monthly_stats) / len(self.monthly_stats)
        
        report_lines.append(f"- **Successful Processing:** {successful_months}/{len(self.monthly_stats)} months ({successful_months/len(self.monthly_stats)*100:.1f}%)")
        report_lines.append(f"- **Requires Reprocessing:** {requires_reprocessing} months ({requires_reprocessing/len(self.monthly_stats)*100:.1f}%)")
        report_lines.append(f"- **Average Quality Score:** {avg_quality_score:.3f}")
        report_lines.append("")
        
        # Quality Score Distribution
        report_lines.append("## Quality Score Distribution")
        
        high_quality = sum(1 for stats in self.monthly_stats if stats.overall_quality_score >= 0.8)
        medium_quality = sum(1 for stats in self.monthly_stats if 0.6 <= stats.overall_quality_score < 0.8)
        low_quality = sum(1 for stats in self.monthly_stats if stats.overall_quality_score < 0.6)
        
        report_lines.append(f"- **High Quality (≥0.8):** {high_quality} months ({high_quality/len(self.monthly_stats)*100:.1f}%)")
        report_lines.append(f"- **Medium Quality (0.6-0.8):** {medium_quality} months ({medium_quality/len(self.monthly_stats)*100:.1f}%)")
        report_lines.append(f"- **Low Quality (<0.6):** {low_quality} months ({low_quality/len(self.monthly_stats)*100:.1f}%)")
        report_lines.append("")
        
        # Performance Trends
        report_lines.append("## Performance Trends")
        
        avg_processing_time = sum(stats.performance_metrics.total_processing_time_minutes for stats in self.monthly_stats) / len(self.monthly_stats)
        avg_memory_usage = sum(stats.performance_metrics.peak_memory_mb for stats in self.monthly_stats) / len(self.monthly_stats)
        avg_data_retention = sum(stats.data_quality.overall_retention_rate for stats in self.monthly_stats) / len(self.monthly_stats)
        
        report_lines.append(f"- **Average Processing Time:** {avg_processing_time:.1f} minutes")
        report_lines.append(f"- **Average Peak Memory:** {avg_memory_usage:.1f} MB")
        report_lines.append(f"- **Average Data Retention:** {avg_data_retention:.1f}%")
        report_lines.append("")
        
        # Win Rate Analysis
        report_lines.append("## Win Rate Analysis")
        
        # Calculate average win rates per mode
        mode_win_rates = {}
        for stats in self.monthly_stats:
            for mode_name, mode_stats in stats.mode_statistics.items():
                if mode_name not in mode_win_rates:
                    mode_win_rates[mode_name] = []
                mode_win_rates[mode_name].append(mode_stats.win_rate)
        
        for mode_name, win_rates in mode_win_rates.items():
            avg_win_rate = sum(win_rates) / len(win_rates)
            min_win_rate = min(win_rates)
            max_win_rate = max(win_rates)
            report_lines.append(f"- **{mode_name}:** {avg_win_rate:.1%} avg (range: {min_win_rate:.1%} - {max_win_rate:.1%})")
        
        report_lines.append("")
        
        # Feature Quality Trends
        report_lines.append("## Feature Quality Trends")
        
        avg_feature_quality = sum(stats.feature_statistics.quality_score for stats in self.monthly_stats) / len(self.monthly_stats)
        avg_feature_completeness = sum(stats.feature_statistics.feature_completeness for stats in self.monthly_stats) / len(self.monthly_stats)
        avg_nan_percentage = sum(stats.feature_statistics.avg_nan_percentage for stats in self.monthly_stats) / len(self.monthly_stats)
        
        report_lines.append(f"- **Average Feature Quality Score:** {avg_feature_quality:.3f}")
        report_lines.append(f"- **Average Feature Completeness:** {avg_feature_completeness:.1f}%")
        report_lines.append(f"- **Average NaN Percentage:** {avg_nan_percentage:.1f}%")
        report_lines.append("")
        
        # Rollover Impact Analysis
        report_lines.append("## Rollover Impact Analysis")
        
        total_rollover_events = sum(stats.total_rollover_events for stats in self.monthly_stats)
        avg_rollover_percentage = sum(stats.rollover_affected_percentage for stats in self.monthly_stats) / len(self.monthly_stats)
        months_with_excessive_rollovers = sum(1 for stats in self.monthly_stats if stats.rollover_affected_percentage > 10.0)
        
        report_lines.append(f"- **Total Rollover Events:** {total_rollover_events}")
        report_lines.append(f"- **Average Rollover Impact:** {avg_rollover_percentage:.2f}% of bars affected")
        report_lines.append(f"- **Months with Excessive Rollovers (>10%):** {months_with_excessive_rollovers}")
        report_lines.append("")
        
        # Reprocessing Recommendations
        if requires_reprocessing > 0:
            report_lines.append("## 🔄 Reprocessing Recommendations")
            
            # Group by reprocessing priority
            high_priority = [stats for stats in self.monthly_stats if stats.requires_reprocessing and stats.overall_quality_score < 0.5]
            medium_priority = [stats for stats in self.monthly_stats if stats.requires_reprocessing and 0.5 <= stats.overall_quality_score < 0.7]
            low_priority = [stats for stats in self.monthly_stats if stats.requires_reprocessing and stats.overall_quality_score >= 0.7]
            
            if high_priority:
                report_lines.append("### High Priority (Quality Score < 0.5)")
                for stats in high_priority:
                    report_lines.append(f"- **{stats.month_str}** (Score: {stats.overall_quality_score:.2f}): {', '.join(stats.reprocessing_reasons[:3])}")
                report_lines.append("")
            
            if medium_priority:
                report_lines.append("### Medium Priority (Quality Score 0.5-0.7)")
                for stats in medium_priority:
                    report_lines.append(f"- **{stats.month_str}** (Score: {stats.overall_quality_score:.2f}): {', '.join(stats.reprocessing_reasons[:2])}")
                report_lines.append("")
            
            if low_priority:
                report_lines.append("### Low Priority (Quality Score ≥ 0.7)")
                for stats in low_priority:
                    report_lines.append(f"- **{stats.month_str}** (Score: {stats.overall_quality_score:.2f}): {stats.reprocessing_reasons[0] if stats.reprocessing_reasons else 'Minor issues'}")
                report_lines.append("")
        
        # Data Quality Issues Summary
        report_lines.append("## Data Quality Issues Summary")
        
        months_with_data_loss = sum(1 for stats in self.monthly_stats if stats.data_quality.excessive_data_loss)
        months_with_rth_issues = sum(1 for stats in self.monthly_stats if stats.data_quality.unusual_rth_percentage)
        total_warnings = sum(len(stats.processing_warnings) for stats in self.monthly_stats)
        total_errors = sum(len(stats.processing_errors) for stats in self.monthly_stats)
        
        report_lines.append(f"- **Months with Excessive Data Loss:** {months_with_data_loss}")
        report_lines.append(f"- **Months with RTH Issues:** {months_with_rth_issues}")
        report_lines.append(f"- **Total Processing Warnings:** {total_warnings}")
        report_lines.append(f"- **Total Processing Errors:** {total_errors}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## 📋 Overall Recommendations")
        
        recommendations = self._generate_overall_recommendations()
        for recommendation in recommendations:
            report_lines.append(f"- {recommendation}")
        
        return "\n".join(report_lines)
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations based on trends across all months"""
        recommendations = []
        
        if not self.monthly_stats:
            return recommendations
        
        # Quality score trend analysis
        avg_quality = sum(stats.overall_quality_score for stats in self.monthly_stats) / len(self.monthly_stats)
        if avg_quality < 0.7:
            recommendations.append("Overall quality scores are below target (0.7). Review and optimize the entire processing pipeline.")
        
        # Processing time analysis
        avg_processing_time = sum(stats.performance_metrics.total_processing_time_minutes for stats in self.monthly_stats) / len(self.monthly_stats)
        if avg_processing_time > 30:
            recommendations.append(f"Average processing time ({avg_processing_time:.1f} min) exceeds target. Consider performance optimizations.")
        
        # Memory usage analysis
        avg_memory = sum(stats.performance_metrics.peak_memory_mb for stats in self.monthly_stats) / len(self.monthly_stats)
        if avg_memory > 6000:
            recommendations.append(f"Average memory usage ({avg_memory:.0f} MB) is high. Implement memory optimization strategies.")
        
        # Data retention analysis
        avg_retention = sum(stats.data_quality.overall_retention_rate for stats in self.monthly_stats) / len(self.monthly_stats)
        if avg_retention < 50:
            recommendations.append(f"Data retention rate ({avg_retention:.1f}%) is low. Review data cleaning and filtering processes.")
        
        # Feature quality analysis
        avg_feature_quality = sum(stats.feature_statistics.quality_score for stats in self.monthly_stats) / len(self.monthly_stats)
        if avg_feature_quality < 0.7:
            recommendations.append("Feature quality scores are below target. Review feature engineering pipeline.")
        
        # Win rate consistency analysis
        mode_win_rate_issues = []
        mode_win_rates = {}
        for stats in self.monthly_stats:
            for mode_name, mode_stats in stats.mode_statistics.items():
                if mode_name not in mode_win_rates:
                    mode_win_rates[mode_name] = []
                mode_win_rates[mode_name].append(mode_stats.win_rate)
        
        for mode_name, win_rates in mode_win_rates.items():
            avg_win_rate = sum(win_rates) / len(win_rates)
            if not (0.05 <= avg_win_rate <= 0.50):
                mode_win_rate_issues.append(mode_name)
        
        if mode_win_rate_issues:
            recommendations.append(f"Win rates for {', '.join(mode_win_rate_issues)} are outside acceptable range (5-50%). Review labeling logic.")
        
        # Rollover impact analysis
        avg_rollover_impact = sum(stats.rollover_affected_percentage for stats in self.monthly_stats) / len(self.monthly_stats)
        if avg_rollover_impact > 10:
            recommendations.append(f"Rollover impact ({avg_rollover_impact:.1f}%) is high. Review rollover detection parameters.")
        
        # Reprocessing frequency analysis
        reprocessing_rate = sum(1 for stats in self.monthly_stats if stats.requires_reprocessing) / len(self.monthly_stats)
        if reprocessing_rate > 0.2:  # More than 20% need reprocessing
            recommendations.append(f"High reprocessing rate ({reprocessing_rate:.1%}). Focus on improving initial processing quality.")
        
        return recommendations
    
    def generate_trend_analysis(self) -> Dict[str, Any]:
        """
        Generate detailed trend analysis across months
        
        Returns:
            Dictionary with trend analysis data
        """
        if len(self.monthly_stats) < 2:
            return {"error": "Need at least 2 months for trend analysis"}
        
        # Sort by month for trend analysis
        sorted_stats = sorted(self.monthly_stats, key=lambda x: x.month_str)
        
        trends = {
            'quality_scores': [stats.overall_quality_score for stats in sorted_stats],
            'processing_times': [stats.performance_metrics.total_processing_time_minutes for stats in sorted_stats],
            'memory_usage': [stats.performance_metrics.peak_memory_mb for stats in sorted_stats],
            'data_retention': [stats.data_quality.overall_retention_rate for stats in sorted_stats],
            'feature_quality': [stats.feature_statistics.quality_score for stats in sorted_stats],
            'rollover_impact': [stats.rollover_affected_percentage for stats in sorted_stats],
            'months': [stats.month_str for stats in sorted_stats]
        }
        
        # Calculate trend directions (improving/declining)
        trend_analysis = {}
        for metric, values in trends.items():
            if metric == 'months':
                continue
            
            if len(values) >= 3:
                # Simple trend calculation: compare first third vs last third
                first_third = sum(values[:len(values)//3]) / (len(values)//3)
                last_third = sum(values[-len(values)//3:]) / (len(values)//3)
                
                if last_third > first_third * 1.05:  # 5% improvement threshold
                    trend_analysis[metric] = 'improving'
                elif last_third < first_third * 0.95:  # 5% decline threshold
                    trend_analysis[metric] = 'declining'
                else:
                    trend_analysis[metric] = 'stable'
            else:
                trend_analysis[metric] = 'insufficient_data'
        
        return {
            'trends': trends,
            'trend_analysis': trend_analysis,
            'summary': {
                'total_months': len(sorted_stats),
                'date_range': f"{sorted_stats[0].month_str} to {sorted_stats[-1].month_str}",
                'improving_metrics': [k for k, v in trend_analysis.items() if v == 'improving'],
                'declining_metrics': [k for k, v in trend_analysis.items() if v == 'declining']
            }
        }


def save_monthly_report(stats: MonthlyProcessingStatistics, output_dir: str = "reports") -> str:
    """
    Save monthly quality report to file
    
    Args:
        stats: MonthlyProcessingStatistics object
        output_dir: Directory to save report
        
    Returns:
        Path to saved report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_content = create_monthly_quality_report(stats)
    report_file = output_path / f"monthly_report_{stats.month_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return str(report_file)


def save_summary_report(monthly_stats_list: List[MonthlyProcessingStatistics], output_dir: str = "reports") -> str:
    """
    Save multi-month summary report to file
    
    Args:
        monthly_stats_list: List of MonthlyProcessingStatistics objects
        output_dir: Directory to save report
        
    Returns:
        Path to saved report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_generator = MultiMonthReportGenerator()
    for stats in monthly_stats_list:
        report_generator.add_monthly_stats(stats)
    
    report_content = report_generator.generate_summary_report()
    report_file = output_path / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return str(report_file)