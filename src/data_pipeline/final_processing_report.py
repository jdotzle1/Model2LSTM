"""
Final Processing Report Generation System

This module implements comprehensive final processing report generation as specified in task 5.3.
It provides detailed analysis of successful vs failed months, data quality summaries across all
processed months, and recommendations for any required reprocessing.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from .monthly_statistics import MonthlyProcessingStatistics, MultiMonthReportGenerator


@dataclass
class ProcessingSummary:
    """Summary of overall processing results"""
    total_months_attempted: int
    successful_months: int
    failed_months: int
    success_rate: float
    total_processing_time_hours: float
    avg_processing_time_minutes: float
    date_range_start: str
    date_range_end: str
    processing_start_time: datetime
    processing_end_time: datetime


@dataclass
class DataQualitySummary:
    """Summary of data quality across all processed months"""
    avg_data_retention_rate: float
    avg_win_rate_consistency: float
    avg_feature_quality_score: float
    months_with_quality_issues: int
    total_rollover_events: int
    avg_rollover_impact_percentage: float
    months_requiring_reprocessing: int


@dataclass
class ReprocessingRecommendation:
    """Recommendation for reprocessing specific months"""
    month_str: str
    priority: str  # 'high', 'medium', 'low'
    quality_score: float
    reasons: List[str]
    estimated_fix_time_minutes: int


class FinalProcessingReportGenerator:
    """
    Generates comprehensive final processing reports with statistics, quality analysis,
    and reprocessing recommendations as specified in requirements 9.6 and 9.7.
    """
    
    def __init__(self, output_dir: str = "/tmp/monthly_processing"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports and read processing logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize multi-month report generator
        self.multi_month_generator = MultiMonthReportGenerator()
        
        # Processing data
        self.monthly_statistics: List[MonthlyProcessingStatistics] = []
        self.processing_summary: Optional[ProcessingSummary] = None
        self.failed_months: List[str] = []
        self.processing_errors: List[Dict[str, Any]] = []
    
    def load_processing_results(self) -> bool:
        """
        Load processing results from logs and statistics files
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Load monthly statistics from JSON files
            self._load_monthly_statistics()
            
            # Load processing summary from logs
            self._load_processing_summary()
            
            # Load failed months and errors
            self._load_failed_months_and_errors()
            
            return True
            
        except Exception as e:
            print(f"Error loading processing results: {e}")
            return False
    
    def _load_monthly_statistics(self) -> None:
        """Load monthly statistics from saved JSON files"""
        stats_dir = self.output_dir / "statistics"
        
        if not stats_dir.exists():
            print(f"Warning: Statistics directory not found at {stats_dir}")
            return
        
        # Load all monthly statistics JSON files
        for stats_file in stats_dir.glob("monthly_stats_*.json"):
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                
                # Store as dict for now (conversion to dataclass is complex with nested objects)
                # The report generation methods handle both dict and dataclass formats
                self.monthly_statistics.append(stats_data)
                
                # For multi-month generator, we'll create a simple object with nested attribute access
                class SimpleStats:
                    def __init__(self, data):
                        for key, value in data.items():
                            if isinstance(value, dict):
                                # Create nested object for dict values
                                nested_obj = SimpleStats(value)
                                setattr(self, key, nested_obj)
                            else:
                                setattr(self, key, value)
                
                simple_stats = SimpleStats(stats_data)
                self.multi_month_generator.add_monthly_stats(simple_stats)
                
            except Exception as e:
                print(f"Warning: Could not load statistics from {stats_file}: {e}")
    
    def _load_processing_summary(self) -> None:
        """Load processing summary from comprehensive processing log"""
        log_file = self.output_dir / "comprehensive_processing.log"
        
        if not log_file.exists():
            print(f"Warning: Processing log not found at {log_file}")
            return
        
        # Parse processing log to extract summary information
        processing_events = []
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        processing_events.append(event)
                    except json.JSONDecodeError:
                        continue
            
            # Extract summary from events
            self._extract_processing_summary_from_events(processing_events)
            
        except Exception as e:
            print(f"Warning: Could not parse processing log: {e}")
    
    def _extract_processing_summary_from_events(self, events: List[Dict[str, Any]]) -> None:
        """Extract processing summary from log events"""
        if not events:
            return
        
        # Find start and end events
        start_events = [e for e in events if 'PROCESSING PIPELINE' in e.get('message', '')]
        end_events = [e for e in events if 'PROCESSING COMPLETE' in e.get('message', '')]
        
        # Count successful and failed months
        success_events = [e for e in events if 'completed successfully' in e.get('message', '')]
        failure_events = [e for e in events if 'failed' in e.get('message', '') and 'month' in e.get('context', {})]
        
        # Extract month information
        months_attempted = set()
        for event in events:
            context = event.get('context', {})
            if 'month' in context:
                months_attempted.add(context['month'])
        
        # Calculate processing time
        if start_events and end_events:
            start_time = datetime.fromisoformat(start_events[0]['timestamp'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_events[-1]['timestamp'].replace('Z', '+00:00'))
            total_time = (end_time - start_time).total_seconds() / 3600
        else:
            start_time = end_time = datetime.now()
            total_time = 0
        
        # Create processing summary
        total_attempted = len(months_attempted)
        successful = len(success_events)
        failed = len(failure_events)
        
        if total_attempted > 0:
            success_rate = (successful / total_attempted) * 100
            avg_time = (total_time * 60) / total_attempted if total_attempted > 0 else 0
        else:
            success_rate = 0
            avg_time = 0
        
        # Determine date range
        sorted_months = sorted(months_attempted) if months_attempted else ['unknown']
        
        self.processing_summary = ProcessingSummary(
            total_months_attempted=total_attempted,
            successful_months=successful,
            failed_months=failed,
            success_rate=success_rate,
            total_processing_time_hours=total_time,
            avg_processing_time_minutes=avg_time,
            date_range_start=sorted_months[0] if sorted_months else 'unknown',
            date_range_end=sorted_months[-1] if sorted_months else 'unknown',
            processing_start_time=start_time,
            processing_end_time=end_time
        )
    
    def _load_failed_months_and_errors(self) -> None:
        """Load failed months and error information"""
        # Load failed months list
        failed_months_file = self.output_dir / "failed_months.txt"
        if failed_months_file.exists():
            try:
                with open(failed_months_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            self.failed_months.append(line)
            except Exception as e:
                print(f"Warning: Could not load failed months: {e}")
        
        # Load processing errors from status log
        status_log = self.output_dir / "processing_status.log"
        if status_log.exists():
            try:
                with open(status_log, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',', 3)
                        if len(parts) >= 4:
                            timestamp, month, level, message = parts
                            if level in ['ERROR', 'CRITICAL']:
                                self.processing_errors.append({
                                    'timestamp': timestamp,
                                    'month': month,
                                    'level': level,
                                    'message': message
                                })
            except Exception as e:
                print(f"Warning: Could not load processing errors: {e}")
    
    def generate_comprehensive_final_report(self) -> str:
        """
        Generate comprehensive final processing report
        
        Returns:
            Formatted final report as string
        """
        report_lines = []
        
        # Header
        report_lines.append("# 📊 Final Data Processing Report")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.extend(self._generate_executive_summary())
        
        # Processing Results Summary
        report_lines.extend(self._generate_processing_results_summary())
        
        # Data Quality Summary
        report_lines.extend(self._generate_data_quality_summary())
        
        # Failed Months Analysis
        if self.failed_months or self.processing_errors:
            report_lines.extend(self._generate_failed_months_analysis())
        
        # Reprocessing Recommendations
        recommendations = self._generate_reprocessing_recommendations()
        if recommendations:
            report_lines.extend(self._format_reprocessing_recommendations(recommendations))
        
        # Multi-Month Trend Analysis
        if len(self.monthly_statistics) > 1:
            report_lines.extend(self._generate_trend_analysis())
        
        # Performance Analysis
        report_lines.extend(self._generate_performance_analysis())
        
        # Next Steps and Recommendations
        report_lines.extend(self._generate_next_steps())
        
        return "\n".join(report_lines)
    
    def _generate_executive_summary(self) -> List[str]:
        """Generate executive summary section"""
        lines = []
        lines.append("## 🎯 Executive Summary")
        lines.append("")
        
        if self.processing_summary:
            lines.append(f"**Processing Period:** {self.processing_summary.date_range_start} to {self.processing_summary.date_range_end}")
            lines.append(f"**Total Runtime:** {self.processing_summary.total_processing_time_hours:.1f} hours")
            lines.append(f"**Success Rate:** {self.processing_summary.success_rate:.1f}% ({self.processing_summary.successful_months}/{self.processing_summary.total_months_attempted} months)")
            lines.append(f"**Average Processing Time:** {self.processing_summary.avg_processing_time_minutes:.1f} minutes per month")
        else:
            lines.append("**Status:** Processing summary not available")
        
        # Overall assessment
        if self.processing_summary and self.processing_summary.success_rate >= 95:
            lines.append("**Overall Assessment:** ✅ Excellent - Ready for model training")
        elif self.processing_summary and self.processing_summary.success_rate >= 85:
            lines.append("**Overall Assessment:** ⚠️ Good - Minor reprocessing needed")
        elif self.processing_summary and self.processing_summary.success_rate >= 70:
            lines.append("**Overall Assessment:** 🔄 Fair - Significant reprocessing required")
        else:
            lines.append("**Overall Assessment:** ❌ Poor - Major issues need resolution")
        
        lines.append("")
        return lines
    
    def _generate_processing_results_summary(self) -> List[str]:
        """Generate processing results summary section"""
        lines = []
        lines.append("## 📈 Processing Results Summary")
        lines.append("")
        
        if self.processing_summary:
            lines.append("### Success/Failure Breakdown")
            lines.append(f"- **Successful Months:** {self.processing_summary.successful_months}")
            lines.append(f"- **Failed Months:** {self.processing_summary.failed_months}")
            lines.append(f"- **Success Rate:** {self.processing_summary.success_rate:.1f}%")
            lines.append("")
            
            # Performance metrics
            lines.append("### Performance Metrics")
            lines.append(f"- **Total Processing Time:** {self.processing_summary.total_processing_time_hours:.1f} hours")
            lines.append(f"- **Average Time per Month:** {self.processing_summary.avg_processing_time_minutes:.1f} minutes")
            
            if self.processing_summary.avg_processing_time_minutes <= 20:
                lines.append("- **Performance Assessment:** ✅ Excellent (≤20 min/month)")
            elif self.processing_summary.avg_processing_time_minutes <= 30:
                lines.append("- **Performance Assessment:** ✅ Good (≤30 min/month)")
            else:
                lines.append("- **Performance Assessment:** ⚠️ Slow (>30 min/month)")
            
            lines.append("")
        
        return lines
    
    def _generate_data_quality_summary(self) -> List[str]:
        """Generate data quality summary section"""
        lines = []
        lines.append("## 🔍 Data Quality Summary")
        lines.append("")
        
        if not self.monthly_statistics:
            lines.append("No monthly statistics available for data quality analysis.")
            lines.append("")
            return lines
        
        # Calculate aggregate quality metrics
        quality_summary = self._calculate_data_quality_summary()
        
        lines.append("### Overall Data Quality")
        lines.append(f"- **Average Data Retention:** {quality_summary.avg_data_retention_rate:.1f}%")
        lines.append(f"- **Average Feature Quality Score:** {quality_summary.avg_feature_quality_score:.3f}")
        lines.append(f"- **Win Rate Consistency:** {quality_summary.avg_win_rate_consistency:.1f}%")
        lines.append("")
        
        lines.append("### Quality Issues")
        lines.append(f"- **Months with Quality Issues:** {quality_summary.months_with_quality_issues}")
        lines.append(f"- **Months Requiring Reprocessing:** {quality_summary.months_requiring_reprocessing}")
        lines.append("")
        
        lines.append("### Rollover Impact Analysis")
        lines.append(f"- **Total Rollover Events:** {quality_summary.total_rollover_events}")
        lines.append(f"- **Average Rollover Impact:** {quality_summary.avg_rollover_impact_percentage:.2f}% of bars affected")
        lines.append("")
        
        return lines
    
    def _calculate_data_quality_summary(self) -> DataQualitySummary:
        """Calculate aggregate data quality metrics"""
        if not self.monthly_statistics:
            return DataQualitySummary(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate averages (handle both dataclass and dict formats)
        avg_retention_values = []
        avg_feature_quality_values = []
        
        for stats in self.monthly_statistics:
            # Handle data quality (could be dataclass or dict)
            if hasattr(stats, 'data_quality'):
                data_quality = stats.data_quality
                if hasattr(data_quality, 'overall_retention_rate'):
                    avg_retention_values.append(data_quality.overall_retention_rate)
                elif isinstance(data_quality, dict) and 'overall_retention_rate' in data_quality:
                    avg_retention_values.append(data_quality['overall_retention_rate'])
            
            # Handle feature statistics (could be dataclass or dict)
            if hasattr(stats, 'feature_statistics'):
                feature_stats = stats.feature_statistics
                if hasattr(feature_stats, 'quality_score'):
                    avg_feature_quality_values.append(feature_stats.quality_score)
                elif isinstance(feature_stats, dict) and 'quality_score' in feature_stats:
                    avg_feature_quality_values.append(feature_stats['quality_score'])
        
        avg_retention = np.mean(avg_retention_values) if avg_retention_values else 0
        avg_feature_quality = np.mean(avg_feature_quality_values) if avg_feature_quality_values else 0
        
        # Calculate win rate consistency (how many modes have reasonable win rates)
        win_rate_consistency_scores = []
        for stats in self.monthly_statistics:
            mode_statistics = stats.mode_statistics if hasattr(stats, 'mode_statistics') else stats.get('mode_statistics', {})
            
            reasonable_modes = 0
            total_modes = 0
            
            for mode_name, mode_stats in mode_statistics.items():
                total_modes += 1
                win_rate = mode_stats.win_rate if hasattr(mode_stats, 'win_rate') else mode_stats.get('win_rate', 0)
                if 0.05 <= win_rate <= 0.50:
                    reasonable_modes += 1
            
            if total_modes > 0:
                consistency = (reasonable_modes / total_modes) * 100
                win_rate_consistency_scores.append(consistency)
        
        avg_win_rate_consistency = np.mean(win_rate_consistency_scores) if win_rate_consistency_scores else 0
        
        # Count quality issues
        months_with_issues = 0
        months_requiring_reprocessing = 0
        total_rollover_events = 0
        rollover_impact_values = []
        
        for stats in self.monthly_statistics:
            # Overall quality score
            quality_score = stats.overall_quality_score if hasattr(stats, 'overall_quality_score') else stats.get('overall_quality_score', 0)
            if quality_score < 0.7:
                months_with_issues += 1
            
            # Reprocessing requirement
            requires_reprocessing = stats.requires_reprocessing if hasattr(stats, 'requires_reprocessing') else stats.get('requires_reprocessing', False)
            if requires_reprocessing:
                months_requiring_reprocessing += 1
            
            # Rollover statistics
            rollover_events = stats.total_rollover_events if hasattr(stats, 'total_rollover_events') else stats.get('total_rollover_events', 0)
            total_rollover_events += rollover_events
            
            rollover_impact = stats.rollover_affected_percentage if hasattr(stats, 'rollover_affected_percentage') else stats.get('rollover_affected_percentage', 0)
            rollover_impact_values.append(rollover_impact)
        
        avg_rollover_impact = np.mean(rollover_impact_values) if rollover_impact_values else 0
        
        return DataQualitySummary(
            avg_data_retention_rate=avg_retention,
            avg_win_rate_consistency=avg_win_rate_consistency,
            avg_feature_quality_score=avg_feature_quality,
            months_with_quality_issues=months_with_issues,
            total_rollover_events=total_rollover_events,
            avg_rollover_impact_percentage=avg_rollover_impact,
            months_requiring_reprocessing=months_requiring_reprocessing
        )
    
    def _generate_failed_months_analysis(self) -> List[str]:
        """Generate failed months analysis section"""
        lines = []
        lines.append("## ❌ Failed Months Analysis")
        lines.append("")
        
        if self.failed_months:
            lines.append("### Failed Months List")
            for month in sorted(self.failed_months):
                lines.append(f"- {month}")
            lines.append("")
        
        if self.processing_errors:
            lines.append("### Error Analysis")
            
            # Group errors by type
            error_types = {}
            for error in self.processing_errors:
                error_type = error.get('level', 'UNKNOWN')
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error)
            
            for error_type, errors in error_types.items():
                lines.append(f"#### {error_type} Errors ({len(errors)} occurrences)")
                for error in errors[:5]:  # Show first 5 errors
                    month = error.get('month', 'unknown')
                    message = error.get('message', 'No message')[:100]
                    lines.append(f"- **{month}:** {message}")
                
                if len(errors) > 5:
                    lines.append(f"- ... and {len(errors) - 5} more")
                lines.append("")
        
        return lines
    
    def _generate_reprocessing_recommendations(self) -> List[ReprocessingRecommendation]:
        """Generate reprocessing recommendations"""
        recommendations = []
        
        for stats in self.monthly_statistics:
            # Handle both dict and dataclass formats
            requires_reprocessing = stats.requires_reprocessing if hasattr(stats, 'requires_reprocessing') else stats.get('requires_reprocessing', False)
            
            if requires_reprocessing:
                # Get values with fallbacks
                month_str = stats.month_str if hasattr(stats, 'month_str') else stats.get('month_str', 'unknown')
                quality_score = stats.overall_quality_score if hasattr(stats, 'overall_quality_score') else stats.get('overall_quality_score', 0.0)
                reprocessing_reasons = stats.reprocessing_reasons if hasattr(stats, 'reprocessing_reasons') else stats.get('reprocessing_reasons', [])
                
                # Determine priority based on quality score
                if quality_score < 0.5:
                    priority = 'high'
                    estimated_time = 45  # High priority gets more time
                elif quality_score < 0.7:
                    priority = 'medium'
                    estimated_time = 30
                else:
                    priority = 'low'
                    estimated_time = 20
                
                recommendations.append(ReprocessingRecommendation(
                    month_str=month_str,
                    priority=priority,
                    quality_score=quality_score,
                    reasons=reprocessing_reasons[:3] if reprocessing_reasons else ['Quality issues detected'],  # Top 3 reasons
                    estimated_fix_time_minutes=estimated_time
                ))
        
        # Add failed months as high priority
        for month in self.failed_months:
            if not any(rec.month_str == month for rec in recommendations):
                recommendations.append(ReprocessingRecommendation(
                    month_str=month,
                    priority='high',
                    quality_score=0.0,
                    reasons=['Processing failed completely'],
                    estimated_fix_time_minutes=60
                ))
        
        # Sort by priority and quality score
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: (priority_order[x.priority], x.quality_score))
        
        return recommendations
    
    def _format_reprocessing_recommendations(self, recommendations: List[ReprocessingRecommendation]) -> List[str]:
        """Format reprocessing recommendations section"""
        lines = []
        lines.append("## 🔄 Reprocessing Recommendations")
        lines.append("")
        
        if not recommendations:
            lines.append("✅ No reprocessing required - all months processed successfully!")
            lines.append("")
            return lines
        
        # Group by priority
        high_priority = [r for r in recommendations if r.priority == 'high']
        medium_priority = [r for r in recommendations if r.priority == 'medium']
        low_priority = [r for r in recommendations if r.priority == 'low']
        
        total_estimated_time = sum(r.estimated_fix_time_minutes for r in recommendations)
        
        lines.append(f"**Total Months Requiring Reprocessing:** {len(recommendations)}")
        lines.append(f"**Estimated Total Reprocessing Time:** {total_estimated_time / 60:.1f} hours")
        lines.append("")
        
        if high_priority:
            lines.append("### 🚨 High Priority (Process First)")
            for rec in high_priority:
                lines.append(f"- **{rec.month_str}** (Score: {rec.quality_score:.2f}, ~{rec.estimated_fix_time_minutes}min)")
                for reason in rec.reasons:
                    lines.append(f"  - {reason}")
            lines.append("")
        
        if medium_priority:
            lines.append("### ⚠️ Medium Priority")
            for rec in medium_priority:
                lines.append(f"- **{rec.month_str}** (Score: {rec.quality_score:.2f}, ~{rec.estimated_fix_time_minutes}min)")
                for reason in rec.reasons[:2]:  # Show top 2 reasons
                    lines.append(f"  - {reason}")
            lines.append("")
        
        if low_priority:
            lines.append("### 💡 Low Priority (Optional)")
            for rec in low_priority:
                lines.append(f"- **{rec.month_str}** (Score: {rec.quality_score:.2f}, ~{rec.estimated_fix_time_minutes}min)")
                if rec.reasons:
                    lines.append(f"  - {rec.reasons[0]}")
            lines.append("")
        
        return lines
    
    def _generate_trend_analysis(self) -> List[str]:
        """Generate trend analysis section"""
        lines = []
        lines.append("## 📊 Trend Analysis")
        lines.append("")
        
        # Use existing multi-month generator for trend analysis
        trend_data = self.multi_month_generator.generate_trend_analysis()
        
        if 'error' in trend_data:
            lines.append(f"Trend analysis not available: {trend_data['error']}")
            lines.append("")
            return lines
        
        lines.append("### Quality Score Trends")
        quality_scores = trend_data.get('quality_scores', [])
        if len(quality_scores) >= 3:
            first_third_avg = np.mean(quality_scores[:len(quality_scores)//3])
            last_third_avg = np.mean(quality_scores[-len(quality_scores)//3:])
            
            if last_third_avg > first_third_avg * 1.05:
                trend = "📈 Improving"
            elif last_third_avg < first_third_avg * 0.95:
                trend = "📉 Declining"
            else:
                trend = "➡️ Stable"
            
            lines.append(f"- **Overall Trend:** {trend}")
            lines.append(f"- **Early Average:** {first_third_avg:.3f}")
            lines.append(f"- **Recent Average:** {last_third_avg:.3f}")
        
        lines.append("")
        return lines
    
    def _generate_performance_analysis(self) -> List[str]:
        """Generate performance analysis section"""
        lines = []
        lines.append("## ⚡ Performance Analysis")
        lines.append("")
        
        if not self.monthly_statistics:
            lines.append("No performance data available.")
            lines.append("")
            return lines
        
        # Calculate performance metrics (handle both dict and dataclass formats)
        processing_times = []
        memory_usage = []
        
        for stats in self.monthly_statistics:
            # Get performance metrics
            if hasattr(stats, 'performance_metrics'):
                perf_metrics = stats.performance_metrics
                if hasattr(perf_metrics, 'total_processing_time_minutes'):
                    processing_times.append(perf_metrics.total_processing_time_minutes)
                elif isinstance(perf_metrics, dict) and 'total_processing_time_minutes' in perf_metrics:
                    processing_times.append(perf_metrics['total_processing_time_minutes'])
                
                if hasattr(perf_metrics, 'peak_memory_mb'):
                    memory_usage.append(perf_metrics.peak_memory_mb)
                elif isinstance(perf_metrics, dict) and 'peak_memory_mb' in perf_metrics:
                    memory_usage.append(perf_metrics['peak_memory_mb'])
            elif isinstance(stats, dict) and 'performance_metrics' in stats:
                perf_metrics = stats['performance_metrics']
                if 'total_processing_time_minutes' in perf_metrics:
                    processing_times.append(perf_metrics['total_processing_time_minutes'])
                if 'peak_memory_mb' in perf_metrics:
                    memory_usage.append(perf_metrics['peak_memory_mb'])
        
        if processing_times:
            lines.append("### Processing Time Analysis")
            lines.append(f"- **Average Processing Time:** {np.mean(processing_times):.1f} minutes")
            lines.append(f"- **Fastest Month:** {np.min(processing_times):.1f} minutes")
            lines.append(f"- **Slowest Month:** {np.max(processing_times):.1f} minutes")
            lines.append(f"- **Standard Deviation:** {np.std(processing_times):.1f} minutes")
            lines.append("")
        
        if memory_usage:
            lines.append("### Memory Usage Analysis")
            lines.append(f"- **Average Peak Memory:** {np.mean(memory_usage):.0f} MB")
            lines.append(f"- **Maximum Memory Used:** {np.max(memory_usage):.0f} MB")
            lines.append(f"- **Memory Efficiency:** {'✅ Good' if np.max(memory_usage) < 8000 else '⚠️ High'}")
            lines.append("")
        
        if not processing_times and not memory_usage:
            lines.append("Performance metrics not available in loaded statistics.")
            lines.append("")
        
        return lines
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps and recommendations section"""
        lines = []
        lines.append("## 🎯 Next Steps and Recommendations")
        lines.append("")
        
        # Determine overall status and recommendations
        if self.processing_summary:
            success_rate = self.processing_summary.success_rate
            
            if success_rate >= 95:
                lines.append("### ✅ Ready for Model Training")
                lines.append("- Processing completed successfully with excellent results")
                lines.append("- Proceed with XGBoost model training using the processed dataset")
                lines.append("- Monitor model performance and validate against expected metrics")
                
            elif success_rate >= 85:
                lines.append("### 🔄 Minor Reprocessing Required")
                lines.append("- Address high-priority reprocessing recommendations above")
                lines.append("- Consider proceeding with model training using successfully processed months")
                lines.append("- Reprocess failed months in parallel with initial model training")
                
            elif success_rate >= 70:
                lines.append("### ⚠️ Significant Reprocessing Required")
                lines.append("- Focus on resolving high and medium priority reprocessing items")
                lines.append("- Investigate common failure patterns to prevent future issues")
                lines.append("- Consider processing in smaller batches to isolate problems")
                
            else:
                lines.append("### 🚨 Major Issues Need Resolution")
                lines.append("- Review and fix fundamental processing pipeline issues")
                lines.append("- Investigate infrastructure and configuration problems")
                lines.append("- Consider reprocessing entire dataset after fixes")
        
        lines.append("")
        
        # General recommendations
        lines.append("### General Recommendations")
        
        if self.monthly_statistics:
            # Calculate average quality (handle both dict and dataclass formats)
            quality_scores = []
            processing_times = []
            reprocessing_count = 0
            
            for stats in self.monthly_statistics:
                # Quality score
                quality_score = stats.overall_quality_score if hasattr(stats, 'overall_quality_score') else stats.get('overall_quality_score', 0)
                quality_scores.append(quality_score)
                
                # Processing time
                if hasattr(stats, 'performance_metrics'):
                    perf_metrics = stats.performance_metrics
                    if hasattr(perf_metrics, 'total_processing_time_minutes'):
                        processing_times.append(perf_metrics.total_processing_time_minutes)
                    elif isinstance(perf_metrics, dict) and 'total_processing_time_minutes' in perf_metrics:
                        processing_times.append(perf_metrics['total_processing_time_minutes'])
                elif isinstance(stats, dict) and 'performance_metrics' in stats:
                    perf_metrics = stats['performance_metrics']
                    if 'total_processing_time_minutes' in perf_metrics:
                        processing_times.append(perf_metrics['total_processing_time_minutes'])
                
                # Reprocessing requirement
                requires_reprocessing = stats.requires_reprocessing if hasattr(stats, 'requires_reprocessing') else stats.get('requires_reprocessing', False)
                if requires_reprocessing:
                    reprocessing_count += 1
            
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                if avg_quality < 0.7:
                    lines.append("- Review and optimize data quality validation thresholds")
            
            if processing_times:
                avg_processing_time = np.mean(processing_times)
                if avg_processing_time > 30:
                    lines.append("- Implement performance optimizations to reduce processing time")
            
            reprocessing_rate = reprocessing_count / len(self.monthly_statistics)
            if reprocessing_rate > 0.2:
                lines.append("- Focus on improving initial processing quality to reduce reprocessing needs")
        
        lines.append("- Maintain regular monitoring of processing pipeline health")
        lines.append("- Document lessons learned for future processing runs")
        lines.append("")
        
        return lines
    
    def save_report_to_file(self, report_content: str, filename: str = None) -> str:
        """
        Save report to file
        
        Args:
            report_content: Report content to save
            filename: Optional custom filename
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"final_processing_report_{timestamp}.md"
        
        report_path = self.output_dir / filename
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"Final processing report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            print(f"Error saving report: {e}")
            return ""
    
    def generate_and_save_final_report(self) -> Tuple[str, str]:
        """
        Generate and save comprehensive final processing report
        
        Returns:
            Tuple of (report_content, report_file_path)
        """
        # Load processing results
        if not self.load_processing_results():
            print("Warning: Could not load complete processing results")
        
        # Generate comprehensive report
        report_content = self.generate_comprehensive_final_report()
        
        # Save report to file
        report_path = self.save_report_to_file(report_content)
        
        return report_content, report_path


def generate_final_processing_report(output_dir: str = "/tmp/monthly_processing") -> Tuple[str, str]:
    """
    Convenience function to generate final processing report
    
    Args:
        output_dir: Directory containing processing results and logs
        
    Returns:
        Tuple of (report_content, report_file_path)
    """
    generator = FinalProcessingReportGenerator(output_dir)
    return generator.generate_and_save_final_report()


if __name__ == "__main__":
    # Generate final processing report
    report_content, report_path = generate_final_processing_report()
    
    if report_path:
        print(f"\n📊 Final Processing Report Generated Successfully!")
        print(f"Report saved to: {report_path}")
        print(f"\nReport preview (first 50 lines):")
        print("=" * 80)
        
        lines = report_content.split('\n')
        for i, line in enumerate(lines[:50]):
            print(line)
        
        if len(lines) > 50:
            print(f"\n... and {len(lines) - 50} more lines")
        
        print("=" * 80)
    else:
        print("❌ Failed to generate final processing report")