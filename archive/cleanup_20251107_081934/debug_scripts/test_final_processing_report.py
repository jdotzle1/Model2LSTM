#!/usr/bin/env python3
"""
Test Final Processing Report Generation

This script tests the final processing report generation functionality with sample data.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.final_processing_report import FinalProcessingReportGenerator
from src.data_pipeline.monthly_statistics import MonthlyProcessingStatistics, ModeStatistics, ProcessingPerformanceMetrics, DataQualityMetrics, FeatureStatistics, RolloverEvent


def create_sample_monthly_statistics(month_str: str, success: bool = True, quality_score: float = 0.8) -> MonthlyProcessingStatistics:
    """Create sample monthly statistics for testing"""
    
    # Create sample mode statistics
    mode_stats = {}
    for mode_name in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 'low_vol_short', 'normal_vol_short', 'high_vol_short']:
        win_rate = 0.25 + (hash(mode_name + month_str) % 20) / 100  # 0.25-0.45 range
        total_winners = 1500 + (hash(mode_name) % 500)
        total_samples = 6000 + (hash(mode_name) % 1000)
        avg_weight = 1.2 + (hash(mode_name) % 50) / 100
        
        mode_stats[mode_name] = ModeStatistics(
            win_rate=win_rate,
            total_winners=total_winners,
            total_samples=total_samples,
            loss_rate=1.0 - win_rate,
            avg_weight=avg_weight,
            weight_std=0.3 + (hash(mode_name) % 20) / 100,
            min_weight=0.5 + (hash(mode_name) % 10) / 100,
            max_weight=2.5 + (hash(mode_name) % 15) / 100,
            weight_percentiles={'p25': 0.8, 'p50': 1.2, 'p75': 1.8, 'p90': 2.5, 'p95': 3.2},
            
            # Data quality metrics
            nan_percentage_labels=0.0 if success else 2.5,
            nan_percentage_weights=0.0 if success else 1.8,
            has_nan_labels=not success,
            has_nan_weights=not success,
            has_infinite_labels=False,
            has_infinite_weights=False,
            
            # Validation flags
            labels_binary=success,
            weights_positive=success,
            win_rate_reasonable=0.05 <= win_rate <= 0.50,
            validation_passed=success and 0.05 <= win_rate <= 0.50,
            
            # Quality flags
            quality_flags=[] if success else ['unusual_win_rate']
        )
    
    # Create sample performance metrics
    processing_time = 15 + (hash(month_str) % 20)
    start_time = datetime.now() - timedelta(minutes=processing_time)
    end_time = datetime.now()
    
    performance = ProcessingPerformanceMetrics(
        processing_start_time=start_time,
        processing_end_time=end_time,
        total_processing_time_minutes=processing_time,
        
        # Stage timing breakdown
        stage_times={'download': 2.5, 'processing': 12.3, 'upload': 1.8},
        slowest_stage='processing',
        slowest_stage_time=12.3,
        
        # Performance indicators
        rows_per_minute=2800 + (hash(month_str) % 500),
        mb_per_minute=150.0 + (hash(month_str) % 50),
        processing_efficiency_score=0.85 + (hash(month_str) % 15) / 100,
        
        # Memory usage tracking
        peak_memory_mb=4500 + (hash(month_str) % 2000),
        final_memory_mb=3200 + (hash(month_str) % 1000),
        memory_reduction_mb=1300 + (hash(month_str) % 500),
        memory_per_row_kb=0.08 + (hash(month_str) % 20) / 1000,
        memory_at_stages={'start': 2500, 'processing': 4500, 'end': 3200},
        
        # Component versions and methods
        component_versions={'weighted_labeling': '2.1.0', 'features': '1.8.0'},
        processing_methods={'labeling': 'vectorized', 'features': 'chunked'}
    )
    
    # Create sample data quality metrics
    raw_bars = 50000 + (hash(month_str) % 5000)
    cleaned_bars = int(raw_bars * 0.95)
    rth_bars = int(cleaned_bars * 0.75)
    final_bars = int(rth_bars * 0.92)
    
    data_quality = DataQualityMetrics(
        raw_bars=raw_bars,
        cleaned_bars=cleaned_bars,
        rth_bars=rth_bars,
        final_bars=final_bars,
        
        # Data retention rates
        cleaning_retention_rate=cleaned_bars / raw_bars * 100,
        rth_retention_rate=rth_bars / cleaned_bars * 100,
        final_retention_rate=final_bars / rth_bars * 100,
        overall_retention_rate=final_bars / raw_bars * 100,
        
        # Data quality issues
        price_issues_fixed=10 + (hash(month_str) % 50),
        negative_volume_fixed=5 + (hash(month_str) % 20),
        invalid_ohlc_fixed=3 + (hash(month_str) % 15),
        timezone_issues=2 + (hash(month_str) % 10),
        
        # Quality flags
        excessive_data_loss=not success,
        unusual_rth_percentage=False,
        data_quality_warnings=[] if success else ['High data loss detected']
    )
    
    # Create sample feature statistics
    feature_stats = FeatureStatistics(
        features_generated=43 if success else 41,
        expected_features=43,
        feature_completeness=100.0 if success else 95.3,
        
        # NaN analysis
        nan_percentages={'volume_slope_5s': 28.5, 'atr_30s': 15.2, 'return_300s': 22.1},
        high_nan_features=['volume_slope_5s'] if hash(month_str) % 4 == 0 else [],
        avg_nan_percentage=12.5 + (hash(month_str) % 10),
        max_nan_percentage=28.0 + (hash(month_str) % 15),
        
        # Outlier detection
        outlier_counts={'atr_30s': 45, 'return_300s': 23, 'volatility_regime': 12},
        features_with_outliers=['atr_30s', 'return_300s'] if hash(month_str) % 3 == 0 else [],
        
        # Value range validation
        value_ranges={'volume_ratio_30s': (0.1, 5.8), 'atr_30s': (0.5, 12.3)},
        suspicious_ranges=[],
        
        # Overall quality score
        quality_score=quality_score + 0.1 if quality_score < 0.9 else 0.95
    )
    
    # Create rollover events
    rollover_events = [
        RolloverEvent(
            timestamp=datetime.now() - timedelta(days=15),
            price_gap=25.5,
            bars_affected=6,
            gap_direction='up',
            detection_method='price_gap_threshold'
        )
    ]
    
    return MonthlyProcessingStatistics(
        month_str=month_str,
        processing_date=datetime.now() - timedelta(days=hash(month_str) % 30),
        
        # Data quality and flow metrics
        data_quality=data_quality,
        
        # Rollover event tracking
        rollover_events=rollover_events,
        total_rollover_events=len(rollover_events),
        bars_excluded_rollover=sum(event.bars_affected for event in rollover_events),
        rollover_affected_percentage=1.5 + (hash(month_str) % 3),
        
        # Mode-specific statistics
        mode_statistics=mode_stats,
        
        # Feature engineering statistics
        feature_statistics=feature_stats,
        
        # Processing performance metrics
        performance_metrics=performance,
        
        # Overall quality assessment
        overall_quality_score=quality_score,
        requires_reprocessing=not success or quality_score < 0.7,
        reprocessing_reasons=['Low quality score'] if quality_score < 0.7 else [],
        
        # Processing status
        processing_successful=success,
        processing_errors=[f'Error in {month_str}'] if not success else [],
        processing_warnings=[f'Warning for {month_str}'] if hash(month_str) % 5 == 0 else []
    )


def create_sample_processing_logs(temp_dir: Path):
    """Create sample processing logs for testing"""
    
    # Create comprehensive processing log
    log_file = temp_dir / "comprehensive_processing.log"
    
    processing_events = [
        {
            'timestamp': '2024-01-01T10:00:00',
            'processing_time': 1704110400.0,
            'level': 'INFO',
            'stage': 'pipeline',
            'stage_event': 'start',
            'message': 'MONTHLY ES DATA PROCESSING PIPELINE',
            'context': {},
            'has_error': False
        },
        {
            'timestamp': '2024-01-01T10:15:00',
            'processing_time': 1704111300.0,
            'level': 'INFO',
            'stage': 'processing',
            'stage_event': 'end',
            'message': '2024-01 completed successfully',
            'context': {'month': '2024-01'},
            'has_error': False
        },
        {
            'timestamp': '2024-01-01T10:30:00',
            'processing_time': 1704112200.0,
            'level': 'ERROR',
            'stage': 'processing',
            'stage_event': 'end',
            'message': '2024-02 failed',
            'context': {'month': '2024-02'},
            'has_error': True
        },
        {
            'timestamp': '2024-01-01T12:00:00',
            'processing_time': 1704117600.0,
            'level': 'INFO',
            'stage': 'pipeline',
            'stage_event': 'end',
            'message': 'MONTHLY PROCESSING COMPLETE',
            'context': {},
            'has_error': False
        }
    ]
    
    with open(log_file, 'w') as f:
        for event in processing_events:
            f.write(json.dumps(event) + '\n')
    
    # Create failed months file
    failed_file = temp_dir / "failed_months.txt"
    with open(failed_file, 'w') as f:
        f.write("# Failed months from processing run\n")
        f.write("# Run completed: 2024-01-01T12:00:00\n")
        f.write("# Success rate: 83.3%\n\n")
        f.write("2024-02\n")
    
    # Create processing status log
    status_file = temp_dir / "processing_status.log"
    with open(status_file, 'w') as f:
        f.write("2024-01-01T10:30:00,2024-02,ERROR,Processing failed with S3 connection timeout\n")


def create_sample_statistics_files(temp_dir: Path):
    """Create sample monthly statistics files"""
    
    stats_dir = temp_dir / "statistics"
    stats_dir.mkdir(exist_ok=True)
    
    # Create sample statistics for multiple months
    months = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06']
    
    for i, month in enumerate(months):
        # Make some months have issues for testing
        success = i != 1  # 2024-02 fails
        quality_score = 0.85 if success else 0.45
        
        if i == 4:  # 2024-05 has medium quality
            quality_score = 0.65
        
        stats = create_sample_monthly_statistics(month, success, quality_score)
        
        stats_file = stats_dir / f"monthly_stats_{month}.json"
        with open(stats_file, 'w') as f:
            json.dump(asdict(stats), f, indent=2, default=str)


def test_final_processing_report():
    """Test the final processing report generation"""
    
    print("🧪 Testing Final Processing Report Generation")
    print("=" * 60)
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"📁 Creating test data in: {temp_path}")
        
        # Create sample data
        create_sample_processing_logs(temp_path)
        create_sample_statistics_files(temp_path)
        
        print("✅ Sample data created")
        
        # Test report generation
        print("🔄 Generating final processing report...")
        
        try:
            generator = FinalProcessingReportGenerator(str(temp_path))
            
            # Test loading processing results
            load_success = generator.load_processing_results()
            print(f"📊 Data loading: {'✅ Success' if load_success else '⚠️ Partial'}")
            
            if generator.monthly_statistics:
                print(f"   • Loaded {len(generator.monthly_statistics)} monthly statistics")
            
            if generator.processing_summary:
                print(f"   • Processing summary: {generator.processing_summary.success_rate:.1f}% success rate")
            
            if generator.failed_months:
                print(f"   • Failed months: {len(generator.failed_months)}")
            
            # Generate comprehensive report
            report_content = generator.generate_comprehensive_final_report()
            
            print(f"📄 Report generated: {len(report_content.split())} words, {len(report_content.split(chr(10)))} lines")
            
            # Save report
            report_path = generator.save_report_to_file(report_content, "test_final_report.md")
            
            if report_path:
                print(f"💾 Report saved to: {report_path}")
                
                # Validate report content
                print("🔍 Validating report content...")
                
                required_sections = [
                    "Executive Summary",
                    "Processing Results Summary", 
                    "Data Quality Summary",
                    "Failed Months Analysis",
                    "Reprocessing Recommendations",
                    "Performance Analysis",
                    "Next Steps"
                ]
                
                missing_sections = []
                for section in required_sections:
                    if section not in report_content:
                        missing_sections.append(section)
                
                if missing_sections:
                    print(f"⚠️  Missing sections: {missing_sections}")
                else:
                    print("✅ All required sections present")
                
                # Test reprocessing recommendations
                recommendations = generator._generate_reprocessing_recommendations()
                print(f"🔄 Reprocessing recommendations: {len(recommendations)} months")
                
                high_priority = len([r for r in recommendations if r.priority == 'high'])
                medium_priority = len([r for r in recommendations if r.priority == 'medium'])
                low_priority = len([r for r in recommendations if r.priority == 'low'])
                
                print(f"   • High priority: {high_priority}")
                print(f"   • Medium priority: {medium_priority}")
                print(f"   • Low priority: {low_priority}")
                
                # Show sample of report
                print("\n" + "=" * 60)
                print("📋 SAMPLE REPORT CONTENT (First 20 lines)")
                print("=" * 60)
                
                lines = report_content.split('\n')
                for i, line in enumerate(lines[:20]):
                    print(f"{i+1:2d}: {line}")
                
                if len(lines) > 20:
                    print(f"... and {len(lines) - 20} more lines")
                
                print("=" * 60)
                print("✅ Final Processing Report Test Completed Successfully!")
                
                return True
            
            else:
                print("❌ Failed to save report")
                return False
        
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False


if __name__ == "__main__":
    success = test_final_processing_report()
    sys.exit(0 if success else 1)