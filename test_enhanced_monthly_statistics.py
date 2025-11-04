#!/usr/bin/env python3
"""
Test Enhanced Monthly Statistics Collection System

This test validates the comprehensive statistics collection enhancements
implemented for task 3.4, including processing metrics, rollover event tracking,
feature quality metrics, and data quality flags.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_data():
    """Create test data with various quality scenarios"""
    
    # Create 1000 bars of test data
    n_bars = 1000
    
    # Generate timestamps (1-second intervals)
    start_time = datetime(2024, 3, 1, 9, 30, 0)  # RTH start
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_bars)]
    
    # Generate realistic ES price data
    base_price = 4750.0
    price_changes = np.random.normal(0, 0.5, n_bars)  # Small random changes
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] + change
        prices.append(max(new_price, 4000))  # Minimum price floor
    
    # Create OHLCV data
    data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLC from price
        spread = np.random.uniform(0.25, 2.0)  # ES tick size is 0.25
        high = price + spread/2
        low = price - spread/2
        open_price = price + np.random.uniform(-spread/4, spread/4)
        close_price = price + np.random.uniform(-spread/4, spread/4)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        volume = np.random.randint(100, 5000)
        
        data.append({
            'timestamp': ts,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    # Add a few rollover events (large price gaps)
    rollover_indices = [200, 500, 800]
    for idx in rollover_indices:
        if idx < len(data):
            # Create 20+ point gap for rollover detection
            gap_size = np.random.uniform(20, 50)
            direction = np.random.choice(['up', 'down'])
            
            if direction == 'up':
                data[idx]['open'] += gap_size
                data[idx]['high'] += gap_size
                data[idx]['low'] += gap_size
                data[idx]['close'] += gap_size
            else:
                data[idx]['open'] -= gap_size
                data[idx]['high'] -= gap_size
                data[idx]['low'] -= gap_size
                data[idx]['close'] -= gap_size
    
    return pd.DataFrame(data)

def test_enhanced_monthly_statistics():
    """Test the enhanced statistics collection system"""
    
    print("🧪 Testing Enhanced Monthly Statistics Collection")
    print("=" * 60)
    
    try:
        # Import required modules
        from src.data_pipeline.monthly_statistics import (
            MonthlyStatisticsCollector, 
            MonthlyProcessingStatistics,
            RolloverEvent,
            create_monthly_quality_report
        )
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, OutputDataFrame
        from src.data_pipeline.features import create_all_features
        
        print("✅ Successfully imported enhanced statistics modules")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 1: Create test data and process it
    print("\n📊 Test 1: Creating and processing test data")
    
    try:
        df = create_test_data()
        print(f"   ✅ Created test data: {len(df)} rows")
        print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Process with weighted labeling
        engine = WeightedLabelingEngine()
        df_labeled = engine.process_dataframe(df, validate_performance=False)
        print(f"   ✅ Weighted labeling complete: {len(df_labeled.columns)} columns")
        
        # Add features
        df_final = create_all_features(df_labeled)
        print(f"   ✅ Feature engineering complete: {len(df_final.columns)} columns")
        
    except Exception as e:
        print(f"   ❌ Data processing failed: {e}")
        return False
    
    # Test 2: Initialize and use MonthlyStatisticsCollector
    print("\n📈 Test 2: Testing MonthlyStatisticsCollector")
    
    try:
        month_str = "2024-03"
        collector = MonthlyStatisticsCollector(month_str)
        
        # Record processing data flow
        collector.record_data_flow('raw', 1200)
        collector.record_data_flow('cleaned', 1100)
        collector.record_data_flow('rth', 1000)
        collector.record_data_flow('final', len(df_final))
        
        # Record processing stages
        collector.start_stage('weighted_labeling')
        collector.end_stage('weighted_labeling', memory_mb=2500.0)
        
        collector.start_stage('feature_engineering')
        collector.end_stage('feature_engineering', memory_mb=3200.0)
        
        # Record rollover events
        collector.record_rollover_event(
            timestamp=datetime(2024, 3, 1, 10, 30, 0),
            price_gap=25.5,
            bars_affected=6,
            gap_direction='up'
        )
        
        collector.record_rollover_event(
            timestamp=datetime(2024, 3, 1, 12, 15, 0),
            price_gap=32.0,
            bars_affected=6,
            gap_direction='down'
        )
        
        # Record component versions
        collector.record_component_version('weighted_labeling', '3.0')
        collector.record_component_version('feature_engineering', '2.1')
        
        # Record data quality fixes
        collector.record_data_quality_fix('price_issues', 15)
        collector.record_data_quality_fix('negative_volume', 3)
        
        print("   ✅ MonthlyStatisticsCollector initialized and configured")
        
    except Exception as e:
        print(f"   ❌ MonthlyStatisticsCollector setup failed: {e}")
        return False
    
    # Test 3: Test enhanced OutputDataFrame.get_statistics()
    print("\n📊 Test 3: Testing enhanced OutputDataFrame.get_statistics()")
    
    try:
        output_data = OutputDataFrame(df_final, ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Prepare test metrics
        processing_metrics = {
            'processing_time_minutes': 15.5,
            'memory_peak_mb': 3200.0,
            'memory_final_mb': 2800.0,
            'rows_per_minute': 64.5,
            'processing_efficiency_score': 0.85,
            'stage_times': {'weighted_labeling': 450.0, 'feature_engineering': 480.0},
            'slowest_stage': 'feature_engineering',
            'slowest_stage_time': 480.0
        }
        
        rollover_events = [
            {'timestamp': datetime(2024, 3, 1, 10, 30, 0), 'price_gap': 25.5, 'bars_affected': 6, 'gap_direction': 'up'},
            {'timestamp': datetime(2024, 3, 1, 12, 15, 0), 'price_gap': 32.0, 'bars_affected': 6, 'gap_direction': 'down'}
        ]
        
        feature_quality = {
            'features_generated': 43,
            'expected_features': 43,
            'feature_completeness': 100.0,
            'avg_nan_percentage': 12.5,
            'max_nan_percentage': 35.0,
            'high_nan_features': ['feature_1', 'feature_2'],
            'features_with_outliers': ['feature_3'],
            'suspicious_ranges': [],
            'quality_score': 0.82
        }
        
        # Get enhanced statistics
        stats = output_data.get_statistics(
            processing_metrics=processing_metrics,
            rollover_events=rollover_events,
            feature_quality=feature_quality
        )
        
        print(f"   ✅ Enhanced statistics collected for {len(stats)} modes")
        
        # Validate enhanced statistics structure
        dataset_summary = stats.get('dataset_summary', {})
        
        # Check processing metrics
        proc_metrics = dataset_summary.get('processing_metrics', {})
        assert proc_metrics.get('processing_time_minutes') == 15.5
        assert proc_metrics.get('memory_peak_mb') == 3200.0
        print("   ✅ Processing metrics validated")
        
        # Check rollover statistics
        rollover_stats = dataset_summary.get('rollover_statistics', {})
        assert rollover_stats.get('total_rollover_events') == 2
        assert rollover_stats.get('avg_price_gap') == 28.75  # (25.5 + 32.0) / 2
        print("   ✅ Rollover statistics validated")
        
        # Check feature quality metrics
        feature_stats = dataset_summary.get('feature_quality', {})
        assert feature_stats.get('features_generated') == 43
        assert feature_stats.get('feature_quality_score') == 0.82
        print("   ✅ Feature quality metrics validated")
        
        # Check mode-specific statistics
        for mode_name in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                         'low_vol_short', 'normal_vol_short', 'high_vol_short']:
            mode_stats = stats.get(mode_name, {})
            assert 'win_rate' in mode_stats
            assert 'avg_weight' in mode_stats
            assert 'validation_passed' in mode_stats
        
        print("   ✅ Mode-specific statistics validated")
        
    except Exception as e:
        print(f"   ❌ Enhanced statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test comprehensive statistics collection
    print("\n📈 Test 4: Testing comprehensive statistics collection")
    
    try:
        # Collect comprehensive statistics
        monthly_stats = collector.collect_comprehensive_statistics(df_final)
        
        print(f"   ✅ Comprehensive statistics collected")
        print(f"   📊 Overall quality score: {monthly_stats.overall_quality_score:.2f}")
        print(f"   🔄 Requires reprocessing: {monthly_stats.requires_reprocessing}")
        print(f"   ⏱️  Processing time: {monthly_stats.performance_metrics.total_processing_time_minutes:.1f} minutes")
        print(f"   💾 Peak memory: {monthly_stats.performance_metrics.peak_memory_mb:.1f} MB")
        print(f"   🎯 Rollover events: {monthly_stats.total_rollover_events}")
        print(f"   🔧 Features generated: {monthly_stats.feature_statistics.features_generated}")
        
        # Validate comprehensive statistics structure
        assert hasattr(monthly_stats, 'month_str')
        assert hasattr(monthly_stats, 'data_quality')
        assert hasattr(monthly_stats, 'performance_metrics')
        assert hasattr(monthly_stats, 'feature_statistics')
        assert hasattr(monthly_stats, 'mode_statistics')
        assert hasattr(monthly_stats, 'rollover_events')
        
        print("   ✅ Comprehensive statistics structure validated")
        
    except Exception as e:
        print(f"   ❌ Comprehensive statistics collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test JSON serialization and quality report generation
    print("\n📄 Test 5: Testing JSON serialization and quality report")
    
    try:
        # Test JSON serialization
        json_str = monthly_stats.to_json()
        json_data = json.loads(json_str)
        
        assert 'month_str' in json_data
        assert 'overall_quality_score' in json_data
        assert 'performance_metrics' in json_data
        
        print("   ✅ JSON serialization successful")
        
        # Test quality report generation
        quality_report = create_monthly_quality_report(monthly_stats)
        
        assert 'Monthly Processing Quality Report' in quality_report
        assert monthly_stats.month_str in quality_report
        assert 'Overall Quality Score' in quality_report
        
        print("   ✅ Quality report generation successful")
        print(f"   📄 Report length: {len(quality_report)} characters")
        
        # Save test outputs
        with tempfile.NamedTemporaryFile(mode='w', suffix='_statistics.json', delete=False) as f:
            f.write(json_str)
            json_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_quality_report.md', delete=False) as f:
            f.write(quality_report)
            report_file = f.name
        
        print(f"   💾 Test outputs saved:")
        print(f"      Statistics JSON: {json_file}")
        print(f"      Quality Report: {report_file}")
        
    except Exception as e:
        print(f"   ❌ JSON/Report test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test feature quality analysis
    print("\n🔧 Test 6: Testing feature quality analysis")
    
    try:
        feature_stats = collector.analyze_feature_quality(df_final)
        
        print(f"   ✅ Feature quality analysis complete")
        print(f"   🔧 Features generated: {feature_stats.features_generated}")
        print(f"   📊 Feature completeness: {feature_stats.feature_completeness:.1f}%")
        print(f"   📈 Quality score: {feature_stats.quality_score:.2f}")
        print(f"   ⚠️  High NaN features: {len(feature_stats.high_nan_features)}")
        print(f"   🔍 Features with outliers: {len(feature_stats.features_with_outliers)}")
        
        # Validate feature statistics
        assert feature_stats.features_generated > 0
        assert 0 <= feature_stats.quality_score <= 1
        assert isinstance(feature_stats.nan_percentages, dict)
        assert isinstance(feature_stats.value_ranges, dict)
        
        print("   ✅ Feature quality analysis validated")
        
    except Exception as e:
        print(f"   ❌ Feature quality analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All Enhanced Monthly Statistics Tests Passed!")
    print("=" * 60)
    
    # Summary
    print("\n📋 Test Summary:")
    print("✅ Enhanced OutputDataFrame.get_statistics() with processing metrics")
    print("✅ Rollover event tracking and statistics collection")
    print("✅ Feature quality metrics and data quality flags")
    print("✅ Processing time and memory usage tracking")
    print("✅ Comprehensive MonthlyProcessingStatistics integration")
    print("✅ JSON serialization and quality report generation")
    print("✅ Feature quality analysis and validation")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_monthly_statistics()
    if success:
        print("\n🎯 Task 3.4 Implementation: COMPLETE")
        print("Enhanced monthly statistics collection system is working correctly!")
    else:
        print("\n❌ Task 3.4 Implementation: FAILED")
        print("Please review the errors above and fix the issues.")
    
    sys.exit(0 if success else 1)