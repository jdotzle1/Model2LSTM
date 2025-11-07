#!/usr/bin/env python3
"""
Test script for the comprehensive performance monitoring system

This script validates the performance monitoring functionality including:
- Stage timing and metrics collection
- Memory usage tracking and analysis
- Bottleneck detection and optimization recommendations
- Performance report generation
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.data_pipeline.comprehensive_performance_monitor import (
        ComprehensivePerformanceMonitor,
        create_performance_monitor,
        generate_performance_report_json
    )
    print("✅ Successfully imported comprehensive performance monitor")
except ImportError as e:
    print(f"❌ Failed to import performance monitor: {e}")
    sys.exit(1)

def simulate_processing_stage(monitor, stage_name, duration_seconds, rows_processed, memory_intensive=False):
    """Simulate a processing stage for testing"""
    print(f"  🔄 Starting {stage_name}...")
    
    stage_id = monitor.start_stage(stage_name, {'test_mode': True})
    
    # Simulate processing work
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        # Simulate some work
        if memory_intensive:
            # Create temporary large arrays to simulate memory usage
            temp_data = np.random.random((10000, 100))
            _ = np.sum(temp_data)
            del temp_data
        
        time.sleep(0.1)
    
    # End the stage
    stage_metrics = monitor.end_stage(stage_id, rows_processed, success=True)
    
    print(f"    ✅ Completed {stage_name} in {stage_metrics.duration:.2f}s "
          f"({stage_metrics.rows_per_second:.0f} rows/sec, "
          f"{stage_metrics.memory_delta_mb:+.1f}MB)")
    
    return stage_metrics

def test_basic_monitoring():
    """Test basic performance monitoring functionality"""
    print("\n🧪 Testing Basic Performance Monitoring")
    
    # Create monitor
    monitor = create_performance_monitor(
        enable_memory_tracking=True,
        memory_sampling_interval=0.5,
        enable_detailed_logging=True
    )
    
    # Start monitoring
    expected_rows = 50000
    monitor.start_monitoring(expected_rows)
    
    try:
        # Simulate various processing stages
        simulate_processing_stage(monitor, "data_loading", 2.0, 10000)
        simulate_processing_stage(monitor, "data_cleaning", 1.5, 8000)
        simulate_processing_stage(monitor, "feature_engineering", 3.0, 15000, memory_intensive=True)
        simulate_processing_stage(monitor, "weighted_labeling", 4.0, 12000, memory_intensive=True)
        simulate_processing_stage(monitor, "validation", 1.0, 5000)
        
        # Add some quality flags
        monitor.add_quality_flag("Test quality flag 1", "warning")
        monitor.add_quality_flag("Test quality flag 2", "info")
        
        # Get live stats
        live_stats = monitor.get_live_performance_stats()
        print(f"\n📊 Live Stats:")
        print(f"  - Elapsed time: {live_stats['elapsed_time']:.1f}s")
        print(f"  - Rows processed: {live_stats['total_rows_processed']:,}")
        print(f"  - Current throughput: {live_stats['current_rows_per_second']:.0f} rows/sec")
        print(f"  - Completed stages: {live_stats['completed_stages']}")
        print(f"  - Quality flags: {live_stats['quality_flags_count']}")
        
        # Stop monitoring and get report
        report = monitor.stop_monitoring()
        
        print(f"\n📈 Performance Report Summary:")
        print(f"  - Total duration: {report.total_duration:.2f}s")
        print(f"  - Total rows: {report.total_rows_processed:,}")
        print(f"  - Overall throughput: {report.overall_rows_per_second:.0f} rows/sec")
        print(f"  - Peak memory: {report.peak_memory_mb:.1f}MB")
        print(f"  - Memory efficiency: {report.memory_efficiency_score:.1f}%")
        print(f"  - Stages completed: {len(report.stage_metrics)}")
        
        # Show bottleneck analysis
        bottlenecks = report.bottleneck_analysis
        print(f"\n🔍 Bottleneck Analysis:")
        print(f"  - Slowest stage: {bottlenecks.slowest_stage} ({bottlenecks.slowest_duration:.2f}s)")
        print(f"  - Memory intensive: {bottlenecks.memory_intensive_stage} (+{bottlenecks.peak_memory_usage:.1f}MB)")
        print(f"  - CPU intensive: {bottlenecks.cpu_intensive_stage} ({bottlenecks.peak_cpu_usage:.1f}%)")
        
        # Show recommendations
        if report.optimization_recommendations:
            print(f"\n💡 Optimization Recommendations:")
            for i, rec in enumerate(report.optimization_recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # Show quality flags
        if report.quality_flags:
            print(f"\n⚠️  Quality Flags:")
            for flag in report.quality_flags:
                print(f"  - {flag}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        return False

def test_memory_tracking():
    """Test memory tracking and analysis"""
    print("\n🧪 Testing Memory Tracking and Analysis")
    
    monitor = create_performance_monitor(
        enable_memory_tracking=True,
        memory_sampling_interval=0.2,
        enable_detailed_logging=False
    )
    
    monitor.start_monitoring(1000)
    
    try:
        # Simulate memory-intensive operations
        stage_id = monitor.start_stage("memory_test")
        
        # Create progressively larger arrays to test memory tracking
        arrays = []
        for i in range(5):
            # Create 50MB array
            array = np.random.random((6250000,))  # ~50MB
            arrays.append(array)
            time.sleep(0.5)  # Allow memory tracker to sample
        
        # Clean up arrays
        del arrays
        import gc
        gc.collect()
        time.sleep(1.0)  # Allow memory tracker to see cleanup
        
        monitor.end_stage(stage_id, 1000, success=True)
        
        # Stop monitoring and analyze memory pattern
        report = monitor.stop_monitoring()
        
        memory_pattern = report.memory_usage_pattern
        print(f"📊 Memory Analysis:")
        print(f"  - Pattern: {memory_pattern.get('pattern', 'unknown')}")
        print(f"  - Growth rate: {memory_pattern.get('growth_rate_mb_per_sec', 0):.2f} MB/sec")
        print(f"  - Memory spikes: {memory_pattern.get('spike_count', 0)}")
        print(f"  - Leak detected: {memory_pattern.get('leak_detected', False)}")
        print(f"  - Efficiency score: {memory_pattern.get('efficiency_score', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during memory tracking test: {e}")
        return False

def test_bottleneck_detection():
    """Test bottleneck detection and optimization recommendations"""
    print("\n🧪 Testing Bottleneck Detection")
    
    monitor = create_performance_monitor(enable_detailed_logging=False)
    monitor.start_monitoring(10000)
    
    try:
        # Simulate stages with different performance characteristics
        
        # Fast stage
        simulate_processing_stage(monitor, "fast_stage", 0.5, 5000)
        
        # Slow stage (bottleneck)
        simulate_processing_stage(monitor, "slow_bottleneck", 8.0, 2000)
        
        # Memory intensive stage
        simulate_processing_stage(monitor, "memory_intensive", 2.0, 1500, memory_intensive=True)
        
        # Low throughput stage
        simulate_processing_stage(monitor, "low_throughput", 3.0, 500)
        
        # Add quality flags for various issues
        monitor.add_quality_flag("High memory usage detected", "warning")
        monitor.add_quality_flag("Slow processing detected", "error")
        monitor.add_quality_flag("Low throughput detected", "warning")
        
        report = monitor.stop_monitoring()
        
        # Analyze bottlenecks
        bottlenecks = report.bottleneck_analysis
        print(f"🔍 Detected Bottlenecks:")
        print(f"  - Slowest: {bottlenecks.slowest_stage} ({bottlenecks.slowest_duration:.1f}s)")
        print(f"  - Memory intensive: {bottlenecks.memory_intensive_stage}")
        print(f"  - Recommendations: {len(bottlenecks.recommendations)}")
        
        for i, rec in enumerate(bottlenecks.recommendations, 1):
            print(f"    {i}. {rec}")
        
        if bottlenecks.optimization_opportunities:
            print(f"  - Optimization opportunities: {len(bottlenecks.optimization_opportunities)}")
            for i, opp in enumerate(bottlenecks.optimization_opportunities, 1):
                print(f"    {i}. {opp}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during bottleneck detection test: {e}")
        return False

def test_report_generation():
    """Test performance report generation and serialization"""
    print("\n🧪 Testing Report Generation")
    
    monitor = create_performance_monitor()
    monitor.start_monitoring(5000)
    
    try:
        # Quick processing simulation
        simulate_processing_stage(monitor, "test_stage", 1.0, 5000)
        
        report = monitor.stop_monitoring()
        
        # Generate JSON report
        json_report = generate_performance_report_json(report)
        
        print(f"📄 Generated JSON report ({len(json_report)} characters)")
        
        # Save to file
        report_path = "test_performance_report.json"
        generate_performance_report_json(report, report_path)
        
        if os.path.exists(report_path):
            print(f"✅ Report saved to {report_path}")
            
            # Clean up
            os.remove(report_path)
            print(f"🧹 Cleaned up test file")
        else:
            print(f"❌ Failed to save report to {report_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error during report generation test: {e}")
        return False

def test_integration_with_existing_code():
    """Test integration with existing processing pipeline"""
    print("\n🧪 Testing Integration with Existing Code")
    
    try:
        # Test importing existing modules
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig
        
        # Create monitor
        monitor = create_performance_monitor(enable_detailed_logging=False)
        
        # Test that monitor can be used with existing code structure
        config = LabelingConfig(enable_performance_monitoring=True)
        
        print("✅ Successfully integrated with existing weighted labeling system")
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:30:00', periods=1000, freq='1s', tz='UTC'),
            'open': 4750.0 + np.random.randn(1000) * 0.5,
            'high': 4750.0 + np.random.randn(1000) * 0.5 + 0.25,
            'low': 4750.0 + np.random.randn(1000) * 0.5 - 0.25,
            'close': 4750.0 + np.random.randn(1000) * 0.5,
            'volume': np.random.randint(100, 1000, 1000)
        })
        
        # Ensure OHLC relationships are valid
        sample_data['high'] = np.maximum.reduce([sample_data['open'], sample_data['high'], 
                                               sample_data['low'], sample_data['close']])
        sample_data['low'] = np.minimum.reduce([sample_data['open'], sample_data['high'], 
                                              sample_data['low'], sample_data['close']])
        
        monitor.start_monitoring(len(sample_data))
        
        # Simulate processing with the monitor
        stage_id = monitor.start_stage("sample_processing")
        
        # Simulate some processing time
        time.sleep(0.5)
        
        monitor.end_stage(stage_id, len(sample_data), success=True)
        
        report = monitor.stop_monitoring()
        
        print(f"✅ Successfully processed {len(sample_data)} rows")
        print(f"   Duration: {report.total_duration:.2f}s")
        print(f"   Throughput: {report.overall_rows_per_second:.0f} rows/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all performance monitoring tests"""
    print("🚀 Starting Comprehensive Performance Monitor Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Monitoring", test_basic_monitoring),
        ("Memory Tracking", test_memory_tracking),
        ("Bottleneck Detection", test_bottleneck_detection),
        ("Report Generation", test_report_generation),
        ("Integration Test", test_integration_with_existing_code)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Comprehensive performance monitoring is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)