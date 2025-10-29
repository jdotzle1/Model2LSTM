"""
Test script for performance monitoring implementation

This script tests the performance monitoring and optimization features
on a small dataset to ensure everything works correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from project.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig
from project.data_pipeline.performance_monitor import PerformanceMonitor


def generate_small_test_data(n_rows: int = 1000) -> pd.DataFrame:
    """Generate small test dataset for validation"""
    np.random.seed(42)
    
    # Generate timestamps (RTH only - ensure we don't exceed RTH hours)
    start_time = pd.Timestamp('2024-01-02 07:30:00')
    # RTH is 7.5 hours = 27,000 seconds, so limit to that per day
    if n_rows <= 27000:
        timestamps = pd.date_range(start_time, periods=n_rows, freq='1s')
    else:
        # For larger datasets, span multiple days
        timestamps = []
        current_date = pd.Timestamp('2024-01-02')
        rows_generated = 0
        
        while rows_generated < n_rows:
            day_start = current_date.replace(hour=7, minute=30, second=0)
            remaining_rows = n_rows - rows_generated
            max_rows_today = min(remaining_rows, 27000)
            
            day_timestamps = pd.date_range(day_start, periods=max_rows_today, freq='1s')
            timestamps.extend(day_timestamps)
            rows_generated += max_rows_today
            
            # Move to next business day
            current_date += pd.Timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += pd.Timedelta(days=1)
        
        timestamps = pd.DatetimeIndex(timestamps[:n_rows])
    
    # Generate realistic price data
    base_price = 4750.0
    tick_size = 0.25
    
    # Random walk with small movements
    price_changes = np.random.normal(0, 0.1, n_rows)
    prices = base_price + np.cumsum(price_changes)
    prices = np.round(prices / tick_size) * tick_size
    
    # Generate OHLC
    opens = prices
    spreads = np.random.exponential(0.5, n_rows) * tick_size
    highs = opens + spreads
    lows = opens - spreads
    closes = opens + np.random.normal(0, 0.1, n_rows)
    closes = np.clip(closes, lows, highs)
    
    # Round to tick size
    opens = np.round(opens / tick_size) * tick_size
    highs = np.round(highs / tick_size) * tick_size
    lows = np.round(lows / tick_size) * tick_size
    closes = np.round(closes / tick_size) * tick_size
    
    # Generate volume
    volumes = np.random.randint(500, 2000, n_rows)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


def test_performance_monitoring():
    """Test performance monitoring functionality"""
    print("Testing Performance Monitoring Implementation")
    print("=" * 50)
    
    # Generate test data
    print("Generating test data...")
    test_df = generate_small_test_data(1000)
    print(f"Generated {len(test_df):,} rows of test data")
    
    # Configure with performance monitoring enabled but relaxed targets for small test
    config = LabelingConfig(
        chunk_size=500,  # Small chunks for testing
        enable_performance_monitoring=True,
        enable_memory_optimization=True,
        enable_progress_tracking=True,
        performance_target_rows_per_minute=10_000,  # Relaxed target for small test
        memory_limit_gb=8.0
    )
    
    # Test the weighted labeling engine
    print("\nTesting WeightedLabelingEngine with performance monitoring...")
    
    try:
        engine = WeightedLabelingEngine(config)
        
        # Verify performance monitor was initialized
        assert engine.performance_monitor is not None, "Performance monitor not initialized"
        print("✓ Performance monitor initialized")
        
        # Process the data
        result_df = engine.process_dataframe(test_df)
        
        # Verify results
        assert len(result_df) == len(test_df), "Output size mismatch"
        print(f"✓ Processed {len(result_df):,} rows successfully")
        
        # Check that new columns were added
        expected_columns = []
        for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                    'low_vol_short', 'normal_vol_short', 'high_vol_short']:
            expected_columns.extend([f'label_{mode}', f'weight_{mode}'])
        
        for col in expected_columns:
            assert col in result_df.columns, f"Missing column: {col}"
        print(f"✓ All {len(expected_columns)} expected columns present")
        
        # Verify label values are 0 or 1
        for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                    'low_vol_short', 'normal_vol_short', 'high_vol_short']:
            label_col = f'label_{mode}'
            unique_values = set(result_df[label_col].unique())
            assert unique_values.issubset({0, 1}), f"Invalid label values in {label_col}: {unique_values}"
        print("✓ All label columns contain only 0 and 1 values")
        
        # Verify weight values are positive
        for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                    'low_vol_short', 'normal_vol_short', 'high_vol_short']:
            weight_col = f'weight_{mode}'
            min_weight = result_df[weight_col].min()
            assert min_weight > 0, f"Non-positive weight found in {weight_col}: {min_weight}"
        print("✓ All weight columns contain only positive values")
        
        # Check performance metrics
        if engine.performance_monitor:
            metrics = engine.performance_monitor.metrics
            print(f"\nPerformance Metrics:")
            print(f"  Rows processed: {metrics.rows_processed:,}")
            print(f"  Processing speed: {metrics.rows_per_minute:,.0f} rows/minute")
            print(f"  Peak memory: {metrics.peak_memory_gb:.3f} GB")
            print(f"  Elapsed time: {metrics.elapsed_time:.2f} seconds")
            
            # Verify metrics make sense
            assert metrics.rows_processed == len(test_df), "Incorrect row count in metrics"
            assert metrics.rows_per_minute > 0, "Invalid processing speed"
            assert metrics.peak_memory_gb > 0, "Invalid memory usage"
            print("✓ Performance metrics are valid")
        
        print(f"\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitor_standalone():
    """Test PerformanceMonitor class standalone"""
    print("\nTesting PerformanceMonitor standalone...")
    
    monitor = PerformanceMonitor(target_rows_per_minute=100_000, memory_limit_gb=4.0)
    
    # Test basic functionality
    monitor.start_monitoring(1000)
    
    # Simulate some processing
    import time
    time.sleep(0.1)
    monitor.update_progress(500, "halfway")
    
    time.sleep(0.1)
    monitor.update_progress(1000, "complete")
    
    # Finish monitoring
    metrics = monitor.finish_monitoring()
    
    # Validate metrics
    assert metrics.rows_processed == 1000, "Incorrect row count"
    assert metrics.elapsed_time > 0.2, "Elapsed time too short"
    assert metrics.rows_per_minute > 0, "Invalid processing speed"
    
    print("✓ PerformanceMonitor standalone test passed")
    
    # Test validation
    validation = monitor.validate_performance_target(10_000)
    print(f"  Validation results: {validation}")
    
    return True


if __name__ == "__main__":
    print("Performance Monitoring Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_performance_monitor_standalone()
    test2_passed = test_performance_monitoring()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"PerformanceMonitor standalone: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"WeightedLabelingEngine integration: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    overall_pass = test1_passed and test2_passed
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")
    
    if overall_pass:
        print("\n🎉 Performance monitoring implementation is working correctly!")
        print("   Ready to test on larger datasets and validate 10M row target.")
    else:
        print("\n⚠️  Issues found in performance monitoring implementation.")
        print("   Please review the errors above and fix before proceeding.")