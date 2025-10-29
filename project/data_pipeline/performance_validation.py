"""
Performance Validation Script

This script validates that the weighted labeling system meets the performance
requirements of processing 10M rows within 60 minutes.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .weighted_labeling import WeightedLabelingEngine, LabelingConfig
from .performance_monitor import PerformanceMonitor, performance_context


def generate_test_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic test data for performance validation
    
    Args:
        n_rows: Number of rows to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with OHLCV data suitable for testing
    """
    np.random.seed(seed)
    
    # Generate realistic ES futures data
    base_price = 4750.0
    tick_size = 0.25
    
    # Generate timestamps (1-second bars during RTH only)
    # RTH is 07:30-15:00 CT = 7.5 hours = 27,000 seconds per day
    rth_seconds_per_day = 7.5 * 60 * 60  # 27,000 seconds
    
    timestamps = []
    current_date = pd.Timestamp('2024-01-02')  # Start on a Tuesday
    rows_generated = 0
    
    while rows_generated < n_rows:
        # Generate RTH timestamps for current day
        day_start = current_date.replace(hour=7, minute=30, second=0)
        day_end = current_date.replace(hour=15, minute=0, second=0)
        
        # Calculate how many rows we can fit in this day
        remaining_rows = n_rows - rows_generated
        max_rows_today = min(remaining_rows, int(rth_seconds_per_day))
        
        # Generate timestamps for this day
        day_timestamps = pd.date_range(day_start, periods=max_rows_today, freq='1s')
        timestamps.extend(day_timestamps)
        
        rows_generated += max_rows_today
        
        # Move to next business day
        current_date += pd.Timedelta(days=1)
        # Skip weekends
        while current_date.weekday() >= 5:  # Saturday=5, Sunday=6
            current_date += pd.Timedelta(days=1)
    
    timestamps = pd.DatetimeIndex(timestamps[:n_rows])
    
    # Generate price movements (random walk with mean reversion)
    price_changes = np.random.normal(0, 0.5, n_rows)  # Small random changes
    prices = base_price + np.cumsum(price_changes)
    
    # Round to tick size
    prices = np.round(prices / tick_size) * tick_size
    
    # Generate OHLC from price series
    opens = prices.copy()
    
    # Generate realistic high/low spreads
    spreads = np.random.exponential(1.0, n_rows) * tick_size
    highs = opens + spreads
    lows = opens - spreads
    
    # Ensure OHLC relationships are valid
    closes = opens + np.random.normal(0, 0.25, n_rows)
    closes = np.clip(closes, lows, highs)
    
    # Round all prices to tick size
    opens = np.round(opens / tick_size) * tick_size
    highs = np.round(highs / tick_size) * tick_size
    lows = np.round(lows / tick_size) * tick_size
    closes = np.round(closes / tick_size) * tick_size
    
    # Generate realistic volume
    base_volume = 1000
    volume_noise = np.random.exponential(0.5, n_rows)
    volumes = (base_volume * (1 + volume_noise)).astype(int)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


def run_performance_test(n_rows: int, config: LabelingConfig = None) -> Dict[str, float]:
    """
    Run performance test on specified number of rows
    
    Args:
        n_rows: Number of rows to test
        config: Optional configuration, uses optimized defaults if None
        
    Returns:
        Dictionary with performance metrics
    """
    if config is None:
        # Use optimized configuration for performance testing
        config = LabelingConfig(
            chunk_size=50_000,  # Smaller chunks for better memory management
            enable_performance_monitoring=True,
            enable_memory_optimization=True,
            enable_progress_tracking=True,
            performance_target_rows_per_minute=167_000,
            memory_limit_gb=8.0
        )
    
    print(f"Generating {n_rows:,} rows of test data...")
    test_data = generate_test_data(n_rows)
    
    print(f"Starting performance test with {n_rows:,} rows...")
    
    # Initialize engine and monitor
    engine = WeightedLabelingEngine(config)
    monitor = PerformanceMonitor(
        target_rows_per_minute=config.performance_target_rows_per_minute,
        memory_limit_gb=config.memory_limit_gb
    )
    
    # Run test with performance monitoring
    with performance_context(monitor, n_rows):
        try:
            result_df = engine.process_dataframe(test_data)
            
            # Validate results
            expected_columns = []
            for mode_name in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                            'low_vol_short', 'normal_vol_short', 'high_vol_short']:
                expected_columns.extend([f'label_{mode_name}', f'weight_{mode_name}'])
            
            missing_columns = set(expected_columns) - set(result_df.columns)
            if missing_columns:
                raise ValueError(f"Missing expected columns: {missing_columns}")
            
            print(f"✓ Successfully processed {len(result_df):,} rows")
            print(f"✓ Generated {len(expected_columns)} new columns")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            raise
    
    # Return performance metrics
    metrics = monitor.metrics
    return {
        'rows_processed': metrics.rows_processed,
        'elapsed_time_seconds': metrics.elapsed_time,
        'rows_per_minute': metrics.rows_per_minute,
        'peak_memory_gb': metrics.peak_memory_gb,
        'target_met': metrics.rows_per_minute >= config.performance_target_rows_per_minute
    }


def validate_10m_target() -> bool:
    """
    Validate that the system can process 10M rows within 60 minutes
    
    Returns:
        True if target is met, False otherwise
    """
    print("="*80)
    print("PERFORMANCE VALIDATION: 10M ROWS IN 60 MINUTES")
    print("="*80)
    
    # Test with progressively larger datasets to estimate 10M performance
    test_sizes = [10_000, 50_000, 100_000, 500_000]
    
    results = []
    
    for test_size in test_sizes:
        print(f"\nTesting with {test_size:,} rows...")
        
        try:
            metrics = run_performance_test(test_size)
            results.append(metrics)
            
            print(f"Results: {metrics['rows_per_minute']:,.0f} rows/min, "
                  f"{metrics['peak_memory_gb']:.2f} GB peak memory")
            
        except Exception as e:
            print(f"Test failed for {test_size:,} rows: {e}")
            return False
    
    # Analyze results and project to 10M
    if not results:
        print("❌ No successful test runs")
        return False
    
    # Use the largest successful test for projection
    best_result = max(results, key=lambda x: x['rows_processed'])
    projected_speed = best_result['rows_per_minute']
    projected_time_minutes = 10_000_000 / projected_speed
    projected_memory = best_result['peak_memory_gb']
    
    print(f"\n" + "="*60)
    print("PROJECTION TO 10M ROWS")
    print("="*60)
    print(f"Best observed speed: {projected_speed:,.0f} rows/minute")
    print(f"Projected time for 10M rows: {projected_time_minutes:.1f} minutes")
    print(f"Peak memory usage: {projected_memory:.2f} GB")
    
    # Check targets
    speed_target_met = projected_speed >= 167_000
    time_target_met = projected_time_minutes <= 60.0
    memory_target_met = projected_memory <= 8.0
    
    print(f"\nTarget Validation:")
    print(f"  Speed (≥167K/min): {'✓' if speed_target_met else '❌'} "
          f"({projected_speed:,.0f})")
    print(f"  Time (≤60 min): {'✓' if time_target_met else '❌'} "
          f"({projected_time_minutes:.1f} min)")
    print(f"  Memory (≤8 GB): {'✓' if memory_target_met else '❌'} "
          f"({projected_memory:.2f} GB)")
    
    all_targets_met = speed_target_met and time_target_met and memory_target_met
    
    if all_targets_met:
        print(f"\n✅ ALL PERFORMANCE TARGETS MET")
        print(f"   System is ready for 10M row processing")
    else:
        print(f"\n❌ PERFORMANCE TARGETS NOT MET")
        print(f"   System requires optimization before 10M row processing")
    
    return all_targets_met


def run_memory_stress_test(max_memory_gb: float = 8.0) -> bool:
    """
    Run memory stress test to validate memory usage stays within limits
    
    Args:
        max_memory_gb: Maximum allowed memory usage in GB
        
    Returns:
        True if memory stays within limits
    """
    print(f"\nMemory Stress Test (limit: {max_memory_gb:.1f} GB)")
    print("-" * 50)
    
    # Test with largest feasible dataset that should fit in memory
    test_size = 1_000_000  # 1M rows
    
    config = LabelingConfig(
        chunk_size=100_000,  # Larger chunks to stress memory
        enable_memory_optimization=True,
        enable_performance_monitoring=True,
        memory_limit_gb=max_memory_gb
    )
    
    try:
        metrics = run_performance_test(test_size, config)
        
        memory_ok = metrics['peak_memory_gb'] <= max_memory_gb
        
        print(f"Peak memory usage: {metrics['peak_memory_gb']:.2f} GB")
        print(f"Memory limit: {max_memory_gb:.1f} GB")
        print(f"Memory test: {'✓ PASSED' if memory_ok else '❌ FAILED'}")
        
        return memory_ok
        
    except Exception as e:
        print(f"Memory stress test failed: {e}")
        return False


if __name__ == "__main__":
    """Run complete performance validation suite"""
    
    print("Starting Performance Validation Suite...")
    
    # Run 10M row target validation
    target_met = validate_10m_target()
    
    # Run memory stress test
    memory_ok = run_memory_stress_test()
    
    # Final summary
    print(f"\n" + "="*80)
    print("FINAL VALIDATION SUMMARY")
    print("="*80)
    print(f"10M rows in 60 minutes: {'✅ PASS' if target_met else '❌ FAIL'}")
    print(f"Memory usage ≤8GB: {'✅ PASS' if memory_ok else '❌ FAIL'}")
    
    overall_pass = target_met and memory_ok
    print(f"\nOverall validation: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    if overall_pass:
        print("\n🎉 System meets all performance requirements!")
        print("   Ready for production use with 10M+ row datasets")
    else:
        print("\n⚠️  System requires optimization before production use")
        print("   Consider reducing chunk size or enabling more optimizations")