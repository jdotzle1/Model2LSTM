#!/usr/bin/env python3
"""
Basic Usage Example for Weighted Labeling System

This example demonstrates how to use the weighted labeling system to process
ES futures data and generate labels and weights for XGBoost model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import sys
import os

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from project.data_pipeline.weighted_labeling import (
    process_weighted_labeling, 
    WeightedLabelingEngine, 
    LabelingConfig,
    TRADING_MODES
)


def create_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """
    Create synthetic ES futures data for demonstration
    
    Args:
        n_rows: Number of 1-second bars to generate
        
    Returns:
        DataFrame with OHLCV data suitable for labeling
    """
    print(f"Creating {n_rows:,} rows of synthetic ES futures data...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate RTH timestamps (07:30-15:00 CT)
    start_time = pd.Timestamp('2024-01-15 07:30:00', tz='UTC')
    timestamps = pd.date_range(start_time, periods=n_rows, freq='1s')
    
    # Generate realistic price movement
    base_price = 4500.0
    price_changes = np.random.normal(0, 0.5, n_rows)  # Small random changes
    price_changes = np.cumsum(price_changes)  # Cumulative for trending
    close_prices = base_price + price_changes
    
    # Generate OHLC with realistic relationships
    open_prices = close_prices + np.random.normal(0, 0.2, n_rows)
    high_prices = np.maximum(open_prices, close_prices) + np.random.exponential(0.3, n_rows)
    low_prices = np.minimum(open_prices, close_prices) - np.random.exponential(0.3, n_rows)
    
    # Round to ES tick size (0.25)
    for prices in [open_prices, high_prices, low_prices, close_prices]:
        prices[:] = np.round(prices / 0.25) * 0.25
    
    # Generate volume
    volume = np.random.randint(100, 2000, n_rows)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    print(f"Generated data from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    return df


def basic_usage_example():
    """Demonstrate basic usage of the weighted labeling system"""
    
    print("=" * 80)
    print("BASIC USAGE EXAMPLE - WEIGHTED LABELING SYSTEM")
    print("=" * 80)
    
    # 1. Create sample data
    df = create_sample_data(1000)
    
    # 2. Process with default configuration
    print("\n1. Processing with default configuration...")
    df_labeled = process_weighted_labeling(df)
    
    print(f"✓ Processing complete!")
    print(f"  Original columns: {len(df.columns)}")
    print(f"  New columns: {len(df_labeled.columns) - len(df.columns)}")
    print(f"  Total columns: {len(df_labeled.columns)}")
    
    # 3. Examine the results
    print("\n2. Examining results...")
    
    # Show new columns
    new_columns = [col for col in df_labeled.columns if col not in df.columns]
    print(f"New columns added: {new_columns}")
    
    # Show statistics for each mode
    print("\n3. Mode Statistics:")
    for mode_name, mode in TRADING_MODES.items():
        label_col = mode.label_column
        weight_col = mode.weight_column
        
        labels = df_labeled[label_col]
        weights = df_labeled[weight_col]
        
        win_rate = labels.mean()
        total_winners = labels.sum()
        avg_weight = weights.mean()
        weight_range = (weights.min(), weights.max())
        
        print(f"  {mode_name}:")
        print(f"    Win rate: {win_rate:.1%} ({total_winners} winners)")
        print(f"    Avg weight: {avg_weight:.3f}")
        print(f"    Weight range: {weight_range[0]:.3f} - {weight_range[1]:.3f}")
    
    return df_labeled


def advanced_configuration_example():
    """Demonstrate advanced configuration options"""
    
    print("\n" + "=" * 80)
    print("ADVANCED CONFIGURATION EXAMPLE")
    print("=" * 80)
    
    # Create larger dataset
    df = create_sample_data(5000)
    
    # Custom configuration for performance and monitoring
    config = LabelingConfig(
        chunk_size=2000,                      # Process in 2K row chunks
        enable_performance_monitoring=True,    # Track performance metrics
        enable_progress_tracking=True,         # Show detailed progress
        enable_memory_optimization=True,       # Use optimizations
        memory_limit_gb=4.0,                  # Conservative memory limit
        progress_update_interval=1000          # Update every 1K rows
    )
    
    print(f"Processing {len(df):,} rows with custom configuration...")
    print(f"Configuration:")
    print(f"  Chunk size: {config.chunk_size:,}")
    print(f"  Memory limit: {config.memory_limit_gb} GB")
    print(f"  Performance monitoring: {config.enable_performance_monitoring}")
    
    # Process with custom configuration
    engine = WeightedLabelingEngine(config)
    df_labeled = engine.process_dataframe(df)
    
    # Show performance metrics if available
    if engine.performance_monitor:
        metrics = engine.performance_monitor.metrics
        print(f"\nPerformance Results:")
        print(f"  Processing speed: {metrics.rows_per_minute:,.0f} rows/minute")
        print(f"  Peak memory: {metrics.peak_memory_gb:.2f} GB")
        print(f"  Total time: {metrics.elapsed_time:.1f} seconds")
    
    return df_labeled


def validation_example(df_labeled: pd.DataFrame):
    """Demonstrate validation and quality assurance"""
    
    print("\n" + "=" * 80)
    print("VALIDATION AND QUALITY ASSURANCE EXAMPLE")
    print("=" * 80)
    
    from project.data_pipeline.validation_utils import run_comprehensive_validation
    
    print("Running comprehensive validation suite...")
    
    # Run all validations
    validation_results = run_comprehensive_validation(df_labeled, print_reports=False)
    
    # Show summary
    overall = validation_results['overall_validation']
    summary = overall['summary']
    
    print(f"\nValidation Summary:")
    print(f"  Overall status: {'✅ PASSED' if overall['passed'] else '❌ FAILED'}")
    print(f"  Label validation: {'✅' if summary['label_validation_passed'] else '❌'}")
    print(f"  Weight validation: {'✅' if summary['weight_validation_passed'] else '❌'}")
    print(f"  Data quality: {'✅' if summary['data_quality_passed'] else '❌'}")
    print(f"  XGBoost ready: {'✅' if summary['xgboost_ready'] else '❌'}")
    
    # Show detailed statistics for one mode
    label_results = validation_results['label_distributions']
    weight_results = validation_results['weight_distributions']
    
    print(f"\nDetailed Statistics (Low Vol Long Mode):")
    mode_stats = label_results['low_vol_long']
    weight_stats = weight_results['low_vol_long']
    
    print(f"  Total samples: {mode_stats['total_samples']:,}")
    print(f"  Winners: {mode_stats['winners']:,} ({mode_stats['win_rate_percentage']:.1f}%)")
    print(f"  Losers: {mode_stats['losers']:,}")
    print(f"  Average weight: {weight_stats['mean_weight']:.3f}")
    print(f"  Weight std dev: {weight_stats['std_weight']:.3f}")
    
    return validation_results


def xgboost_preparation_example(df_labeled: pd.DataFrame):
    """Demonstrate preparing data for XGBoost training"""
    
    print("\n" + "=" * 80)
    print("XGBOOST PREPARATION EXAMPLE")
    print("=" * 80)
    
    # For this example, we'll add some mock features
    print("Adding mock features for demonstration...")
    
    # Add some simple technical indicators as features
    df_with_features = df_labeled.copy()
    
    # Simple moving averages
    df_with_features['sma_5'] = df_with_features['close'].rolling(5).mean()
    df_with_features['sma_20'] = df_with_features['close'].rolling(20).mean()
    
    # Price ratios
    df_with_features['close_to_sma5'] = df_with_features['close'] / df_with_features['sma_5']
    df_with_features['close_to_sma20'] = df_with_features['close'] / df_with_features['sma_20']
    
    # Volume features
    df_with_features['volume_ma'] = df_with_features['volume'].rolling(10).mean()
    df_with_features['volume_ratio'] = df_with_features['volume'] / df_with_features['volume_ma']
    
    # Remove NaN values from rolling calculations
    df_with_features = df_with_features.dropna()
    
    print(f"Dataset with features: {len(df_with_features):,} rows, {len(df_with_features.columns)} columns")
    
    # Prepare data for XGBoost training (example for low_vol_long mode)
    mode_name = 'low_vol_long'
    
    # Define feature columns (exclude OHLCV, timestamp, labels, weights)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    exclude_cols.extend([col for col in df_with_features.columns if col.startswith(('label_', 'weight_'))])
    
    feature_cols = [col for col in df_with_features.columns if col not in exclude_cols]
    
    # Extract features, labels, and weights for the specific mode
    X = df_with_features[feature_cols]
    y = df_with_features[f'label_{mode_name}']
    sample_weights = df_with_features[f'weight_{mode_name}']
    
    print(f"\nXGBoost Data Preparation for {mode_name}:")
    print(f"  Features: {X.shape[1]} columns, {X.shape[0]} samples")
    print(f"  Feature columns: {feature_cols}")
    print(f"  Labels: {y.shape[0]} samples, win rate: {y.mean():.1%}")
    print(f"  Sample weights: range {sample_weights.min():.3f} - {sample_weights.max():.3f}")
    
    # Show sample of the prepared data
    print(f"\nSample of prepared data:")
    sample_data = pd.DataFrame({
        'features_shape': [X.shape],
        'labels_sum': [y.sum()],
        'weights_mean': [sample_weights.mean()],
        'ready_for_training': [True]
    })
    print(sample_data.to_string(index=False))
    
    return X, y, sample_weights


def performance_benchmarking_example():
    """Demonstrate performance benchmarking"""
    
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKING EXAMPLE")
    print("=" * 80)
    
    # Test different dataset sizes
    sizes = [1000, 5000, 10000]
    results = []
    
    for size in sizes:
        print(f"\nBenchmarking {size:,} rows...")
        
        # Create test data
        df = create_sample_data(size)
        
        # Configure for performance monitoring
        config = LabelingConfig(
            enable_performance_monitoring=True,
            enable_progress_tracking=False,  # Reduce output for benchmarking
            chunk_size=min(5000, size)
        )
        
        # Process and measure
        engine = WeightedLabelingEngine(config)
        df_labeled = engine.process_dataframe(df)
        
        # Collect metrics
        if engine.performance_monitor:
            metrics = engine.performance_monitor.metrics
            results.append({
                'dataset_size': size,
                'processing_time': metrics.elapsed_time,
                'rows_per_minute': metrics.rows_per_minute,
                'peak_memory_gb': metrics.peak_memory_gb,
                'target_met': metrics.rows_per_minute >= 167_000
            })
    
    # Show benchmark results
    print(f"\nBenchmark Results:")
    print(f"{'Size':<10} {'Time (s)':<10} {'Speed (rows/min)':<15} {'Memory (GB)':<12} {'Target Met':<10}")
    print("-" * 65)
    
    for result in results:
        target_icon = "✅" if result['target_met'] else "❌"
        print(f"{result['dataset_size']:<10,} "
              f"{result['processing_time']:<10.1f} "
              f"{result['rows_per_minute']:<15,.0f} "
              f"{result['peak_memory_gb']:<12.2f} "
              f"{target_icon}")
    
    # Projection for 10M rows
    if results:
        avg_speed = sum(r['rows_per_minute'] for r in results) / len(results)
        projected_time_hours = 10_000_000 / (avg_speed * 60)
        
        print(f"\nProjection for 10M rows:")
        print(f"  Average speed: {avg_speed:,.0f} rows/minute")
        print(f"  Projected time: {projected_time_hours:.1f} hours")
        print(f"  Target (60 min): {'✅ Met' if projected_time_hours <= 1.0 else '❌ Exceeded'}")


def main():
    """Run all examples"""
    
    print("WEIGHTED LABELING SYSTEM - COMPREHENSIVE USAGE EXAMPLES")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Basic usage
        df_labeled = basic_usage_example()
        
        # 2. Advanced configuration
        df_advanced = advanced_configuration_example()
        
        # 3. Validation
        validation_results = validation_example(df_labeled)
        
        # 4. XGBoost preparation
        X, y, weights = xgboost_preparation_example(df_labeled)
        
        # 5. Performance benchmarking
        performance_benchmarking_example()
        
        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Use your own ES futures data instead of synthetic data")
        print("2. Add proper feature engineering before XGBoost training")
        print("3. Train separate XGBoost models for each volatility mode")
        print("4. Validate model performance on out-of-sample data")
        print("5. Deploy for real-time inference")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())