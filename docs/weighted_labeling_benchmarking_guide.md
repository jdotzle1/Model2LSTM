# Weighted Labeling System - Performance Benchmarking Guide

## Overview

This guide provides comprehensive benchmarking tools and methodologies to measure, analyze, and optimize the performance of the weighted labeling system. It includes automated benchmarking scripts, performance analysis tools, and optimization recommendations.

## Performance Requirements

The weighted labeling system is designed to meet these performance targets:

- **Speed**: 167,000 rows/minute (10M rows in 60 minutes)
- **Memory**: Less than 8GB peak usage
- **Scalability**: Linear scaling with dataset size
- **Reliability**: Consistent performance across different data characteristics

## Benchmarking Framework

### Basic Benchmarking Script

```python
import pandas as pd
import numpy as np
import time
import psutil
from typing import Dict, List, Tuple
from dataclasses import dataclass

from project.data_pipeline.weighted_labeling import (
    WeightedLabelingEngine, 
    LabelingConfig,
    process_weighted_labeling
)

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    dataset_size: int
    chunk_size: int
    processing_time: float
    rows_per_minute: float
    peak_memory_gb: float
    target_met: bool
    memory_ok: bool
    configuration: str
    error: str = None

def create_benchmark_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Create standardized benchmark data for consistent testing
    
    Args:
        n_rows: Number of rows to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic ES futures data
    """
    np.random.seed(seed)
    
    # Generate RTH timestamps
    start_time = pd.Timestamp('2024-01-01 08:00:00', tz='UTC')
    timestamps = pd.date_range(start_time, periods=n_rows, freq='1s')
    
    # Generate realistic price data with trends and volatility
    base_price = 4500.0
    
    # Add trend component
    trend = np.linspace(0, 50, n_rows)  # 50 point trend over dataset
    
    # Add random walk
    random_walk = np.cumsum(np.random.normal(0, 0.5, n_rows))
    
    # Add intraday patterns
    time_of_day = np.arange(n_rows) % (6.5 * 3600)  # 6.5 hour trading day
    intraday_pattern = 10 * np.sin(2 * np.pi * time_of_day / (6.5 * 3600))
    
    # Combine components
    close_prices = base_price + trend + random_walk + intraday_pattern
    
    # Generate OHLC with realistic relationships
    open_prices = close_prices + np.random.normal(0, 0.2, n_rows)
    high_prices = np.maximum(open_prices, close_prices) + np.random.exponential(0.3, n_rows)
    low_prices = np.minimum(open_prices, close_prices) - np.random.exponential(0.3, n_rows)
    
    # Round to tick size
    for prices in [open_prices, high_prices, low_prices, close_prices]:
        prices[:] = np.round(prices / 0.25) * 0.25
    
    # Generate volume with realistic patterns
    base_volume = 1000
    volume_trend = np.random.exponential(base_volume, n_rows)
    volume = np.maximum(100, volume_trend).astype(int)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })

def benchmark_single_configuration(df: pd.DataFrame, 
                                 config: LabelingConfig,
                                 config_name: str) -> BenchmarkResult:
    """
    Benchmark a single configuration
    
    Args:
        df: Test dataset
        config: Configuration to test
        config_name: Descriptive name for configuration
        
    Returns:
        BenchmarkResult with performance metrics
    """
    try:
        # Record initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**3)
        
        # Process with timing
        start_time = time.time()
        engine = WeightedLabelingEngine(config)
        df_labeled = engine.process_dataframe(df)
        end_time = time.time()
        
        # Calculate metrics
        processing_time = end_time - start_time
        rows_per_minute = (len(df) / processing_time) * 60 if processing_time > 0 else 0
        
        # Get peak memory from monitor if available
        if engine.performance_monitor:
            peak_memory_gb = engine.performance_monitor.metrics.peak_memory_gb
        else:
            peak_memory_gb = process.memory_info().rss / (1024**3)
        
        # Check targets
        target_met = rows_per_minute >= config.performance_target_rows_per_minute
        memory_ok = peak_memory_gb <= config.memory_limit_gb
        
        return BenchmarkResult(
            dataset_size=len(df),
            chunk_size=config.chunk_size,
            processing_time=processing_time,
            rows_per_minute=rows_per_minute,
            peak_memory_gb=peak_memory_gb,
            target_met=target_met,
            memory_ok=memory_ok,
            configuration=config_name
        )
        
    except Exception as e:
        return BenchmarkResult(
            dataset_size=len(df),
            chunk_size=config.chunk_size,
            processing_time=0,
            rows_per_minute=0,
            peak_memory_gb=0,
            target_met=False,
            memory_ok=False,
            configuration=config_name,
            error=str(e)
        )

def run_comprehensive_benchmark() -> List[BenchmarkResult]:
    """
    Run comprehensive benchmark across multiple configurations and dataset sizes
    
    Returns:
        List of benchmark results
    """
    print("=" * 80)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Test configurations
    configurations = {
        'default': LabelingConfig(),
        
        'high_performance': LabelingConfig(
            chunk_size=200_000,
            enable_memory_optimization=True,
            enable_parallel_processing=True,
            memory_limit_gb=12.0
        ),
        
        'memory_optimized': LabelingConfig(
            chunk_size=50_000,
            enable_memory_optimization=True,
            memory_limit_gb=4.0,
            enable_parallel_processing=False
        ),
        
        'development': LabelingConfig(
            chunk_size=25_000,
            enable_performance_monitoring=False,
            enable_progress_tracking=True,
            timeout_seconds=300
        )
    }
    
    # Test dataset sizes
    dataset_sizes = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    
    results = []
    
    for size in dataset_sizes:
        print(f"\nTesting dataset size: {size:,} rows")
        
        # Create test data
        df = create_benchmark_data(size)
        
        for config_name, config in configurations.items():
            print(f"  Testing configuration: {config_name}")
            
            result = benchmark_single_configuration(df, config, config_name)
            results.append(result)
            
            if result.error:
                print(f"    ❌ Failed: {result.error}")
            else:
                status = "✅" if result.target_met and result.memory_ok else "⚠️"
                print(f"    {status} {result.rows_per_minute:,.0f} rows/min, "
                      f"{result.peak_memory_gb:.2f} GB")
    
    return results

def analyze_benchmark_results(results: List[BenchmarkResult]) -> Dict:
    """
    Analyze benchmark results and provide insights
    
    Args:
        results: List of benchmark results
        
    Returns:
        Dictionary with analysis results
    """
    # Filter successful results
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    
    analysis = {
        'summary': {
            'total_tests': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) if results else 0
        },
        'performance_analysis': {},
        'memory_analysis': {},
        'configuration_analysis': {},
        'scaling_analysis': {},
        'recommendations': []
    }
    
    if not successful:
        analysis['recommendations'].append("All tests failed - check system requirements")
        return analysis
    
    # Performance analysis
    speeds = [r.rows_per_minute for r in successful]
    analysis['performance_analysis'] = {
        'min_speed': min(speeds),
        'max_speed': max(speeds),
        'avg_speed': sum(speeds) / len(speeds),
        'target_met_count': sum(1 for r in successful if r.target_met),
        'target_met_rate': sum(1 for r in successful if r.target_met) / len(successful)
    }
    
    # Memory analysis
    memories = [r.peak_memory_gb for r in successful]
    analysis['memory_analysis'] = {
        'min_memory': min(memories),
        'max_memory': max(memories),
        'avg_memory': sum(memories) / len(memories),
        'memory_ok_count': sum(1 for r in successful if r.memory_ok),
        'memory_ok_rate': sum(1 for r in successful if r.memory_ok) / len(successful)
    }
    
    # Configuration analysis
    config_performance = {}
    for result in successful:
        config = result.configuration
        if config not in config_performance:
            config_performance[config] = []
        config_performance[config].append(result.rows_per_minute)
    
    analysis['configuration_analysis'] = {
        config: {
            'avg_speed': sum(speeds) / len(speeds),
            'max_speed': max(speeds),
            'test_count': len(speeds)
        }
        for config, speeds in config_performance.items()
    }
    
    # Scaling analysis
    size_performance = {}
    for result in successful:
        size = result.dataset_size
        if size not in size_performance:
            size_performance[size] = []
        size_performance[size].append(result.rows_per_minute)
    
    analysis['scaling_analysis'] = {
        size: {
            'avg_speed': sum(speeds) / len(speeds),
            'best_speed': max(speeds),
            'config_count': len(speeds)
        }
        for size, speeds in size_performance.items()
    }
    
    # Generate recommendations
    perf_analysis = analysis['performance_analysis']
    mem_analysis = analysis['memory_analysis']
    
    if perf_analysis['target_met_rate'] < 0.5:
        analysis['recommendations'].append(
            f"Performance target met only {perf_analysis['target_met_rate']:.0%} of the time - "
            "consider optimizing configuration or upgrading hardware"
        )
    
    if mem_analysis['memory_ok_rate'] < 0.8:
        analysis['recommendations'].append(
            f"Memory limit exceeded in {(1-mem_analysis['memory_ok_rate']):.0%} of tests - "
            "consider reducing chunk size or increasing memory limit"
        )
    
    # Find best configuration
    best_config = max(
        analysis['configuration_analysis'].items(),
        key=lambda x: x[1]['avg_speed']
    )
    analysis['recommendations'].append(
        f"Best performing configuration: {best_config[0]} "
        f"({best_config[1]['avg_speed']:,.0f} rows/min average)"
    )
    
    return analysis

def print_benchmark_report(results: List[BenchmarkResult], analysis: Dict):
    """Print comprehensive benchmark report"""
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS REPORT")
    print("=" * 80)
    
    # Summary
    summary = analysis['summary']
    print(f"\nTest Summary:")
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    
    # Performance overview
    perf = analysis['performance_analysis']
    print(f"\nPerformance Overview:")
    print(f"  Speed range: {perf['min_speed']:,.0f} - {perf['max_speed']:,.0f} rows/min")
    print(f"  Average speed: {perf['avg_speed']:,.0f} rows/min")
    print(f"  Target met: {perf['target_met_count']}/{len([r for r in results if r.error is None])} "
          f"({perf['target_met_rate']:.1%})")
    
    # Memory overview
    mem = analysis['memory_analysis']
    print(f"\nMemory Overview:")
    print(f"  Memory range: {mem['min_memory']:.2f} - {mem['max_memory']:.2f} GB")
    print(f"  Average memory: {mem['avg_memory']:.2f} GB")
    print(f"  Memory OK: {mem['memory_ok_count']}/{len([r for r in results if r.error is None])} "
          f"({mem['memory_ok_rate']:.1%})")
    
    # Configuration comparison
    print(f"\nConfiguration Comparison:")
    config_analysis = analysis['configuration_analysis']
    for config, stats in config_analysis.items():
        print(f"  {config}:")
        print(f"    Average speed: {stats['avg_speed']:,.0f} rows/min")
        print(f"    Best speed: {stats['max_speed']:,.0f} rows/min")
        print(f"    Tests: {stats['test_count']}")
    
    # Scaling analysis
    print(f"\nScaling Analysis:")
    scaling = analysis['scaling_analysis']
    print(f"{'Dataset Size':<12} {'Avg Speed':<15} {'Best Speed':<15} {'Configs':<8}")
    print("-" * 55)
    for size in sorted(scaling.keys()):
        stats = scaling[size]
        print(f"{size:<12,} {stats['avg_speed']:<15,.0f} {stats['best_speed']:<15,.0f} {stats['config_count']:<8}")
    
    # Detailed results table
    print(f"\nDetailed Results:")
    print(f"{'Size':<8} {'Config':<15} {'Chunk':<8} {'Speed':<12} {'Memory':<8} {'Status':<8}")
    print("-" * 70)
    
    for result in sorted(results, key=lambda x: (x.dataset_size, x.configuration)):
        if result.error:
            status = "FAILED"
            speed_str = "N/A"
            memory_str = "N/A"
        else:
            status = "✅" if result.target_met and result.memory_ok else "⚠️"
            speed_str = f"{result.rows_per_minute:,.0f}"
            memory_str = f"{result.peak_memory_gb:.2f}"
        
        print(f"{result.dataset_size:<8,} {result.configuration:<15} "
              f"{result.chunk_size:<8,} {speed_str:<12} {memory_str:<8} {status:<8}")
    
    # Recommendations
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Projection for 10M rows
    if analysis['performance_analysis']['avg_speed'] > 0:
        avg_speed = analysis['performance_analysis']['avg_speed']
        projected_hours = 10_000_000 / (avg_speed * 60)
        
        print(f"\nProjection for 10M Rows:")
        print(f"  Based on average speed: {avg_speed:,.0f} rows/min")
        print(f"  Projected time: {projected_hours:.1f} hours")
        print(f"  Target (60 min): {'✅ Met' if projected_hours <= 1.0 else '❌ Exceeded'}")

def benchmark_memory_scaling():
    """Benchmark memory usage scaling with dataset size"""
    
    print("\n" + "=" * 80)
    print("MEMORY SCALING BENCHMARK")
    print("=" * 80)
    
    sizes = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    memory_results = []
    
    for size in sizes:
        print(f"Testing memory usage for {size:,} rows...")
        
        df = create_benchmark_data(size)
        
        # Test with memory monitoring
        config = LabelingConfig(
            chunk_size=min(25_000, size),
            enable_performance_monitoring=True,
            enable_memory_optimization=True
        )
        
        engine = WeightedLabelingEngine(config)
        df_labeled = engine.process_dataframe(df)
        
        if engine.performance_monitor:
            memory_gb = engine.performance_monitor.metrics.peak_memory_gb
            memory_results.append({
                'size': size,
                'memory_gb': memory_gb,
                'memory_per_row_mb': (memory_gb * 1024) / size
            })
            
            print(f"  Peak memory: {memory_gb:.2f} GB ({(memory_gb * 1024) / size:.3f} MB/row)")
    
    # Analyze memory scaling
    if len(memory_results) >= 2:
        print(f"\nMemory Scaling Analysis:")
        
        # Calculate scaling factor
        first = memory_results[0]
        last = memory_results[-1]
        
        size_ratio = last['size'] / first['size']
        memory_ratio = last['memory_gb'] / first['memory_gb']
        
        print(f"  Size increased: {size_ratio:.1f}x ({first['size']:,} → {last['size']:,})")
        print(f"  Memory increased: {memory_ratio:.1f}x ({first['memory_gb']:.2f} → {last['memory_gb']:.2f} GB)")
        print(f"  Scaling efficiency: {(size_ratio / memory_ratio):.2f} (1.0 = linear)")
        
        # Memory per row analysis
        memory_per_row = [r['memory_per_row_mb'] for r in memory_results]
        avg_memory_per_row = sum(memory_per_row) / len(memory_per_row)
        
        print(f"  Average memory per row: {avg_memory_per_row:.3f} MB")
        
        # Project memory for 10M rows
        projected_memory_gb = (avg_memory_per_row * 10_000_000) / 1024
        print(f"  Projected memory for 10M rows: {projected_memory_gb:.1f} GB")

def benchmark_chunk_size_optimization():
    """Find optimal chunk size for current system"""
    
    print("\n" + "=" * 80)
    print("CHUNK SIZE OPTIMIZATION BENCHMARK")
    print("=" * 80)
    
    # Create test dataset
    df = create_benchmark_data(50_000)
    
    # Test different chunk sizes
    chunk_sizes = [10_000, 25_000, 50_000, 100_000, 200_000]
    chunk_results = []
    
    for chunk_size in chunk_sizes:
        print(f"Testing chunk size: {chunk_size:,}")
        
        config = LabelingConfig(
            chunk_size=chunk_size,
            enable_performance_monitoring=True,
            enable_progress_tracking=False
        )
        
        try:
            engine = WeightedLabelingEngine(config)
            df_labeled = engine.process_dataframe(df)
            
            if engine.performance_monitor:
                metrics = engine.performance_monitor.metrics
                chunk_results.append({
                    'chunk_size': chunk_size,
                    'speed': metrics.rows_per_minute,
                    'memory': metrics.peak_memory_gb,
                    'time': metrics.elapsed_time,
                    'success': True
                })
                
                print(f"  Speed: {metrics.rows_per_minute:,.0f} rows/min, "
                      f"Memory: {metrics.peak_memory_gb:.2f} GB")
        
        except Exception as e:
            chunk_results.append({
                'chunk_size': chunk_size,
                'error': str(e),
                'success': False
            })
            print(f"  ❌ Failed: {e}")
    
    # Find optimal chunk size
    successful = [r for r in chunk_results if r['success']]
    
    if successful:
        # Find best speed within memory constraints
        memory_ok = [r for r in successful if r['memory'] <= 8.0]
        
        if memory_ok:
            optimal = max(memory_ok, key=lambda x: x['speed'])
            print(f"\nOptimal chunk size: {optimal['chunk_size']:,}")
            print(f"  Speed: {optimal['speed']:,.0f} rows/min")
            print(f"  Memory: {optimal['memory']:.2f} GB")
        else:
            print(f"\n⚠️ No configuration met memory constraints")
            best_speed = max(successful, key=lambda x: x['speed'])
            print(f"Best speed (ignoring memory): {best_speed['chunk_size']:,} "
                  f"({best_speed['speed']:,.0f} rows/min)")

def main():
    """Run complete benchmarking suite"""
    
    print("WEIGHTED LABELING SYSTEM - PERFORMANCE BENCHMARKING SUITE")
    print("=" * 80)
    print(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.1f} GB RAM")
    print(f"Timestamp: {pd.Timestamp.now()}")
    
    try:
        # 1. Comprehensive benchmark
        results = run_comprehensive_benchmark()
        analysis = analyze_benchmark_results(results)
        print_benchmark_report(results, analysis)
        
        # 2. Memory scaling benchmark
        benchmark_memory_scaling()
        
        # 3. Chunk size optimization
        benchmark_chunk_size_optimization()
        
        print("\n" + "=" * 80)
        print("✅ BENCHMARKING SUITE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

## Quick Benchmarking Tools

### Simple Performance Test

```python
def quick_performance_test(df: pd.DataFrame) -> Dict:
    """
    Quick performance test for immediate feedback
    
    Args:
        df: Your dataset (or sample)
        
    Returns:
        Dictionary with performance metrics
    """
    print(f"Quick performance test on {len(df):,} rows...")
    
    start_time = time.time()
    df_labeled = process_weighted_labeling(df)
    end_time = time.time()
    
    elapsed = end_time - start_time
    speed = (len(df) / elapsed) * 60 if elapsed > 0 else 0
    
    # Project to 10M rows
    projected_hours = 10_000_000 / (speed * 60) if speed > 0 else float('inf')
    
    results = {
        'dataset_size': len(df),
        'processing_time': elapsed,
        'rows_per_minute': speed,
        'projected_time_hours': projected_hours,
        'target_met': speed >= 167_000,
        'within_60min_target': projected_hours <= 1.0
    }
    
    print(f"Results:")
    print(f"  Processing time: {elapsed:.1f} seconds")
    print(f"  Speed: {speed:,.0f} rows/minute")
    print(f"  Projected time for 10M rows: {projected_hours:.1f} hours")
    print(f"  Target met: {'✅' if results['target_met'] else '❌'}")
    
    return results

# Usage
quick_results = quick_performance_test(df.head(10_000))
```

### Memory Usage Test

```python
def memory_usage_test(df: pd.DataFrame) -> Dict:
    """
    Test memory usage patterns
    
    Args:
        df: Your dataset
        
    Returns:
        Dictionary with memory metrics
    """
    import psutil
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024**3)
    
    print(f"Memory usage test on {len(df):,} rows...")
    print(f"Initial memory: {initial_memory:.2f} GB")
    
    # Process with memory monitoring
    config = LabelingConfig(enable_performance_monitoring=True)
    engine = WeightedLabelingEngine(config)
    
    df_labeled = engine.process_dataframe(df)
    
    final_memory = process.memory_info().rss / (1024**3)
    
    if engine.performance_monitor:
        peak_memory = engine.performance_monitor.metrics.peak_memory_gb
    else:
        peak_memory = final_memory
    
    results = {
        'initial_memory_gb': initial_memory,
        'peak_memory_gb': peak_memory,
        'final_memory_gb': final_memory,
        'memory_increase_gb': peak_memory - initial_memory,
        'memory_per_row_mb': ((peak_memory - initial_memory) * 1024) / len(df),
        'memory_ok': peak_memory <= 8.0
    }
    
    print(f"Results:")
    print(f"  Peak memory: {peak_memory:.2f} GB")
    print(f"  Memory increase: {results['memory_increase_gb']:.2f} GB")
    print(f"  Memory per row: {results['memory_per_row_mb']:.3f} MB")
    print(f"  Within limit: {'✅' if results['memory_ok'] else '❌'}")
    
    return results

# Usage
memory_results = memory_usage_test(df.head(25_000))
```

This benchmarking guide provides comprehensive tools for measuring and optimizing the performance of the weighted labeling system across different configurations and environments.