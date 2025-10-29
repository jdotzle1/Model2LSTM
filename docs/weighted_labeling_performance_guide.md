# Weighted Labeling System - Performance Tuning Guide

## Performance Requirements

The weighted labeling system is designed to meet these performance targets:

- **Speed**: Process 10M rows within 60 minutes (167,000 rows/minute)
- **Memory**: Use less than 8GB RAM during processing
- **Scalability**: Handle datasets up to 15 years of 1-second ES futures data

## Performance Monitoring

### Built-in Performance Tracking

```python
from project.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig

# Enable comprehensive performance monitoring
config = LabelingConfig(
    enable_performance_monitoring=True,
    enable_progress_tracking=True,
    memory_limit_gb=8.0,
    performance_target_rows_per_minute=167_000
)

engine = WeightedLabelingEngine(config)
df_labeled = engine.process_dataframe(df)

# Performance metrics are automatically validated against targets
```

### Manual Performance Monitoring

```python
from project.data_pipeline.performance_monitor import PerformanceMonitor, performance_context

# Create performance monitor
monitor = PerformanceMonitor(
    target_rows_per_minute=167_000,
    memory_limit_gb=8.0
)

# Use context manager for automatic monitoring
with performance_context(monitor, len(df)) as perf:
    df_labeled = process_weighted_labeling(df)
    
    # Monitor will automatically print performance report
```

### Performance Validation

```python
from project.data_pipeline.performance_monitor import validate_performance_requirements

# Validate that performance targets are met
try:
    validate_performance_requirements(monitor, total_rows=10_000_000)
    print("✅ Performance requirements met!")
except PerformanceError as e:
    print(f"❌ Performance issue: {e}")
```

## Optimization Strategies

### 1. Chunk Size Optimization

The chunk size significantly impacts both memory usage and processing speed:

```python
# For different dataset sizes
def get_optimal_chunk_size(dataset_size, available_memory_gb):
    """Calculate optimal chunk size based on dataset size and memory"""
    
    if dataset_size < 10_000:
        return dataset_size  # Process in single chunk
    elif dataset_size < 100_000:
        return 25_000        # Small chunks for medium datasets
    elif available_memory_gb >= 16:
        return 200_000       # Large chunks for high-memory systems
    elif available_memory_gb >= 8:
        return 100_000       # Standard chunks for normal systems
    else:
        return 50_000        # Small chunks for low-memory systems

# Example usage
chunk_size = get_optimal_chunk_size(len(df), 8.0)
config = LabelingConfig(chunk_size=chunk_size)
```

### 2. Memory Optimization

Enable all memory optimizations for large datasets:

```python
config = LabelingConfig(
    enable_memory_optimization=True,    # Enable numpy vectorization
    chunk_size=100_000,                 # Appropriate chunk size
    memory_limit_gb=8.0,               # Set memory limit
    progress_update_interval=10_000     # Regular garbage collection
)
```

### 3. Vectorization Settings

The system automatically uses vectorized calculations for datasets > 1000 rows:

```python
# Force vectorization on/off for testing
from project.data_pipeline.weighted_labeling import LabelCalculator, WeightCalculator

# Disable vectorization for debugging
label_calc = LabelCalculator(mode, enable_vectorization=False)
weight_calc = WeightCalculator(mode, enable_vectorization=False)

# Enable vectorization for performance (default for large datasets)
label_calc = LabelCalculator(mode, enable_vectorization=True)
weight_calc = WeightCalculator(mode, enable_vectorization=True)
```

## Performance Benchmarking

### Benchmark Script

```python
import time
import pandas as pd
import numpy as np
from project.data_pipeline.weighted_labeling import process_weighted_labeling, LabelingConfig

def benchmark_performance(dataset_sizes, chunk_sizes):
    """Benchmark performance across different configurations"""
    
    results = []
    
    for size in dataset_sizes:
        print(f"\nBenchmarking {size:,} rows...")
        
        # Create test data
        df = create_test_data(size)
        
        for chunk_size in chunk_sizes:
            print(f"  Testing chunk size: {chunk_size:,}")
            
            config = LabelingConfig(
                chunk_size=chunk_size,
                enable_performance_monitoring=True,
                enable_memory_optimization=True
            )
            
            # Measure performance
            start_time = time.time()
            start_memory = get_memory_usage()
            
            try:
                engine = WeightedLabelingEngine(config)
                df_labeled = engine.process_dataframe(df)
                
                end_time = time.time()
                peak_memory = engine.performance_monitor.metrics.peak_memory_gb
                
                elapsed = end_time - start_time
                rows_per_minute = (size / elapsed) * 60
                
                results.append({
                    'dataset_size': size,
                    'chunk_size': chunk_size,
                    'elapsed_time': elapsed,
                    'rows_per_minute': rows_per_minute,
                    'peak_memory_gb': peak_memory,
                    'target_met': rows_per_minute >= 167_000,
                    'memory_ok': peak_memory <= 8.0
                })
                
                print(f"    Speed: {rows_per_minute:,.0f} rows/min")
                print(f"    Memory: {peak_memory:.2f} GB")
                print(f"    Target: {'✅' if rows_per_minute >= 167_000 else '❌'}")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results.append({
                    'dataset_size': size,
                    'chunk_size': chunk_size,
                    'error': str(e)
                })
    
    return pd.DataFrame(results)

# Run benchmark
dataset_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
chunk_sizes = [25_000, 50_000, 100_000, 200_000]

benchmark_results = benchmark_performance(dataset_sizes, chunk_sizes)
print("\nBenchmark Results:")
print(benchmark_results)
```

### Performance Analysis

```python
def analyze_performance_results(results_df):
    """Analyze benchmark results to find optimal configurations"""
    
    # Find configurations that meet targets
    successful = results_df[
        (results_df['target_met'] == True) & 
        (results_df['memory_ok'] == True)
    ]
    
    if len(successful) == 0:
        print("❌ No configurations met performance targets")
        return
    
    # Find optimal configuration for each dataset size
    optimal_configs = []
    
    for size in results_df['dataset_size'].unique():
        size_results = successful[successful['dataset_size'] == size]
        
        if len(size_results) > 0:
            # Choose configuration with best speed
            best = size_results.loc[size_results['rows_per_minute'].idxmax()]
            optimal_configs.append({
                'dataset_size': size,
                'optimal_chunk_size': best['chunk_size'],
                'speed': best['rows_per_minute'],
                'memory': best['peak_memory_gb']
            })
    
    optimal_df = pd.DataFrame(optimal_configs)
    print("Optimal Configurations:")
    print(optimal_df)
    
    return optimal_df
```

## Hardware-Specific Optimizations

### High-Memory Systems (16GB+ RAM)

```python
# Optimize for high-memory systems
config = LabelingConfig(
    chunk_size=500_000,                 # Very large chunks
    memory_limit_gb=12.0,              # Higher memory limit
    enable_memory_optimization=True,    # Still enable optimizations
    enable_parallel_processing=True     # Parallel mode processing
)
```

### Low-Memory Systems (4GB RAM)

```python
# Optimize for memory-constrained systems
config = LabelingConfig(
    chunk_size=25_000,                  # Small chunks
    memory_limit_gb=3.0,               # Conservative memory limit
    enable_memory_optimization=True,    # Essential for low memory
    progress_update_interval=5_000,     # Frequent cleanup
    enable_parallel_processing=False    # Reduce memory overhead
)
```

### SSD vs HDD Storage

```python
# For SSD storage (fast I/O)
config = LabelingConfig(
    chunk_size=200_000,                 # Larger chunks OK
    enable_performance_monitoring=True   # Track I/O performance
)

# For HDD storage (slower I/O)
config = LabelingConfig(
    chunk_size=50_000,                  # Smaller chunks to reduce I/O
    progress_update_interval=10_000     # Less frequent updates
)
```

## Troubleshooting Performance Issues

### Slow Processing Speed

**Symptoms**: Processing speed < 167,000 rows/minute

**Solutions**:
1. Increase chunk size if memory allows
2. Enable memory optimization
3. Check for memory swapping
4. Verify data is properly sorted by timestamp

```python
# Diagnostic code
def diagnose_slow_processing(df):
    """Diagnose causes of slow processing"""
    
    print("Performance Diagnostics:")
    
    # Check data size
    print(f"Dataset size: {len(df):,} rows")
    
    # Check timestamp sorting
    is_sorted = df['timestamp'].is_monotonic_increasing
    print(f"Timestamp sorted: {is_sorted}")
    
    if not is_sorted:
        print("⚠ Data not sorted - this will slow processing")
        print("Solution: df = df.sort_values('timestamp')")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print(f"Missing values found: {missing_counts[missing_counts > 0]}")
    
    # Check memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"DataFrame memory usage: {memory_mb:.1f} MB")
    
    # Recommend chunk size
    recommended_chunk = min(100_000, len(df) // 10)
    print(f"Recommended chunk size: {recommended_chunk:,}")
```

### High Memory Usage

**Symptoms**: Memory usage > 8GB or out-of-memory errors

**Solutions**:
1. Reduce chunk size
2. Enable memory optimization
3. Process in smaller batches
4. Check for memory leaks

```python
# Memory optimization example
def process_with_memory_monitoring(df):
    """Process data with aggressive memory monitoring"""
    
    import gc
    import psutil
    
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024**3  # GB
    
    print(f"Initial memory: {get_memory_usage():.2f} GB")
    
    # Use small chunks and aggressive cleanup
    config = LabelingConfig(
        chunk_size=25_000,
        enable_memory_optimization=True,
        memory_limit_gb=6.0  # Conservative limit
    )
    
    engine = WeightedLabelingEngine(config)
    
    # Process with manual memory monitoring
    df_labeled = engine.process_dataframe(df)
    
    # Force garbage collection
    gc.collect()
    
    print(f"Final memory: {get_memory_usage():.2f} GB")
    
    return df_labeled
```

### Performance Validation Failures

**Symptoms**: PerformanceError exceptions

**Solutions**:
1. Check hardware specifications
2. Optimize configuration parameters
3. Consider distributed processing

```python
# Handle performance validation gracefully
def process_with_fallback(df):
    """Process data with fallback configurations if performance targets not met"""
    
    configs = [
        # Optimal configuration
        LabelingConfig(chunk_size=200_000, memory_limit_gb=8.0),
        
        # Conservative configuration
        LabelingConfig(chunk_size=100_000, memory_limit_gb=6.0),
        
        # Minimal configuration
        LabelingConfig(
            chunk_size=50_000, 
            memory_limit_gb=4.0,
            enable_performance_monitoring=False  # Skip validation
        )
    ]
    
    for i, config in enumerate(configs):
        try:
            print(f"Trying configuration {i+1}...")
            engine = WeightedLabelingEngine(config)
            return engine.process_dataframe(df)
            
        except PerformanceError as e:
            print(f"Configuration {i+1} failed: {e}")
            if i == len(configs) - 1:
                print("All configurations failed - processing anyway")
                # Last config with validation disabled
                return engine.process_dataframe(df)
            continue
```

## Production Deployment Considerations

### EC2 Instance Recommendations

**For 10M+ row datasets**:
- Instance type: c5.2xlarge or better (8+ vCPUs, 16+ GB RAM)
- Storage: SSD with sufficient space for input + output data
- Memory: 16GB+ recommended for optimal performance

**Configuration for EC2**:
```python
# Production EC2 configuration
config = LabelingConfig(
    chunk_size=200_000,
    memory_limit_gb=12.0,
    enable_performance_monitoring=True,
    enable_memory_optimization=True,
    performance_target_rows_per_minute=200_000  # Higher target on EC2
)
```

### Monitoring in Production

```python
# Production monitoring setup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def production_processing(df, output_path):
    """Production-ready processing with comprehensive monitoring"""
    
    config = LabelingConfig(
        enable_performance_monitoring=True,
        enable_progress_tracking=True,
        memory_limit_gb=12.0
    )
    
    try:
        logger.info(f"Starting processing of {len(df):,} rows")
        
        engine = WeightedLabelingEngine(config)
        df_labeled = engine.process_dataframe(df)
        
        # Log performance metrics
        if engine.performance_monitor:
            metrics = engine.performance_monitor.metrics
            logger.info(f"Processing completed:")
            logger.info(f"  Speed: {metrics.rows_per_minute:,.0f} rows/minute")
            logger.info(f"  Memory: {metrics.peak_memory_gb:.2f} GB peak")
            logger.info(f"  Time: {metrics.elapsed_time:.1f} seconds")
        
        # Save results
        df_labeled.to_parquet(output_path)
        logger.info(f"Results saved to {output_path}")
        
        return df_labeled
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
```

This performance guide provides comprehensive strategies for optimizing the weighted labeling system across different hardware configurations and dataset sizes.