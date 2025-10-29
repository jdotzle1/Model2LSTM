# Weighted Labeling System - Configuration Guide

## Overview

The weighted labeling system provides extensive configuration options to optimize performance, memory usage, and processing behavior for different environments and dataset sizes. This guide covers all configuration parameters and provides recommendations for common scenarios.

## Configuration Class

```python
from project.data_pipeline.weighted_labeling import LabelingConfig

@dataclass
class LabelingConfig:
    # Processing Configuration
    chunk_size: int = 100_000
    timeout_seconds: int = 900
    
    # Performance Configuration
    performance_target_rows_per_minute: int = 167_000
    memory_limit_gb: float = 8.0
    
    # Feature Flags
    enable_parallel_processing: bool = True
    enable_progress_tracking: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    
    # Progress Configuration
    progress_update_interval: int = 10_000
    
    # Weight Calculation Parameters
    decay_rate: float = 0.05
    tick_size: float = 0.25
```

## Configuration Parameters

### Processing Configuration

#### `chunk_size: int = 100_000`

Controls how many rows are processed at once for large datasets.

**Impact:**
- **Memory Usage**: Larger chunks use more memory but may be more efficient
- **Processing Speed**: Optimal chunk size depends on available memory
- **Progress Granularity**: Smaller chunks provide more frequent progress updates

**Recommendations:**
```python
# High-memory systems (16GB+ RAM)
config = LabelingConfig(chunk_size=200_000)

# Standard systems (8GB RAM)
config = LabelingConfig(chunk_size=100_000)

# Low-memory systems (4GB RAM)
config = LabelingConfig(chunk_size=50_000)

# Development/testing
config = LabelingConfig(chunk_size=10_000)
```

#### `timeout_seconds: int = 900`

Maximum time (in seconds) to look forward for target/stop hits.

**Default**: 900 seconds (15 minutes)

**Impact:**
- **Label Accuracy**: Longer timeout captures more potential winners
- **Processing Speed**: Longer timeout increases computation time
- **Memory Usage**: Longer lookforward requires more data in memory

**Recommendations:**
```python
# Standard trading (15 minutes)
config = LabelingConfig(timeout_seconds=900)

# Faster trading (10 minutes)
config = LabelingConfig(timeout_seconds=600)

# Scalping (5 minutes)
config = LabelingConfig(timeout_seconds=300)

# Development/testing (2 minutes)
config = LabelingConfig(timeout_seconds=120)
```

### Performance Configuration

#### `performance_target_rows_per_minute: int = 167_000`

Target processing speed for performance validation.

**Default**: 167,000 rows/minute (10M rows in 60 minutes)

**Usage:**
```python
# Strict performance requirements
config = LabelingConfig(performance_target_rows_per_minute=200_000)

# Relaxed performance requirements
config = LabelingConfig(performance_target_rows_per_minute=100_000)

# Disable performance validation
config = LabelingConfig(enable_performance_monitoring=False)
```

#### `memory_limit_gb: float = 8.0`

Maximum memory usage limit in GB.

**Impact:**
- **Processing Strategy**: Determines chunk size and optimization level
- **Error Prevention**: Prevents out-of-memory errors
- **Performance Monitoring**: Triggers warnings when limit approached

**Recommendations:**
```python
# High-memory systems
config = LabelingConfig(memory_limit_gb=16.0)

# Standard systems
config = LabelingConfig(memory_limit_gb=8.0)

# Low-memory systems
config = LabelingConfig(memory_limit_gb=4.0)

# Cloud instances (adjust based on instance type)
config = LabelingConfig(memory_limit_gb=12.0)  # c5.2xlarge
```

### Feature Flags

#### `enable_parallel_processing: bool = True`

Enables parallel processing of multiple trading modes.

**Benefits:**
- Faster processing on multi-core systems
- Better CPU utilization

**Considerations:**
- May increase memory usage
- Can complicate debugging

```python
# Production (recommended)
config = LabelingConfig(enable_parallel_processing=True)

# Development/debugging
config = LabelingConfig(enable_parallel_processing=False)
```

#### `enable_progress_tracking: bool = True`

Shows detailed progress information during processing.

**Benefits:**
- Visibility into processing status
- Helps identify bottlenecks
- Useful for long-running operations

**Considerations:**
- Slight performance overhead
- May produce verbose output

```python
# Interactive use
config = LabelingConfig(enable_progress_tracking=True)

# Batch processing/logging
config = LabelingConfig(enable_progress_tracking=False)
```

#### `enable_performance_monitoring: bool = True`

Tracks performance metrics and validates against targets.

**Benefits:**
- Automatic performance validation
- Detailed timing and memory metrics
- Helps optimize configuration

**Considerations:**
- Small performance overhead
- May raise PerformanceError if targets not met

```python
# Production monitoring
config = LabelingConfig(enable_performance_monitoring=True)

# Skip validation for small datasets
config = LabelingConfig(enable_performance_monitoring=False)
```

#### `enable_memory_optimization: bool = True`

Enables memory optimization techniques.

**Optimizations:**
- Vectorized numpy calculations
- Garbage collection management
- Efficient data structures

**Recommendations:**
```python
# Always enable for production
config = LabelingConfig(enable_memory_optimization=True)

# Disable only for debugging
config = LabelingConfig(enable_memory_optimization=False)
```

### Progress Configuration

#### `progress_update_interval: int = 10_000`

How often to update progress (in rows processed).

**Impact:**
- **Feedback Frequency**: More frequent updates provide better visibility
- **Performance**: Very frequent updates can slow processing
- **Memory Cleanup**: Triggers garbage collection at intervals

```python
# Frequent updates (good for development)
config = LabelingConfig(progress_update_interval=5_000)

# Standard updates
config = LabelingConfig(progress_update_interval=10_000)

# Infrequent updates (good for batch processing)
config = LabelingConfig(progress_update_interval=50_000)
```

### Weight Calculation Parameters

#### `decay_rate: float = 0.05`

Monthly decay rate for time-based weight calculation.

**Formula**: `time_decay = exp(-decay_rate × months_ago)`

**Impact:**
- **Higher values**: More aggressive decay, recent data heavily favored
- **Lower values**: Gentler decay, historical data retains more weight

```python
# Aggressive decay (favor very recent data)
config = LabelingConfig(decay_rate=0.10)

# Standard decay
config = LabelingConfig(decay_rate=0.05)

# Gentle decay (historical data important)
config = LabelingConfig(decay_rate=0.02)

# No decay (all data equal weight by time)
config = LabelingConfig(decay_rate=0.0)
```

#### `tick_size: float = 0.25`

ES futures tick size in points.

**Default**: 0.25 (standard ES tick size)

**Note**: Should not be changed unless processing different instruments.

## Configuration Scenarios

### Development and Testing

```python
# Optimized for development work
dev_config = LabelingConfig(
    chunk_size=10_000,                    # Small chunks for quick iteration
    enable_progress_tracking=True,         # See detailed progress
    enable_performance_monitoring=False,   # Skip performance validation
    enable_parallel_processing=False,      # Easier debugging
    progress_update_interval=2_000,        # Frequent updates
    timeout_seconds=300                    # Shorter timeout for speed
)
```

### Production Processing

```python
# Optimized for production performance
prod_config = LabelingConfig(
    chunk_size=200_000,                   # Large chunks for efficiency
    enable_performance_monitoring=True,    # Validate performance
    enable_memory_optimization=True,       # All optimizations
    enable_parallel_processing=True,       # Use all cores
    memory_limit_gb=12.0,                 # Higher limit for production
    performance_target_rows_per_minute=200_000  # Strict target
)
```

### Memory-Constrained Environments

```python
# Optimized for low-memory systems
low_mem_config = LabelingConfig(
    chunk_size=25_000,                    # Small chunks
    memory_limit_gb=3.0,                  # Conservative limit
    enable_memory_optimization=True,       # Essential optimizations
    enable_parallel_processing=False,      # Reduce memory overhead
    progress_update_interval=5_000         # Frequent cleanup
)
```

### Cloud/EC2 Processing

```python
# Optimized for EC2 c5.2xlarge instance
ec2_config = LabelingConfig(
    chunk_size=150_000,                   # Balanced for 8 vCPUs
    memory_limit_gb=14.0,                 # Leave headroom from 16GB
    enable_performance_monitoring=True,    # Monitor cloud performance
    enable_parallel_processing=True,       # Use all vCPUs
    performance_target_rows_per_minute=250_000  # Higher target on EC2
)
```

### Batch Processing

```python
# Optimized for unattended batch processing
batch_config = LabelingConfig(
    chunk_size=100_000,                   # Standard chunks
    enable_progress_tracking=False,        # Reduce output
    enable_performance_monitoring=True,    # Validate results
    enable_memory_optimization=True,       # Prevent memory issues
    progress_update_interval=50_000        # Infrequent updates
)
```

## Dynamic Configuration

### Auto-sizing Based on Dataset

```python
def get_optimal_config(dataset_size: int, available_memory_gb: float) -> LabelingConfig:
    """
    Generate optimal configuration based on dataset characteristics
    
    Args:
        dataset_size: Number of rows in dataset
        available_memory_gb: Available system memory
        
    Returns:
        Optimized LabelingConfig
    """
    # Determine chunk size based on memory and dataset size
    if available_memory_gb >= 16:
        chunk_size = min(200_000, dataset_size // 10)
    elif available_memory_gb >= 8:
        chunk_size = min(100_000, dataset_size // 20)
    else:
        chunk_size = min(50_000, dataset_size // 50)
    
    # Adjust performance monitoring for small datasets
    enable_perf_monitoring = dataset_size >= 10_000
    
    # Adjust progress tracking based on dataset size
    if dataset_size < 10_000:
        progress_interval = 1_000
    elif dataset_size < 100_000:
        progress_interval = 5_000
    else:
        progress_interval = 10_000
    
    return LabelingConfig(
        chunk_size=chunk_size,
        memory_limit_gb=available_memory_gb * 0.8,  # Leave 20% headroom
        enable_performance_monitoring=enable_perf_monitoring,
        progress_update_interval=progress_interval,
        enable_parallel_processing=available_memory_gb >= 4.0
    )

# Usage
import psutil
available_memory = psutil.virtual_memory().available / (1024**3)  # GB
config = get_optimal_config(len(df), available_memory)
```

### Environment-Based Configuration

```python
import os

def get_environment_config() -> LabelingConfig:
    """Get configuration based on environment variables"""
    
    # Check if running in development
    is_development = os.getenv('ENVIRONMENT', 'production') == 'development'
    
    # Check available memory
    memory_limit = float(os.getenv('MEMORY_LIMIT_GB', '8.0'))
    
    # Check performance requirements
    performance_target = int(os.getenv('PERFORMANCE_TARGET', '167000'))
    
    if is_development:
        return LabelingConfig(
            chunk_size=10_000,
            enable_performance_monitoring=False,
            enable_progress_tracking=True,
            memory_limit_gb=memory_limit
        )
    else:
        return LabelingConfig(
            chunk_size=100_000,
            enable_performance_monitoring=True,
            enable_progress_tracking=False,
            memory_limit_gb=memory_limit,
            performance_target_rows_per_minute=performance_target
        )
```

## Configuration Validation

### Validate Configuration Before Processing

```python
def validate_config(config: LabelingConfig, dataset_size: int) -> List[str]:
    """
    Validate configuration and return list of warnings/recommendations
    
    Args:
        config: Configuration to validate
        dataset_size: Size of dataset to process
        
    Returns:
        List of validation messages
    """
    warnings = []
    
    # Check chunk size vs dataset size
    if config.chunk_size > dataset_size:
        warnings.append(f"Chunk size ({config.chunk_size:,}) larger than dataset ({dataset_size:,})")
    
    # Check memory vs chunk size
    estimated_memory_gb = (config.chunk_size * 50) / (1024**3)  # Rough estimate
    if estimated_memory_gb > config.memory_limit_gb:
        warnings.append(f"Chunk size may exceed memory limit: {estimated_memory_gb:.1f}GB > {config.memory_limit_gb:.1f}GB")
    
    # Check timeout vs performance
    if config.timeout_seconds > 1800:  # 30 minutes
        warnings.append(f"Long timeout ({config.timeout_seconds}s) may slow processing significantly")
    
    # Check performance monitoring for small datasets
    if config.enable_performance_monitoring and dataset_size < 10_000:
        warnings.append("Performance monitoring overhead may dominate for small datasets")
    
    return warnings

# Usage
config = LabelingConfig(chunk_size=500_000, memory_limit_gb=4.0)
warnings = validate_config(config, len(df))

if warnings:
    print("Configuration warnings:")
    for warning in warnings:
        print(f"  ⚠ {warning}")
```

## Performance Tuning

### Iterative Optimization

```python
def find_optimal_chunk_size(df: pd.DataFrame, 
                           memory_limit_gb: float = 8.0) -> int:
    """
    Find optimal chunk size through testing
    
    Args:
        df: Sample of your dataset (e.g., first 10K rows)
        memory_limit_gb: Memory constraint
        
    Returns:
        Optimal chunk size
    """
    test_sizes = [25_000, 50_000, 100_000, 200_000]
    results = []
    
    for chunk_size in test_sizes:
        config = LabelingConfig(
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            enable_performance_monitoring=True,
            enable_progress_tracking=False
        )
        
        try:
            engine = WeightedLabelingEngine(config)
            df_labeled = engine.process_dataframe(df)
            
            if engine.performance_monitor:
                metrics = engine.performance_monitor.metrics
                results.append({
                    'chunk_size': chunk_size,
                    'speed': metrics.rows_per_minute,
                    'memory': metrics.peak_memory_gb,
                    'success': True
                })
        except Exception as e:
            results.append({
                'chunk_size': chunk_size,
                'error': str(e),
                'success': False
            })
    
    # Find best performing configuration
    successful = [r for r in results if r['success']]
    if successful:
        best = max(successful, key=lambda x: x['speed'])
        return best['chunk_size']
    else:
        return 25_000  # Conservative fallback

# Usage
optimal_chunk = find_optimal_chunk_size(df.head(10_000))
config = LabelingConfig(chunk_size=optimal_chunk)
```

This configuration guide provides comprehensive coverage of all configuration options with practical examples for different use cases and environments.