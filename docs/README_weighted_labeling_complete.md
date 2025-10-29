# Weighted Labeling System - Complete Documentation

## Overview

The Weighted Labeling System is a comprehensive solution for generating binary labels and training weights for XGBoost models in ES futures trading. It processes OHLCV data to create 12 columns (6 labels + 6 weights) for 6 volatility-based trading modes, designed to meet strict performance requirements of processing 10 million rows within 60 minutes.

## Quick Start

### Installation and Setup

```python
# Import the main components
from project.data_pipeline.weighted_labeling import (
    process_weighted_labeling,
    WeightedLabelingEngine, 
    LabelingConfig
)
```

### Basic Usage (30 seconds)

```python
import pandas as pd

# Load your ES futures data
df = pd.read_parquet('your_es_data.parquet')

# Process with default settings
df_labeled = process_weighted_labeling(df)

# Result: Original columns + 12 new columns (6 labels + 6 weights)
print(f"Added {len(df_labeled.columns) - len(df.columns)} new columns")

# Check results
for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long']:
    win_rate = df_labeled[f'label_{mode}'].mean()
    avg_weight = df_labeled[f'weight_{mode}'].mean()
    print(f"{mode}: {win_rate:.1%} win rate, avg weight: {avg_weight:.3f}")
```

### Validation (1 minute)

```python
from project.data_pipeline.validation_utils import run_comprehensive_validation

# Validate results
validation_results = run_comprehensive_validation(df_labeled)

if validation_results['overall_validation']['passed']:
    print("✅ Data ready for XGBoost training")
else:
    print("❌ Issues found - check validation report")
```

## System Architecture

### Trading Modes

The system generates labels and weights for 6 volatility-based trading modes:

| Mode | Direction | Stop | Target | Risk:Reward |
|------|-----------|------|--------|-------------|
| `low_vol_long` | Long | 6 ticks | 12 ticks | 1:2 |
| `normal_vol_long` | Long | 8 ticks | 16 ticks | 1:2 |
| `high_vol_long` | Long | 10 ticks | 20 ticks | 1:2 |
| `low_vol_short` | Short | 6 ticks | 12 ticks | 1:2 |
| `normal_vol_short` | Short | 8 ticks | 16 ticks | 1:2 |
| `high_vol_short` | Short | 10 ticks | 20 ticks | 1:2 |

### Label Generation Process

1. **Entry Determination**: Uses next bar's open price as entry
2. **Target/Stop Calculation**: Based on mode-specific tick distances
3. **Lookforward Analysis**: Checks up to 15 minutes for target/stop hits
4. **Win/Loss Assignment**: Binary labels (1 = target hit first, 0 = stop hit or timeout)

### Weight Calculation Components

**For Winners (label = 1):**
- **Quality Weight**: Based on Maximum Adverse Excursion (MAE)
  - Formula: `2.0 - (1.5 × mae_ratio)`, clipped to [0.5, 2.0]
  - Lower MAE = higher weight (better entry timing)

- **Velocity Weight**: Based on speed to target
  - Formula: `2.0 - (1.5 × (seconds - 300) / 600)`, clipped to [0.5, 2.0]
  - Faster to target = higher weight

- **Time Decay**: Based on data recency
  - Formula: `exp(-0.05 × months_ago)`
  - More recent data = higher weight

- **Final Weight**: `quality × velocity × time_decay`

**For Losers (label = 0):**
- **Final Weight**: `1.0 × 1.0 × time_decay` (only time decay applied)

## Performance Requirements

- **Speed**: 167,000 rows/minute (10M rows in 60 minutes)
- **Memory**: Less than 8GB peak usage
- **Scalability**: Handles datasets up to 15 years of 1-second ES data
- **Accuracy**: Tick-precise target/stop detection with contract roll handling

## Documentation Structure

### 📚 Core Documentation

1. **[Usage Guide](weighted_labeling_usage_guide.md)** - Complete usage examples and integration
2. **[API Reference](weighted_labeling_api_reference.md)** - Comprehensive API documentation
3. **[Configuration Guide](weighted_labeling_configuration_guide.md)** - All configuration options and scenarios
4. **[Performance Guide](weighted_labeling_performance_guide.md)** - Performance tuning and optimization
5. **[Troubleshooting Guide](weighted_labeling_troubleshooting.md)** - Common issues and solutions

### 🔧 Advanced Documentation

6. **[Benchmarking Guide](weighted_labeling_benchmarking_guide.md)** - Performance measurement and analysis
7. **[Requirements Specification](../specs/labeling-revision/requirements.md)** - Detailed system requirements
8. **[Design Document](../specs/labeling-revision/design.md)** - Technical architecture and design decisions

### 💡 Examples and Tools

9. **[Basic Usage Example](../examples/basic_usage_example.py)** - Runnable examples for all features
10. **[Validation Tools](validation_utils.py)** - Quality assurance and validation utilities
11. **[Performance Monitoring](performance_monitor.py)** - Performance tracking and optimization

## Common Use Cases

### Development and Testing

```python
# Quick test on small dataset
df_sample = df.head(1000)
df_labeled = process_weighted_labeling(df_sample)

# Development configuration
dev_config = LabelingConfig(
    chunk_size=10_000,
    enable_progress_tracking=True,
    enable_performance_monitoring=False,
    timeout_seconds=300  # Faster for testing
)

engine = WeightedLabelingEngine(dev_config)
df_labeled = engine.process_dataframe(df_sample)
```

### Production Processing

```python
# Production configuration for large datasets
prod_config = LabelingConfig(
    chunk_size=200_000,
    enable_performance_monitoring=True,
    enable_memory_optimization=True,
    memory_limit_gb=12.0
)

engine = WeightedLabelingEngine(prod_config)
df_labeled = engine.process_dataframe(df)

# Validate performance
if engine.performance_monitor:
    metrics = engine.performance_monitor.metrics
    print(f"Speed: {metrics.rows_per_minute:,.0f} rows/min")
    print(f"Memory: {metrics.peak_memory_gb:.2f} GB")
```

### XGBoost Integration

```python
# Prepare data for XGBoost training
def prepare_xgboost_data(df_labeled, mode_name, feature_columns):
    """Prepare data for XGBoost training for a specific mode"""
    
    X = df_labeled[feature_columns]
    y = df_labeled[f'label_{mode_name}']
    sample_weights = df_labeled[f'weight_{mode_name}']
    
    # Remove NaN values
    valid_mask = ~X.isnull().any(axis=1)
    return X[valid_mask], y[valid_mask], sample_weights[valid_mask]

# Example for low volatility long mode
feature_cols = ['sma_5', 'sma_20', 'volume_ratio', 'rsi']  # Your features
X, y, weights = prepare_xgboost_data(df_labeled, 'low_vol_long', feature_cols)

# Train XGBoost model
import xgboost as xgb

model = xgb.XGBClassifier(objective='binary:logistic')
model.fit(X, y, sample_weight=weights)
```

### Memory-Constrained Environments

```python
# Configuration for low-memory systems
low_mem_config = LabelingConfig(
    chunk_size=25_000,
    memory_limit_gb=3.0,
    enable_memory_optimization=True,
    enable_parallel_processing=False,
    progress_update_interval=5_000
)

# Process in smaller batches if needed
def process_large_dataset_safely(df, batch_size=100_000):
    """Process very large datasets in batches"""
    results = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        batch_labeled = process_weighted_labeling(batch, low_mem_config)
        results.append(batch_labeled)
        
        # Memory cleanup
        del batch
        import gc
        gc.collect()
    
    return pd.concat(results, ignore_index=True)
```

## Quality Assurance

### Automated Validation

The system includes comprehensive validation to ensure data quality:

```python
# Run all validation checks
validation_results = run_comprehensive_validation(df_labeled)

# Individual validation components
from project.data_pipeline.validation_utils import (
    LabelDistributionValidator,
    WeightDistributionValidator, 
    DataQualityChecker
)

# Check label distributions (win rates should be 5-50%)
label_validator = LabelDistributionValidator()
label_results = label_validator.validate_label_distributions(df_labeled)

# Check weight distributions (all positive, reasonable ranges)
weight_validator = WeightDistributionValidator()
weight_results = weight_validator.validate_weight_distributions(df_labeled)

# Check data quality (no NaN/infinite values)
quality_checker = DataQualityChecker()
quality_results = quality_checker.check_data_quality(df_labeled)
```

### Expected Output Characteristics

**Label Distributions:**
- Win rates typically 10-40% depending on market conditions
- All values exactly 0 or 1
- No NaN values

**Weight Distributions:**
- All positive values
- Winners: weights in range [0.5, 2.0] × time_decay
- Losers: weights equal to time_decay only
- Recent data has higher weights

**Performance Metrics:**
- Processing speed ≥ 167,000 rows/minute
- Memory usage ≤ 8GB peak
- Linear scaling with dataset size

## Troubleshooting Quick Reference

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **ValidationError** | "Missing required columns" | Check input DataFrame has timestamp, OHLCV, volume |
| **ValidationError** | "Non-RTH data found" | Filter data to 07:30-15:00 CT only |
| **PerformanceError** | "Speed target not met" | Increase chunk_size, enable optimizations |
| **ProcessingError** | "Memory limit exceeded" | Reduce chunk_size, enable memory optimization |
| **Invalid OHLC** | "Invalid OHLC relationships" | Fix high/low to contain open/close prices |

### Performance Optimization

```python
# Find optimal configuration for your system
def optimize_for_system(df_sample):
    """Find optimal configuration through testing"""
    
    configs = [
        LabelingConfig(chunk_size=50_000),
        LabelingConfig(chunk_size=100_000),
        LabelingConfig(chunk_size=200_000),
    ]
    
    best_config = None
    best_speed = 0
    
    for config in configs:
        try:
            engine = WeightedLabelingEngine(config)
            df_labeled = engine.process_dataframe(df_sample)
            
            if engine.performance_monitor:
                speed = engine.performance_monitor.metrics.rows_per_minute
                if speed > best_speed:
                    best_speed = speed
                    best_config = config
        except:
            continue
    
    return best_config

# Use optimized configuration
optimal_config = optimize_for_system(df.head(10_000))
```

## Integration Examples

### Complete Pipeline Example

```python
def complete_labeling_pipeline(input_file: str, output_file: str):
    """Complete pipeline from raw data to labeled output"""
    
    print("1. Loading data...")
    df = pd.read_parquet(input_file)
    print(f"   Loaded {len(df):,} rows")
    
    print("2. Data validation and preprocessing...")
    # Filter to RTH
    df['time'] = df['timestamp'].dt.time
    rth_mask = (df['time'] >= pd.Timestamp('07:30:00').time()) & \
               (df['time'] <= pd.Timestamp('15:00:00').time())
    df = df[rth_mask].drop('time', axis=1)
    print(f"   RTH data: {len(df):,} rows")
    
    print("3. Processing with weighted labeling...")
    config = LabelingConfig(
        enable_performance_monitoring=True,
        enable_progress_tracking=True
    )
    
    engine = WeightedLabelingEngine(config)
    df_labeled = engine.process_dataframe(df)
    
    print("4. Validation...")
    validation_results = run_comprehensive_validation(df_labeled, print_reports=False)
    
    if validation_results['overall_validation']['passed']:
        print("   ✅ Validation passed")
    else:
        print("   ⚠️ Validation issues found")
    
    print("5. Saving results...")
    df_labeled.to_parquet(output_file)
    print(f"   Saved to {output_file}")
    
    # Performance summary
    if engine.performance_monitor:
        metrics = engine.performance_monitor.metrics
        print(f"\nPerformance Summary:")
        print(f"   Speed: {metrics.rows_per_minute:,.0f} rows/minute")
        print(f"   Memory: {metrics.peak_memory_gb:.2f} GB")
        print(f"   Time: {metrics.elapsed_time:.1f} seconds")
    
    return df_labeled

# Usage
df_labeled = complete_labeling_pipeline('raw_data.parquet', 'labeled_data.parquet')
```

### EC2 Deployment Example

```python
# EC2 optimized configuration
def get_ec2_config():
    """Configuration optimized for EC2 c5.2xlarge instance"""
    return LabelingConfig(
        chunk_size=200_000,           # Large chunks for 8 vCPUs
        memory_limit_gb=14.0,         # Leave headroom from 16GB
        enable_performance_monitoring=True,
        enable_parallel_processing=True,
        performance_target_rows_per_minute=250_000  # Higher target on EC2
    )

# Process large dataset on EC2
def process_on_ec2(s3_input_path: str, s3_output_path: str):
    """Process large dataset on EC2 with S3 integration"""
    
    # Download from S3
    df = pd.read_parquet(s3_input_path)
    
    # Process with EC2-optimized config
    config = get_ec2_config()
    engine = WeightedLabelingEngine(config)
    df_labeled = engine.process_dataframe(df)
    
    # Upload to S3
    df_labeled.to_parquet(s3_output_path)
    
    return engine.performance_monitor.metrics if engine.performance_monitor else None
```

## Next Steps

After successfully processing your data with the weighted labeling system:

1. **Feature Engineering**: Add technical indicators and market microstructure features
2. **Model Training**: Train separate XGBoost models for each volatility mode
3. **Backtesting**: Validate model performance on out-of-sample data
4. **Production Deployment**: Deploy models for real-time inference
5. **Monitoring**: Set up performance monitoring and model drift detection

## Support and Resources

- **Examples**: See `examples/basic_usage_example.py` for runnable code
- **Validation**: Use `run_comprehensive_validation()` to check data quality
- **Performance**: Use benchmarking tools in the performance guide
- **Troubleshooting**: Check the troubleshooting guide for common issues

The weighted labeling system is designed to be production-ready with comprehensive error handling, performance monitoring, and quality assurance built in.