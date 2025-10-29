# Weighted Labeling System - Usage Guide

## Overview

The Weighted Labeling System is a comprehensive solution for generating binary labels and training weights for XGBoost models in ES futures trading. It processes OHLCV data to create 12 columns (6 labels + 6 weights) for 6 volatility-based trading modes.

## Quick Start

### Basic Usage

```python
import pandas as pd
from project.data_pipeline.weighted_labeling import process_weighted_labeling

# Load your OHLCV data
df = pd.read_parquet('your_data.parquet')

# Process with default configuration
df_labeled = process_weighted_labeling(df)

# Result: Original columns + 12 new columns
print(f"Original columns: {len(df.columns)}")
print(f"Processed columns: {len(df_labeled.columns)}")
```

### Advanced Configuration

```python
from project.data_pipeline.weighted_labeling import (
    WeightedLabelingEngine, 
    LabelingConfig
)

# Custom configuration
config = LabelingConfig(
    chunk_size=50_000,                    # Process in 50K row chunks
    enable_performance_monitoring=True,    # Track speed and memory
    enable_progress_tracking=True,         # Show progress updates
    memory_limit_gb=6.0                   # Custom memory limit
)

# Create engine with custom config
engine = WeightedLabelingEngine(config)
df_labeled = engine.process_dataframe(df)
```

## Input Data Requirements

### Required Columns

Your DataFrame must contain these columns:

```python
required_columns = [
    'timestamp',  # pd.Timestamp, datetime64
    'open',       # float, ES futures open price
    'high',       # float, ES futures high price  
    'low',        # float, ES futures low price
    'close',      # float, ES futures close price
    'volume'      # int/float, trading volume
]
```

### Data Quality Requirements

1. **RTH Only**: Data must be Regular Trading Hours (07:30-15:00 CT)
2. **Valid OHLC**: High ≥ max(Open, Close), Low ≤ min(Open, Close)
3. **Positive Prices**: All prices > 0
4. **Non-negative Volume**: Volume ≥ 0
5. **No Missing Values**: No NaN in required columns
6. **Proper Timestamps**: Sorted datetime index

### Example Data Preparation

```python
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_parquet('raw_es_data.parquet')

# Convert timestamp if needed
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter to RTH only (07:30-15:00 CT)
df['time'] = df['timestamp'].dt.time
rth_start = pd.Timestamp('07:30:00').time()
rth_end = pd.Timestamp('15:00:00').time()
df = df[(df['time'] >= rth_start) & (df['time'] <= rth_end)]

# Validate OHLC relationships
df = df[
    (df['high'] >= df['open']) & 
    (df['high'] >= df['close']) &
    (df['low'] <= df['open']) & 
    (df['low'] <= df['close']) &
    (df['open'] > 0) & 
    (df['volume'] >= 0)
]

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Prepared {len(df):,} rows for labeling")
```

## Output Format

### Generated Columns

The system adds 12 new columns to your DataFrame:

**Label Columns (Binary: 0 or 1):**
- `label_low_vol_long`: Low volatility long trades (6 tick stop, 12 tick target)
- `label_normal_vol_long`: Normal volatility long trades (8 tick stop, 16 tick target)  
- `label_high_vol_long`: High volatility long trades (10 tick stop, 20 tick target)
- `label_low_vol_short`: Low volatility short trades (6 tick stop, 12 tick target)
- `label_normal_vol_short`: Normal volatility short trades (8 tick stop, 16 tick target)
- `label_high_vol_short`: High volatility short trades (10 tick stop, 20 tick target)

**Weight Columns (Positive floats):**
- `weight_low_vol_long`: Training weights for low vol long mode
- `weight_normal_vol_long`: Training weights for normal vol long mode
- `weight_high_vol_long`: Training weights for high vol long mode
- `weight_low_vol_short`: Training weights for low vol short mode
- `weight_normal_vol_short`: Training weights for normal vol short mode
- `weight_high_vol_short`: Training weights for high vol short mode

### Example Output

```python
# Check the results
print("Label columns:")
label_cols = [col for col in df_labeled.columns if col.startswith('label_')]
for col in label_cols:
    win_rate = df_labeled[col].mean()
    print(f"  {col}: {win_rate:.1%} win rate")

print("\nWeight columns:")
weight_cols = [col for col in df_labeled.columns if col.startswith('weight_')]
for col in weight_cols:
    avg_weight = df_labeled[col].mean()
    weight_range = (df_labeled[col].min(), df_labeled[col].max())
    print(f"  {col}: avg={avg_weight:.3f}, range={weight_range[0]:.3f}-{weight_range[1]:.3f}")
```

## Configuration Options

### LabelingConfig Parameters

```python
@dataclass
class LabelingConfig:
    # Processing Configuration
    chunk_size: int = 100_000              # Rows per chunk for large datasets
    timeout_seconds: int = 900             # Max lookforward time (15 minutes)
    
    # Performance Configuration  
    performance_target_rows_per_minute: int = 167_000  # Target speed
    memory_limit_gb: float = 8.0           # Memory usage limit
    
    # Feature Flags
    enable_parallel_processing: bool = True     # Process modes in parallel
    enable_progress_tracking: bool = True       # Show progress updates
    enable_performance_monitoring: bool = True  # Track performance metrics
    enable_memory_optimization: bool = True     # Use memory optimizations
    
    # Progress Configuration
    progress_update_interval: int = 10_000     # Progress update frequency
    
    # Weight Calculation Parameters
    decay_rate: float = 0.05               # Monthly time decay rate
    tick_size: float = 0.25                # ES tick size in points
```

### Performance Tuning

**For Large Datasets (>1M rows):**
```python
config = LabelingConfig(
    chunk_size=200_000,                    # Larger chunks for efficiency
    enable_memory_optimization=True,       # Enable all optimizations
    enable_performance_monitoring=True,    # Track performance
    memory_limit_gb=12.0                  # Higher memory limit if available
)
```

**For Memory-Constrained Environments:**
```python
config = LabelingConfig(
    chunk_size=50_000,                     # Smaller chunks
    memory_limit_gb=4.0,                   # Lower memory limit
    enable_memory_optimization=True,       # Essential for low memory
    progress_update_interval=5_000         # More frequent cleanup
)
```

**For Development/Testing:**
```python
config = LabelingConfig(
    chunk_size=10_000,                     # Small chunks for testing
    enable_progress_tracking=True,         # See detailed progress
    enable_performance_monitoring=False,   # Skip performance overhead
    enable_parallel_processing=False       # Easier debugging
)
```

## Processing Sample Datasets

### Small Sample (1,000 rows)

```python
# Create test data
import numpy as np

def create_test_data(n_rows=1000):
    """Create synthetic ES futures data for testing"""
    np.random.seed(42)
    
    # Generate timestamps (1-second bars, RTH only)
    start_date = pd.Timestamp('2024-01-01 07:30:00', tz='UTC')
    timestamps = pd.date_range(start_date, periods=n_rows, freq='1s')
    
    # Generate realistic price data
    base_price = 4500.0
    price_changes = np.random.normal(0, 0.5, n_rows).cumsum()
    close_prices = base_price + price_changes
    
    # Generate OHLC
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': close_prices + np.random.normal(0, 0.2, n_rows),
        'high': close_prices + np.random.exponential(0.5, n_rows),
        'low': close_prices - np.random.exponential(0.5, n_rows),
        'close': close_prices,
        'volume': np.random.randint(100, 5000, n_rows)
    })
    
    # Ensure valid OHLC relationships
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    # Round to tick size
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = (df[col] / 0.25).round() * 0.25
    
    return df

# Process test data
df_test = create_test_data(1000)
df_labeled = process_weighted_labeling(df_test)

print(f"Processed {len(df_labeled):,} rows")
print(f"Added {len(df_labeled.columns) - len(df_test.columns)} new columns")
```

### Medium Sample (100,000 rows)

```python
# Process with performance monitoring
config = LabelingConfig(
    chunk_size=25_000,
    enable_performance_monitoring=True,
    enable_progress_tracking=True
)

df_medium = create_test_data(100_000)
engine = WeightedLabelingEngine(config)

# Process and get performance metrics
df_labeled = engine.process_dataframe(df_medium)

# Check performance
if engine.performance_monitor:
    metrics = engine.performance_monitor.metrics
    print(f"Processing speed: {metrics.rows_per_minute:,.0f} rows/minute")
    print(f"Peak memory: {metrics.peak_memory_gb:.2f} GB")
    print(f"Total time: {metrics.elapsed_time:.1f} seconds")
```

### Large Sample (1M+ rows)

```python
# For production-scale processing
config = LabelingConfig(
    chunk_size=100_000,
    enable_memory_optimization=True,
    enable_performance_monitoring=True,
    memory_limit_gb=8.0
)

# Load large dataset
df_large = pd.read_parquet('large_es_dataset.parquet')
print(f"Loading {len(df_large):,} rows...")

# Process with monitoring
engine = WeightedLabelingEngine(config)
df_labeled = engine.process_dataframe(df_large)

# Save results
output_file = 'weighted_labeled_large_dataset.parquet'
df_labeled.to_parquet(output_file)
print(f"Saved results to {output_file}")
```

## Validation and Quality Assurance

### Basic Validation

```python
from project.data_pipeline.validation_utils import run_comprehensive_validation

# Run all validation checks
validation_results = run_comprehensive_validation(df_labeled)

# Check if validation passed
if validation_results['overall_validation']['passed']:
    print("✅ All validations passed - data ready for XGBoost training")
else:
    print("❌ Validation failed - check issues before training")
    
    # Show specific issues
    summary = validation_results['overall_validation']['summary']
    if not summary['label_validation_passed']:
        print("  - Label validation issues")
    if not summary['weight_validation_passed']:
        print("  - Weight validation issues")
    if not summary['data_quality_passed']:
        print("  - Data quality issues")
```

### Individual Validation Components

```python
from project.data_pipeline.validation_utils import (
    LabelDistributionValidator,
    WeightDistributionValidator,
    DataQualityChecker
)

# Check label distributions
label_validator = LabelDistributionValidator()
label_results = label_validator.validate_label_distributions(df_labeled)
label_validator.print_label_distribution_report(label_results)

# Check weight distributions  
weight_validator = WeightDistributionValidator()
weight_results = weight_validator.validate_weight_distributions(df_labeled)
weight_validator.print_weight_distribution_report(weight_results)

# Check data quality
quality_checker = DataQualityChecker()
quality_results = quality_checker.check_data_quality(df_labeled)
quality_checker.print_data_quality_report(quality_results)
```

## Integration with XGBoost Training

### Preparing Data for XGBoost

```python
# Separate features, labels, and weights for each mode
def prepare_xgboost_data(df_labeled, mode_name):
    """Prepare data for XGBoost training for a specific mode"""
    
    # Get feature columns (exclude timestamp, OHLCV, labels, weights)
    feature_cols = [col for col in df_labeled.columns 
                   if not col.startswith(('label_', 'weight_')) 
                   and col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Get mode-specific columns
    label_col = f'label_{mode_name}'
    weight_col = f'weight_{mode_name}'
    
    # Extract data
    X = df_labeled[feature_cols]
    y = df_labeled[label_col]
    sample_weights = df_labeled[weight_col]
    
    # Remove any rows with NaN in features
    valid_mask = ~X.isnull().any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    sample_weights = sample_weights[valid_mask]
    
    return X, y, sample_weights

# Example for low volatility long mode
X, y, weights = prepare_xgboost_data(df_labeled, 'low_vol_long')

print(f"Features: {X.shape}")
print(f"Labels: {y.shape} (win rate: {y.mean():.1%})")
print(f"Weights: {weights.shape} (avg: {weights.mean():.3f})")
```

### XGBoost Training Example

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Prepare data for all modes
modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
         'low_vol_short', 'normal_vol_short', 'high_vol_short']

models = {}

for mode in modes:
    print(f"\nTraining {mode} model...")
    
    # Prepare data
    X, y, weights = prepare_xgboost_data(df_labeled, mode)
    
    # Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost model with sample weights
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(
        X_train, y_train, 
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[w_test],
        verbose=False
    )
    
    models[mode] = model
    
    # Evaluate
    train_score = model.score(X_train, y_train, sample_weight=w_train)
    test_score = model.score(X_test, y_test, sample_weight=w_test)
    
    print(f"  Train accuracy: {train_score:.3f}")
    print(f"  Test accuracy: {test_score:.3f}")

print(f"\nTrained {len(models)} XGBoost models successfully!")
```

## Next Steps

After processing your data with the weighted labeling system:

1. **Validate Results**: Always run comprehensive validation
2. **Feature Engineering**: Add technical indicators and market features
3. **Model Training**: Train separate XGBoost models for each volatility mode
4. **Backtesting**: Test model performance on out-of-sample data
5. **Production Deployment**: Deploy models for real-time inference

For more advanced usage, see:
- [Performance Tuning Guide](weighted_labeling_performance_guide.md)
- [Troubleshooting Guide](weighted_labeling_troubleshooting.md)
- [API Reference](weighted_labeling_api_reference.md)