# Weighted Labeling System - Troubleshooting Guide

## Common Issues and Solutions

### Input Data Issues

#### Issue: ValidationError - Missing Required Columns

**Error Message**: `ValidationError: Missing required columns: {'timestamp', 'volume'}`

**Cause**: Input DataFrame is missing one or more required columns.

**Solution**:
```python
# Check your DataFrame columns
print("Available columns:", df.columns.tolist())

# Required columns
required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
missing = set(required) - set(df.columns)

if missing:
    print(f"Missing columns: {missing}")
    
    # Add missing columns if possible
    if 'volume' in missing:
        df['volume'] = 1000  # Default volume if not available
    
    # Or rename existing columns
    if 'datetime' in df.columns and 'timestamp' in missing:
        df = df.rename(columns={'datetime': 'timestamp'})
```

#### Issue: ValidationError - Invalid Data Types

**Error Message**: `ValidationError: timestamp column must be datetime type`

**Cause**: Columns have incorrect data types.

**Solution**:
```python
# Fix timestamp column
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Fix numeric columns
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for NaN values created by conversion
print("NaN values after conversion:")
print(df[numeric_cols].isnull().sum())

# Remove rows with NaN if necessary
df = df.dropna(subset=numeric_cols)
```

#### Issue: ValidationError - Non-RTH Data

**Error Message**: `ValidationError: Found 5000 bars outside RTH (07:30-15:00 CT)`

**Cause**: Data contains non-Regular Trading Hours data.

**Solution**:
```python
# Filter to RTH only (07:30-15:00 CT)
df['time'] = df['timestamp'].dt.time
rth_start = pd.Timestamp('07:30:00').time()
rth_end = pd.Timestamp('15:00:00').time()

print(f"Original data: {len(df):,} rows")

# Filter to RTH
df_rth = df[(df['time'] >= rth_start) & (df['time'] <= rth_end)].copy()
df_rth = df_rth.drop('time', axis=1)  # Remove helper column

print(f"RTH data: {len(df_rth):,} rows")
print(f"Removed: {len(df) - len(df_rth):,} non-RTH rows")

# Use RTH data for processing
df_labeled = process_weighted_labeling(df_rth)
```

#### Issue: ValidationError - Invalid OHLC Relationships

**Error Message**: `ValidationError: Found 150 bars with invalid OHLC relationships`

**Cause**: High/Low prices don't properly contain Open/Close prices.

**Solution**:
```python
# Identify invalid OHLC bars
invalid_ohlc = (
    (df['high'] < df['low']) |
    (df['high'] < df['open']) |
    (df['high'] < df['close']) |
    (df['low'] > df['open']) |
    (df['low'] > df['close'])
)

print(f"Invalid OHLC bars: {invalid_ohlc.sum()}")

if invalid_ohlc.any():
    # Option 1: Fix OHLC relationships
    df.loc[:, 'high'] = df[['high', 'open', 'close']].max(axis=1)
    df.loc[:, 'low'] = df[['low', 'open', 'close']].min(axis=1)
    
    # Option 2: Remove invalid bars
    # df = df[~invalid_ohlc]
    
    print("OHLC relationships fixed")
```

### Processing Issues

#### Issue: ProcessingError - Memory Limit Exceeded

**Error Message**: `ProcessingError: Memory usage (12.5 GB) exceeds limit (8.0 GB)`

**Cause**: Dataset too large for available memory.

**Solutions**:

1. **Reduce chunk size**:
```python
config = LabelingConfig(
    chunk_size=25_000,  # Smaller chunks
    memory_limit_gb=6.0  # Conservative limit
)
```

2. **Process in batches**:
```python
def process_in_batches(df, batch_size=100_000):
    """Process large dataset in batches"""
    results = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        print(f"Processing batch {i//batch_size + 1}: rows {i:,} to {i+len(batch):,}")
        
        batch_labeled = process_weighted_labeling(batch)
        results.append(batch_labeled)
        
        # Clean up memory
        del batch
        import gc
        gc.collect()
    
    return pd.concat(results, ignore_index=True)

# Use batch processing
df_labeled = process_in_batches(df, batch_size=50_000)
```

3. **Use memory optimization**:
```python
config = LabelingConfig(
    enable_memory_optimization=True,
    chunk_size=50_000,
    progress_update_interval=5_000  # Frequent cleanup
)
```

#### Issue: PerformanceError - Speed Target Not Met

**Error Message**: `PerformanceError: Speed requirement not met: 85,000 < 167,000 rows/minute`

**Cause**: Processing too slow for performance requirements.

**Solutions**:

1. **Optimize configuration**:
```python
# Try larger chunk size if memory allows
config = LabelingConfig(
    chunk_size=200_000,  # Larger chunks
    enable_memory_optimization=True,
    enable_parallel_processing=True
)
```

2. **Check data quality**:
```python
# Ensure data is sorted (improves cache performance)
df = df.sort_values('timestamp').reset_index(drop=True)

# Check for excessive missing values
print("Missing value counts:")
print(df.isnull().sum())

# Remove unnecessary columns
essential_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = df[essential_cols]
```

3. **Disable performance validation for testing**:
```python
config = LabelingConfig(
    enable_performance_monitoring=False  # Skip validation
)
```

#### Issue: Contract Roll Detection Problems

**Error Message**: Processing seems to skip many bars or produces unexpected results.

**Cause**: Contract roll events causing large price jumps.

**Solution**:
```python
# Check for large price jumps (potential contract rolls)
price_changes = df['close'].diff().abs()
large_jumps = price_changes > 20.0  # 20 point jumps

print(f"Potential contract rolls detected: {large_jumps.sum()}")

if large_jumps.any():
    print("Contract roll dates:")
    roll_dates = df[large_jumps]['timestamp'].dt.date.unique()
    for date in roll_dates:
        print(f"  {date}")
    
    # The system automatically handles rolls, but you can verify
    # by checking that bars around roll dates are excluded from labeling
```

### Output Validation Issues

#### Issue: Label Values Not Binary

**Error Message**: `ValidationError: label_low_vol_long must contain only 0 or 1`

**Cause**: Label calculation produced invalid values.

**Diagnosis**:
```python
# Check label value distributions
label_cols = [col for col in df_labeled.columns if col.startswith('label_')]

for col in label_cols:
    unique_vals = df_labeled[col].unique()
    print(f"{col}: {unique_vals}")
    
    if not set(unique_vals).issubset({0, 1}):
        print(f"  ❌ Invalid values in {col}")
        
        # Check for NaN values
        nan_count = df_labeled[col].isnull().sum()
        if nan_count > 0:
            print(f"  NaN values: {nan_count}")
```

**Solution**:
```python
# Fix invalid label values
for col in label_cols:
    # Convert any non-binary values to 0
    df_labeled[col] = df_labeled[col].fillna(0).astype(int)
    df_labeled[col] = df_labeled[col].clip(0, 1)
```

#### Issue: Non-Positive Weights

**Error Message**: `ValidationError: weight_low_vol_long must contain only positive values`

**Cause**: Weight calculation produced zero or negative values.

**Diagnosis**:
```python
# Check weight distributions
weight_cols = [col for col in df_labeled.columns if col.startswith('weight_')]

for col in weight_cols:
    weights = df_labeled[col]
    print(f"{col}:")
    print(f"  Min: {weights.min():.6f}")
    print(f"  Max: {weights.max():.6f}")
    print(f"  Non-positive count: {(weights <= 0).sum()}")
    print(f"  NaN count: {weights.isnull().sum()}")
```

**Solution**:
```python
# Fix non-positive weights
for col in weight_cols:
    # Replace non-positive weights with minimum positive value
    min_positive = df_labeled[col][df_labeled[col] > 0].min()
    df_labeled[col] = df_labeled[col].clip(lower=min_positive)
    
    # Fill NaN with 1.0 (neutral weight)
    df_labeled[col] = df_labeled[col].fillna(1.0)
```

#### Issue: Unreasonable Win Rates

**Error Message**: Win rates outside expected range (5-50%).

**Cause**: Data quality issues or incorrect parameters.

**Diagnosis**:
```python
# Analyze win rates by mode
for mode_name in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                  'low_vol_short', 'normal_vol_short', 'high_vol_short']:
    label_col = f'label_{mode_name}'
    if label_col in df_labeled.columns:
        win_rate = df_labeled[label_col].mean()
        total_winners = df_labeled[label_col].sum()
        
        print(f"{mode_name}:")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Winners: {total_winners:,}")
        
        if win_rate < 0.05:
            print(f"  ⚠ Win rate too low (<5%)")
        elif win_rate > 0.50:
            print(f"  ⚠ Win rate too high (>50%)")
```

**Solutions**:

1. **Check data timeframe**:
```python
# Ensure sufficient data for lookforward
print(f"Data timespan: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Total duration: {df['timestamp'].max() - df['timestamp'].min()}")

# Need at least 15 minutes of data after each potential entry
min_required_bars = 900  # 15 minutes of 1-second bars
if len(df) < min_required_bars * 2:
    print(f"⚠ Dataset may be too small for reliable labeling")
```

2. **Check market conditions**:
```python
# Analyze price volatility
price_changes = df['close'].pct_change().abs()
avg_volatility = price_changes.mean()
print(f"Average price volatility: {avg_volatility:.4f}")

if avg_volatility < 0.0001:
    print("⚠ Very low volatility - may result in few winners")
elif avg_volatility > 0.01:
    print("⚠ Very high volatility - may result in many timeouts")
```

### Performance Issues

#### Issue: Slow Processing on Large Datasets

**Symptoms**: Processing takes much longer than expected.

**Diagnostic Steps**:
```python
import time
import psutil

def diagnose_performance_issues(df):
    """Diagnose performance bottlenecks"""
    
    print("Performance Diagnostics:")
    print(f"Dataset size: {len(df):,} rows")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check system resources
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"CPU count: {psutil.cpu_count()}")
    
    # Check data characteristics
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Check for data quality issues that slow processing
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠ Duplicate rows: {duplicates:,}")
    
    # Check timestamp sorting
    is_sorted = df['timestamp'].is_monotonic_increasing
    print(f"Timestamp sorted: {is_sorted}")
    
    if not is_sorted:
        print("⚠ Data not sorted - this significantly slows processing")
        print("Solution: df = df.sort_values('timestamp').reset_index(drop=True)")
    
    # Test small sample processing speed
    sample_size = min(1000, len(df))
    sample_df = df.head(sample_size)
    
    start_time = time.time()
    sample_labeled = process_weighted_labeling(sample_df)
    elapsed = time.time() - start_time
    
    rows_per_second = sample_size / elapsed
    projected_time_hours = len(df) / (rows_per_second * 3600)
    
    print(f"Sample processing speed: {rows_per_second:.0f} rows/second")
    print(f"Projected time for full dataset: {projected_time_hours:.1f} hours")
    
    if projected_time_hours > 2:
        print("⚠ Projected processing time is very long")
        print("Recommendations:")
        print("  - Use larger chunk sizes if memory allows")
        print("  - Enable memory optimization")
        print("  - Consider processing on more powerful hardware")

# Run diagnostics
diagnose_performance_issues(df)
```

#### Issue: Memory Leaks During Processing

**Symptoms**: Memory usage continuously increases during processing.

**Solution**:
```python
import gc

def process_with_memory_monitoring(df):
    """Process data with aggressive memory management"""
    
    def get_memory_mb():
        process = psutil.Process()
        return process.memory_info().rss / 1024**2
    
    print(f"Initial memory: {get_memory_mb():.1f} MB")
    
    # Use small chunks with frequent cleanup
    config = LabelingConfig(
        chunk_size=25_000,
        enable_memory_optimization=True,
        progress_update_interval=5_000
    )
    
    # Process in smaller batches with manual cleanup
    batch_size = 50_000
    results = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        
        print(f"Processing batch {i//batch_size + 1}, memory: {get_memory_mb():.1f} MB")
        
        batch_labeled = process_weighted_labeling(batch, config)
        results.append(batch_labeled)
        
        # Aggressive cleanup
        del batch, batch_labeled
        gc.collect()
    
    final_result = pd.concat(results, ignore_index=True)
    
    # Final cleanup
    del results
    gc.collect()
    
    print(f"Final memory: {get_memory_mb():.1f} MB")
    
    return final_result
```

### Integration Issues

#### Issue: XGBoost Training Fails

**Error**: Issues when using labeled data for XGBoost training.

**Common Causes and Solutions**:

1. **Missing feature columns**:
```python
# Ensure feature engineering was completed
feature_cols = [col for col in df_labeled.columns 
               if not col.startswith(('label_', 'weight_')) 
               and col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

print(f"Feature columns available: {len(feature_cols)}")

if len(feature_cols) == 0:
    print("❌ No feature columns found")
    print("Solution: Run feature engineering before XGBoost training")
```

2. **Inconsistent data types**:
```python
# Ensure all features are numeric
for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(df_labeled[col]):
        print(f"⚠ Non-numeric feature: {col}")
        df_labeled[col] = pd.to_numeric(df_labeled[col], errors='coerce')
```

3. **NaN values in features**:
```python
# Check for NaN in features
nan_counts = df_labeled[feature_cols].isnull().sum()
problematic_features = nan_counts[nan_counts > 0]

if len(problematic_features) > 0:
    print("Features with NaN values:")
    print(problematic_features)
    
    # Handle NaN values
    df_labeled[feature_cols] = df_labeled[feature_cols].fillna(method='ffill')
    df_labeled[feature_cols] = df_labeled[feature_cols].fillna(0)
```

## Advanced Troubleshooting

### Contract Roll Detection Issues

**Issue**: Unexpected results around contract expiration dates.

**Symptoms**: 
- Large price gaps in data
- Unusual win/loss patterns around specific dates
- Processing warnings about contract rolls

**Solution**:
```python
# Check for contract roll events
def detect_contract_rolls(df, threshold=20.0):
    """Detect potential contract roll events"""
    price_changes = df['close'].diff().abs()
    large_jumps = price_changes > threshold
    
    if large_jumps.any():
        roll_dates = df[large_jumps]['timestamp'].dt.date.unique()
        print(f"Potential contract rolls detected on:")
        for date in roll_dates:
            print(f"  {date}")
        
        # Show price changes around rolls
        for i in df[large_jumps].index:
            if i > 0:
                prev_close = df.loc[i-1, 'close']
                curr_close = df.loc[i, 'close']
                change = abs(curr_close - prev_close)
                print(f"  {df.loc[i, 'timestamp']}: {prev_close:.2f} → {curr_close:.2f} "
                      f"(change: {change:.2f} points)")

# Check your data
detect_contract_rolls(df)

# The system automatically handles rolls by excluding affected bars
# No manual intervention needed, but you can verify the detection
```

### Data Quality Diagnostics

**Issue**: Inconsistent or poor quality results.

**Comprehensive Data Quality Check**:
```python
def comprehensive_data_diagnostics(df):
    """Perform comprehensive data quality diagnostics"""
    
    print("=" * 60)
    print("COMPREHENSIVE DATA QUALITY DIAGNOSTICS")
    print("=" * 60)
    
    # Basic statistics
    print(f"\n1. Dataset Overview:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Duration: {df['timestamp'].max() - df['timestamp'].min()}")
    
    # Check for gaps in data
    print(f"\n2. Timestamp Analysis:")
    time_diffs = df['timestamp'].diff().dt.total_seconds()
    expected_interval = 1  # 1 second bars
    
    gaps = time_diffs[time_diffs > expected_interval * 2]  # More than 2 seconds
    print(f"   Expected interval: {expected_interval} seconds")
    print(f"   Actual intervals: {time_diffs.min():.1f}s - {time_diffs.max():.1f}s")
    print(f"   Data gaps (>2s): {len(gaps)}")
    
    if len(gaps) > 0:
        print(f"   Largest gap: {gaps.max():.1f} seconds")
        gap_locations = df[time_diffs > expected_interval * 2]['timestamp']
        print(f"   Gap locations (first 5): {list(gap_locations.head())}")
    
    # Price analysis
    print(f"\n3. Price Analysis:")
    for col in ['open', 'high', 'low', 'close']:
        prices = df[col]
        print(f"   {col}: {prices.min():.2f} - {prices.max():.2f} "
              f"(mean: {prices.mean():.2f}, std: {prices.std():.2f})")
    
    # Check tick alignment
    print(f"\n4. Tick Alignment Check:")
    tick_size = 0.25
    for col in ['open', 'high', 'low', 'close']:
        misaligned = (df[col] % tick_size != 0).sum()
        if misaligned > 0:
            print(f"   ⚠️ {col}: {misaligned} prices not aligned to {tick_size} tick size")
        else:
            print(f"   ✅ {col}: All prices aligned to tick size")
    
    # Volume analysis
    print(f"\n5. Volume Analysis:")
    volume = df['volume']
    print(f"   Range: {volume.min():,} - {volume.max():,}")
    print(f"   Mean: {volume.mean():,.0f}")
    print(f"   Zero volume bars: {(volume == 0).sum()}")
    
    # OHLC relationship validation
    print(f"\n6. OHLC Relationship Validation:")
    invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
    invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
    
    print(f"   Invalid highs: {invalid_high}")
    print(f"   Invalid lows: {invalid_low}")
    
    if invalid_high > 0 or invalid_low > 0:
        print(f"   ⚠️ OHLC relationship issues found")
    else:
        print(f"   ✅ All OHLC relationships valid")
    
    # RTH validation
    print(f"\n7. RTH Validation:")
    times = df['timestamp'].dt.time
    rth_start = pd.Timestamp('07:30:00').time()
    rth_end = pd.Timestamp('15:00:00').time()
    
    rth_mask = (times >= rth_start) & (times <= rth_end)
    non_rth_count = (~rth_mask).sum()
    
    print(f"   RTH bars: {rth_mask.sum():,}")
    print(f"   Non-RTH bars: {non_rth_count:,}")
    
    if non_rth_count > 0:
        print(f"   ⚠️ Non-RTH data found - filter before processing")
    else:
        print(f"   ✅ All data within RTH")
    
    # Missing data check
    print(f"\n8. Missing Data Check:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"   ⚠️ {col}: {missing:,} missing values ({missing/len(df)*100:.2f}%)")
        else:
            print(f"   ✅ {col}: No missing values")
    
    print(f"\n" + "=" * 60)

# Run diagnostics
comprehensive_data_diagnostics(df)
```

### Performance Profiling

**Issue**: Need to identify performance bottlenecks.

**Detailed Performance Profiling**:
```python
import cProfile
import pstats
from io import StringIO

def profile_processing(df_sample):
    """Profile processing performance to identify bottlenecks"""
    
    print("Running performance profiling...")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the processing
    profiler.enable()
    df_labeled = process_weighted_labeling(df_sample)
    profiler.disable()
    
    # Analyze results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("Top 20 functions by cumulative time:")
    print(s.getvalue())
    
    return df_labeled

# Profile on sample data
df_sample = df.head(5000)
df_labeled = profile_processing(df_sample)
```

### Memory Leak Detection

**Issue**: Memory usage continuously increases during processing.

**Memory Leak Detection and Prevention**:
```python
import psutil
import gc
import tracemalloc

def detect_memory_leaks(df, chunk_size=10000):
    """Detect potential memory leaks during processing"""
    
    print("Starting memory leak detection...")
    
    # Start memory tracing
    tracemalloc.start()
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss / (1024**2)  # MB
    print(f"Initial memory: {initial_memory:.1f} MB")
    
    memory_history = []
    
    # Process in small chunks and monitor memory
    for i in range(0, min(len(df), 50000), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        
        # Process chunk
        chunk_labeled = process_weighted_labeling(chunk)
        
        # Monitor memory
        current_memory = process.memory_info().rss / (1024**2)
        memory_history.append({
            'chunk': i // chunk_size + 1,
            'rows_processed': i + len(chunk),
            'memory_mb': current_memory,
            'memory_increase': current_memory - initial_memory
        })
        
        print(f"Chunk {len(memory_history)}: {current_memory:.1f} MB "
              f"(+{current_memory - initial_memory:.1f} MB)")
        
        # Clean up
        del chunk, chunk_labeled
        gc.collect()
    
    # Analyze memory pattern
    print(f"\nMemory Analysis:")
    memory_increases = [h['memory_increase'] for h in memory_history]
    
    if len(memory_increases) > 1:
        # Check if memory is consistently increasing
        increasing_trend = all(
            memory_increases[i] >= memory_increases[i-1] 
            for i in range(1, len(memory_increases))
        )
        
        if increasing_trend:
            print(f"⚠️ Potential memory leak detected")
            print(f"   Memory increased from {memory_increases[0]:.1f} MB to {memory_increases[-1]:.1f} MB")
            
            # Get memory snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            print(f"\nTop memory allocations:")
            for stat in top_stats[:5]:
                print(f"   {stat}")
        else:
            print(f"✅ No memory leak detected")
            print(f"   Memory usage appears stable")
    
    tracemalloc.stop()

# Run memory leak detection
detect_memory_leaks(df)
```

## Getting Help

### Enable Debug Mode

```python
# Enable detailed logging for debugging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('weighted_labeling')

# Process with debug information
config = LabelingConfig(
    enable_progress_tracking=True,
    enable_performance_monitoring=True,
    chunk_size=10_000  # Small chunks for easier debugging
)

try:
    df_labeled = process_weighted_labeling(df, config)
except Exception as e:
    logger.error(f"Processing failed: {e}")
    import traceback
    traceback.print_exc()
```

### System Requirements Check

```python
def check_system_requirements():
    """Check if system meets minimum requirements"""
    
    import psutil
    import sys
    
    print("System Requirements Check:")
    print("=" * 40)
    
    # Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("⚠️ Python 3.8+ recommended")
    else:
        print("✅ Python version OK")
    
    # Memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total RAM: {memory_gb:.1f} GB")
    
    if memory_gb < 8:
        print("⚠️ 8GB+ RAM recommended for optimal performance")
    else:
        print("✅ Memory OK")
    
    # CPU
    cpu_count = psutil.cpu_count()
    print(f"CPU cores: {cpu_count}")
    
    if cpu_count < 4:
        print("⚠️ 4+ CPU cores recommended")
    else:
        print("✅ CPU OK")
    
    # Available memory
    available_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available RAM: {available_gb:.1f} GB")
    
    if available_gb < 4:
        print("⚠️ Low available memory - close other applications")
    else:
        print("✅ Available memory OK")

# Check system
check_system_requirements()
```

### Create Minimal Reproduction Case

```python
def create_minimal_test_case():
    """Create minimal test case for debugging"""
    
    # Create small synthetic dataset
    np.random.seed(42)
    n_rows = 1000
    
    timestamps = pd.date_range('2024-01-01 08:00:00', periods=n_rows, freq='1s')
    base_price = 4500.0
    
    df_test = pd.DataFrame({
        'timestamp': timestamps,
        'open': base_price + np.random.normal(0, 1, n_rows),
        'high': base_price + np.random.normal(2, 1, n_rows),
        'low': base_price + np.random.normal(-2, 1, n_rows),
        'close': base_price + np.random.normal(0, 1, n_rows),
        'volume': np.random.randint(100, 1000, n_rows)
    })
    
    # Ensure valid OHLC
    df_test['high'] = np.maximum(df_test['high'], 
                                np.maximum(df_test['open'], df_test['close']))
    df_test['low'] = np.minimum(df_test['low'], 
                               np.minimum(df_test['open'], df_test['close']))
    
    return df_test

# Test with minimal case
df_test = create_minimal_test_case()
try:
    df_test_labeled = process_weighted_labeling(df_test)
    print("✅ Minimal test case passed")
except Exception as e:
    print(f"❌ Minimal test case failed: {e}")
    # This helps isolate whether the issue is with your data or the system
```

### Contact Information

If you continue to experience issues:

1. **Check the validation results** using `run_comprehensive_validation.py`
2. **Review the performance guide** for optimization tips
3. **Create a minimal reproduction case** to isolate the issue
4. **Check system requirements** (Python version, dependencies, hardware)

For additional support, provide:
- Error messages and full stack traces
- Dataset characteristics (size, date range, source)
- System specifications (RAM, CPU, storage type)
- Configuration used
- Results from diagnostic functions above