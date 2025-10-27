# Feature Engineering System Design

## Overview

The Feature Engineering System transforms raw OHLCV ES futures data into 43 trading features across 7 categories for LSTM model training. The system prioritizes minimal viable code with direct pandas operations while ensuring data leakage prevention and scalability from laptop (947K bars) to SageMaker (88M bars).

## Architecture

### High-Level Flow
```
Raw Dataset (39 columns) → Feature Engineering → Enhanced Dataset (82 columns)
├── OHLCV + Labels (39)     ├── 43 Features        ├── OHLCV + Labels + Features
└── 947K-88M bars           └── 7 Categories       └── Ready for LSTM training
```

### Core Design Principles
1. **Minimal Code**: <400 lines total, 1-3 lines per feature calculation
2. **Data Leakage Prevention**: Only use bars 0 to N-1 to predict bar N
3. **Direct Operations**: Pure pandas/numpy, no classes or abstractions
4. **Memory Efficiency**: Process in chunks for large datasets
5. **Identical Results**: Deterministic across laptop and SageMaker

## Components and Interfaces

### 1. Main Processing Function
```python
def create_all_features(df):
    """
    Add 43 features to existing labeled dataset
    
    Input: DataFrame with OHLCV + labels (39 columns)
    Output: DataFrame with OHLCV + labels + features (82 columns)
    """
```

### 2. Feature Category Functions
```python
# Volume Features (4)
def add_volume_features(df): pass

# Price Context Features (5) 
def add_price_context_features(df): pass

# Consolidation Features (10)
def add_consolidation_features(df): pass

# Return Features (5)
def add_return_features(df): pass

# Volatility Features (6)
def add_volatility_features(df): pass

# Microstructure Features (6)
def add_microstructure_features(df): pass

# Time Features (7)
def add_time_features(df): pass
```

### 3. Utility Functions (Minimal)
```python
def linear_slope(series): pass  # For slope calculations
def count_retouches(highs, lows, threshold_high, threshold_low): pass  # With 30s cooldown
def get_session_period(timestamp): pass  # UTC to Central Time conversion
```

## Data Models

### Input Schema (39 columns)
```python
# OHLCV Data (5 columns)
['timestamp', 'open', 'high', 'low', 'close', 'volume']

# Label Data (34 columns - 6 profiles × 5-6 columns each)
# Long profiles: long_2to1_small_*, long_2to1_medium_*, long_2to1_large_*
# Short profiles: short_2to1_small_*, short_2to1_medium_*, short_2to1_large_*
# Each profile: label, target_hit_bar, stop_hit_bar, mae, timeout_bar, [win_sequence_id]
```

### Output Schema (82 columns)
```python
# Input columns (39) + Feature columns (43)

# Volume Features (4)
['volume_ratio_30s', 'volume_slope_30s', 'volume_slope_5s', 'volume_exhaustion']

# Price Context Features (5)
['vwap', 'distance_from_vwap_pct', 'vwap_slope', 'distance_from_rth_high', 'distance_from_rth_low']

# Consolidation Features (10)
['short_range_high', 'short_range_low', 'short_range_size', 'position_in_short_range',
 'medium_range_high', 'medium_range_low', 'medium_range_size', 'range_compression_ratio',
 'short_range_retouches', 'medium_range_retouches']

# Return Features (5)
['return_30s', 'return_60s', 'return_300s', 'momentum_acceleration', 'momentum_consistency']

# Volatility Features (6)
['atr_30s', 'atr_300s', 'volatility_regime', 'volatility_acceleration', 'volatility_breakout', 'atr_percentile']

# Microstructure Features (6)
['bar_range', 'relative_bar_size', 'uptick_pct_30s', 'uptick_pct_60s', 'bar_flow_consistency', 'directional_strength']

# Time Features (7)
['is_eth', 'is_pre_open', 'is_rth_open', 'is_morning', 'is_lunch', 'is_afternoon', 'is_rth_close']
```

### Data Types and Ranges
```python
# Continuous features: float64
# Binary features (time): int8 (0 or 1)
# Expected ranges per feature documented in #[[file:docs/feature_summary.md]]
```

## Implementation Details

### 1. Volume Features Implementation
```python
def add_volume_features(df):
    df['volume_ratio_30s'] = df['volume'] / df['volume'].rolling(30).mean()
    
    vol_ma = df['volume'].rolling(5).mean()
    df['volume_slope_30s'] = vol_ma.rolling(30).apply(lambda x: linear_slope(x))
    df['volume_slope_5s'] = df['volume'].rolling(5).apply(lambda x: linear_slope(x))
    df['volume_exhaustion'] = df['volume_ratio_30s'] * df['volume_slope_5s']
```

### 2. Data Leakage Prevention Strategy
- **Rolling calculations**: Use `.rolling(window)` with historical data only
- **Session calculations**: Reset daily, use only completed bars
- **Lookback windows**: 300s (5min), 900s (15min) use bars [N-window:N-1]
- **Current bar data**: Only use bar N-1 OHLCV for predictions about bar N

### 3. Memory Management
```python
def create_all_features(df, chunk_size=100000):
    """Process large datasets in chunks to manage memory"""
    if len(df) <= chunk_size:
        return _process_chunk(df)
    
    # Process in chunks, handle rolling window overlaps
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[max(0, i-1000):i+chunk_size]  # Overlap for rolling calcs
        processed = _process_chunk(chunk)
        chunks.append(processed.iloc[1000:] if i > 0 else processed)
    
    return pd.concat(chunks, ignore_index=True)
```

### 4. Session Period Detection
```python
def get_session_period(timestamp):
    """Convert UTC to Central Time and identify ES session period"""
    import pytz
    central_tz = pytz.timezone('US/Central')
    ct_time = timestamp.astimezone(central_tz)
    ct_decimal = ct_time.hour + ct_time.minute/60.0
    
    # Return period name based on Central Time ranges
    if (15.0 <= ct_decimal < 24.0) or (0.0 <= ct_decimal < 7.5):
        return 'eth'
    elif 7.5 <= ct_decimal < 8.5:
        return 'pre_open'
    # ... etc for all 7 periods
```

### 5. Retouch Counting with Cooldown
```python
def count_retouches(highs, lows, threshold_high, threshold_low, cooldown=30):
    """Count distinct retouch events with 30-second cooldown"""
    retouches = 0
    last_retouch = -cooldown - 1
    
    for i, (high, low) in enumerate(zip(highs, lows)):
        in_zone = (high >= threshold_high) or (low <= threshold_low)
        if in_zone and (i - last_retouch >= cooldown):
            retouches += 1
            last_retouch = i
    
    return retouches
```

## Error Handling

### Input Validation
```python
def validate_input(df):
    """Basic validation - fail fast on critical issues"""
    assert not df.empty, "DataFrame cannot be empty"
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    assert not missing, f"Missing required columns: {missing}"
    print(f"Processing {len(df):,} bars")
```

### Edge Case Handling
- **Insufficient data**: Rolling calculations return NaN for first N bars (expected)
- **Division by zero**: Pandas handles gracefully with inf/NaN
- **Missing values**: Forward fill where appropriate, otherwise leave as NaN
- **Timezone issues**: Use pytz for robust DST handling

## Testing Strategy

### Unit Tests (Single File)
```python
def test_volume_features():
    """Test volume feature calculations with known inputs"""
    # Simple synthetic data with expected outputs
    
def test_data_leakage():
    """Verify no future data is used in calculations"""
    # Check that feature[i] only uses data[0:i]
    
def test_session_periods():
    """Test timezone conversion and period identification"""
    # Test known UTC timestamps → expected periods

def test_integration():
    """End-to-end test on small sample dataset"""
    # Load 1000-bar sample, verify all 43 features created
```

### Performance Validation
```python
def test_performance():
    """Validate processing time on laptop"""
    # Process 947K bars, assert < 10 minutes
    # Memory usage stays reasonable
```

## Deployment Considerations

### SageMaker Compatibility
- **Container**: Standard pandas/numpy environment
- **Input**: S3 Parquet files
- **Output**: S3 Parquet files  
- **Scaling**: Process in chunks, handle memory limits
- **Monitoring**: Basic print statements for progress

### File Structure
```
project/data_pipeline/
├── features.py              # Main implementation (~350 lines)
├── __init__.py             # Empty
└── utils.py                # Optional: shared utilities if needed

tests/
└── test_features.py        # All tests (~150 lines)
```

### Dependencies
```python
# Minimal dependencies
import pandas as pd
import numpy as np
import pytz  # For timezone handling
```

## Success Criteria

1. **Functionality**: All 43 features calculated correctly with proper data leakage prevention
2. **Performance**: 947K bars in <10 minutes (laptop), 88M bars in <4 hours (SageMaker)  
3. **Code Quality**: <400 lines, readable by junior developers, minimal abstractions
4. **Integration**: Seamlessly adds features to existing 39-column labeled dataset
5. **Reliability**: Identical results across multiple runs and environments

## Future Considerations

### Potential Optimizations (Not in MVP)
- **Cython/Numba**: For compute-intensive features if needed
- **Parallel processing**: Multi-core utilization for large datasets
- **Feature caching**: Store intermediate calculations
- **Advanced chunking**: Smarter memory management

### Model Integration
- **Feature scaling**: Prepare features for LSTM training
- **Sequence creation**: Convert to time series sequences
- **Train/test splits**: Temporal splitting without data leakage

The design prioritizes simplicity and correctness over premature optimization, ensuring a robust foundation for ES futures trading model development.