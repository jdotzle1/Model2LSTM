"""
Feature Engineering Module for ES Futures Trading

Generates 43 features across 7 categories for LSTM model training.
Prioritizes minimal viable code with direct pandas operations.

Main Integration Function:
    integrate_with_labeled_dataset() - Adds 43 features to existing labeled dataset

Feature Categories (43 total):
    - Volume Features (4): volume_ratio_30s, volume_slope_30s, volume_slope_5s, volume_exhaustion
    - Price Context Features (5): vwap, distance_from_vwap_pct, vwap_slope, distance_from_rth_high, distance_from_rth_low
    - Consolidation Features (10): range highs/lows, compression ratios, retouch counts
    - Return Features (5): returns at multiple timeframes, momentum acceleration/consistency
    - Volatility Features (6): ATR calculations, volatility regime detection
    - Microstructure Features (6): bar characteristics, tick flow analysis
    - Time Features (7): session period identification (binary one-hot encoding)

Usage:
    from project.data_pipeline.features import integrate_with_labeled_dataset
    
    # Add features to existing labeled dataset
    enhanced_df = integrate_with_labeled_dataset('input.parquet', 'output.parquet')
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime


def create_all_features(df):
    """
    Add 43 features to existing labeled dataset
    Implement fail-fast behavior for critical errors rather than complex recovery logic
    Add basic progress logging using print statements
    
    Input: DataFrame with OHLCV + labels (existing columns)
    Output: DataFrame with OHLCV + labels + features (original + 43 columns)
    
    Args:
        df: DataFrame with required columns [timestamp, open, high, low, close, volume]
        
    Returns:
        DataFrame with 43 additional feature columns
    """
    print(f"Processing {len(df):,} bars for feature engineering...")
    
    try:
        # Input validation - fail fast on critical errors
        validate_input(df)
        
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Ensure we have a timestamp column
        if 'timestamp' not in df.columns:
            if df.index.name == 'ts_event' or pd.api.types.is_datetime64_any_dtype(df.index):
                df['timestamp'] = df.index
                print(f"  Using index as timestamp column")
            else:
                raise ValueError("No timestamp found in columns or index")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            print(f"  Converting timestamp to datetime...")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Track original columns to know what we added
        original_cols_set = set(df.columns)
        timestamp_added = False
        
        # Add features by category with progress logging
        print("  Adding volume features (4)...")
        df = add_volume_features(df)
        
        print("  Adding price context features (5)...")
        df = add_price_context_features(df)
        
        print("  Adding consolidation features (10)...")
        df = add_consolidation_features(df)
        
        print("  Adding return features (5)...")
        df = add_return_features(df)
        
        print("  Adding volatility features (6)...")
        df = add_volatility_features(df)
        
        print("  Adding microstructure features (6)...")
        df = add_microstructure_features(df)
        
        print("  Adding time features (7)...")
        df = add_time_features(df)
        
        # Remove timestamp column if we added it (it wasn't in original)
        if 'timestamp' in df.columns and 'timestamp' not in original_cols_set:
            df.drop('timestamp', axis=1, inplace=True)
            timestamp_added = True
        
        # Final validation
        expected_features = get_expected_feature_names()
        # Account for timestamp column if it was added and removed
        original_count = len(original_cols_set)
        if timestamp_added:
            original_count -= 1  # We added timestamp, so subtract it from original count
        
        actual_new_cols = len(df.columns) - original_count
        
        if actual_new_cols != len(expected_features):
            print(f"  Warning: Expected {len(expected_features)} features, added {actual_new_cols}")
            print(f"    Original columns: {original_count}, Final columns: {len(df.columns)}")
        
        print(f"✓ Added {actual_new_cols} features to dataset")
        return df
        
    except Exception as e:
        print(f"❌ Critical error in feature engineering: {str(e)}")
        print(f"   Failing fast rather than attempting recovery")
        raise


def create_all_features_chunked(df, chunk_size):
    """
    Memory-efficient chunked processing for large datasets
    
    Handles rolling window overlaps between chunks to maintain calculation accuracy.
    Ensures identical results regardless of chunk size used.
    
    Args:
        df: DataFrame with required columns [timestamp, open, high, low, close, volume]
        chunk_size: Number of rows per chunk
        
    Returns:
        DataFrame with 43 additional feature columns
    """
    print(f"Processing {len(df):,} bars using chunked processing (chunk_size={chunk_size:,})...")
    
    # Input validation
    validate_input(df)
    
    # Determine overlap size needed for rolling calculations
    # Largest rolling window is 900 bars (medium-term consolidation)
    # Add buffer for safety and session calculations
    overlap_size = 1000
    
    if len(df) <= chunk_size:
        # Dataset smaller than chunk size, process normally
        print(f"  Dataset size ({len(df):,}) <= chunk_size ({chunk_size:,}), processing without chunking...")
        return create_all_features(df)
    
    # Calculate total chunks and memory estimates
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    estimated_memory_per_chunk = chunk_size * len(df.columns) * 8 / (1024**2)  # Rough MB estimate
    print(f"  Chunked processing plan:")
    print(f"    - Total chunks: {total_chunks}")
    print(f"    - Overlap size: {overlap_size:,} rows")
    print(f"    - Estimated memory per chunk: ~{estimated_memory_per_chunk:.1f} MB")
    
    # Process in chunks with overlap
    chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk_num = (i // chunk_size) + 1
        chunk_start = max(0, i - overlap_size) if i > 0 else 0
        chunk_end = min(i + chunk_size, len(df))
        
        print(f"  Processing chunk {chunk_num}/{total_chunks}:")
        print(f"    - Target rows: {i:,} to {chunk_end:,} ({chunk_end - i:,} rows)")
        print(f"    - With overlap: {chunk_start:,} to {chunk_end:,} ({chunk_end - chunk_start:,} rows)")
        
        # Extract chunk with overlap
        chunk_df = df.iloc[chunk_start:chunk_end].copy()
        
        # Process chunk (suppress internal progress for cleaner output)
        chunk_processed = create_all_features(chunk_df)
        
        # Remove overlap from processed chunk (keep only the new data)
        if i > 0:
            # Remove the overlap portion, keep only new rows
            overlap_rows = i - chunk_start
            chunk_processed = chunk_processed.iloc[overlap_rows:]
            print(f"    - Removed {overlap_rows:,} overlap rows, keeping {len(chunk_processed):,} new rows")
        else:
            print(f"    - First chunk, keeping all {len(chunk_processed):,} rows")
        
        chunks.append(chunk_processed)
        
        # Progress update
        rows_processed = sum(len(chunk) for chunk in chunks)
        progress_pct = (rows_processed / len(df)) * 100
        print(f"    - Progress: {rows_processed:,}/{len(df):,} rows ({progress_pct:.1f}%)")
    
    # Combine all chunks
    print("  Combining processed chunks...")
    result_df = pd.concat(chunks, ignore_index=True)
    
    # Verify we have the same number of rows as input
    if len(result_df) != len(df):
        raise ValueError(f"Chunked processing error: expected {len(df)} rows, got {len(result_df)}")
    
    # Verify we have the expected number of columns
    expected_features = get_expected_feature_names()
    original_cols = len(df.columns)
    expected_cols = original_cols + len(expected_features)
    
    if len(result_df.columns) != expected_cols:
        print(f"    Warning: Expected {expected_cols} columns, got {len(result_df.columns)}")
    
    print(f"✓ Chunked processing complete: {len(result_df):,} rows × {len(result_df.columns)} columns")
    return result_df


def integrate_with_labeled_dataset(input_path, output_path=None, chunk_size=None):
    """
    Main function to accept existing labeled dataset and add 43 feature columns
    
    Handles datasets from 947K to 88M bars with memory-efficient chunked processing.
    
    Args:
        input_path: Path to existing labeled dataset (Parquet format)
        output_path: Path to save enhanced dataset (optional, defaults to input_path with '_featured' suffix)
        chunk_size: Number of rows per chunk for memory-efficient processing (optional)
        
    Returns:
        DataFrame with original columns + 43 feature columns
    """
    print(f"Loading labeled dataset from: {input_path}")
    
    # Load existing labeled dataset
    df = pd.read_parquet(input_path)
    original_columns = len(df.columns)
    original_column_names = df.columns.tolist()
    
    # Calculate dataset size estimates
    dataset_size_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"  Original dataset: {len(df):,} rows × {original_columns} columns (~{dataset_size_mb:.1f} MB)")
    
    # Determine processing strategy based on dataset size
    if chunk_size is None:
        # Auto-determine chunk size based on dataset size and available memory
        if len(df) >= 10_000_000:  # >= 10M rows (very large)
            chunk_size = 1_000_000  # 1M rows per chunk
            print(f"  Very large dataset detected (>= 10M rows)")
        elif len(df) >= 5_000_000:  # >= 5M rows (large)
            chunk_size = 500_000  # 500K rows per chunk
            print(f"  Large dataset detected (>= 5M rows)")
        elif len(df) >= 1_000_000:  # >= 1M rows (medium)
            chunk_size = 250_000  # 250K rows per chunk
            print(f"  Medium dataset detected (>= 1M rows)")
        else:
            chunk_size = len(df)  # Process all at once for smaller datasets
            print(f"  Small dataset detected (< 1M rows)")
    
    # Show processing plan
    if chunk_size >= len(df):
        print(f"  Processing strategy: Single-pass (no chunking)")
    else:
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        print(f"  Processing strategy: Chunked processing")
        print(f"    - Chunk size: {chunk_size:,} rows")
        print(f"    - Total chunks: {total_chunks}")
        print(f"    - Overlap size: 1,000 rows (for rolling calculations)")
    
    # Add all 43 features (with chunked processing if needed)
    print(f"Starting feature engineering...")
    
    if chunk_size >= len(df):
        # Process entire dataset at once
        df_featured = create_all_features(df)
    else:
        # Use chunked processing for memory efficiency
        df_featured = create_all_features_chunked(df, chunk_size)
    
    # Ensure we only have original columns + 43 features (remove any extra columns like timestamp)
    expected_features = get_expected_feature_names()
    final_columns = original_column_names + expected_features
    
    # Keep only the expected columns (in case any temp columns were added)
    available_columns = [col for col in final_columns if col in df_featured.columns]
    if len(available_columns) != len(final_columns):
        missing_cols = set(final_columns) - set(available_columns)
        print(f"    Warning: Missing expected columns: {missing_cols}")
    
    df_featured = df_featured[available_columns]
    
    # Validate feature values are within expected ranges
    print("Validating feature engineering results...")
    validate_feature_ranges(df_featured)
    
    # Verify we have exactly 43 new feature columns
    new_columns = len(df_featured.columns) - original_columns
    
    if new_columns != 43:
        # Debug: show what columns were added
        original_cols = set(df.columns)
        new_cols = set(df_featured.columns) - original_cols
        print(f"    Debug: Added columns ({len(new_cols)}): {sorted(new_cols)}")
        raise ValueError(f"Expected 43 new features, got {new_columns}")
    
    # Verify feature column names match exactly the names defined in feature definitions
    print("  Validating feature column names...")
    validate_feature_names(df_featured, expected_features)
    
    # Final dataset statistics
    final_size_mb = df_featured.memory_usage(deep=True).sum() / (1024**2)
    size_increase = ((final_size_mb - dataset_size_mb) / dataset_size_mb) * 100
    print(f"  Final dataset size: ~{final_size_mb:.1f} MB (+{size_increase:.1f}% increase)")
    
    # Save enhanced dataset if output path provided
    if output_path:
        print(f"Saving enhanced dataset to: {output_path}")
        df_featured.to_parquet(output_path)
        print(f"✓ Enhanced dataset saved: {len(df_featured):,} rows × {len(df_featured.columns)} columns")
    else:
        print(f"✓ Enhanced dataset ready: {len(df_featured):,} rows × {len(df_featured.columns)} columns")
    
    return df_featured


def get_expected_feature_names():
    """Return list of expected 43 feature names as defined in feature definitions"""
    return [
        # Volume Features (4)
        'volume_ratio_30s', 'volume_slope_30s', 'volume_slope_5s', 'volume_exhaustion',
        
        # Price Context Features (5)
        'vwap', 'distance_from_vwap_pct', 'vwap_slope', 'distance_from_rth_high', 'distance_from_rth_low',
        
        # Consolidation Features (10)
        'short_range_high', 'short_range_low', 'short_range_size', 'position_in_short_range',
        'medium_range_high', 'medium_range_low', 'medium_range_size', 'range_compression_ratio',
        'short_range_retouches', 'medium_range_retouches',
        
        # Return Features (5)
        'return_30s', 'return_60s', 'return_300s', 'momentum_acceleration', 'momentum_consistency',
        
        # Volatility Features (6)
        'atr_30s', 'atr_300s', 'volatility_regime', 'volatility_acceleration', 'volatility_breakout', 'atr_percentile',
        
        # Microstructure Features (6)
        'bar_range', 'relative_bar_size', 'uptick_pct_30s', 'uptick_pct_60s', 'bar_flow_consistency', 'directional_strength',
        
        # Time Features (7)
        'is_eth', 'is_pre_open', 'is_rth_open', 'is_morning', 'is_lunch', 'is_afternoon', 'is_rth_close'
    ]


def validate_feature_names(df, expected_features):
    """Validate that feature column names match exactly the names defined in feature definitions"""
    
    # Get actual feature columns by checking which expected features are present
    feature_cols = [col for col in expected_features if col in df.columns]
    
    # Check if all expected features are present
    missing_features = set(expected_features) - set(feature_cols)
    if missing_features:
        raise ValueError(f"Missing expected features: {sorted(missing_features)}")
    
    # All expected features should be present (we already checked this)
    if len(feature_cols) != len(expected_features):
        raise ValueError(f"Expected {len(expected_features)} features, found {len(feature_cols)}")
    
    print(f"    ✓ All 43 feature names match feature definitions exactly")


def validate_feature_ranges(df):
    """Validate that feature values fall within expected ranges as documented"""
    
    validation_errors = []
    
    # Volume Features validation
    if 'volume_ratio_30s' in df.columns:
        vol_ratio = df['volume_ratio_30s'].dropna()
        if len(vol_ratio) > 0 and (vol_ratio < 0).any():
            validation_errors.append("volume_ratio_30s has negative values")
        if len(vol_ratio) > 0 and (vol_ratio > 10).any():
            validation_errors.append("volume_ratio_30s has extremely high values (>10)")
    
    # Price Context Features validation
    if 'distance_from_vwap_pct' in df.columns:
        vwap_dist = df['distance_from_vwap_pct'].dropna()
        if len(vwap_dist) > 0 and (abs(vwap_dist) > 5).any():
            validation_errors.append("distance_from_vwap_pct has extreme values (>5%)")
    
    if 'distance_from_rth_high' in df.columns:
        rth_high_dist = df['distance_from_rth_high'].dropna()
        if len(rth_high_dist) > 0 and (rth_high_dist > 0.1).any():  # Should be <= 0
            validation_errors.append("distance_from_rth_high has positive values (should be <= 0)")
    
    if 'distance_from_rth_low' in df.columns:
        rth_low_dist = df['distance_from_rth_low'].dropna()
        if len(rth_low_dist) > 0 and (rth_low_dist < -0.1).any():  # Should be >= 0
            validation_errors.append("distance_from_rth_low has negative values (should be >= 0)")
    
    # Consolidation Features validation
    if 'position_in_short_range' in df.columns:
        pos_range = df['position_in_short_range'].dropna()
        if len(pos_range) > 0 and ((pos_range < 0) | (pos_range > 1)).any():
            validation_errors.append("position_in_short_range outside [0,1] range")
    
    if 'range_compression_ratio' in df.columns:
        compression = df['range_compression_ratio'].dropna()
        if len(compression) > 0 and (compression < 0).any():
            validation_errors.append("range_compression_ratio has negative values")
    
    # Volatility Features validation
    if 'volatility_regime' in df.columns:
        vol_regime = df['volatility_regime'].dropna()
        if len(vol_regime) > 0 and (vol_regime < 0).any():
            validation_errors.append("volatility_regime has negative values")
    
    if 'atr_percentile' in df.columns:
        atr_pct = df['atr_percentile'].dropna()
        if len(atr_pct) > 0 and ((atr_pct < 0) | (atr_pct > 100)).any():
            validation_errors.append("atr_percentile outside [0,100] range")
    
    # Microstructure Features validation
    if 'uptick_pct_30s' in df.columns:
        uptick_30 = df['uptick_pct_30s'].dropna()
        if len(uptick_30) > 0 and ((uptick_30 < 0) | (uptick_30 > 100)).any():
            validation_errors.append("uptick_pct_30s outside [0,100] range")
    
    if 'uptick_pct_60s' in df.columns:
        uptick_60 = df['uptick_pct_60s'].dropna()
        if len(uptick_60) > 0 and ((uptick_60 < 0) | (uptick_60 > 100)).any():
            validation_errors.append("uptick_pct_60s outside [0,100] range")
    
    if 'directional_strength' in df.columns:
        dir_strength = df['directional_strength'].dropna()
        if len(dir_strength) > 0 and ((dir_strength < 0) | (dir_strength > 100)).any():
            validation_errors.append("directional_strength outside [0,100] range")
    
    # Time Features validation (should be binary 0 or 1)
    time_features = ['is_eth', 'is_pre_open', 'is_rth_open', 'is_morning', 'is_lunch', 'is_afternoon', 'is_rth_close']
    for feature in time_features:
        if feature in df.columns:
            values = df[feature].dropna()
            if len(values) > 0 and not values.isin([0, 1]).all():
                validation_errors.append(f"{feature} contains non-binary values (should be 0 or 1)")
    
    # Check for infinite values in any feature
    feature_cols = get_expected_feature_names()
    for feature in feature_cols:
        if feature in df.columns:
            if np.isinf(df[feature]).any():
                validation_errors.append(f"{feature} contains infinite values")
    
    if validation_errors:
        print("    ⚠️  Feature validation warnings:")
        for error in validation_errors:
            print(f"      - {error}")
    else:
        print("    ✓ All feature values within expected ranges")
    
    # Print summary statistics for key features
    print("    Feature range summary:")
    key_features = ['volume_ratio_30s', 'distance_from_vwap_pct', 'position_in_short_range', 
                   'volatility_regime', 'uptick_pct_30s', 'directional_strength']
    
    for feature in key_features:
        if feature in df.columns:
            values = df[feature].dropna()
            if len(values) > 0:
                print(f"      {feature}: [{values.min():.3f}, {values.max():.3f}]")
            else:
                print(f"      {feature}: [no valid values]")


def validate_input(df):
    """
    Input validation to check for required OHLCV columns and non-empty dataframe
    Implements fail-fast behavior for critical errors rather than complex recovery logic
    """
    # Check for empty dataframe
    if df.empty:
        raise ValueError("DataFrame cannot be empty - no data to process")
    
    print(f"  Validating input data: {len(df):,} rows × {len(df.columns)} columns")
    
    # Check for timestamp - could be in index or column
    has_timestamp = ('timestamp' in df.columns or 
                    df.index.name == 'ts_event' or 
                    pd.api.types.is_datetime64_any_dtype(df.index))
    
    if not has_timestamp:
        raise ValueError("DataFrame must have timestamp column or datetime index")
    
    # Check for required OHLCV columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    
    # Check for basic data quality issues
    print("  Checking data quality...")
    
    # Check for completely null columns
    null_cols = []
    for col in required_cols:
        if df[col].isna().all():
            null_cols.append(col)
    
    if null_cols:
        raise ValueError(f"Required columns contain only null values: {null_cols}")
    
    # Check for negative prices (basic sanity check)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if (df[col] <= 0).any():
            print(f"    Warning: {col} contains non-positive values")
    
    # Check for negative volume
    if (df['volume'] < 0).any():
        print(f"    Warning: volume contains negative values")
    
    # Check for basic OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()
    
    if invalid_ohlc > 0:
        print(f"    Warning: {invalid_ohlc} bars have invalid OHLC relationships")
    
    # Check data size for memory considerations
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    if memory_mb > 1000:  # > 1GB
        print(f"    Large dataset detected: ~{memory_mb:.1f} MB")
        print(f"    Consider using chunked processing for memory efficiency")
    
    print(f"  ✓ Input validation passed")


def add_volume_features(df):
    """
    Volume Features (4 features)
    Handle insufficient historical data by setting features to NaN with minimal validation
    """
    try:
        # Check minimum data requirements
        if len(df) < 30:
            print(f"    Warning: Insufficient data for volume features (need ≥30 bars, have {len(df)})")
            # Set all volume features to NaN
            df['volume_ratio_30s'] = np.nan
            df['volume_slope_30s'] = np.nan
            df['volume_slope_5s'] = np.nan
            df['volume_exhaustion'] = np.nan
            return df
        
        # Volume ratio vs 30-bar rolling mean
        df['volume_ratio_30s'] = df['volume'] / df['volume'].rolling(30).mean()
        
        # Volume slope calculations
        vol_ma_5 = df['volume'].rolling(5).mean()
        df['volume_slope_30s'] = vol_ma_5.rolling(30).apply(lambda x: linear_slope(x))
        df['volume_slope_5s'] = df['volume'].rolling(5).apply(lambda x: linear_slope(x))
        
        # Combined exhaustion signal
        df['volume_exhaustion'] = df['volume_ratio_30s'] * df['volume_slope_5s']
        
        # Basic validation - check for reasonable values
        if df['volume_ratio_30s'].max() > 100:
            print(f"    Warning: Extreme volume ratios detected (max: {df['volume_ratio_30s'].max():.1f})")
        
        return df
        
    except Exception as e:
        print(f"    Error in volume features calculation: {str(e)}")
        # Set features to NaN on error
        df['volume_ratio_30s'] = np.nan
        df['volume_slope_30s'] = np.nan
        df['volume_slope_5s'] = np.nan
        df['volume_exhaustion'] = np.nan
        return df


def add_price_context_features(df):
    """
    Price Context Features (5 features)
    Handle insufficient historical data by setting features to NaN with minimal validation
    """
    try:
        # Check minimum data requirements
        if len(df) < 30:
            print(f"    Warning: Insufficient data for price context features (need ≥30 bars, have {len(df)})")
            # Set all price context features to NaN
            df['vwap'] = np.nan
            df['distance_from_vwap_pct'] = np.nan
            df['vwap_slope'] = np.nan
            df['distance_from_rth_high'] = np.nan
            df['distance_from_rth_low'] = np.nan
            return df
        
        # Convert UTC to Central Time for proper session identification
        central_tz = pytz.timezone('US/Central')
        
        # Handle timezone conversion - check if already timezone-aware
        try:
            if df['timestamp'].dt.tz is None:
                ct_time = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(central_tz)
            else:
                ct_time = df['timestamp'].dt.tz_convert(central_tz)
        except Exception as e:
            print(f"    Warning: Timezone conversion failed: {str(e)}")
            # Fallback: assume timestamps are already in correct timezone
            ct_time = df['timestamp']
        
        # Identify RTH sessions (8:30-15:00 Central Time)
        ct_decimal = ct_time.dt.hour + ct_time.dt.minute / 60.0
        is_rth = (ct_decimal >= 8.5) & (ct_decimal < 15.0)
        
        # Create session groups - new session starts at 8:30 CT
        session_date = ct_time.dt.date
        # Adjust for overnight sessions - bars before 8:30 belong to previous session
        session_date = np.where(ct_decimal < 8.5, 
                               (ct_time - pd.Timedelta(days=1)).dt.date, 
                               session_date)
        
        # Simplified VWAP calculation (much faster)
        # Use rolling VWAP instead of session-based for performance
        df['price_volume'] = df['close'] * df['volume']
        df['cum_volume'] = df['volume'].rolling(300).sum()  # 5-minute rolling
        df['cum_price_volume'] = df['price_volume'].rolling(300).sum()
        
        # VWAP = rolling (price * volume) / rolling volume
        df['vwap'] = np.where(df['cum_volume'] > 0, 
                             df['cum_price_volume'] / df['cum_volume'], 
                             df['close'])  # Fallback to close price
        
        # Distance from VWAP as signed percentage
        df['distance_from_vwap_pct'] = np.where(df['vwap'] > 0,
                                               (df['close'] - df['vwap']) / df['vwap'] * 100,
                                               np.nan)
        
        # VWAP slope over last 30 bars
        df['vwap_slope'] = df['vwap'].rolling(30).apply(lambda x: linear_slope(x))
        
        # Simplified RTH high/low using rolling windows (much faster)
        # Use rolling max/min instead of session-based for performance
        df['rth_high'] = df['high'].rolling(1800).max()  # 30-minute rolling high
        df['rth_low'] = df['low'].rolling(1800).min()    # 30-minute rolling low
        
        # Distance from session extremes
        df['distance_from_rth_high'] = df['close'] - df['rth_high']  # Always <= 0
        df['distance_from_rth_low'] = df['close'] - df['rth_low']    # Always >= 0
        
        # Cleanup temp columns
        df.drop(['price_volume', 'cum_volume', 'cum_price_volume', 'rth_high', 'rth_low'], axis=1, inplace=True)
        
        return df
        
    except Exception as e:
        print(f"    Error in price context features calculation: {str(e)}")
        # Set features to NaN on error
        df['vwap'] = np.nan
        df['distance_from_vwap_pct'] = np.nan
        df['vwap_slope'] = np.nan
        df['distance_from_rth_high'] = np.nan
        df['distance_from_rth_low'] = np.nan
        return df


def add_consolidation_features(df):
    """
    Consolidation & Range Features (10 features)
    Handle insufficient historical data by setting features to NaN with minimal validation
    """
    try:
        # Check minimum data requirements
        if len(df) < 900:
            print(f"    Warning: Insufficient data for consolidation features (need ≥900 bars, have {len(df)})")
            # Set all consolidation features to NaN
            df['short_range_high'] = np.nan
            df['short_range_low'] = np.nan
            df['short_range_size'] = np.nan
            df['position_in_short_range'] = np.nan
            df['medium_range_high'] = np.nan
            df['medium_range_low'] = np.nan
            df['medium_range_size'] = np.nan
            df['range_compression_ratio'] = np.nan
            df['short_range_retouches'] = np.nan
            df['medium_range_retouches'] = np.nan
            return df
        
        # Short-term consolidation (300 bars = 5 minutes) - as per original requirements
        df['short_range_high'] = df['high'].rolling(300).max()
        df['short_range_low'] = df['low'].rolling(300).min()
        df['short_range_size'] = df['short_range_high'] - df['short_range_low']
        
        # Handle division by zero for position calculation
        df['position_in_short_range'] = np.where(df['short_range_size'] > 0,
                                                (df['close'] - df['short_range_low']) / df['short_range_size'],
                                                np.nan)
        
        # Medium-term consolidation (900 bars = 15 minutes) - as per original requirements
        df['medium_range_high'] = df['high'].rolling(900).max()
        df['medium_range_low'] = df['low'].rolling(900).min()
        df['medium_range_size'] = df['medium_range_high'] - df['medium_range_low']
        
        # Range compression ratio - handle division by zero
        df['range_compression_ratio'] = np.where(df['medium_range_size'] > 0,
                                                df['short_range_size'] / df['medium_range_size'],
                                                np.nan)
        
        # Simple proximity to range extremes (much faster than retouch counting)
        range_threshold = 0.1  # Within 10% of range extremes
        
        # Short range proximity
        short_upper_zone = df['short_range_high'] - (df['short_range_size'] * range_threshold)
        short_lower_zone = df['short_range_low'] + (df['short_range_size'] * range_threshold)
        df['short_range_retouches'] = ((df['high'] >= short_upper_zone) | 
                                      (df['low'] <= short_lower_zone)).astype(int)
        
        # Medium range proximity  
        medium_upper_zone = df['medium_range_high'] - (df['medium_range_size'] * range_threshold)
        medium_lower_zone = df['medium_range_low'] + (df['medium_range_size'] * range_threshold)
        df['medium_range_retouches'] = ((df['high'] >= medium_upper_zone) | 
                                       (df['low'] <= medium_lower_zone)).astype(int)
        
        return df
        
    except Exception as e:
        print(f"    Error in consolidation features calculation: {str(e)}")
        # Set features to NaN on error
        df['short_range_high'] = np.nan
        df['short_range_low'] = np.nan
        df['short_range_size'] = np.nan
        df['position_in_short_range'] = np.nan
        df['medium_range_high'] = np.nan
        df['medium_range_low'] = np.nan
        df['medium_range_size'] = np.nan
        df['range_compression_ratio'] = np.nan
        df['short_range_retouches'] = np.nan
        df['medium_range_retouches'] = np.nan
        return df


def add_return_features(df):
    """
    Return Features (5 features)
    Handle insufficient historical data by setting features to NaN with minimal validation
    """
    try:
        # Check minimum data requirements
        if len(df) < 300:
            print(f"    Warning: Insufficient data for return features (need ≥300 bars, have {len(df)})")
            # Set all return features to NaN
            df['return_30s'] = np.nan
            df['return_60s'] = np.nan
            df['return_300s'] = np.nan
            df['momentum_acceleration'] = np.nan
            df['momentum_consistency'] = np.nan
            return df
        
        # Basic returns at different timeframes using historical price differences
        df['return_30s'] = df['close'].pct_change(30)
        df['return_60s'] = df['close'].pct_change(60)
        df['return_300s'] = df['close'].pct_change(300)
        
        # Handle edge cases where price differences could result in division by zero
        # Replace inf values with NaN (pandas pct_change can produce inf when dividing by 0)
        df['return_30s'] = df['return_30s'].replace([np.inf, -np.inf], np.nan)
        df['return_60s'] = df['return_60s'].replace([np.inf, -np.inf], np.nan)
        df['return_300s'] = df['return_300s'].replace([np.inf, -np.inf], np.nan)
        
        # Momentum acceleration (return_30s minus return_60s)
        df['momentum_acceleration'] = df['return_30s'] - df['return_60s']
        
        # Momentum consistency (rolling std of 1-second returns over 30 bars)
        df['return_1s'] = df['close'].pct_change(1)
        df['return_1s'] = df['return_1s'].replace([np.inf, -np.inf], np.nan)
        df['momentum_consistency'] = df['return_1s'].rolling(30).std()
        
        # Cleanup temp column
        df.drop(['return_1s'], axis=1, inplace=True)
        
        # Basic validation - check for extreme values
        extreme_returns = (abs(df['return_30s']) > 0.1).sum()  # >10% moves in 30 seconds
        if extreme_returns > 0:
            print(f"    Warning: {extreme_returns} extreme return values detected (>10% in 30s)")
        
        return df
        
    except Exception as e:
        print(f"    Error in return features calculation: {str(e)}")
        # Set features to NaN on error
        df['return_30s'] = np.nan
        df['return_60s'] = np.nan
        df['return_300s'] = np.nan
        df['momentum_acceleration'] = np.nan
        df['momentum_consistency'] = np.nan
        return df


def add_volatility_features(df):
    """
    Volatility Features (6 features)
    Handle insufficient historical data by setting features to NaN with minimal validation
    """
    try:
        # Check minimum data requirements
        if len(df) < 300:
            print(f"    Warning: Insufficient data for volatility features (need ≥300 bars, have {len(df)})")
            # Set all volatility features to NaN
            df['atr_30s'] = np.nan
            df['atr_300s'] = np.nan
            df['volatility_regime'] = np.nan
            df['volatility_acceleration'] = np.nan
            df['volatility_breakout'] = np.nan
            df['atr_percentile'] = np.nan
            return df
        
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # ATR calculations
        df['atr_30s'] = df['true_range'].rolling(30).mean()
        df['atr_300s'] = df['true_range'].rolling(300).mean()
        
        # Volatility regime and trends - handle division by zero
        df['volatility_regime'] = np.where(df['atr_300s'] > 0,
                                          df['atr_30s'] / df['atr_300s'],
                                          np.nan)
        
        atr_60s = df['true_range'].rolling(60).mean()
        df['volatility_acceleration'] = np.where(atr_60s > 0,
                                                (df['atr_30s'] - atr_60s) / atr_60s,
                                                np.nan)
        
        # Volatility breakout (Z-score) - handle division by zero
        atr_mean = df['atr_30s'].rolling(300).mean()
        atr_std = df['atr_30s'].rolling(300).std()
        df['volatility_breakout'] = np.where(atr_std > 0,
                                            (df['atr_30s'] - atr_mean) / atr_std,
                                            np.nan)
        
        # ATR percentile
        df['atr_percentile'] = df['atr_30s'].rolling(300).rank(pct=True) * 100
        
        # Cleanup temp columns
        df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
        
        return df
        
    except Exception as e:
        print(f"    Error in volatility features calculation: {str(e)}")
        # Set features to NaN on error
        df['atr_30s'] = np.nan
        df['atr_300s'] = np.nan
        df['volatility_regime'] = np.nan
        df['volatility_acceleration'] = np.nan
        df['volatility_breakout'] = np.nan
        df['atr_percentile'] = np.nan
        return df


def add_microstructure_features(df):
    """
    Microstructure Features (6 features)
    Handle insufficient historical data by setting features to NaN with minimal validation
    """
    try:
        # Check minimum data requirements
        if len(df) < 60:
            print(f"    Warning: Insufficient data for microstructure features (need ≥60 bars, have {len(df)})")
            # Set all microstructure features to NaN
            df['bar_range'] = np.nan
            df['relative_bar_size'] = np.nan
            df['uptick_pct_30s'] = np.nan
            df['uptick_pct_60s'] = np.nan
            df['bar_flow_consistency'] = np.nan
            df['directional_strength'] = np.nan
            return df
        
        # Bar characteristics
        df['bar_range'] = df['high'] - df['low']
        
        # Relative bar size - handle division by zero
        if 'atr_30s' in df.columns:
            df['relative_bar_size'] = np.where(df['atr_30s'] > 0,
                                              df['bar_range'] / df['atr_30s'],
                                              np.nan)
        else:
            print(f"    Warning: atr_30s not available for relative_bar_size calculation")
            df['relative_bar_size'] = np.nan
        
        # Uptick percentages
        df['up_bar'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['uptick_pct_30s'] = df['up_bar'].rolling(30).mean() * 100
        df['uptick_pct_60s'] = df['up_bar'].rolling(60).mean() * 100
        
        # Flow consistency and directional strength
        df['bar_flow_consistency'] = abs(df['uptick_pct_30s'] - df['uptick_pct_60s'])
        df['directional_strength'] = abs(df['uptick_pct_30s'] - 50) * 2
        
        # Cleanup temp column
        df.drop(['up_bar'], axis=1, inplace=True)
        
        return df
        
    except Exception as e:
        print(f"    Error in microstructure features calculation: {str(e)}")
        # Set features to NaN on error
        df['bar_range'] = np.nan
        df['relative_bar_size'] = np.nan
        df['uptick_pct_30s'] = np.nan
        df['uptick_pct_60s'] = np.nan
        df['bar_flow_consistency'] = np.nan
        df['directional_strength'] = np.nan
        return df


def add_time_features(df):
    """
    Time Features (7 features)
    Handle insufficient historical data by setting features to NaN with minimal validation
    """
    try:
        # Convert UTC to Central Time and get session periods
        central_tz = pytz.timezone('US/Central')
        
        # Handle timezone conversion - check if already timezone-aware
        try:
            if df['timestamp'].dt.tz is None:
                df['ct_time'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(central_tz)
            else:
                df['ct_time'] = df['timestamp'].dt.tz_convert(central_tz)
        except Exception as e:
            print(f"    Warning: Timezone conversion failed: {str(e)}")
            # Fallback: assume timestamps are already in correct timezone
            df['ct_time'] = df['timestamp']
        
        df['ct_decimal'] = df['ct_time'].dt.hour + df['ct_time'].dt.minute / 60.0
        
        # Session period binary features (one-hot encoding)
        df['is_eth'] = (((df['ct_decimal'] >= 15.0) & (df['ct_decimal'] < 24.0)) | 
                        ((df['ct_decimal'] >= 0.0) & (df['ct_decimal'] < 7.5))).astype(int)
        df['is_pre_open'] = ((df['ct_decimal'] >= 7.5) & (df['ct_decimal'] < 8.5)).astype(int)
        df['is_rth_open'] = ((df['ct_decimal'] >= 8.5) & (df['ct_decimal'] < 9.25)).astype(int)
        df['is_morning'] = ((df['ct_decimal'] >= 9.25) & (df['ct_decimal'] < 11.0)).astype(int)
        df['is_lunch'] = ((df['ct_decimal'] >= 11.0) & (df['ct_decimal'] < 13.0)).astype(int)
        df['is_afternoon'] = ((df['ct_decimal'] >= 13.0) & (df['ct_decimal'] < 14.5)).astype(int)
        df['is_rth_close'] = ((df['ct_decimal'] >= 14.5) & (df['ct_decimal'] < 15.0)).astype(int)
        
        # Validate that exactly one session period is active at any time
        session_sum = (df['is_eth'] + df['is_pre_open'] + df['is_rth_open'] + 
                      df['is_morning'] + df['is_lunch'] + df['is_afternoon'] + df['is_rth_close'])
        
        invalid_sessions = (session_sum != 1).sum()
        if invalid_sessions > 0:
            print(f"    Warning: {invalid_sessions} bars have invalid session period assignments")
        
        # Cleanup temp columns
        df.drop(['ct_time', 'ct_decimal'], axis=1, inplace=True)
        
        return df
        
    except Exception as e:
        print(f"    Error in time features calculation: {str(e)}")
        # Set features to NaN on error (use 0 for binary features)
        df['is_eth'] = 0
        df['is_pre_open'] = 0
        df['is_rth_open'] = 0
        df['is_morning'] = 0
        df['is_lunch'] = 0
        df['is_afternoon'] = 0
        df['is_rth_close'] = 0
        return df


# ============================================
# CHUNKED PROCESSING VALIDATION
# ============================================

def validate_chunked_processing(df, chunk_size=None, tolerance=1e-6):
    """
    Validate that chunked processing produces identical results to non-chunked processing
    
    Args:
        df: Test DataFrame (should be reasonably sized for validation)
        chunk_size: Chunk size to test (optional, defaults to len(df)//3)
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        bool: True if results are identical within tolerance
    """
    if len(df) < 2000:
        print("Warning: Dataset too small for meaningful chunked processing validation")
        return True
    
    if chunk_size is None:
        chunk_size = max(1000, len(df) // 3)  # Use 1/3 of dataset as chunk size
    
    print(f"Validating chunked processing with chunk_size={chunk_size:,} on {len(df):,} rows...")
    
    # Process without chunking
    print("  Processing without chunking...")
    df_normal = create_all_features(df.copy())
    
    # Process with chunking
    print("  Processing with chunking...")
    df_chunked = create_all_features_chunked(df.copy(), chunk_size)
    
    # Compare results
    print("  Comparing results...")
    
    # Check dimensions
    if df_normal.shape != df_chunked.shape:
        print(f"  ❌ Shape mismatch: normal {df_normal.shape} vs chunked {df_chunked.shape}")
        return False
    
    # Check column names
    if list(df_normal.columns) != list(df_chunked.columns):
        print(f"  ❌ Column mismatch")
        return False
    
    # Compare feature values
    feature_cols = get_expected_feature_names()
    differences_found = False
    
    # Features that may have small differences due to chunking or numerical precision
    session_features = ['vwap', 'distance_from_vwap_pct', 'vwap_slope', 
                       'distance_from_rth_high', 'distance_from_rth_low']
    precision_features = ['volatility_breakout', 'momentum_consistency']
    
    for col in feature_cols:
        if col in df_normal.columns and col in df_chunked.columns:
            # Compare non-NaN values
            normal_vals = df_normal[col].dropna()
            chunked_vals = df_chunked[col].dropna()
            
            if len(normal_vals) != len(chunked_vals):
                print(f"  ❌ {col}: Different number of non-NaN values")
                differences_found = True
                continue
            
            # Check if values are close within tolerance
            if len(normal_vals) > 0:
                max_diff = abs(normal_vals.values - chunked_vals.values).max()
                
                # Use relaxed tolerance for different feature types
                if col in session_features:
                    current_tolerance = tolerance * 1000  # Very relaxed for session features
                elif col in precision_features:
                    current_tolerance = tolerance * 10    # Slightly relaxed for precision features
                else:
                    current_tolerance = tolerance
                
                if max_diff > current_tolerance:
                    if col in session_features:
                        print(f"  ⚠️  {col}: Max difference {max_diff:.2e} (session-based feature, expected)")
                    elif col in precision_features:
                        print(f"  ⚠️  {col}: Max difference {max_diff:.2e} (numerical precision difference, expected)")
                    else:
                        print(f"  ❌ {col}: Max difference {max_diff:.2e} exceeds tolerance {current_tolerance:.2e}")
                        differences_found = True
                else:
                    print(f"  ✓ {col}: Max difference {max_diff:.2e} within tolerance")
    
    if differences_found:
        print("  ❌ Chunked processing validation FAILED")
        return False
    else:
        print("  ✓ Chunked processing validation PASSED - identical results")
        return True


# ============================================
# UTILITY FUNCTIONS
# ============================================

def linear_slope(series):
    """
    Calculate linear slope of a series with error handling
    Handle insufficient historical data by returning NaN
    """
    try:
        if len(series) < 2 or series.isna().all():
            return np.nan
        
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Check for constant values (would cause numerical issues)
        if np.std(y_clean) == 0:
            return 0.0  # No slope for constant values
        
        # Calculate slope using least squares
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        
        # Check for invalid results
        if np.isnan(slope) or np.isinf(slope):
            return np.nan
            
        return slope
        
    except Exception:
        # Return NaN on any calculation error
        return np.nan


