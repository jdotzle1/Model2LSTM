"""
Corrected Contract Filtering and RTH Processing

Pipeline:
1. Identify primary contract per day (by volume)
2. Remove all other contracts
3. Filter to RTH only (07:30-15:00 CT)
4. Fill gaps to create 1-second resolution
"""
import pandas as pd
import numpy as np
import pytz
from datetime import time as dt_time, datetime, timedelta
from typing import Tuple, Dict


# RTH Definition: 07:30 CT to 15:00 CT
RTH_START = dt_time(7, 30)
RTH_END = dt_time(15, 0)
CENTRAL_TZ = pytz.timezone('US/Central')


def filter_primary_contract_by_volume(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Keep only the primary (highest volume) contract for each trading day
    
    Args:
        df: DataFrame with timestamp, symbol (or instrument_id), volume columns
        
    Returns:
        Tuple of (filtered_df, stats_dict)
    """
    print("Step 1: Filtering to primary contract per day...")
    
    stats = {
        'original_rows': len(df),
        'original_dates': 0,
        'filtered_rows': 0,
        'filtered_dates': 0,
        'removed_rows': 0,
        'days_with_multiple_contracts': 0
    }
    
    # Ensure timestamp is datetime with timezone
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
    
    # Convert to Central Time for daily grouping
    df['timestamp_ct'] = df['timestamp'].dt.tz_convert(CENTRAL_TZ)
    df['date'] = df['timestamp_ct'].dt.date
    
    stats['original_dates'] = df['date'].nunique()
    
    # Determine which column to use for contract identification
    if 'symbol' in df.columns:
        contract_col = 'symbol'
    elif 'instrument_id' in df.columns:
        contract_col = 'instrument_id'
    else:
        print("   ⚠️  No symbol or instrument_id column found!")
        print("   Using price gap detection as fallback...")
        return _filter_by_price_gaps(df, stats)
    
    # Calculate total volume per contract per day
    daily_contract_volumes = df.groupby(['date', contract_col])['volume'].sum().reset_index()
    daily_contract_volumes.columns = ['date', 'contract', 'total_volume']
    
    # Find primary contract (highest volume) for each day
    idx = daily_contract_volumes.groupby('date')['total_volume'].idxmax()
    primary_contracts = daily_contract_volumes.loc[idx][['date', 'contract']].copy()
    
    # Count days with multiple contracts
    contracts_per_day = daily_contract_volumes.groupby('date')['contract'].count()
    stats['days_with_multiple_contracts'] = (contracts_per_day > 1).sum()
    
    # Filter to keep only primary contract bars
    df_filtered = df.merge(primary_contracts, left_on=['date', contract_col], 
                           right_on=['date', 'contract'], how='inner')
    
    # Clean up temporary columns
    df_filtered = df_filtered.drop(columns=['timestamp_ct', 'date', 'contract'], errors='ignore')
    
    stats['filtered_rows'] = len(df_filtered)
    stats['filtered_dates'] = df_filtered['timestamp'].dt.date.nunique()
    stats['removed_rows'] = stats['original_rows'] - stats['filtered_rows']
    stats['removal_percentage'] = (stats['removed_rows'] / stats['original_rows'] * 100) if stats['original_rows'] > 0 else 0
    
    print(f"   Original: {stats['original_rows']:,} rows across {stats['original_dates']} days")
    print(f"   Filtered: {stats['filtered_rows']:,} rows across {stats['filtered_dates']} days")
    print(f"   Removed: {stats['removed_rows']:,} rows ({stats['removal_percentage']:.1f}%)")
    print(f"   Days with multiple contracts: {stats['days_with_multiple_contracts']}")
    
    return df_filtered, stats


def _filter_by_price_gaps(df: pd.DataFrame, stats: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Fallback: Use price gaps to detect contract segments, then select by volume
    """
    print("   Using price gap detection (no symbol column available)...")
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp').copy()
    
    # Detect large price gaps (>5 points) as potential contract switches
    df_sorted['price_change'] = df_sorted['close'].diff().abs()
    df_sorted['is_gap'] = df_sorted['price_change'] > 5.0
    
    # Assign segment IDs
    df_sorted['segment'] = df_sorted['is_gap'].cumsum()
    
    # For each day, calculate volume per segment
    daily_segment_volumes = df_sorted.groupby(['date', 'segment'])['volume'].sum().reset_index()
    daily_segment_volumes.columns = ['date', 'segment', 'total_volume']
    
    # Find primary segment (highest volume) for each day
    idx = daily_segment_volumes.groupby('date')['total_volume'].idxmax()
    primary_segments = daily_segment_volumes.loc[idx][['date', 'segment']].copy()
    
    # Filter to keep only primary segment bars
    df_filtered = df_sorted.merge(primary_segments, on=['date', 'segment'], how='inner')
    
    # Clean up temporary columns
    df_filtered = df_filtered.drop(columns=['timestamp_ct', 'date', 'price_change', 
                                             'is_gap', 'segment'], errors='ignore')
    
    stats['filtered_rows'] = len(df_filtered)
    stats['filtered_dates'] = df_filtered['timestamp'].dt.date.nunique()
    stats['removed_rows'] = stats['original_rows'] - stats['filtered_rows']
    stats['removal_percentage'] = (stats['removed_rows'] / stats['original_rows'] * 100) if stats['original_rows'] > 0 else 0
    
    print(f"   Filtered: {stats['filtered_rows']:,} rows")
    print(f"   Removed: {stats['removed_rows']:,} rows ({stats['removal_percentage']:.1f}%)")
    
    return df_filtered, stats


def filter_rth(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Filter to RTH only (07:30 CT to 15:00 CT)
    
    Handles daylight savings time automatically using pytz
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        Tuple of (filtered_df, stats_dict)
    """
    print("\nStep 2: Filtering to RTH (07:30-15:00 CT)...")
    
    stats = {
        'original_rows': len(df),
        'rth_rows': 0,
        'eth_rows': 0,
        'removal_percentage': 0
    }
    
    # Ensure timestamp has timezone
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
    
    # Convert to Central Time (handles DST automatically)
    df['timestamp_ct'] = df['timestamp'].dt.tz_convert(CENTRAL_TZ)
    df['time_ct'] = df['timestamp_ct'].dt.time
    
    # Filter to RTH
    rth_mask = (df['time_ct'] >= RTH_START) & (df['time_ct'] < RTH_END)
    df_rth = df[rth_mask].copy()
    
    # Clean up temporary columns
    df_rth = df_rth.drop(columns=['timestamp_ct', 'time_ct'])
    
    stats['rth_rows'] = len(df_rth)
    stats['eth_rows'] = stats['original_rows'] - stats['rth_rows']
    stats['removal_percentage'] = (stats['eth_rows'] / stats['original_rows'] * 100) if stats['original_rows'] > 0 else 0
    
    print(f"   Original: {stats['original_rows']:,} rows")
    print(f"   RTH: {stats['rth_rows']:,} rows")
    print(f"   ETH removed: {stats['eth_rows']:,} rows ({stats['removal_percentage']:.1f}%)")
    
    return df_rth, stats


def fill_gaps_to_1_second(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Fill gaps to create 1-second resolution
    
    For each trading day:
    - Create complete 1-second timestamp range (07:30-15:00 CT)
    - Merge with actual data
    - Forward fill OHLC for missing seconds
    - Set volume=0 for missing seconds
    
    Args:
        df: DataFrame with timestamp, open, high, low, close, volume
        
    Returns:
        Tuple of (filled_df, stats_dict)
    """
    print("\nStep 3: Filling gaps to 1-second resolution...")
    
    stats = {
        'original_rows': len(df),
        'filled_rows': 0,
        'added_rows': 0,
        'trading_days': 0,
        'expected_rows_per_day': 27000  # 7.5 hours * 3600 seconds
    }
    
    # Ensure timestamp has timezone
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
    
    # Get unique trading days
    df['timestamp_ct'] = df['timestamp'].dt.tz_convert(CENTRAL_TZ)
    df['date'] = df['timestamp_ct'].dt.date
    trading_days = sorted(df['date'].unique())
    stats['trading_days'] = len(trading_days)
    
    print(f"   Processing {stats['trading_days']} trading days...")
    
    filled_dfs = []
    
    for i, date in enumerate(trading_days, 1):
        if i % 5 == 0 or i == len(trading_days):
            print(f"   Progress: {i}/{len(trading_days)} days", end='\r')
        
        # Create complete 1-second range for this day (in Central Time)
        start_ct = CENTRAL_TZ.localize(datetime.combine(date, RTH_START))
        end_ct = CENTRAL_TZ.localize(datetime.combine(date, RTH_END))
        
        # Generate all timestamps (excluding end time)
        full_range_ct = pd.date_range(start_ct, end_ct, freq='1s', inclusive='left')
        
        # Convert to UTC for consistency
        full_range_utc = full_range_ct.tz_convert(pytz.UTC)
        
        # Get actual data for this day
        day_data = df[df['date'] == date].copy()
        day_data = day_data.drop(columns=['timestamp_ct', 'date'])
        
        # Create complete timestamp DataFrame
        complete_df = pd.DataFrame({'timestamp': full_range_utc})
        
        # Merge with actual data
        complete_df = complete_df.merge(day_data, on='timestamp', how='left')
        
        # Forward fill OHLC
        complete_df['close'] = complete_df['close'].ffill().bfill()  # Backfill for first row if needed
        complete_df['open'] = complete_df['open'].fillna(complete_df['close'])
        complete_df['high'] = complete_df['high'].fillna(complete_df['close'])
        complete_df['low'] = complete_df['low'].fillna(complete_df['close'])
        
        # Set volume=0 for filled gaps
        complete_df['volume'] = complete_df['volume'].fillna(0).astype(int)
        
        # Forward fill symbol/instrument_id if present
        if 'symbol' in complete_df.columns:
            complete_df['symbol'] = complete_df['symbol'].ffill()
        if 'instrument_id' in complete_df.columns:
            complete_df['instrument_id'] = complete_df['instrument_id'].ffill()
        
        filled_dfs.append(complete_df)
    
    print()  # New line after progress
    
    # Combine all days
    df_filled = pd.concat(filled_dfs, ignore_index=True)
    
    stats['filled_rows'] = len(df_filled)
    stats['added_rows'] = stats['filled_rows'] - stats['original_rows']
    stats['avg_rows_per_day'] = stats['filled_rows'] / stats['trading_days'] if stats['trading_days'] > 0 else 0
    
    print(f"   Original: {stats['original_rows']:,} rows")
    print(f"   Filled: {stats['filled_rows']:,} rows")
    print(f"   Added: {stats['added_rows']:,} rows")
    print(f"   Avg per day: {stats['avg_rows_per_day']:.0f} rows (expected: {stats['expected_rows_per_day']})")
    
    return df_filled, stats


def process_complete_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the complete corrected pipeline
    
    1. Filter to primary contract per day (by volume)
    2. Filter to RTH only (07:30-15:00 CT)
    3. Fill gaps to 1-second resolution
    
    Args:
        df: Raw DataFrame from Databento
        
    Returns:
        Tuple of (processed_df, combined_stats)
    """
    print("=" * 80)
    print("CORRECTED DATA PIPELINE")
    print("=" * 80)
    
    combined_stats = {}
    
    # Step 1: Contract filtering
    df, contract_stats = filter_primary_contract_by_volume(df)
    combined_stats['contract_filtering'] = contract_stats
    
    # Step 2: RTH filtering
    df, rth_stats = filter_rth(df)
    combined_stats['rth_filtering'] = rth_stats
    
    # Step 3: Gap filling
    df, gap_stats = fill_gaps_to_1_second(df)
    combined_stats['gap_filling'] = gap_stats
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Final output: {len(df):,} rows across {gap_stats['trading_days']} days")
    print(f"Average: {gap_stats['avg_rows_per_day']:.0f} rows per day")
    print(f"Expected: {gap_stats['expected_rows_per_day']} rows per day (27,000 = 7.5 hours)")
    
    return df, combined_stats
