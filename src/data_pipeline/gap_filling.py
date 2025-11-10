"""
Gap Filling for 1-Second OHLCV Data

Databento omits rows where there's no trading activity (volume = 0).
This module fills those gaps with placeholder bars for ML training.
"""
import pandas as pd
import numpy as np
from datetime import timedelta

def fill_1second_gaps(df: pd.DataFrame, forward_fill_price: bool = True) -> pd.DataFrame:
    """
    Fill gaps in 1-second OHLCV data where Databento omitted zero-volume bars
    
    Args:
        df: DataFrame with timestamp, open, high, low, close, volume columns
        forward_fill_price: If True, use last close for OHLC in gap bars (default: True)
                           If False, use NaN for OHLC in gap bars
    
    Returns:
        DataFrame with gaps filled (one row per second)
    """
    print(f"Filling 1-second gaps...")
    print(f"   Original rows: {len(df):,}")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Create complete 1-second range
    start_time = df['timestamp'].min().floor('S')  # Round down to nearest second
    end_time = df['timestamp'].max().ceil('S')     # Round up to nearest second
    
    # Generate all seconds in range
    all_seconds = pd.date_range(start=start_time, end=end_time, freq='1S')
    
    print(f"   Time range: {start_time} to {end_time}")
    print(f"   Expected rows (1-second): {len(all_seconds):,}")
    
    # Create complete DataFrame with all seconds
    df_complete = pd.DataFrame({'timestamp': all_seconds})
    
    # Merge with original data
    df_filled = df_complete.merge(df, on='timestamp', how='left')
    
    # Fill missing values
    # Volume: 0 for gaps (no trading)
    df_filled['volume'] = df_filled['volume'].fillna(0)
    
    if forward_fill_price:
        # OHLC: Forward fill from last known price
        # This represents "no change" - price stays at last traded level
        df_filled['close'] = df_filled['close'].ffill()
        df_filled['open'] = df_filled['close']   # Open = Close for no-trade bars
        df_filled['high'] = df_filled['close']   # High = Close for no-trade bars
        df_filled['low'] = df_filled['close']    # Low = Close for no-trade bars
        
        # For the very first bars (before any trading), backfill
        df_filled['close'] = df_filled['close'].bfill()
        df_filled['open'] = df_filled['open'].bfill()
        df_filled['high'] = df_filled['high'].bfill()
        df_filled['low'] = df_filled['low'].bfill()
    else:
        # Leave OHLC as NaN for gap bars
        # Model will need to handle NaN values
        pass
    
    # Fill other columns if they exist
    if 'symbol' in df_filled.columns:
        df_filled['symbol'] = df_filled['symbol'].ffill().bfill()
    
    if 'instrument_id' in df_filled.columns:
        df_filled['instrument_id'] = df_filled['instrument_id'].ffill().bfill()
    
    gaps_filled = len(df_filled) - len(df)
    print(f"   Filled rows: {len(df_filled):,}")
    print(f"   Gaps filled: {gaps_filled:,} ({gaps_filled/len(df_filled)*100:.1f}%)")
    
    return df_filled


def fill_gaps_with_session_awareness(df: pd.DataFrame, 
                                     rth_only: bool = True,
                                     forward_fill_price: bool = True) -> pd.DataFrame:
    """
    Fill gaps with session awareness (don't fill overnight gaps)
    
    Args:
        df: DataFrame with timestamp column
        rth_only: If True, only fill gaps during RTH (9:30-16:00 ET)
        forward_fill_price: If True, forward fill prices in gaps
    
    Returns:
        DataFrame with gaps filled only within trading sessions
    """
    import pytz
    
    print(f"Filling gaps with session awareness...")
    print(f"   RTH only: {rth_only}")
    
    # Convert to Central Time
    central_tz = pytz.timezone('US/Central')
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
    
    df['timestamp_ct'] = df['timestamp'].dt.tz_convert(central_tz)
    df['date'] = df['timestamp_ct'].dt.date
    df['time'] = df['timestamp_ct'].dt.time
    
    if rth_only:
        # Filter to RTH only first
        from datetime import time as dt_time
        rth_start = dt_time(9, 30)
        rth_end = dt_time(16, 0)
        df = df[(df['time'] >= rth_start) & (df['time'] < rth_end)].copy()
        print(f"   Filtered to RTH: {len(df):,} rows")
    
    # Group by date and fill gaps within each session
    filled_dfs = []
    
    for date, group in df.groupby('date'):
        # Fill gaps for this session
        group_filled = fill_1second_gaps(group, forward_fill_price=forward_fill_price)
        filled_dfs.append(group_filled)
    
    # Combine all sessions
    df_final = pd.concat(filled_dfs, ignore_index=True)
    
    print(f"   Final rows: {len(df_final):,}")
    
    return df_final


def validate_gap_filling(df_original: pd.DataFrame, df_filled: pd.DataFrame):
    """
    Validate that gap filling worked correctly
    """
    print(f"\nVALIDATION:")
    
    # Check intervals
    df_filled_sorted = df_filled.sort_values('timestamp')
    intervals = df_filled_sorted['timestamp'].diff().dt.total_seconds().dropna()
    
    one_sec = ((intervals >= 0.99) & (intervals <= 1.01)).sum()
    pct_one_sec = (one_sec / len(intervals)) * 100
    
    print(f"   1-second intervals: {pct_one_sec:.1f}%")
    
    if pct_one_sec > 99:
        print(f"   Perfect! All intervals are 1 second")
    elif pct_one_sec > 95:
        print(f"   Mostly 1-second, some gaps remain")
    else:
        print(f"   Still has large gaps")
    
    # Check volume
    zero_volume = (df_filled['volume'] == 0).sum()
    pct_zero = (zero_volume / len(df_filled)) * 100
    
    print(f"   Zero-volume bars: {zero_volume:,} ({pct_zero:.1f}%)")
    print(f"   These represent periods with no trading activity")
    
    # Check for NaN
    nan_counts = df_filled.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n   NaN values found:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"      {col}: {count:,}")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("GAP FILLING MODULE")
    print("=" * 80)
    print("\nThis module fills gaps in 1-second OHLCV data")
    print("where Databento omitted zero-volume bars.")
    print("\nUsage:")
    print("   from src.data_pipeline.gap_filling import fill_1second_gaps")
    print("   df_filled = fill_1second_gaps(df)")
