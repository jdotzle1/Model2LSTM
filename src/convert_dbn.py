#!/usr/bin/env python3
"""
DBN to Parquet Conversion Module

Converts Databento DBN files to Parquet format with optional RTH filtering.
RTH (Regular Trading Hours) for ES futures: 07:30-15:00 Central Time
"""

import databento as db
import pandas as pd
import numpy as np
import pytz
from datetime import time


def convert_dbn_file(dbn_path, rth_only=True):
    """
    Convert a DBN file to Parquet format with optional RTH filtering
    
    Args:
        dbn_path: Path to the DBN.ZST file
        rth_only: If True, filter to Regular Trading Hours only (07:30-15:00 CT)
        
    Returns:
        pandas.DataFrame: Converted data
    """
    print(f"Reading DBN file: {dbn_path}")
    
    # Load the DBN file
    store = db.DBNStore.from_file(dbn_path)
    df = store.to_df()
    
    print(f"Loaded {len(df):,} bars")
    print(f"Columns: {df.columns.tolist()}")
    
    # DBN files have timestamp in the index
    if 'timestamp' not in df.columns and df.index.name == 'ts_event':
        df = df.reset_index()
        df = df.rename(columns={'ts_event': 'timestamp'})
        print(f"Converted index to timestamp column")
    
    # Keep only OHLCV columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[required_cols]
    
    if rth_only:
        df = filter_rth_only(df)
        print(f"After RTH filtering: {len(df):,} bars")
    
    return df


def filter_rth_only(df):
    """
    Filter DataFrame to Regular Trading Hours only with session boundary markers
    
    RTH for ES futures: 07:30-15:00 Central Time
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        pandas.DataFrame: Filtered to RTH only with session info
    """
    print("Applying RTH filter (07:30-15:00 Central Time)...")
    
    # Ensure timestamp is datetime
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    
    # Convert to Central Time if not already
    central_tz = pytz.timezone('US/Central')
    
    # If timestamp is timezone-naive, assume it's already in Central Time
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(central_tz)
    else:
        # Convert to Central Time
        df['timestamp'] = df['timestamp'].dt.tz_convert(central_tz)
    
    # Define RTH hours (07:30-15:00 Central)
    rth_start = time(7, 30)  # 07:30 CT
    rth_end = time(15, 0)    # 15:00 CT
    
    # Filter to RTH only
    df_time = df['timestamp'].dt.time
    rth_mask = (df_time >= rth_start) & (df_time < rth_end)
    
    df_filtered = df[rth_mask].copy()
    
    # Add session boundary information for proper handling of gaps
    df_filtered = add_session_boundaries(df_filtered)
    
    # Convert back to UTC for consistency (optional)
    utc_tz = pytz.UTC
    df_filtered['timestamp'] = df_filtered['timestamp'].dt.tz_convert(utc_tz)
    
    print(f"RTH filtering removed {len(df) - len(df_filtered):,} bars outside 07:30-15:00 CT")
    
    return df_filtered


def add_session_boundaries(df):
    """
    Add session boundary markers to handle day transitions properly
    
    This helps rolling calculations and lookforward logic respect session boundaries
    """
    # Create session date (trading day)
    ct_time = df['timestamp']
    ct_decimal = ct_time.dt.hour + ct_time.dt.minute / 60.0
    
    # Session date - bars before 7:30 belong to previous session
    session_date = np.where(ct_decimal < 7.5, 
                           (ct_time - pd.Timedelta(days=1)).dt.date, 
                           ct_time.dt.date)
    
    df['session_date'] = pd.to_datetime(session_date)
    
    # Mark session start (first bar of each session)
    df['session_start'] = (df['session_date'] != df['session_date'].shift(1))
    
    # Mark session end (last bar of each session) 
    df['session_end'] = (df['session_date'] != df['session_date'].shift(-1))
    
    # Add session number for easy grouping
    df['session_number'] = df['session_start'].cumsum()
    
    print(f"Identified {df['session_number'].max()} trading sessions")
    
    return df


def main():
    """Example usage"""
    # October 2025 data file
    dbn_path = r"C:\Users\jdotzler\Downloads\glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst"
    
    try:
        # Convert with RTH filtering
        df = convert_dbn_file(dbn_path, rth_only=True)
        
        # Save as Parquet
        output_path = "oct2025_raw.parquet"
        df.to_parquet(output_path, index=False)
        print(f"\n✓ Saved to {output_path}")
        
        # Show sample data
        print("\nSample data:")
        print(df.head())
        
        # Show time range
        print(f"\nTime range:")
        print(f"Start: {df['timestamp'].min()}")
        print(f"End: {df['timestamp'].max()}")
        print(f"Total bars: {len(df):,}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()