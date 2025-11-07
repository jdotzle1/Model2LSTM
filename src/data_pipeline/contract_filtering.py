"""
Contract Roll Detection and Filtering

Detects and filters out contract roll periods to ensure clean, single-contract data.
Uses volume-based detection at the daily level to avoid intra-day switching.
"""
import pandas as pd
import numpy as np
import pytz
from datetime import time as dt_time
from typing import Tuple, Dict, List


def detect_and_filter_contracts(df: pd.DataFrame, min_daily_volume: int = 50000) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect contract rolls and filter to keep only the dominant contract per day
    
    Strategy:
    1. Group data by trading day
    2. Detect contract boundaries using price gaps and volume patterns
    3. For each day, identify the dominant contract (highest volume)
    4. Filter out all bars from non-dominant contracts
    
    Args:
        df: DataFrame with timestamp, open, high, low, close, volume columns
        min_daily_volume: Minimum daily volume to consider a day valid (default: 50K)
        
    Returns:
        Tuple of (filtered_df, stats_dict)
    """
    print("🔄 Detecting and filtering contract rolls...")
    
    stats = {
        'original_rows': len(df),
        'original_dates': 0,
        'contracts_detected': 0,
        'filtered_rows': 0,
        'filtered_dates': 0,
        'removed_rows': 0,
        'daily_contract_info': {}
    }
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert to Central Time for daily grouping
    central_tz = pytz.timezone('US/Central')
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
    
    df['timestamp_ct'] = df['timestamp'].dt.tz_convert(central_tz)
    df['date'] = df['timestamp_ct'].dt.date
    
    stats['original_dates'] = df['date'].nunique()
    
    # Step 1: Detect contract segments within each day using price gaps
    df_sorted = df.sort_values('timestamp').copy()
    df_sorted['price_change'] = df_sorted['close'].diff()
    df_sorted['is_gap'] = abs(df_sorted['price_change']) > 5.0  # >5 point gap indicates contract switch
    
    # Assign contract segment IDs
    df_sorted['contract_segment'] = df_sorted['is_gap'].cumsum()
    
    # Step 2: For each day, calculate volume per contract segment
    daily_contract_volumes = df_sorted.groupby(['date', 'contract_segment']).agg({
        'volume': 'sum',
        'close': ['count', 'mean']
    }).reset_index()
    
    daily_contract_volumes.columns = ['date', 'contract_segment', 'total_volume', 'bar_count', 'avg_price']
    
    # Step 3: For each day, identify the dominant contract (highest volume)
    dominant_contracts = daily_contract_volumes.loc[
        daily_contract_volumes.groupby('date')['total_volume'].idxmax()
    ][['date', 'contract_segment', 'total_volume', 'bar_count']].copy()
    
    dominant_contracts.columns = ['date', 'dominant_segment', 'dominant_volume', 'dominant_bars']
    
    # Step 4: Filter to keep only dominant contract bars
    df_filtered = df_sorted.merge(
        dominant_contracts[['date', 'dominant_segment']], 
        on='date', 
        how='left'
    )
    
    # Keep only bars from dominant contract
    df_filtered = df_filtered[df_filtered['contract_segment'] == df_filtered['dominant_segment']].copy()
    
    # Step 5: Remove days with very low volume (likely holidays or data issues)
    daily_volumes = df_filtered.groupby('date')['volume'].sum()
    valid_dates = daily_volumes[daily_volumes >= min_daily_volume].index
    df_filtered = df_filtered[df_filtered['date'].isin(valid_dates)].copy()
    
    # Clean up temporary columns
    df_filtered = df_filtered.drop(columns=['timestamp_ct', 'date', 'price_change', 'is_gap', 
                                             'contract_segment', 'dominant_segment'], errors='ignore')
    
    # Calculate statistics
    stats['filtered_rows'] = len(df_filtered)
    stats['filtered_dates'] = df_filtered['timestamp'].dt.date.nunique()
    stats['removed_rows'] = stats['original_rows'] - stats['filtered_rows']
    stats['removal_percentage'] = (stats['removed_rows'] / stats['original_rows'] * 100) if stats['original_rows'] > 0 else 0
    
    # Detailed daily contract info
    for _, row in dominant_contracts.iterrows():
        date = row['date']
        total_bars_that_day = df_sorted[df_sorted['date'] == date].shape[0]
        removed_bars = total_bars_that_day - row['dominant_bars']
        
        stats['daily_contract_info'][str(date)] = {
            'dominant_volume': int(row['dominant_volume']),
            'dominant_bars': int(row['dominant_bars']),
            'total_bars': int(total_bars_that_day),
            'removed_bars': int(removed_bars),
            'removal_pct': (removed_bars / total_bars_that_day * 100) if total_bars_that_day > 0 else 0
        }
    
    # Count how many days had contract filtering
    days_with_filtering = sum(1 for info in stats['daily_contract_info'].values() if info['removed_bars'] > 0)
    stats['days_with_contract_filtering'] = days_with_filtering
    
    print(f"   ✅ Contract filtering complete:")
    print(f"      Original: {stats['original_rows']:,} rows across {stats['original_dates']} days")
    print(f"      Filtered: {stats['filtered_rows']:,} rows across {stats['filtered_dates']} days")
    print(f"      Removed: {stats['removed_rows']:,} rows ({stats['removal_percentage']:.1f}%)")
    print(f"      Days with contract filtering: {days_with_filtering}")
    
    return df_filtered, stats


def validate_contract_filtering(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict:
    """
    Validate that contract filtering improved data quality
    
    Args:
        df_before: DataFrame before filtering
        df_after: DataFrame after filtering
        
    Returns:
        Dict with validation metrics
    """
    validation = {}
    
    # Check for large price gaps
    df_before_sorted = df_before.sort_values('timestamp')
    df_after_sorted = df_after.sort_values('timestamp')
    
    gaps_before = abs(df_before_sorted['close'].diff()) > 5.0
    gaps_after = abs(df_after_sorted['close'].diff()) > 5.0
    
    validation['large_gaps_before'] = gaps_before.sum()
    validation['large_gaps_after'] = gaps_after.sum()
    validation['gaps_removed'] = validation['large_gaps_before'] - validation['large_gaps_after']
    validation['gap_reduction_pct'] = (validation['gaps_removed'] / validation['large_gaps_before'] * 100) if validation['large_gaps_before'] > 0 else 0
    
    # Check price continuity
    validation['max_gap_before'] = abs(df_before_sorted['close'].diff()).max()
    validation['max_gap_after'] = abs(df_after_sorted['close'].diff()).max()
    
    print(f"\n   📊 Validation Results:")
    print(f"      Large gaps (>5 pts) before: {validation['large_gaps_before']}")
    print(f"      Large gaps (>5 pts) after: {validation['large_gaps_after']}")
    print(f"      Gaps removed: {validation['gaps_removed']} ({validation['gap_reduction_pct']:.1f}%)")
    print(f"      Max gap before: {validation['max_gap_before']:.2f} points")
    print(f"      Max gap after: {validation['max_gap_after']:.2f} points")
    
    return validation


def get_contract_filtering_summary(stats: Dict) -> str:
    """
    Generate a human-readable summary of contract filtering
    
    Args:
        stats: Statistics dictionary from detect_and_filter_contracts
        
    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("Contract Filtering Summary:")
    summary.append(f"  Total rows removed: {stats['removed_rows']:,} ({stats['removal_percentage']:.1f}%)")
    summary.append(f"  Days processed: {stats['filtered_dates']}")
    summary.append(f"  Days with filtering: {stats['days_with_contract_filtering']}")
    
    # Show days with significant filtering
    significant_days = [(date, info) for date, info in stats['daily_contract_info'].items() 
                       if info['removal_pct'] > 5.0]
    
    if significant_days:
        summary.append(f"\n  Days with >5% filtering:")
        for date, info in sorted(significant_days)[:10]:  # Show first 10
            summary.append(f"    {date}: {info['removed_bars']:,} bars removed ({info['removal_pct']:.1f}%)")
    
    return "\n".join(summary)
