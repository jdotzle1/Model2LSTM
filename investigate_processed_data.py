"""
Investigate processed parquet file to understand data quality and labeling issues
"""
import pandas as pd
import numpy as np
import pytz
from datetime import time as dt_time
from pathlib import Path

def investigate_processed_file(parquet_path):
    """Investigate processed parquet file for data quality issues"""
    
    print("=" * 80)
    print("PROCESSED DATA INVESTIGATION - July 2010")
    print("=" * 80)
    print(f"\nFile: {parquet_path}\n")
    
    # Read parquet file
    print("📖 Reading parquet file...")
    df = pd.read_parquet(parquet_path)
    
    print(f"\n📋 DATAFRAME INFO:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check columns
    print(f"\n📊 COLUMN STRUCTURE:")
    original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    label_cols = [c for c in df.columns if c.startswith('label_')]
    weight_cols = [c for c in df.columns if c.startswith('weight_')]
    feature_cols = [c for c in df.columns if c not in original_cols and not c.startswith(('label_', 'weight_'))]
    
    print(f"   Original columns: {len(original_cols)}")
    print(f"   Label columns: {len(label_cols)} - {label_cols}")
    print(f"   Weight columns: {len(weight_cols)} - {weight_cols}")
    print(f"   Feature columns: {len(feature_cols)}")
    
    # Timestamp analysis
    print(f"\n📅 TIMESTAMP ANALYSIS:")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert to Central Time
    central_tz = pytz.timezone('US/Central')
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
    df['timestamp_ct'] = df['timestamp'].dt.tz_convert(central_tz)
    df['date'] = df['timestamp_ct'].dt.date
    df['time'] = df['timestamp_ct'].dt.time
    
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Unique dates: {df['date'].nunique()}")
    print(f"   Total bars: {len(df):,}")
    
    # Check if all data is RTH
    rth_start = dt_time(7, 30)
    rth_end = dt_time(15, 0)
    rth_mask = (df['time'] >= rth_start) & (df['time'] < rth_end)
    non_rth_count = (~rth_mask).sum()
    
    if non_rth_count > 0:
        print(f"   ⚠️  NON-RTH DATA FOUND: {non_rth_count:,} bars ({non_rth_count/len(df)*100:.1f}%)")
        print(f"      This could explain labeling issues!")
    else:
        print(f"   ✅ All data is within RTH (7:30 AM - 3:00 PM CT)")
    
    # Daily statistics
    print(f"\n📊 DAILY STATISTICS:")
    daily_stats = df.groupby('date').agg({
        'volume': ['sum', 'count', 'mean'],
        'close': ['min', 'max', 'mean']
    })
    daily_stats.columns = ['total_volume', 'bar_count', 'avg_volume', 'min_price', 'max_price', 'avg_price']
    daily_stats['price_range'] = daily_stats['max_price'] - daily_stats['min_price']
    
    print(f"\n   {'Date':<12} {'Bars':>8} {'Total Vol':>12} {'Price Range':>12} {'Avg Price':>10}")
    print(f"   {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    
    for date, row in daily_stats.iterrows():
        print(f"   {date} {row['bar_count']:>8,.0f} {row['total_volume']:>12,.0f} {row['price_range']:>12.2f} {row['avg_price']:>10.2f}")
    
    # Check for volume pattern changes (contract rolls)
    print(f"\n🔄 VOLUME PATTERN ANALYSIS:")
    daily_stats['volume_change_pct'] = daily_stats['total_volume'].pct_change() * 100
    daily_stats['bar_count_change_pct'] = daily_stats['bar_count'].pct_change() * 100
    
    significant_changes = daily_stats[
        (abs(daily_stats['volume_change_pct']) > 30) | 
        (abs(daily_stats['bar_count_change_pct']) > 30)
    ]
    
    if len(significant_changes) > 0:
        print(f"   ⚠️  Days with >30% volume change (POTENTIAL CONTRACT ROLL):")
        for date, row in significant_changes.iterrows():
            print(f"      {date}: Volume: {row['volume_change_pct']:+.1f}%, Bars: {row['bar_count_change_pct']:+.1f}%")
    else:
        print(f"   ✅ No significant volume changes (likely single contract)")
    
    # Price continuity check
    print(f"\n💰 PRICE CONTINUITY CHECK:")
    df_sorted = df.sort_values('timestamp').copy()
    df_sorted['price_change'] = df_sorted['close'].diff()
    df_sorted['price_change_pct'] = df_sorted['close'].pct_change() * 100
    
    # Large gaps (>5 points)
    large_gaps = df_sorted[abs(df_sorted['price_change']) > 5.0]
    
    if len(large_gaps) > 0:
        print(f"   ⚠️  LARGE PRICE GAPS DETECTED: {len(large_gaps)} gaps >5 points")
        print(f"      This suggests CONTRACT ROLL or DATA CORRUPTION!")
        print(f"\n      First 10 gaps:")
        print(f"      {'Timestamp':<25} {'Price Change':>15} {'% Change':>12}")
        print(f"      {'-'*25} {'-'*15} {'-'*12}")
        for idx, row in large_gaps.head(10).iterrows():
            print(f"      {str(row['timestamp_ct']):<25} {row['price_change']:>15.2f} {row['price_change_pct']:>12.2f}%")
    else:
        print(f"   ✅ No large price gaps detected")
    
    # Labeling analysis
    print(f"\n🏷️  LABELING ANALYSIS:")
    print(f"\n   Win Rates:")
    for col in label_cols:
        win_rate = df[col].mean()
        total_labels = df[col].notna().sum()
        wins = (df[col] == 1).sum()
        losses = (df[col] == 0).sum()
        print(f"      {col:<25}: {win_rate:>6.1%} ({wins:>6,} wins / {losses:>6,} losses / {total_labels:>6,} total)")
    
    # Check for suspicious patterns
    print(f"\n   🔍 Suspicious Pattern Check:")
    long_win_rate = df['label_low_vol_long'].mean()
    short_win_rate = df['label_low_vol_short'].mean()
    
    if long_win_rate < 0.10 and short_win_rate > 0.70:
        print(f"      ⚠️  EXTREME INVERSION DETECTED!")
        print(f"         Long win rate: {long_win_rate:.1%} (TOO LOW)")
        print(f"         Short win rate: {short_win_rate:.1%} (TOO HIGH)")
        print(f"         This suggests SYSTEMATIC LABELING ERROR or CONTRACT ROLL ISSUE")
    elif long_win_rate > 0.70 and short_win_rate < 0.10:
        print(f"      ⚠️  EXTREME INVERSION DETECTED (opposite direction)!")
        print(f"         Long win rate: {long_win_rate:.1%} (TOO HIGH)")
        print(f"         Short win rate: {short_win_rate:.1%} (TOO LOW)")
    else:
        print(f"      ✅ Win rates within reasonable range")
    
    # Manual verification sample
    print(f"\n📝 MANUAL VERIFICATION SAMPLE:")
    print(f"   Selecting 5 random bars for manual verification...\n")
    
    # Sample 5 random bars with enough lookforward
    sample_indices = df[df.index < len(df) - 900].sample(5).index
    
    for idx in sample_indices:
        bar = df.loc[idx]
        next_900 = df.loc[idx:idx+900]
        
        print(f"   Bar {idx} - {bar['timestamp_ct']}")
        print(f"      Entry: O={bar['open']:.2f}, H={bar['high']:.2f}, L={bar['low']:.2f}, C={bar['close']:.2f}, V={bar['volume']:.0f}")
        
        # Low vol long (6 tick stop, 12 tick target)
        entry = bar['close']
        long_target = entry + 3.0  # 12 ticks = 3 points
        long_stop = entry - 1.5    # 6 ticks = 1.5 points
        
        # Check if target or stop hit
        target_hit = (next_900['high'] >= long_target).any()
        stop_hit = (next_900['low'] <= long_stop).any()
        
        if target_hit and stop_hit:
            # Find which came first
            target_bar = next_900[next_900['high'] >= long_target].index[0]
            stop_bar = next_900[next_900['low'] <= long_stop].index[0]
            if target_bar < stop_bar:
                actual_result = "WIN (target first)"
            else:
                actual_result = "LOSS (stop first)"
        elif target_hit:
            actual_result = "WIN (target hit)"
        elif stop_hit:
            actual_result = "LOSS (stop hit)"
        else:
            actual_result = "TIMEOUT (neither hit)"
        
        labeled_result = "WIN" if bar['label_low_vol_long'] == 1 else "LOSS"
        match = "✅" if (actual_result.startswith("WIN") and labeled_result == "WIN") or \
                       (actual_result.startswith("LOSS") and labeled_result == "LOSS") or \
                       (actual_result.startswith("TIMEOUT") and labeled_result == "LOSS") else "❌"
        
        print(f"      Low Vol Long: Target={long_target:.2f}, Stop={long_stop:.2f}")
        print(f"      Actual: {actual_result}")
        print(f"      Labeled: {labeled_result}")
        print(f"      Match: {match}")
        print()
    
    print("=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    
    return df, daily_stats

if __name__ == "__main__":
    parquet_path = r"C:\Users\jdotzler\Desktop\monthly_2010-07_20251107_152756.parquet"
    
    if not Path(parquet_path).exists():
        print(f"❌ File not found: {parquet_path}")
    else:
        df, daily_stats = investigate_processed_file(parquet_path)
        
        print(f"\n💡 KEY FINDINGS TO LOOK FOR:")
        print(f"   1. Non-RTH data (should be 0%)")
        print(f"   2. Large price gaps (indicates contract roll)")
        print(f"   3. Volume pattern changes (indicates contract roll)")
        print(f"   4. Extreme win rate inversions (indicates labeling bug)")
        print(f"   5. Manual verification mismatches (confirms labeling bug)")
