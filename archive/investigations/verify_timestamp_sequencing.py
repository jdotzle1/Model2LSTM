"""
Verify that timestamps are sequential and check for gaps
"""
import pandas as pd
import numpy as np

def verify_timestamp_sequencing(parquet_path):
    """Check if timestamps are properly sequential"""
    
    print("=" * 80)
    print("TIMESTAMP SEQUENCING VERIFICATION")
    print("=" * 80)
    print(f"\nFile: {parquet_path}\n")
    
    # Load data
    print("📖 Loading data...")
    df = pd.read_parquet(parquet_path)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   Loaded: {len(df):,} rows")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Check timestamp differences
    print(f"\n⏱️  TIMESTAMP ANALYSIS:")
    df['time_diff'] = df['timestamp'].diff()
    df['time_diff_seconds'] = df['time_diff'].dt.total_seconds()
    
    # Expected: 1 second between bars
    print(f"\n   Time differences (seconds):")
    print(f"      Min: {df['time_diff_seconds'].min():.2f}")
    print(f"      Max: {df['time_diff_seconds'].max():.2f}")
    print(f"      Mean: {df['time_diff_seconds'].mean():.2f}")
    print(f"      Median: {df['time_diff_seconds'].median():.2f}")
    print(f"      Mode: {df['time_diff_seconds'].mode().values[0] if len(df['time_diff_seconds'].mode()) > 0 else 'N/A':.2f}")
    
    # Check for 1-second intervals
    one_second_bars = (df['time_diff_seconds'] == 1.0).sum()
    one_second_pct = one_second_bars / len(df) * 100
    
    print(f"\n   1-second intervals: {one_second_bars:,} ({one_second_pct:.1f}%)")
    
    if one_second_pct < 80:
        print(f"      ⚠️  Less than 80% are 1-second intervals!")
        print(f"         This suggests missing data or irregular sampling")
    else:
        print(f"      ✅ Most bars are 1-second apart")
    
    # Check for gaps
    print(f"\n   🔍 GAP ANALYSIS:")
    gaps = df[df['time_diff_seconds'] > 60]  # Gaps > 1 minute
    
    if len(gaps) > 0:
        print(f"      Found {len(gaps)} gaps >1 minute:")
        print(f"      {'Index':<8} {'Timestamp':<30} {'Gap (seconds)':>15}")
        print(f"      {'-'*8} {'-'*30} {'-'*15}")
        for idx, row in gaps.head(20).iterrows():
            print(f"      {idx:<8} {str(row['timestamp']):<30} {row['time_diff_seconds']:>15.0f}")
    else:
        print(f"      ✅ No gaps >1 minute found")
    
    # Check for duplicate timestamps
    print(f"\n   🔍 DUPLICATE TIMESTAMP CHECK:")
    duplicates = df[df['timestamp'].duplicated(keep=False)]
    
    if len(duplicates) > 0:
        print(f"      ⚠️  Found {len(duplicates)} rows with duplicate timestamps!")
        print(f"      Sample duplicates:")
        print(duplicates.head(10)[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
    else:
        print(f"      ✅ No duplicate timestamps")
    
    # Check if timestamps are truly sequential
    print(f"\n   🔍 SEQUENTIAL ORDER CHECK:")
    is_sorted = df['timestamp'].is_monotonic_increasing
    
    if is_sorted:
        print(f"      ✅ Timestamps are in sequential order")
    else:
        print(f"      ❌ Timestamps are NOT in sequential order!")
        # Find out-of-order timestamps
        out_of_order = df[df['timestamp'] <= df['timestamp'].shift(1)]
        print(f"      Found {len(out_of_order)} out-of-order timestamps")
    
    # Sample consecutive bars to verify they're moving forward
    print(f"\n📝 SAMPLE CONSECUTIVE BARS:")
    print(f"   {'Index':<8} {'Timestamp':<30} {'Time Diff':>12} {'Close':>10}")
    print(f"   {'-'*8} {'-'*30} {'-'*12} {'-'*10}")
    
    for idx in [0, 100, 1000, 10000, 50000, 100000]:
        if idx < len(df) - 1:
            row = df.iloc[idx]
            next_row = df.iloc[idx + 1]
            time_diff = (next_row['timestamp'] - row['timestamp']).total_seconds()
            print(f"   {idx:<8} {str(row['timestamp']):<30} {time_diff:>12.1f}s {row['close']:>10.2f}")
            print(f"   {idx+1:<8} {str(next_row['timestamp']):<30} {' ':>12} {next_row['close']:>10.2f}")
            print()
    
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    print(f"\n💡 SUMMARY:")
    if one_second_pct > 80 and is_sorted and len(duplicates) == 0:
        print(f"   ✅ Timestamps are properly sequential")
        print(f"   ✅ Data appears to be 1-second bars")
    else:
        print(f"   ⚠️  Timestamp issues detected:")
        if one_second_pct < 80:
            print(f"      - Only {one_second_pct:.1f}% are 1-second intervals")
        if not is_sorted:
            print(f"      - Timestamps are not in order")
        if len(duplicates) > 0:
            print(f"      - {len(duplicates)} duplicate timestamps found")
    
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        parquet_path = sys.argv[1]
        print(f"Checking: {parquet_path}")
    else:
        # Check the July 2010 processed data
        parquet_path = r"C:\Users\jdotzler\Desktop\monthly_2010-07_20251107_152756.parquet"
        print("Checking July 2010 data...")
    
    df = verify_timestamp_sequencing(parquet_path)
