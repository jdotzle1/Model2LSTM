"""
Verify the actual timestamps in the raw data - check if we're misinterpreting them
"""
import pandas as pd
import numpy as np

def verify_raw_timestamps(parquet_path):
    """Check the actual timestamp values in the raw data"""
    
    print("=" * 80)
    print("RAW TIMESTAMP VERIFICATION")
    print("=" * 80)
    print(f"\nFile: {parquet_path}\n")
    
    # Load data
    print("📖 Loading data...")
    df = pd.read_parquet(parquet_path)
    
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check timestamp column
    if 'timestamp' in df.columns:
        print(f"\n⏱️  TIMESTAMP COLUMN ANALYSIS:")
        print(f"   Data type: {df['timestamp'].dtype}")
        print(f"   First 10 timestamps:")
        for i in range(min(10, len(df))):
            ts = df['timestamp'].iloc[i]
            print(f"      {i}: {ts} (type: {type(ts)})")
        
        # Check if timestamps are datetime
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            print(f"\n   ✅ Timestamps are datetime64")
            
            # Sort and check differences
            df_sorted = df.sort_values('timestamp').reset_index(drop=True)
            df_sorted['time_diff'] = df_sorted['timestamp'].diff()
            df_sorted['time_diff_seconds'] = df_sorted['time_diff'].dt.total_seconds()
            
            print(f"\n   Time Differences (first 20):")
            print(f"   {'Index':<8} {'Timestamp':<35} {'Diff (seconds)':>15}")
            print(f"   {'-'*8} {'-'*35} {'-'*15}")
            for i in range(1, min(21, len(df_sorted))):
                ts = df_sorted['timestamp'].iloc[i]
                diff = df_sorted['time_diff_seconds'].iloc[i]
                print(f"   {i:<8} {str(ts):<35} {diff:>15.6f}")
            
            # Check the actual nanosecond precision
            print(f"\n   🔍 NANOSECOND PRECISION CHECK:")
            print(f"   First 5 timestamps with full precision:")
            for i in range(min(5, len(df))):
                ts = pd.Timestamp(df['timestamp'].iloc[i])
                print(f"      {i}: {ts} (nanosecond: {ts.nanosecond})")
        else:
            print(f"   ⚠️  Timestamps are NOT datetime64: {df['timestamp'].dtype}")
            print(f"   Sample values:")
            print(df['timestamp'].head(10))
    else:
        print(f"   ❌ No 'timestamp' column found!")
        print(f"   Available columns: {list(df.columns)}")
    
    # Check if there's an index that might be timestamps
    print(f"\n📋 INDEX ANALYSIS:")
    print(f"   Index name: {df.index.name}")
    print(f"   Index dtype: {df.index.dtype}")
    print(f"   First 5 index values:")
    for i in range(min(5, len(df))):
        print(f"      {i}: {df.index[i]}")
    
    print(f"\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    return df

if __name__ == "__main__":
    # Check October 2025 raw data
    parquet_path = r"C:\Users\jdotzler\Desktop\monthly_2025-10_raw.parquet"
    
    print("Checking October 2025 raw data...")
    df = verify_raw_timestamps(parquet_path)
    
    print(f"\n💡 KEY QUESTION:")
    print(f"   Are the timestamps being created artificially by the conversion script?")
    print(f"   Or are they coming from the actual DBN data?")
