"""
Test the corrected DBN conversion on July 2010 data
"""
import sys
import pandas as pd

print("=" * 80)
print("TESTING CORRECTED DBN CONVERSION - July 2010")
print("=" * 80)

try:
    import databento as db
    print("✅ databento available\n")
    
    dbn_path = r"C:\Users\jdotzler\Desktop\glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst"
    output_path = r"C:\Users\jdotzler\Desktop\july_2010_CORRECTED.parquet"
    
    print(f"📖 Reading DBN file...")
    print(f"   File: {dbn_path}\n")
    
    store = db.DBNStore.from_file(dbn_path)
    metadata = store.metadata
    
    print(f"📊 Metadata:")
    print(f"   Schema: {metadata.schema}")
    print(f"   Dataset: {metadata.dataset}")
    print(f"   Symbols: {metadata.symbols}\n")
    
    print(f"🔄 Converting to DataFrame...")
    df = store.to_df()
    
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Index: {df.index.name} ({df.index.dtype})\n")
    
    # CORRECTED: Use ts_event index directly
    print(f"✅ USING CORRECTED APPROACH:")
    if df.index.name == 'ts_event' and pd.api.types.is_datetime64_any_dtype(df.index):
        print(f"   Using ts_event index (actual timestamps from DBN)\n")
        df['timestamp'] = df.index
    else:
        print(f"   ⚠️  ts_event index not found, using fallback\n")
    
    # Reset index to make timestamp a column
    df = df.reset_index(drop=True)
    
    # Analyze timestamps
    print(f"⏱️  TIMESTAMP ANALYSIS:")
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    df_sorted['time_diff'] = df_sorted['timestamp'].diff()
    df_sorted['time_diff_seconds'] = df_sorted['time_diff'].dt.total_seconds()
    
    print(f"   First 20 timestamps:")
    for i in range(min(20, len(df_sorted))):
        ts = df_sorted['timestamp'].iloc[i]
        if i > 0:
            diff = df_sorted['time_diff_seconds'].iloc[i]
            print(f"      {i}: {ts} (diff: {diff:.3f}s)")
        else:
            print(f"      {i}: {ts}")
    
    print(f"\n   Time Difference Statistics:")
    print(f"      Min: {df_sorted['time_diff_seconds'].min():.6f} seconds")
    print(f"      Max: {df_sorted['time_diff_seconds'].max():.6f} seconds")
    print(f"      Mean: {df_sorted['time_diff_seconds'].mean():.6f} seconds")
    print(f"      Median: {df_sorted['time_diff_seconds'].median():.6f} seconds")
    print(f"      Mode: {df_sorted['time_diff_seconds'].mode().values[0] if len(df_sorted['time_diff_seconds'].mode()) > 0 else 'N/A':.6f} seconds")
    
    # Check for 1-second intervals
    one_second = (df_sorted['time_diff_seconds'] == 1.0).sum()
    one_second_pct = one_second / len(df_sorted) * 100
    print(f"\n      1-second intervals: {one_second:,} ({one_second_pct:.1f}%)")
    
    if one_second_pct > 60:
        print(f"      ✅ GOOD! Most bars are 1-second apart")
    else:
        print(f"      ⚠️  WARNING: Less than 60% are 1-second intervals")
    
    # Save corrected data
    print(f"\n💾 Saving corrected data...")
    df.to_parquet(output_path, index=False)
    print(f"   Saved: {output_path}")
    print(f"   Size: {len(df):,} rows\n")
    
    print("=" * 80)
    print("✅ CONVERSION COMPLETE")
    print("=" * 80)
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Verify timestamps look correct (60-80% should be 1-second)")
    print(f"   2. Run labeling and feature engineering on this corrected data")
    print(f"   3. Check if win rates are now reasonable (15-30%)")
    
except ImportError:
    print("❌ databento not available")
    print("Run with: py -3.12 test_corrected_conversion.py")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
