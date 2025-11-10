"""
Check what timestamp data is actually in the DBN file
"""
import sys

# Use Python 3.12 which has databento
print("Checking actual DBN timestamp columns...")

try:
    import databento as db
    import pandas as pd
    
    dbn_path = r"C:\Users\jdotzler\Downloads\glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst"
    
    print(f"\n📖 Reading DBN file: {dbn_path}")
    store = db.DBNStore.from_file(dbn_path)
    
    print(f"\n🔄 Converting to DataFrame WITHOUT custom timestamps...")
    df = store.to_df()
    
    print(f"\n📋 DataFrame Info:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Index name: {df.index.name}")
    print(f"   Index dtype: {df.index.dtype}")
    
    print(f"\n📊 First 20 rows:")
    print(df.head(20))
    
    print(f"\n🔍 Index values (first 20):")
    for i in range(min(20, len(df))):
        idx_val = df.index[i]
        print(f"   {i}: {idx_val} (type: {type(idx_val)})")
    
    # Check if index is timestamps
    if pd.api.types.is_datetime64_any_dtype(df.index):
        print(f"\n   ✅ Index IS datetime - these are the actual timestamps!")
        
        # Calculate time differences
        time_diffs = df.index.to_series().diff()
        time_diffs_seconds = time_diffs.dt.total_seconds()
        
        print(f"\n   Time differences (first 20):")
        for i in range(1, min(21, len(df))):
            diff = time_diffs_seconds.iloc[i]
            print(f"      {i}: {diff:.6f} seconds")
        
        print(f"\n   Statistics:")
        print(f"      Min: {time_diffs_seconds.min():.6f} seconds")
        print(f"      Max: {time_diffs_seconds.max():.6f} seconds")
        print(f"      Mean: {time_diffs_seconds.mean():.6f} seconds")
        print(f"      Median: {time_diffs_seconds.median():.6f} seconds")
        print(f"      Mode: {time_diffs_seconds.mode().values[0] if len(time_diffs_seconds.mode()) > 0 else 'N/A':.6f} seconds")
        
        # Check for 1-second intervals
        one_second = (time_diffs_seconds == 1.0).sum()
        one_second_pct = one_second / len(df) * 100
        print(f"\n      1-second intervals: {one_second:,} ({one_second_pct:.1f}%)")
        
    else:
        print(f"\n   ❌ Index is NOT datetime: {df.index.dtype}")
        print(f"   Need to check for timestamp columns in the data")
    
    # Check for timestamp-related columns
    print(f"\n🔍 Checking for timestamp columns:")
    ts_cols = [col for col in df.columns if 'ts' in col.lower() or 'time' in col.lower()]
    if ts_cols:
        print(f"   Found timestamp-related columns: {ts_cols}")
        for col in ts_cols:
            print(f"\n   Column: {col}")
            print(f"      Type: {df[col].dtype}")
            print(f"      First 5 values: {df[col].head().tolist()}")
    else:
        print(f"   No timestamp-related columns found")
    
except ImportError:
    print("❌ databento not available in current Python version")
    print("Run with: py -3.12 check_dbn_actual_timestamps.py")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
