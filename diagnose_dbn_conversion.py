"""
Diagnose DBN conversion issues - check if OHLC data is being read correctly
"""
import sys
import os

# Try to import databento
try:
    import databento as db
    print("✅ databento module available")
except ImportError:
    print("❌ databento module not available - attempting to install...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "databento==0.38.0"])
    import databento as db
    print("✅ databento installed")

import pandas as pd
import numpy as np

def diagnose_dbn_file(dbn_path):
    """Diagnose DBN file conversion issues"""
    
    print("=" * 80)
    print("DBN CONVERSION DIAGNOSIS - July 2010")
    print("=" * 80)
    print(f"\nFile: {dbn_path}\n")
    
    # Load DBN file
    print("📖 Loading DBN file...")
    store = db.DBNStore.from_file(dbn_path)
    metadata = store.metadata
    
    print(f"\n📊 METADATA:")
    print(f"   Schema: {metadata.schema}")
    print(f"   Dataset: {metadata.dataset}")
    print(f"   Stype In: {metadata.stype_in}")
    print(f"   Stype Out: {metadata.stype_out}")
    print(f"   Symbols: {metadata.symbols}")
    print(f"   Start: {pd.to_datetime(metadata.start, unit='ns', utc=True)}")
    print(f"   End: {pd.to_datetime(metadata.end, unit='ns', utc=True)}")
    
    # Convert to DataFrame
    print(f"\n🔄 Converting to DataFrame...")
    df = store.to_df()
    
    print(f"\n📋 DATAFRAME INFO:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Index: {df.index.name}")
    print(f"   Dtypes:")
    for col in df.columns:
        print(f"      {col}: {df[col].dtype}")
    
    # Check OHLC data quality
    print(f"\n💰 OHLC DATA QUALITY:")
    
    if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        print(f"   ✅ All OHLC columns present")
        
        # Check for zero ranges (H=L)
        df['range'] = df['high'] - df['low']
        zero_range = (df['range'] == 0).sum()
        zero_range_pct = zero_range / len(df) * 100
        
        print(f"\n   Range Analysis:")
        print(f"      Bars with zero range (H=L): {zero_range:,} ({zero_range_pct:.1f}%)")
        print(f"      Min range: {df['range'].min():.4f}")
        print(f"      Max range: {df['range'].max():.4f}")
        print(f"      Mean range: {df['range'].mean():.4f}")
        print(f"      Median range: {df['range'].median():.4f}")
        
        # Check OHLC relationships
        print(f"\n   OHLC Relationship Validation:")
        high_valid = (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['high'] >= df['low'])
        low_valid = (df['low'] <= df['open']) & (df['low'] <= df['close']) & (df['low'] <= df['high'])
        
        invalid_high = (~high_valid).sum()
        invalid_low = (~low_valid).sum()
        
        print(f"      Invalid high values: {invalid_high:,} ({invalid_high/len(df)*100:.1f}%)")
        print(f"      Invalid low values: {invalid_low:,} ({invalid_low/len(df)*100:.1f}%)")
        
        if invalid_high > 0 or invalid_low > 0:
            print(f"      ⚠️  OHLC relationships are BROKEN!")
            print(f"         This indicates data corruption or incorrect schema")
        
        # Check for up/down bars
        df['is_up'] = df['close'] > df['open']
        df['is_down'] = df['close'] < df['open']
        df['is_flat'] = df['close'] == df['open']
        
        up_bars = df['is_up'].sum()
        down_bars = df['is_down'].sum()
        flat_bars = df['is_flat'].sum()
        
        print(f"\n   Bar Direction:")
        print(f"      Up bars (C>O): {up_bars:,} ({up_bars/len(df)*100:.1f}%)")
        print(f"      Down bars (C<O): {down_bars:,} ({down_bars/len(df)*100:.1f}%)")
        print(f"      Flat bars (C=O): {flat_bars:,} ({flat_bars/len(df)*100:.1f}%)")
        print(f"      Up/Down ratio: {up_bars/down_bars if down_bars > 0 else 'N/A':.2f}")
        
        if down_bars > up_bars * 5:  # More than 5x down bars
            print(f"      ⚠️  EXTREME DIRECTIONAL BIAS - Data likely corrupted!")
    
    else:
        print(f"   ❌ Missing OHLC columns!")
        print(f"      Available columns: {list(df.columns)}")
    
    # Sample data
    print(f"\n📝 SAMPLE DATA (First 20 rows):")
    print(df.head(20))
    
    print(f"\n📝 SAMPLE DATA (Random 10 rows):")
    print(df.sample(min(10, len(df))))
    
    # Check for specific issues
    print(f"\n🔍 SPECIFIC ISSUE CHECKS:")
    
    # Check if all prices are the same
    if 'close' in df.columns:
        unique_closes = df['close'].nunique()
        print(f"   Unique close prices: {unique_closes:,}")
        if unique_closes < len(df) * 0.01:  # Less than 1% unique
            print(f"      ⚠️  Very few unique prices - possible data corruption!")
    
    # Check for price precision
    if 'close' in df.columns:
        # ES should have 0.25 tick size
        df['close_mod'] = df['close'] % 0.25
        non_quarter_ticks = (df['close_mod'] != 0).sum()
        print(f"   Non-quarter tick prices: {non_quarter_ticks:,} ({non_quarter_ticks/len(df)*100:.1f}%)")
        if non_quarter_ticks > len(df) * 0.01:
            print(f"      ⚠️  Many non-quarter tick prices - check tick size!")
    
    print(f"\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    print(f"\n💡 RECOMMENDATIONS:")
    if zero_range_pct > 50:
        print(f"   ❌ CRITICAL: {zero_range_pct:.1f}% of bars have zero range (H=L)")
        print(f"      This is NOT normal for 1-second OHLC data")
        print(f"      Possible causes:")
        print(f"         1. Wrong schema (not OHLCV-1s)")
        print(f"         2. Data corruption during download")
        print(f"         3. Incorrect aggregation")
    
    if down_bars > up_bars * 5:
        print(f"   ❌ CRITICAL: Extreme directional bias ({down_bars/len(df)*100:.1f}% down bars)")
        print(f"      This is impossible in real market data")
        print(f"      Possible causes:")
        print(f"         1. Open/Close columns swapped")
        print(f"         2. Data corruption")
        print(f"         3. Wrong data type")
    
    return df, metadata

if __name__ == "__main__":
    dbn_path = r"C:\Users\jdotzler\Desktop\glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst"
    
    try:
        df, metadata = diagnose_dbn_file(dbn_path)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
