"""
Investigate raw DBN file to understand contract structure and volume patterns
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import databento as db
    HAS_DATABENTO = True
except ImportError:
    HAS_DATABENTO = False
    print("⚠️  databento module not found, will try alternative approach")

import pandas as pd
import pytz
from datetime import time as dt_time
from pathlib import Path

def investigate_dbn_file(dbn_path):
    """Investigate raw DBN file for contract information"""
    
    print("=" * 80)
    print("DBN FILE INVESTIGATION")
    print("=" * 80)
    print(f"\nFile: {dbn_path}\n")
    
    if not HAS_DATABENTO:
        print("❌ databento module required. Install with: pip install databento")
        return None, None, None
    
    # Read DBN file
    print("📖 Reading DBN file...")
    store = db.DBNStore.from_file(dbn_path)
    metadata = store.metadata
    
    print(f"\n📊 METADATA:")
    print(f"   Schema: {metadata.schema}")
    print(f"   Dataset: {metadata.dataset}")
    print(f"   Start: {metadata.start}")
    print(f"   End: {metadata.end}")
    print(f"   Symbols: {metadata.symbols}")
    print(f"   Stype In: {metadata.stype_in}")
    print(f"   Stype Out: {metadata.stype_out}")
    
    # Convert to DataFrame
    print(f"\n🔄 Converting to DataFrame...")
    df = store.to_df()
    
    print(f"\n📋 DATAFRAME INFO:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Index: {df.index.name}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check for symbol/instrument information
    print(f"\n🔍 SYMBOL/INSTRUMENT INFORMATION:")
    
    # Check if there's a symbol column
    if 'symbol' in df.columns:
        print(f"   ✅ Symbol column found!")
        unique_symbols = df['symbol'].unique()
        print(f"   Unique symbols: {unique_symbols}")
        
        for symbol in unique_symbols:
            symbol_count = (df['symbol'] == symbol).sum()
            symbol_pct = symbol_count / len(df) * 100
            print(f"      {symbol}: {symbol_count:,} bars ({symbol_pct:.1f}%)")
    else:
        print(f"   ❌ No 'symbol' column found")
    
    # Check if there's an instrument_id
    if 'instrument_id' in df.columns:
        print(f"   ✅ Instrument ID column found!")
        unique_instruments = df['instrument_id'].unique()
        print(f"   Unique instrument IDs: {unique_instruments}")
        
        for inst_id in unique_instruments:
            inst_count = (df['instrument_id'] == inst_id).sum()
            inst_pct = inst_count / len(df) * 100
            print(f"      ID {inst_id}: {inst_count:,} bars ({inst_pct:.1f}%)")
    else:
        print(f"   ❌ No 'instrument_id' column found")
    
    # Check metadata symbols
    if metadata.symbols:
        print(f"\n   📝 Metadata symbols: {metadata.symbols}")
    
    # Analyze timestamps and volume patterns
    print(f"\n📅 TIMESTAMP ANALYSIS:")
    
    # Create timestamp column
    if hasattr(df.index, 'astype'):
        try:
            start_ns = metadata.start
            end_ns = metadata.end
            timestamps = pd.date_range(
                start=pd.to_datetime(start_ns, unit='ns', utc=True),
                end=pd.to_datetime(end_ns, unit='ns', utc=True),
                periods=len(df)
            )
            df['timestamp'] = timestamps
        except:
            df['timestamp'] = pd.to_datetime(df.index, unit='ns', utc=True)
    
    # Convert to Central Time for RTH analysis
    central_tz = pytz.timezone('US/Central')
    df['timestamp_ct'] = df['timestamp'].dt.tz_convert(central_tz)
    df['date'] = df['timestamp_ct'].dt.date
    df['time'] = df['timestamp_ct'].dt.time
    
    # Filter to RTH (7:30 AM - 3:00 PM Central)
    rth_start = dt_time(7, 30)
    rth_end = dt_time(15, 0)
    rth_mask = (df['time'] >= rth_start) & (df['time'] < rth_end)
    df_rth = df[rth_mask].copy()
    
    print(f"   Total bars: {len(df):,}")
    print(f"   RTH bars: {len(df_rth):,} ({len(df_rth)/len(df)*100:.1f}%)")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Unique dates: {df['date'].nunique()}")
    
    # Daily volume analysis
    print(f"\n📊 DAILY VOLUME ANALYSIS (RTH only):")
    daily_volume = df_rth.groupby('date')['volume'].agg(['sum', 'count', 'mean'])
    daily_volume.columns = ['total_volume', 'bar_count', 'avg_volume_per_bar']
    
    print(f"\n   Daily Statistics:")
    print(f"   {'Date':<12} {'Total Volume':>15} {'Bar Count':>12} {'Avg Vol/Bar':>15}")
    print(f"   {'-'*12} {'-'*15} {'-'*12} {'-'*15}")
    
    for date, row in daily_volume.iterrows():
        print(f"   {date} {row['total_volume']:>15,.0f} {row['bar_count']:>12,.0f} {row['avg_volume_per_bar']:>15,.1f}")
    
    # Check for volume pattern changes (potential contract rolls)
    print(f"\n🔄 VOLUME PATTERN ANALYSIS (Potential Contract Rolls):")
    daily_volume['volume_change_pct'] = daily_volume['total_volume'].pct_change() * 100
    daily_volume['bar_count_change_pct'] = daily_volume['bar_count'].pct_change() * 100
    
    # Flag days with significant volume changes
    significant_changes = daily_volume[
        (abs(daily_volume['volume_change_pct']) > 30) | 
        (abs(daily_volume['bar_count_change_pct']) > 30)
    ]
    
    if len(significant_changes) > 0:
        print(f"\n   ⚠️  Days with >30% volume change (potential contract roll):")
        for date, row in significant_changes.iterrows():
            print(f"      {date}: Volume change: {row['volume_change_pct']:+.1f}%, Bar count change: {row['bar_count_change_pct']:+.1f}%")
    else:
        print(f"   ✅ No significant volume changes detected (likely single contract)")
    
    # Price analysis
    print(f"\n💰 PRICE ANALYSIS:")
    print(f"   Close price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"   Close price mean: {df['close'].mean():.2f}")
    
    # Check for price gaps (potential contract rolls)
    df_sorted = df.sort_values('timestamp')
    df_sorted['price_change'] = df_sorted['close'].diff()
    large_gaps = df_sorted[abs(df_sorted['price_change']) > 5.0]
    
    if len(large_gaps) > 0:
        print(f"\n   ⚠️  Large price gaps (>5 points) detected: {len(large_gaps)}")
        print(f"      First 5 gaps:")
        for idx, row in large_gaps.head(5).iterrows():
            print(f"         {row['timestamp_ct']}: {row['price_change']:+.2f} points")
    else:
        print(f"   ✅ No large price gaps detected")
    
    # Sample data
    print(f"\n📝 SAMPLE DATA (First 10 rows):")
    print(df_rth.head(10)[['timestamp_ct', 'open', 'high', 'low', 'close', 'volume']])
    
    print(f"\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    
    return df, df_rth, daily_volume

if __name__ == "__main__":
    dbn_path = r"C:\Users\jdotzler\Desktop\glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst"
    
    if not Path(dbn_path).exists():
        print(f"❌ File not found: {dbn_path}")
    else:
        df, df_rth, daily_volume = investigate_dbn_file(dbn_path)
        
        print(f"\n💡 KEY FINDINGS:")
        print(f"   - Check if 'symbol' or 'instrument_id' columns exist")
        print(f"   - Look for volume pattern changes indicating contract rolls")
        print(f"   - Verify RTH filtering is working correctly")
        print(f"   - Identify which contract(s) are present in the data")
