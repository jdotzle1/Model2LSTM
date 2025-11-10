"""
Test October 2025 with the full corrected pipeline
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import pytz
from datetime import time as dt_time

print("=" * 80)
print("TESTING OCTOBER 2025 WITH CORRECTED PIPELINE")
print("=" * 80)

dbn_path = r"C:\Users\jdotzler\Desktop\oct2025_extracted\glbx-mdp3-20250922-20251021.ohlcv-1s.dbn.zst"
output_path = r"C:\Users\jdotzler\Desktop\oct2025_FINAL_TEST.parquet"

# Step 1: Convert DBN with correct timestamps
print(f"\n📖 Step 1: Converting DBN file...")
try:
    import databento as db
    
    store = db.DBNStore.from_file(dbn_path)
    df = store.to_df()
    
    print(f"   Rows: {len(df):,}")
    print(f"   Using ts_event index: {df.index.name == 'ts_event'}")
    
    # Use actual timestamps from ts_event index
    if df.index.name == 'ts_event':
        df['timestamp'] = df.index
        df = df.reset_index(drop=True)
        print(f"   ✅ Using actual timestamps from DBN\n")
    else:
        print(f"   ❌ ts_event not found!\n")
        sys.exit(1)
        
except ImportError:
    print("❌ Run with: py -3.12 test_oct2025_full_pipeline.py")
    sys.exit(1)

# Step 2: Clean data
print(f"🧹 Step 2: Cleaning data...")
df_clean = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)].copy()
removed = len(df) - len(df_clean)
print(f"   Removed {removed:,} rows with bad prices")
print(f"   Remaining: {len(df_clean):,} rows\n")

# Step 3: Filter to RTH
print(f"🕐 Step 3: Filtering to RTH (7:30 AM - 3:00 PM Central)...")
central_tz = pytz.timezone('US/Central')

df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
if df_clean['timestamp'].dt.tz is None:
    df_clean['timestamp'] = df_clean['timestamp'].dt.tz_localize(pytz.UTC)

df_clean['timestamp_ct'] = df_clean['timestamp'].dt.tz_convert(central_tz)
df_clean['time'] = df_clean['timestamp_ct'].dt.time

rth_start = dt_time(7, 30)
rth_end = dt_time(15, 0)
rth_mask = (df_clean['time'] >= rth_start) & (df_clean['time'] < rth_end)
df_rth = df_clean[rth_mask].copy()

print(f"   RTH rows: {len(df_rth):,} ({len(df_rth)/len(df_clean)*100:.1f}%)\n")

# Step 4: Contract filtering
print(f"🔄 Step 4: Contract filtering...")
try:
    from src.data_pipeline.contract_filtering import detect_and_filter_contracts
    
    df_filtered, stats = detect_and_filter_contracts(df_rth)
    
    print(f"   Removed: {stats['removed_rows']:,} rows ({stats['removal_percentage']:.1f}%)")
    print(f"   Remaining: {len(df_filtered):,} rows\n")
    
except Exception as e:
    print(f"   ⚠️  Contract filtering failed: {e}")
    df_filtered = df_rth

# Step 5: Weighted labeling
print(f"🏷️  Step 5: Weighted labeling...")
try:
    from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
    
    engine = WeightedLabelingEngine()
    df_labeled = engine.process_dataframe(df_filtered, validate_performance=False)
    
    print(f"   ✅ Labeling complete\n")
    
    # Check win rates
    print(f"📊 WIN RATES:")
    label_cols = [c for c in df_labeled.columns if c.startswith('label_')]
    
    for col in label_cols:
        win_rate = df_labeled[col].mean()
        wins = (df_labeled[col] == 1).sum()
        losses = (df_labeled[col] == 0).sum()
        print(f"   {col:<25}: {win_rate:>6.1%} ({wins:>7,} wins / {losses:>7,} losses)")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    long_win_rate = df_labeled['label_low_vol_long'].mean()
    short_win_rate = df_labeled['label_low_vol_short'].mean()
    
    print(f"   Long win rate: {long_win_rate:.1%}")
    print(f"   Short win rate: {short_win_rate:.1%}")
    
    if 0.15 <= long_win_rate <= 0.35 and 0.15 <= short_win_rate <= 0.35:
        print(f"\n   ✅✅✅ WIN RATES ARE REASONABLE! ✅✅✅")
        print(f"   The corrected pipeline WORKS!")
        print(f"   July 2010 data is the problem, not the pipeline!")
    elif long_win_rate < 0.10 or short_win_rate > 0.70:
        print(f"\n   ❌ Win rates still problematic")
        print(f"   The pipeline may still have issues")
    else:
        print(f"\n   ⚙️  Win rates are improved but not ideal")
    
    # Save
    df_labeled.to_parquet(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
    
except Exception as e:
    print(f"   ❌ Labeling failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
