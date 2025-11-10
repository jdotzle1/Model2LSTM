"""
Test labeling on the corrected July 2010 data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import pytz
from datetime import time as dt_time

print("=" * 80)
print("TESTING LABELING ON CORRECTED DATA - July 2010")
print("=" * 80)

# Load corrected data
corrected_path = r"C:\Users\jdotzler\Desktop\july_2010_CORRECTED.parquet"
print(f"\n📖 Loading corrected data: {corrected_path}")

df = pd.read_parquet(corrected_path)
print(f"   Loaded: {len(df):,} rows\n")

# Clean data (remove zero/negative prices)
print(f"🧹 Cleaning data...")
price_cols = ['open', 'high', 'low', 'close']
for col in price_cols:
    bad_prices = (df[col] <= 0).sum()
    if bad_prices > 0:
        print(f"   Removing {bad_prices:,} rows with zero/negative {col}")

# Remove rows with any zero/negative prices
df_clean = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)].copy()
print(f"   Cleaned: {len(df_clean):,} rows ({len(df_clean)/len(df)*100:.1f}%)\n")

# Filter to RTH only (9:30 AM - 4:00 PM ET = 7:30 AM - 3:00 PM CT)
print(f"🕐 Filtering to RTH (7:30 AM - 3:00 PM Central)...")
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

# Apply weighted labeling
print(f"🏷️  Applying weighted labeling...")
try:
    from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
    
    engine = WeightedLabelingEngine()
    df_labeled = engine.process_dataframe(df_rth, validate_performance=False)
    
    print(f"   ✅ Labeling complete\n")
    
    # Check win rates
    print(f"📊 WIN RATES:")
    label_cols = [c for c in df_labeled.columns if c.startswith('label_')]
    
    for col in label_cols:
        win_rate = df_labeled[col].mean()
        wins = (df_labeled[col] == 1).sum()
        losses = (df_labeled[col] == 0).sum()
        print(f"   {col:<25}: {win_rate:>6.1%} ({wins:>7,} wins / {losses:>7,} losses)")
    
    # Check if reasonable
    print(f"\n🔍 ANALYSIS:")
    long_win_rate = df_labeled['label_low_vol_long'].mean()
    short_win_rate = df_labeled['label_low_vol_short'].mean()
    
    if 0.15 <= long_win_rate <= 0.35 and 0.15 <= short_win_rate <= 0.35:
        print(f"   ✅ WIN RATES ARE REASONABLE!")
        print(f"      Long: {long_win_rate:.1%} (expected 15-30%)")
        print(f"      Short: {short_win_rate:.1%} (expected 15-30%)")
        print(f"      ✅ TIMESTAMP FIX WORKED!")
    else:
        print(f"   ⚠️  Win rates still problematic:")
        print(f"      Long: {long_win_rate:.1%}")
        print(f"      Short: {short_win_rate:.1%}")
        if long_win_rate < 0.10 or short_win_rate > 0.70:
            print(f"      May still have issues to investigate")
    
    # Save labeled data
    output_path = r"C:\Users\jdotzler\Desktop\july_2010_CORRECTED_LABELED.parquet"
    df_labeled.to_parquet(output_path, index=False)
    print(f"\n💾 Saved labeled data: {output_path}")
    
except Exception as e:
    print(f"   ❌ Labeling failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
