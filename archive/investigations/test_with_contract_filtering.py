"""
Test with contract filtering THEN labeling
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import pytz
from datetime import time as dt_time

print("=" * 80)
print("TESTING WITH CONTRACT FILTERING - July 2010")
print("=" * 80)

# Load corrected data
corrected_path = r"C:\Users\jdotzler\Desktop\july_2010_CORRECTED.parquet"
print(f"\n📖 Loading corrected data: {corrected_path}")

df = pd.read_parquet(corrected_path)
print(f"   Loaded: {len(df):,} rows\n")

# Clean data
print(f"🧹 Cleaning data...")
df_clean = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)].copy()
print(f"   Cleaned: {len(df_clean):,} rows\n")

# Filter to RTH
print(f"🕐 Filtering to RTH...")
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
print(f"   RTH rows: {len(df_rth):,}\n")

# Apply contract filtering
print(f"🔄 Applying contract filtering...")
try:
    from src.data_pipeline.contract_filtering import detect_and_filter_contracts, validate_contract_filtering
    
    df_before = df_rth.copy()
    df_filtered, stats = detect_and_filter_contracts(df_rth)
    
    print(f"\n   Contract filtering results:")
    print(f"      Removed: {stats['removed_rows']:,} rows ({stats['removal_percentage']:.1f}%)")
    print(f"      Remaining: {len(df_filtered):,} rows")
    print(f"      Days with filtering: {stats['days_with_contract_filtering']}\n")
    
    # Validate
    validation = validate_contract_filtering(df_before, df_filtered)
    
except Exception as e:
    print(f"   ❌ Contract filtering failed: {e}")
    df_filtered = df_rth
    import traceback
    traceback.print_exc()

# Apply weighted labeling
print(f"\n🏷️  Applying weighted labeling...")
try:
    from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
    
    engine = WeightedLabelingEngine()
    df_labeled = engine.process_dataframe(df_filtered, validate_performance=False)
    
    print(f"   ✅ Labeling complete\n")
    
    # Check win rates
    print(f"📊 WIN RATES AFTER CONTRACT FILTERING:")
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
        print(f"      Long: {long_win_rate:.1%}")
        print(f"      Short: {short_win_rate:.1%}")
    else:
        print(f"   ⚠️  Win rates still problematic:")
        print(f"      Long: {long_win_rate:.1%}")
        print(f"      Short: {short_win_rate:.1%}")
    
except Exception as e:
    print(f"   ❌ Labeling failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
