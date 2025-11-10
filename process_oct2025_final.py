"""
Process October 2025 data with corrected pipeline using Python 3.13
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import databento as db
import pandas as pd
import pytz
from datetime import time as dt_time
from src.data_pipeline.corrected_contract_filtering import process_complete_pipeline

print("=" * 80)
print("PROCESSING OCTOBER 2025 DATA - FINAL")
print("=" * 80)

# Load DBN file
dbn_file = r"C:\Users\jdotzler\Downloads\glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst"

print(f"\n📖 Loading: {dbn_file}")
store = db.DBNStore.from_file(dbn_file)
df_raw = store.to_df()

# Reset index
if df_raw.index.name == 'ts_event':
    df_raw = df_raw.reset_index()
    df_raw = df_raw.rename(columns={'ts_event': 'timestamp'})

print(f"   Loaded: {len(df_raw):,} rows")
print(f"   Columns: {df_raw.columns.tolist()}")

# Check for symbol column
if 'symbol' in df_raw.columns:
    print(f"\n   Top 5 symbols by volume:")
    symbol_volumes = df_raw.groupby('symbol')['volume'].sum().sort_values(ascending=False)
    for symbol, vol in symbol_volumes.head(5).items():
        count = (df_raw['symbol'] == symbol).sum()
        print(f"      {symbol}: {vol:,} volume, {count:,} bars")

# Run corrected pipeline
print("\n" + "=" * 80)
df_processed, stats = process_complete_pipeline(df_raw)

# Validation
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

central_tz = pytz.timezone('US/Central')
df_processed['timestamp_ct'] = df_processed['timestamp'].dt.tz_convert(central_tz)
df_processed['date'] = df_processed['timestamp_ct'].dt.date

trading_days = sorted(df_processed['date'].unique())

print(f"\nTrading days: {len(trading_days)}")
print(f"Total rows: {len(df_processed):,}")
print(f"Expected: ~{len(trading_days) * 27000:,} ({len(trading_days)} × 27,000)")

# Save
output_file = "oct2025_processed_FINAL.parquet"
df_processed_clean = df_processed.drop(columns=['timestamp_ct', 'date'], errors='ignore')
df_processed_clean.to_parquet(output_file)

print(f"\n💾 Saved: {output_file}")
print(f"   Size: {os.path.getsize(output_file) / (1024**2):.1f} MB")

print("\n" + "=" * 80)
print("COMPLETE ✅")
print("=" * 80)
