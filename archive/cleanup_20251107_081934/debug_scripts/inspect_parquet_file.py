#!/usr/bin/env python3
"""
Inspect Parquet File

Let's look at the actual parquet file to see what's really happening
with the labeling data.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def inspect_parquet_file():
    """Inspect the parquet file from the desktop"""
    
    file_path = r"C:\Users\jdotzler\Desktop\monthly_2011-06_20251106_213935.parquet"
    
    print("🔍 INSPECTING PARQUET FILE")
    print("=" * 60)
    print(f"File: {file_path}")
    print()
    
    try:
        # Load the data
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        print()
        
        # Show basic info
        print("📊 BASIC INFO:")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Columns: {list(df.columns)}")
        print()
        
        # Check label columns
        label_cols = [col for col in df.columns if col.startswith('label_')]
        weight_cols = [col for col in df.columns if col.startswith('weight_')]
        
        print(f"📋 LABEL COLUMNS ({len(label_cols)}):")
        for col in label_cols:
            print(f"  {col}")
        print()
        
        print(f"⚖️ WEIGHT COLUMNS ({len(weight_cols)}):")
        for col in weight_cols:
            print(f"  {col}")
        print()
        
        # Analyze label values
        print("🎯 LABEL ANALYSIS:")
        for col in label_cols:
            unique_vals = sorted(df[col].unique())
            win_rate = df[col].mean()
            wins = df[col].sum()
            total = len(df)
            
            print(f"  {col}:")
            print(f"    Values: {unique_vals}")
            print(f"    Win rate: {win_rate:.1%} ({wins:,} wins / {total:,} total)")
            
            # Flag suspicious values
            if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                print(f"    ⚠️  NON-BINARY VALUES!")
            
            if 'short' in col and win_rate > 0.6:
                print(f"    🚨 SUSPICIOUS: Short win rate too high!")
            elif 'long' in col and win_rate > 0.6:
                print(f"    🚨 SUSPICIOUS: Long win rate too high!")
        
        print()
        
        # Sample data analysis
        print("📋 SAMPLE DATA (first 10 rows):")
        sample_cols = ['timestamp', 'open', 'high', 'low', 'close'] + label_cols[:3]
        print(df[sample_cols].head(10))
        print()
        
        # Price movement analysis
        print("📈 PRICE MOVEMENT ANALYSIS:")
        df['price_change'] = df['close'].diff()
        
        up_moves = (df['price_change'] > 0).sum()
        down_moves = (df['price_change'] < 0).sum()
        flat_moves = (df['price_change'] == 0).sum()
        
        print(f"  Up moves: {up_moves:,}")
        print(f"  Down moves: {down_moves:,}")
        print(f"  Flat moves: {flat_moves:,}")
        
        if down_moves > 0:
            print(f"  Up/Down ratio: {up_moves/down_moves:.2f}")
        
        # Check if market was trending
        overall_change = df['close'].iloc[-1] - df['close'].iloc[0]
        print(f"  Overall price change: {overall_change:+.2f} points")
        
        if overall_change > 10:
            print("  📈 RISING MARKET - shorts should lose more")
        elif overall_change < -10:
            print("  📉 FALLING MARKET - longs should lose more")
        else:
            print("  ➡️ SIDEWAYS MARKET - mixed results expected")
        
        print()
        
        # Detailed short trade analysis
        print("🔍 DETAILED SHORT TRADE ANALYSIS:")
        short_cols = [col for col in label_cols if 'short' in col]
        
        for col in short_cols:
            print(f"\n  {col}:")
            
            # Sample some winning trades
            winners = df[df[col] == 1].head(5)
            print(f"    Sample winning trades:")
            
            for idx, row in winners.iterrows():
                print(f"      {row['timestamp']}: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}")
                
                # Quick sanity check for normal vol short
                if 'normal_vol' in col:
                    entry_price = row['open']  # Simplified
                    target_price = entry_price - (16 * 0.25)  # 4 points down
                    stop_price = entry_price + (8 * 0.25)     # 2 points up
                    
                    target_hit = row['low'] <= target_price
                    stop_hit = row['high'] >= stop_price
                    
                    print(f"        Entry: {entry_price:.2f}, Target: {target_price:.2f}, Stop: {stop_price:.2f}")
                    print(f"        Target hit: {target_hit}, Stop hit: {stop_hit}")
                    
                    if not target_hit:
                        print(f"        ⚠️  TARGET NOT HIT - why is this a winner?")
                    if target_hit and stop_hit:
                        print(f"        ⚠️  BOTH HIT - should be conservative loss")
        
        # Summary
        print(f"\n🎯 SUMMARY:")
        short_rates = [df[col].mean() for col in label_cols if 'short' in col]
        long_rates = [df[col].mean() for col in label_cols if 'long' in col]
        
        avg_short = np.mean(short_rates) if short_rates else 0
        avg_long = np.mean(long_rates) if long_rates else 0
        
        print(f"  Average short win rate: {avg_short:.1%}")
        print(f"  Average long win rate: {avg_long:.1%}")
        
        if avg_short > 0.6:
            print(f"  🚨 SHORT WIN RATES TOO HIGH - definitely a bug!")
        elif avg_short < 0.2:
            print(f"  ⚠️  SHORT WIN RATES TOO LOW - might be over-corrected")
        else:
            print(f"  ✅ SHORT WIN RATES REASONABLE")
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution"""
    inspect_parquet_file()


if __name__ == "__main__":
    main()