#!/usr/bin/env python3
"""
Debug Trending Market Bug

This script investigates why the algorithm fails in trending markets,
showing inverted results (shorts win in rising markets).
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.weighted_labeling import TRADING_MODES, LabelCalculator

def debug_trending_market():
    """Debug the algorithm's behavior in a trending market"""
    
    print("🔍 DEBUGGING TRENDING MARKET BUG")
    print("=" * 60)
    
    # Create simple trending data: 5 bars, steadily rising
    data = {
        'timestamp': [datetime(2011, 6, 10, 9, 30, i) for i in range(5)],
        'open': [1300.00, 1300.00, 1300.25, 1300.50, 1300.75],
        'high': [1300.25, 1300.50, 1300.75, 1301.00, 1301.25],
        'low': [1299.75, 1299.75, 1300.00, 1300.25, 1300.50],
        'close': [1300.00, 1300.25, 1300.50, 1300.75, 1301.00],  # Rising trend
        'volume': [100, 200, 150, 180, 160]
    }
    
    df = pd.DataFrame(data)
    
    print("Trending market data (rising):")
    print(df[['timestamp', 'open', 'high', 'low', 'close']])
    print()
    
    # Test short trade (should lose in rising market)
    print("🔻 SHORT TRADE ANALYSIS:")
    short_mode = TRADING_MODES['normal_vol_short']
    short_calc = LabelCalculator(short_mode, roll_detection_threshold=50.0)
    
    # Manually trace through each potential entry
    for i in range(len(df) - 1):  # Can't enter on last bar
        print(f"\n  Entry signal at bar {i}:")
        
        # Entry price is next bar's open
        entry_price = df.iloc[i + 1]['open']
        target_price = entry_price - (16 * 0.25)  # Short target goes down
        stop_price = entry_price + (8 * 0.25)     # Short stop goes up
        
        print(f"    Entry price (bar {i+1} open): {entry_price:.2f}")
        print(f"    Target: {target_price:.2f}, Stop: {stop_price:.2f}")
        
        # Check what happens in the entry bar and subsequent bars
        for j in range(i + 1, len(df)):
            bar = df.iloc[j]
            target_hit = bar['low'] <= target_price
            stop_hit = bar['high'] >= stop_price
            
            print(f"    Bar {j}: H={bar['high']:.2f}, L={bar['low']:.2f}")
            print(f"      Target hit: {target_hit}, Stop hit: {stop_hit}")
            
            if target_hit or stop_hit:
                if target_hit and not stop_hit:
                    print(f"      → WIN (target hit)")
                elif stop_hit and not target_hit:
                    print(f"      → LOSS (stop hit)")
                elif target_hit and stop_hit:
                    print(f"      → LOSS (both hit, conservative)")
                break
        else:
            print(f"      → LOSS (timeout)")
    
    # Now test with the actual calculator
    print(f"\n📊 ACTUAL CALCULATOR RESULTS:")
    short_labels, _, _ = short_calc.calculate_labels(df)
    print(f"Short labels: {short_labels}")
    
    wins = short_labels.sum()
    total = len(short_labels) - 1  # Last bar can't be entry
    win_rate = wins / total if total > 0 else 0
    
    print(f"Short win rate: {win_rate:.1%} ({wins}/{total})")
    
    if win_rate > 0.5:
        print("❌ BUG: Shorts winning in rising market!")
    else:
        print("✅ CORRECT: Shorts losing in rising market")

def debug_entry_price_calculation():
    """Debug if there's an issue with entry price calculation"""
    
    print(f"\n🔍 DEBUGGING ENTRY PRICE CALCULATION")
    print("=" * 60)
    
    # Simple 3-bar test
    data = {
        'timestamp': [datetime(2011, 6, 10, 9, 30, i) for i in range(3)],
        'open': [1300.00, 1300.50, 1301.00],    # Rising opens
        'high': [1300.25, 1300.75, 1301.25],
        'low': [1299.75, 1300.25, 1300.75],
        'close': [1300.00, 1300.50, 1301.00],
        'volume': [100, 200, 150]
    }
    
    df = pd.DataFrame(data)
    
    print("Test data:")
    print(df[['open', 'high', 'low', 'close']])
    print()
    
    # Test entry at bar 0 (signal bar)
    print("Entry signal at bar 0:")
    print(f"  Entry price should be bar 1 open: {df.iloc[1]['open']:.2f}")
    
    # Short trade calculation
    entry_price = df.iloc[1]['open']  # 1300.50
    target_price = entry_price - (16 * 0.25)  # 1296.50
    stop_price = entry_price + (8 * 0.25)     # 1302.50
    
    print(f"  Short target: {target_price:.2f}")
    print(f"  Short stop: {stop_price:.2f}")
    
    # Check bar 1 (entry bar)
    bar1 = df.iloc[1]
    target_hit = bar1['low'] <= target_price  # 1300.25 <= 1296.50 = False
    stop_hit = bar1['high'] >= stop_price     # 1300.75 >= 1302.50 = False
    
    print(f"  Bar 1: H={bar1['high']:.2f}, L={bar1['low']:.2f}")
    print(f"  Target hit: {target_hit}")
    print(f"  Stop hit: {stop_hit}")
    print(f"  Expected: Neither hit, continue to next bar")
    
    # Check bar 2
    bar2 = df.iloc[2]
    target_hit_2 = bar2['low'] <= target_price  # 1300.75 <= 1296.50 = False
    stop_hit_2 = bar2['high'] >= stop_price     # 1301.25 >= 1302.50 = False
    
    print(f"  Bar 2: H={bar2['high']:.2f}, L={bar2['low']:.2f}")
    print(f"  Target hit: {target_hit_2}")
    print(f"  Stop hit: {stop_hit_2}")
    print(f"  Expected: Neither hit, timeout → LOSS")
    
    # Test with calculator
    short_mode = TRADING_MODES['normal_vol_short']
    short_calc = LabelCalculator(short_mode, roll_detection_threshold=50.0)
    
    labels, _, _ = short_calc.calculate_labels(df)
    print(f"\nCalculator result: {labels}")
    print(f"Bar 0 label: {labels[0]} ({'WIN' if labels[0] == 1 else 'LOSS'})")
    
    if labels[0] == 0:
        print("✅ CORRECT: Short trade loses in rising market")
    else:
        print("❌ BUG: Short trade wins when it shouldn't")

def main():
    """Main execution"""
    
    debug_trending_market()
    debug_entry_price_calculation()
    
    print(f"\n🎯 CONCLUSION:")
    print("If shorts are winning in rising markets, there's definitely a bug.")
    print("The issue might be in:")
    print("1. Entry price calculation (wrong bar)")
    print("2. Target/stop price calculation (wrong direction)")
    print("3. Hit detection logic (wrong comparisons)")
    print("4. Win/loss determination (inverted logic)")


if __name__ == "__main__":
    main()