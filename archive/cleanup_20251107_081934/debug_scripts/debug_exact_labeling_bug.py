#!/usr/bin/env python3
"""
Debug Exact Labeling Bug

This script creates a very specific test case to isolate the exact bug
in the labeling logic that's causing 100% short wins and 0% long wins.
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

def debug_single_trade():
    """Debug a single trade in extreme detail"""
    
    print("🔍 DEBUGGING SINGLE TRADE IN DETAIL")
    print("=" * 60)
    
    # Create a very simple case: 2 bars, clear short winner
    data = {
        'timestamp': [
            datetime(2011, 6, 10, 9, 30, 0),   # Bar 0 (signal bar)
            datetime(2011, 6, 10, 9, 30, 1),   # Bar 1 (entry bar)
        ],
        'open': [1300.00, 1300.00],     # Entry at 1300.00
        'high': [1300.25, 1300.50],    # Bar 1 high = 1300.50 (doesn't hit stop)
        'low': [1299.75, 1295.00],     # Bar 1 low = 1295.00 (hits target easily)
        'close': [1300.00, 1296.00],
        'volume': [100, 200]
    }
    
    df = pd.DataFrame(data)
    mode = TRADING_MODES['normal_vol_short']  # 8 tick stop, 16 tick target
    
    print("Test data:")
    print(df)
    print()
    
    print(f"Mode: {mode.name}")
    print(f"Direction: {mode.direction}")
    print(f"Stop: {mode.stop_ticks} ticks, Target: {mode.target_ticks} ticks")
    print()
    
    # Manual calculation
    entry_price = 1300.00
    target_price = entry_price - (16 * 0.25)  # 1296.00
    stop_price = entry_price + (8 * 0.25)     # 1302.00
    
    print("Manual calculation:")
    print(f"  Entry: {entry_price:.2f}")
    print(f"  Target: {target_price:.2f} (short target goes DOWN)")
    print(f"  Stop: {stop_price:.2f} (short stop goes UP)")
    print()
    
    # Check bar 1 (entry bar)
    bar1_high = 1300.50
    bar1_low = 1295.00
    
    target_hit = bar1_low <= target_price  # 1295.00 <= 1296.00 = True
    stop_hit = bar1_high >= stop_price     # 1300.50 >= 1302.00 = False
    
    print("Bar 1 analysis:")
    print(f"  High: {bar1_high:.2f}, Low: {bar1_low:.2f}")
    print(f"  Target hit: {target_hit} ({bar1_low:.2f} <= {target_price:.2f})")
    print(f"  Stop hit: {stop_hit} ({bar1_high:.2f} >= {stop_price:.2f})")
    print(f"  Expected result: WIN (target hit, stop not hit)")
    print()
    
    # Test with calculator
    calculator = LabelCalculator(mode, roll_detection_threshold=50.0)  # High threshold
    
    # Check rollover detection first
    roll_affected_bars = calculator._detect_contract_rolls(df)
    print(f"Rollover affected bars: {roll_affected_bars}")
    
    if roll_affected_bars[0]:
        print("❌ Bar 0 incorrectly marked as rollover-affected!")
        return
    
    # Calculate labels
    labels, mae_ticks, seconds_to_target = calculator.calculate_labels(df)
    
    print(f"Calculator results:")
    print(f"  Labels: {labels}")
    print(f"  MAE ticks: {mae_ticks}")
    print(f"  Seconds to target: {seconds_to_target}")
    print()
    
    if labels[0] == 1:
        print("✅ CORRECT: Calculator correctly identified winner")
    else:
        print("❌ BUG: Calculator incorrectly marked as loser")
        
        # Debug the _check_single_entry method
        print("\nDebugging _check_single_entry:")
        
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df['timestamp'].values
        n_bars = len(df)
        
        result = calculator._check_single_entry(
            0, opens, highs, lows, timestamps, n_bars, roll_affected_bars
        )
        
        print(f"  _check_single_entry result: {result}")

def test_both_directions():
    """Test both long and short trades with identical setups"""
    
    print(f"\n🔄 TESTING BOTH DIRECTIONS")
    print("=" * 60)
    
    # Test data: price goes from 1300 to 1296 (4-point drop)
    data = {
        'timestamp': [
            datetime(2011, 6, 10, 9, 30, 0),
            datetime(2011, 6, 10, 9, 30, 1),
        ],
        'open': [1300.00, 1300.00],
        'high': [1300.25, 1300.50],
        'low': [1299.75, 1296.00],  # 4-point drop
        'close': [1300.00, 1296.50],
        'volume': [100, 200]
    }
    
    df = pd.DataFrame(data)
    
    # Test short trade (should win - price dropped)
    print("SHORT TRADE TEST:")
    short_mode = TRADING_MODES['normal_vol_short']
    short_calc = LabelCalculator(short_mode, roll_detection_threshold=50.0)
    
    short_labels, _, _ = short_calc.calculate_labels(df)
    print(f"  Short result: {'WIN' if short_labels[0] == 1 else 'LOSS'}")
    print(f"  Expected: WIN (price dropped 4 points)")
    
    # Test long trade (should lose - price dropped)
    print("\nLONG TRADE TEST:")
    long_mode = TRADING_MODES['normal_vol_long']
    long_calc = LabelCalculator(long_mode, roll_detection_threshold=50.0)
    
    long_labels, _, _ = long_calc.calculate_labels(df)
    print(f"  Long result: {'WIN' if long_labels[0] == 1 else 'LOSS'}")
    print(f"  Expected: LOSS (price dropped 4 points)")
    
    # Analysis
    print(f"\nANALYSIS:")
    if short_labels[0] == 1 and long_labels[0] == 0:
        print("✅ CORRECT: Short wins, long loses (price dropped)")
    elif short_labels[0] == 0 and long_labels[0] == 1:
        print("❌ INVERTED: Long wins, short loses (logic inverted)")
    elif short_labels[0] == 1 and long_labels[0] == 1:
        print("❌ BOTH WIN: Something wrong with loss detection")
    else:
        print("❌ BOTH LOSE: Something wrong with win detection")

def main():
    """Main execution"""
    
    debug_single_trade()
    test_both_directions()
    
    print(f"\n🎯 CONCLUSION:")
    print("If the single trade test shows the bug, we can isolate the exact issue.")
    print("If both directions test shows inversion, we know the logic is backwards.")


if __name__ == "__main__":
    main()