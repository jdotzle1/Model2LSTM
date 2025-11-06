#!/usr/bin/env python3
"""
Debug Exact Issue

This script identifies the exact bug in the labeling algorithm
by stepping through the code line by line.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.weighted_labeling import TRADING_MODES, LabelCalculator

def debug_exact_algorithm():
    """Debug the exact algorithm step by step"""
    
    print("🐛 DEBUGGING EXACT ALGORITHM")
    print("=" * 60)
    
    # Create simple test case
    data = {
        'timestamp': [
            datetime(2011, 6, 10, 10, 0, 0),  # Bar 0 (signal bar)
            datetime(2011, 6, 10, 10, 0, 1),  # Bar 1 (entry bar)
            datetime(2011, 6, 10, 10, 0, 2),  # Bar 2 (outcome bar)
        ],
        'open': [1280.00, 1280.00, 1277.00],   # Entry at bar 1 open = 1280.00
        'high': [1280.25, 1280.50, 1277.50],  # Bar 1 high = 1280.50 (doesn't hit stop 1282.00)
        'low': [1279.75, 1276.00, 1276.50],   # Bar 1 low = 1276.00 (hits target 1276.00)
        'close': [1280.00, 1277.00, 1277.25],
        'volume': [100, 200, 150]
    }
    
    df = pd.DataFrame(data)
    mode = TRADING_MODES['normal_vol_short']
    
    print("Test data:")
    print(df)
    print()
    
    print(f"Mode: {mode.name}")
    print(f"Stop: {mode.stop_ticks} ticks, Target: {mode.target_ticks} ticks")
    print()
    
    # Manual step-through of algorithm
    print("Manual algorithm step-through:")
    print()
    
    # Step 1: Check entry_idx = 0 (signal bar)
    entry_idx = 0
    print(f"Processing entry_idx = {entry_idx} (signal bar)")
    
    # Step 2: Entry price is next bar's open
    if entry_idx + 1 >= len(df):
        print("  No next bar available - should return label=0")
        return
    
    entry_price = df.iloc[entry_idx + 1]['open']
    print(f"  Entry price (bar {entry_idx + 1} open): {entry_price:.2f}")
    
    # Step 3: Calculate target and stop
    target_price = entry_price - (mode.target_ticks * 0.25)  # Short target goes down
    stop_price = entry_price + (mode.stop_ticks * 0.25)      # Short stop goes up
    print(f"  Target price: {target_price:.2f}")
    print(f"  Stop price: {stop_price:.2f}")
    
    # Step 4: Look forward from entry bar
    start_idx = entry_idx + 1  # Start from entry bar (bar 1)
    end_idx = min(start_idx + 900, len(df))  # Up to timeout or end of data
    print(f"  Search range: bars {start_idx} to {end_idx-1}")
    
    # Step 5: Check each bar for hits
    for j in range(start_idx, end_idx):
        print(f"  Checking bar {j}:")
        bar = df.iloc[j]
        print(f"    High: {bar['high']:.2f}, Low: {bar['low']:.2f}")
        
        # Check hits
        target_hit = bar['low'] <= target_price
        stop_hit = bar['high'] >= stop_price
        print(f"    Target hit: {target_hit} ({bar['low']:.2f} <= {target_price:.2f})")
        print(f"    Stop hit: {stop_hit} ({bar['high']:.2f} >= {stop_price:.2f})")
        
        if target_hit and stop_hit:
            print(f"    Both hit - return LOSS (conservative)")
            return 0
        elif target_hit:
            print(f"    Target hit first - return WIN")
            return 1
        elif stop_hit:
            print(f"    Stop hit first - return LOSS")
            return 0
        else:
            print(f"    Neither hit - continue")
    
    print(f"  Timeout - return LOSS")
    return 0

def test_with_actual_calculator():
    """Test with the actual calculator to see what happens"""
    
    print(f"\n🧪 TESTING WITH ACTUAL CALCULATOR")
    print("=" * 60)
    
    # Same test data
    data = {
        'timestamp': [
            datetime(2011, 6, 10, 10, 0, 0),
            datetime(2011, 6, 10, 10, 0, 1),
            datetime(2011, 6, 10, 10, 0, 2),
        ],
        'open': [1280.00, 1280.00, 1277.00],
        'high': [1280.25, 1280.50, 1277.50],
        'low': [1279.75, 1276.00, 1276.50],
        'close': [1280.00, 1277.00, 1277.25],
        'volume': [100, 200, 150]
    }
    
    df = pd.DataFrame(data)
    mode = TRADING_MODES['normal_vol_short']
    calculator = LabelCalculator(mode, roll_detection_threshold=2.0)
    
    # Get the actual result
    labels, mae_ticks, seconds_to_target = calculator.calculate_labels(df)
    
    print(f"Actual calculator results:")
    print(f"Labels: {labels}")
    print(f"MAE ticks: {mae_ticks}")
    print(f"Seconds to target: {seconds_to_target}")
    print()
    
    print(f"Expected: [1, 0, 0] (bar 0 should win)")
    print(f"Actual: {labels}")
    
    if labels[0] == 1:
        print("✅ Calculator working correctly")
    else:
        print("❌ Calculator has a bug!")
        
        # Let's debug the calculator's internal method
        print(f"\nDebugging calculator's _check_single_entry method:")
        
        # Convert to numpy arrays like the calculator does
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df['timestamp'].values
        n_bars = len(df)
        roll_affected_bars = np.zeros(n_bars, dtype=bool)  # No rollover
        
        result = calculator._check_single_entry(
            0, opens, highs, lows, timestamps, n_bars, roll_affected_bars
        )
        
        print(f"_check_single_entry result: {result}")

def main():
    """Main execution"""
    
    manual_result = debug_exact_algorithm()
    print(f"\nManual calculation result: {manual_result}")
    
    test_with_actual_calculator()
    
    print(f"\n🎯 CONCLUSION:")
    print("If manual calculation gives 1 but calculator gives 0,")
    print("then there's definitely a bug in the implementation!")


if __name__ == "__main__":
    main()