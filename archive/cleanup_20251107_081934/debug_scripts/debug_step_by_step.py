#!/usr/bin/env python3
"""
Debug Step by Step

This script steps through the exact algorithm execution to see
where the bug is occurring.
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

def debug_step_by_step():
    """Debug the algorithm step by step"""
    
    print("🔍 STEP-BY-STEP DEBUGGING")
    print("=" * 60)
    
    # Create obvious winner case
    data = {
        'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(3)],
        'open': [1300.00, 1300.00, 1290.00],    # Entry at 1300
        'high': [1300.50, 1300.25, 1290.50],   # Doesn't hit stop (1302)
        'low': [1299.50, 1284.00, 1289.50],    # Hits target (1296) easily
        'close': [1300.00, 1285.00, 1290.00],
        'volume': [100, 200, 150]
    }
    
    df = pd.DataFrame(data)
    mode = TRADING_MODES['normal_vol_short']
    calculator = LabelCalculator(mode, roll_detection_threshold=10.0)
    
    print("Test data:")
    print(df)
    print()
    
    # Step through manually
    print("Manual step-through:")
    
    # Step 1: Check rollover detection
    roll_affected_bars = calculator._detect_contract_rolls(df)
    print(f"1. Rollover affected bars: {roll_affected_bars}")
    
    if roll_affected_bars[0]:
        print("   🚨 Bar 0 is marked as rollover-affected!")
        print("   This would cause it to get label=0")
        return
    
    # Step 2: Call _check_single_entry directly
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    timestamps = df['timestamp'].values
    n_bars = len(df)
    
    print(f"2. Calling _check_single_entry(0, ...)")
    result = calculator._check_single_entry(
        0, opens, highs, lows, timestamps, n_bars, roll_affected_bars
    )
    
    print(f"   Result: {result}")
    
    if result['label'] == 1:
        print("   ✅ _check_single_entry correctly returns WIN")
    else:
        print("   ❌ _check_single_entry incorrectly returns LOSS")
        
        # Debug why it returned loss
        print(f"   Debugging _check_single_entry:")
        debug_check_single_entry(calculator, 0, opens, highs, lows, timestamps, n_bars, roll_affected_bars)
    
    # Step 3: Call full calculate_labels
    print(f"3. Calling calculate_labels()")
    labels, mae_ticks, seconds_to_target = calculator.calculate_labels(df)
    print(f"   Labels: {labels}")
    print(f"   MAE: {mae_ticks}")
    print(f"   Seconds: {seconds_to_target}")
    
    if labels[0] == 1:
        print("   ✅ calculate_labels correctly returns WIN")
    else:
        print("   ❌ calculate_labels incorrectly returns LOSS")
        print("   This means the bug is in the main loop, not _check_single_entry")

def debug_check_single_entry(calculator, entry_idx, opens, highs, lows, timestamps, n_bars, roll_affected_bars):
    """Debug the _check_single_entry method in detail"""
    
    print(f"     Debugging _check_single_entry for entry_idx={entry_idx}:")
    
    # Check bounds
    if entry_idx + 1 >= n_bars:
        print(f"     Entry idx + 1 ({entry_idx + 1}) >= n_bars ({n_bars}) - return LOSS")
        return
    
    # Entry price
    entry_price = opens[entry_idx + 1]
    entry_time = timestamps[entry_idx + 1]
    print(f"     Entry price: {entry_price:.2f}")
    print(f"     Entry time: {entry_time}")
    
    # Target/stop prices
    target_price, stop_price = calculator._calculate_target_stop_prices(entry_price)
    print(f"     Target: {target_price:.2f}, Stop: {stop_price:.2f}")
    
    # Search range
    start_idx = entry_idx + 1
    end_idx = min(start_idx + 900, n_bars)  # TIMEOUT_SECONDS = 900
    print(f"     Search range: {start_idx} to {end_idx-1}")
    
    # Check each bar
    for j in range(start_idx, end_idx):
        print(f"     Checking bar {j}:")
        
        # Rollover check
        if roll_affected_bars[j]:
            print(f"       Bar {j} is rollover-affected - return LOSS")
            return
        
        # Hit detection
        target_hit, stop_hit = calculator._check_price_hits(
            highs[j], lows[j], target_price, stop_price
        )
        
        print(f"       Bar {j}: H={highs[j]:.2f}, L={lows[j]:.2f}")
        print(f"       Target hit: {target_hit}, Stop hit: {stop_hit}")
        
        # Decision logic
        if target_hit and stop_hit:
            print(f"       Both hit - return LOSS (conservative)")
            return
        elif target_hit:
            print(f"       Target hit first - return WIN")
            return
        elif stop_hit:
            print(f"       Stop hit first - return LOSS")
            return
        else:
            print(f"       Neither hit - continue")
    
    print(f"     Timeout - return LOSS")

def test_different_scenarios():
    """Test different scenarios to isolate the issue"""
    
    print(f"\n🧪 TESTING DIFFERENT SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'Single bar winner',
            'data': {
                'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(2)],
                'open': [1300.00, 1300.00],
                'high': [1300.50, 1300.00],  # Doesn't hit stop
                'low': [1299.50, 1296.00],   # Hits target exactly
                'close': [1300.00, 1297.00],
                'volume': [100, 200]
            }
        },
        {
            'name': 'Clear loser',
            'data': {
                'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(2)],
                'open': [1300.00, 1300.00],
                'high': [1300.50, 1302.00],  # Hits stop exactly
                'low': [1299.50, 1299.00],   # Doesn't hit target
                'close': [1300.00, 1301.00],
                'volume': [100, 200]
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}:")
        df = pd.DataFrame(scenario['data'])
        mode = TRADING_MODES['normal_vol_short']
        calculator = LabelCalculator(mode, roll_detection_threshold=10.0)
        
        labels, _, _ = calculator.calculate_labels(df)
        print(f"   Result: {'WIN' if labels[0] == 1 else 'LOSS'}")

def main():
    """Main execution"""
    
    debug_step_by_step()
    test_different_scenarios()
    
    print(f"\n🎯 DEBUGGING CONCLUSION:")
    print("If _check_single_entry returns WIN but calculate_labels returns LOSS,")
    print("then the bug is in the main loop of calculate_labels.")
    print("If _check_single_entry returns LOSS, then the bug is deeper in the logic.")


if __name__ == "__main__":
    main()