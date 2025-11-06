#!/usr/bin/env python3
"""
Debug Rollover Detection Bug

This script checks if the rollover detection is incorrectly marking
all bars as affected, which would cause all labels to be 0.
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

def debug_rollover_detection():
    """Debug the rollover detection method"""
    
    print("🔍 DEBUGGING ROLLOVER DETECTION")
    print("=" * 60)
    
    # Create test data with no rollover gaps
    data = {
        'timestamp': [
            datetime(2011, 6, 10, 10, 0, 0),
            datetime(2011, 6, 10, 10, 0, 1),
            datetime(2011, 6, 10, 10, 0, 2),
        ],
        'open': [1280.00, 1280.00, 1277.00],
        'high': [1280.25, 1280.50, 1277.50],
        'low': [1279.75, 1276.00, 1276.50],
        'close': [1280.00, 1277.00, 1277.25],  # Price changes: 0, -3.00, +0.25
        'volume': [100, 200, 150]
    }
    
    df = pd.DataFrame(data)
    mode = TRADING_MODES['normal_vol_short']
    calculator = LabelCalculator(mode, roll_detection_threshold=2.0)
    
    print("Test data:")
    print(df)
    print()
    
    # Check price changes
    price_changes = df['close'].diff().abs()
    print("Price changes (absolute):")
    print(price_changes)
    print()
    
    # Test rollover detection
    roll_affected_bars = calculator._detect_contract_rolls(df)
    print(f"Rollover threshold: {calculator.roll_detection_threshold}")
    print(f"Roll affected bars: {roll_affected_bars}")
    print()
    
    # Check which bars are marked as affected
    for i, affected in enumerate(roll_affected_bars):
        status = "AFFECTED" if affected else "OK"
        print(f"Bar {i}: {status}")
    
    # The issue: if bar 0 is marked as affected, it will get label=0
    if roll_affected_bars[0]:
        print(f"\n🚨 BUG FOUND!")
        print(f"Bar 0 is incorrectly marked as rollover-affected!")
        print(f"This causes it to get label=0 regardless of actual outcome")
        
        # Check why it's marked as affected
        if len(df) >= 2:
            price_change = abs(df.iloc[1]['close'] - df.iloc[0]['close'])
            print(f"Price change bar 0->1: {price_change:.2f} points")
            print(f"Threshold: {calculator.roll_detection_threshold:.2f} points")
            if price_change > calculator.roll_detection_threshold:
                print(f"Price change > threshold - marked as rollover")
            else:
                print(f"Price change <= threshold - should NOT be marked as rollover")
    else:
        print(f"\n✅ Bar 0 is correctly NOT marked as rollover-affected")

def test_with_smaller_threshold():
    """Test with an even smaller rollover threshold"""
    
    print(f"\n🧪 TESTING WITH SMALLER THRESHOLD")
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
    
    # Test different thresholds
    thresholds = [10.0, 5.0, 2.0, 1.0, 0.5]
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        calculator = LabelCalculator(mode, roll_detection_threshold=threshold)
        
        roll_affected_bars = calculator._detect_contract_rolls(df)
        labels, _, _ = calculator.calculate_labels(df)
        
        print(f"  Roll affected: {roll_affected_bars}")
        print(f"  Labels: {labels}")
        print(f"  Bar 0 label: {labels[0]} ({'WIN' if labels[0] == 1 else 'LOSS'})")

def test_rollover_detection_logic():
    """Test the rollover detection logic directly"""
    
    print(f"\n🔍 TESTING ROLLOVER DETECTION LOGIC")
    print("=" * 60)
    
    # Create data with known price change
    data = {
        'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(5)],
        'open': [1280.00, 1280.00, 1277.00, 1277.25, 1277.50],
        'high': [1280.25, 1280.50, 1277.50, 1277.75, 1278.00],
        'low': [1279.75, 1276.00, 1276.50, 1277.00, 1277.25],
        'close': [1280.00, 1277.00, 1277.25, 1277.50, 1277.75],  # Changes: 0, -3.00, +0.25, +0.25, +0.25
        'volume': [100, 200, 150, 120, 110]
    }
    
    df = pd.DataFrame(data)
    
    # Manual calculation
    price_changes = df['close'].diff().abs()
    print("Manual price change calculation:")
    for i, change in enumerate(price_changes):
        print(f"  Bar {i}: {change:.2f} points")
    
    # Test with 2.0 threshold
    threshold = 2.0
    roll_events = price_changes > threshold
    print(f"\nRoll events (threshold {threshold}):")
    for i, is_roll in enumerate(roll_events):
        print(f"  Bar {i}: {'ROLL' if is_roll else 'OK'}")
    
    # The issue might be in how affected bars are calculated
    print(f"\nAffected bars calculation:")
    affected_bars = np.zeros(len(df), dtype=bool)
    
    for i in range(len(roll_events)):
        if roll_events.iloc[i]:
            print(f"  Roll event at bar {i}")
            # Mark current bar and next 5 bars as affected
            start_idx = max(0, i - 1)  # Include previous bar
            end_idx = min(len(df), i + 6)  # Include next 5 bars
            affected_bars[start_idx:end_idx] = True
            print(f"    Marking bars {start_idx} to {end_idx-1} as affected")
    
    print(f"\nFinal affected bars: {affected_bars}")

def main():
    """Main execution"""
    
    debug_rollover_detection()
    test_with_smaller_threshold()
    test_rollover_detection_logic()
    
    print(f"\n🎯 HYPOTHESIS:")
    print("The 3-point price drop (1280 -> 1277) is being detected")
    print("as a rollover event with the 2.0 threshold, causing bar 0")
    print("to be marked as affected and getting label=0.")
    print()
    print("🚀 SOLUTION:")
    print("Either increase the threshold back to 5.0+ points,")
    print("or fix the rollover detection logic to be more precise.")


if __name__ == "__main__":
    main()