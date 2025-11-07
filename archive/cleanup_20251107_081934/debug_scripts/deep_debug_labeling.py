#!/usr/bin/env python3
"""
Deep Debug Labeling Logic

This script performs a very detailed investigation of the labeling logic
by creating synthetic data and stepping through the exact algorithm.
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

def create_detailed_test_scenarios():
    """Create very specific test scenarios to debug the logic"""
    
    print("🔍 DETAILED LABELING LOGIC DEBUG")
    print("=" * 60)
    
    # Create test data with known outcomes
    base_time = datetime(2011, 6, 10, 10, 0, 0)
    scenarios = []
    
    # Scenario 1: Clear short winner - price drops significantly
    scenarios.append({
        'name': 'Clear Short Winner',
        'bars': [
            # Signal bar
            {'timestamp': base_time, 'open': 1280.00, 'high': 1280.25, 'low': 1279.75, 'close': 1280.00, 'volume': 100},
            # Entry bar (next bar open = entry price)
            {'timestamp': base_time + timedelta(seconds=1), 'open': 1280.00, 'high': 1280.00, 'low': 1276.00, 'close': 1276.50, 'volume': 200},
            # Follow-up bars
            {'timestamp': base_time + timedelta(seconds=2), 'open': 1276.50, 'high': 1277.00, 'low': 1276.00, 'close': 1276.25, 'volume': 150},
        ],
        'expected_result': 'WIN',
        'explanation': 'Entry 1280.00, Target 1276.00 (normal vol), price hit 1276.00'
    })
    
    # Scenario 2: Clear short loser - price rises to stop
    scenarios.append({
        'name': 'Clear Short Loser',
        'bars': [
            # Signal bar
            {'timestamp': base_time + timedelta(minutes=1), 'open': 1280.00, 'high': 1280.25, 'low': 1279.75, 'close': 1280.00, 'volume': 100},
            # Entry bar - price goes up to stop
            {'timestamp': base_time + timedelta(minutes=1, seconds=1), 'open': 1280.00, 'high': 1282.25, 'low': 1279.75, 'close': 1281.50, 'volume': 200},
            # Follow-up bars
            {'timestamp': base_time + timedelta(minutes=1, seconds=2), 'open': 1281.50, 'high': 1282.00, 'low': 1281.00, 'close': 1281.25, 'volume': 150},
        ],
        'expected_result': 'LOSS',
        'explanation': 'Entry 1280.00, Stop 1282.00 (normal vol), price hit 1282.25'
    })
    
    # Scenario 3: Marginal case - price barely hits target
    scenarios.append({
        'name': 'Marginal Short Winner',
        'bars': [
            # Signal bar
            {'timestamp': base_time + timedelta(minutes=2), 'open': 1280.00, 'high': 1280.25, 'low': 1279.75, 'close': 1280.00, 'volume': 100},
            # Entry bar - price barely hits target
            {'timestamp': base_time + timedelta(minutes=2, seconds=1), 'open': 1280.00, 'high': 1280.50, 'low': 1276.00, 'close': 1277.00, 'volume': 200},
            # Follow-up bars
            {'timestamp': base_time + timedelta(minutes=2, seconds=2), 'open': 1277.00, 'high': 1277.50, 'low': 1276.50, 'close': 1277.25, 'volume': 150},
        ],
        'expected_result': 'WIN',
        'explanation': 'Entry 1280.00, Target 1276.00, price hit exactly 1276.00'
    })
    
    return scenarios

def test_labeling_step_by_step():
    """Test labeling logic step by step with detailed output"""
    
    scenarios = create_detailed_test_scenarios()
    mode = TRADING_MODES['normal_vol_short']  # 8 tick stop, 16 tick target
    
    print(f"Testing mode: {mode.name}")
    print(f"Stop: {mode.stop_ticks} ticks ({mode.stop_ticks * 0.25:.2f} points)")
    print(f"Target: {mode.target_ticks} ticks ({mode.target_ticks * 0.25:.2f} points)")
    print()
    
    for scenario in scenarios:
        print(f"📊 SCENARIO: {scenario['name']}")
        print(f"   Expected: {scenario['expected_result']}")
        print(f"   Logic: {scenario['explanation']}")
        
        # Create DataFrame from scenario
        df = pd.DataFrame(scenario['bars'])
        
        # Create calculator
        calculator = LabelCalculator(mode, roll_detection_threshold=2.0)
        
        # Calculate labels
        labels, mae_ticks, seconds_to_target = calculator.calculate_labels(df)
        
        # Analyze results
        print(f"   Results:")
        for i, label in enumerate(labels):
            if i < len(df) - 1:  # Skip last bar (can't be entry)
                entry_bar = df.iloc[i + 1] if i + 1 < len(df) else None
                if entry_bar is not None:
                    entry_price = entry_bar['open']
                    target_price = entry_price - (mode.target_ticks * 0.25)
                    stop_price = entry_price + (mode.stop_ticks * 0.25)
                    
                    print(f"     Bar {i}: Label={label}")
                    print(f"       Entry: {entry_price:.2f}")
                    print(f"       Target: {target_price:.2f}, Stop: {stop_price:.2f}")
                    print(f"       Bar range: H={entry_bar['high']:.2f}, L={entry_bar['low']:.2f}")
                    
                    # Check hits manually
                    target_hit = entry_bar['low'] <= target_price
                    stop_hit = entry_bar['high'] >= stop_price
                    print(f"       Target hit: {target_hit}, Stop hit: {stop_hit}")
                    
                    if label == 1 and scenario['expected_result'] == 'WIN':
                        print(f"       ✅ CORRECT")
                    elif label == 0 and scenario['expected_result'] == 'LOSS':
                        print(f"       ✅ CORRECT")
                    else:
                        print(f"       ❌ INCORRECT!")
        print()

def investigate_potential_bugs():
    """Investigate potential bugs in the implementation"""
    
    print("🐛 POTENTIAL BUG INVESTIGATION")
    print("=" * 60)
    
    bugs_to_check = [
        "1. OFF-BY-ONE ERROR in bar indexing",
        "   - Signal bar vs entry bar confusion",
        "   - Looking at wrong bar for entry price",
        "",
        "2. TICK SIZE CALCULATION ERROR", 
        "   - Using wrong tick size (not 0.25)",
        "   - Rounding errors in price calculations",
        "",
        "3. HIT DETECTION LOGIC ERROR",
        "   - Wrong comparison operators",
        "   - Floating point precision issues",
        "",
        "4. TIMEOUT LOGIC ERROR",
        "   - Looking too far forward",
        "   - Including too many bars in search",
        "",
        "5. ROLLOVER DETECTION STILL BROKEN",
        "   - Detection not actually excluding bars",
        "   - Wrong bars being marked as affected"
    ]
    
    for bug in bugs_to_check:
        print(bug)
    
    print("\n🔍 SPECIFIC CHECKS TO PERFORM:")
    print("1. Verify entry price calculation (next bar open)")
    print("2. Verify target/stop price math")
    print("3. Verify hit detection comparisons")
    print("4. Check if rollover bars are actually excluded")
    print("5. Manual calculation of 10 sample trades")

def create_minimal_reproduction():
    """Create minimal reproduction of the issue"""
    
    print(f"\n🧪 MINIMAL REPRODUCTION")
    print("=" * 60)
    
    # Single bar test
    print("Creating single-bar test case:")
    
    # Create minimal data
    data = {
        'timestamp': [datetime(2011, 6, 10, 10, 0, 0), datetime(2011, 6, 10, 10, 0, 1)],
        'open': [1280.00, 1280.00],  # Entry at 1280.00
        'high': [1280.25, 1280.50],  # High doesn't hit stop (1282.00)
        'low': [1279.75, 1276.00],   # Low hits target (1276.00)
        'close': [1280.00, 1277.00],
        'volume': [100, 200]
    }
    
    df = pd.DataFrame(data)
    mode = TRADING_MODES['normal_vol_short']
    calculator = LabelCalculator(mode)
    
    print(f"Data:")
    print(df)
    print()
    
    labels, mae_ticks, seconds_to_target = calculator.calculate_labels(df)
    
    print(f"Results:")
    print(f"Labels: {labels}")
    print(f"Expected: [1, 0] (first bar wins, second bar can't be entry)")
    print()
    
    # Manual calculation
    entry_price = 1280.00
    target_price = entry_price - (16 * 0.25)  # 1276.00
    stop_price = entry_price + (8 * 0.25)     # 1282.00
    
    print(f"Manual calculation:")
    print(f"Entry: {entry_price:.2f}")
    print(f"Target: {target_price:.2f}")
    print(f"Stop: {stop_price:.2f}")
    print(f"Bar 1 range: H={data['high'][1]:.2f}, L={data['low'][1]:.2f}")
    print(f"Target hit: {data['low'][1] <= target_price} ({data['low'][1]:.2f} <= {target_price:.2f})")
    print(f"Stop hit: {data['high'][1] >= stop_price} ({data['high'][1]:.2f} >= {stop_price:.2f})")
    print()
    
    if labels[0] == 1:
        print("✅ Algorithm correctly identified winner")
    else:
        print("❌ Algorithm failed - this should be a winner!")

def main():
    """Main execution"""
    
    test_labeling_step_by_step()
    investigate_potential_bugs()
    create_minimal_reproduction()
    
    print(f"\n🚨 CRITICAL NEXT STEPS:")
    print("1. Run the minimal reproduction above")
    print("2. If it fails, there's a fundamental bug in the algorithm")
    print("3. If it works, the issue might be in the data or rollover detection")
    print("4. Consider that 66% win rates might actually be correct for 2011")


if __name__ == "__main__":
    main()