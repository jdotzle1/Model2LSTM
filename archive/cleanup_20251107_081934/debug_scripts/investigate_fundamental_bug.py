#!/usr/bin/env python3
"""
Investigate Fundamental Bug

Since the rollover fix didn't work, there must be a fundamental
issue with the labeling logic itself. This script investigates
the core algorithm more deeply.
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

def investigate_fundamental_issue():
    """Investigate the fundamental issue with labeling"""
    
    print("🚨 FUNDAMENTAL BUG INVESTIGATION")
    print("=" * 60)
    
    print("📊 CURRENT SITUATION:")
    print("   Even after rollover threshold fix (5.0 points):")
    print("   - Short win rates still 66-68%")
    print("   - This is impossible for 2:1 R/R strategy")
    print("   - Must be fundamental logic error")
    print()
    
    print("🔍 POSSIBLE ROOT CAUSES:")
    print("1. 🔄 LABELING LOGIC INVERTED")
    print("   - Winners marked as losers, losers as winners")
    print("   - Would explain high short rates")
    print()
    print("2. 📊 TARGET/STOP CALCULATION WRONG")
    print("   - Target too easy to hit")
    print("   - Stop too hard to hit")
    print()
    print("3. 🎯 HIT DETECTION INVERTED")
    print("   - Checking wrong conditions")
    print("   - Logic operators reversed")
    print()
    print("4. 📈 MARKET DATA ISSUE")
    print("   - Data quality problems")
    print("   - Artificial price patterns")
    print()
    print("5. ⏰ TIMING/INDEXING BUG")
    print("   - Looking at wrong bars")
    print("   - Off-by-one errors")

def test_simple_scenarios():
    """Test very simple scenarios to isolate the bug"""
    
    print(f"\n🧪 TESTING SIMPLE SCENARIOS")
    print("=" * 60)
    
    # Test 1: Obvious short winner
    print("TEST 1: Obvious Short Winner")
    data1 = create_obvious_short_winner()
    result1 = test_scenario(data1, "Should be WIN")
    
    # Test 2: Obvious short loser  
    print("\nTEST 2: Obvious Short Loser")
    data2 = create_obvious_short_loser()
    result2 = test_scenario(data2, "Should be LOSS")
    
    # Test 3: Marginal case
    print("\nTEST 3: Marginal Case")
    data3 = create_marginal_case()
    result3 = test_scenario(data3, "Should be WIN")
    
    # Analysis
    print(f"\n📊 RESULTS ANALYSIS:")
    print(f"Test 1 (obvious winner): {'✅ CORRECT' if result1 == 1 else '❌ WRONG'}")
    print(f"Test 2 (obvious loser): {'✅ CORRECT' if result2 == 0 else '❌ WRONG'}")
    print(f"Test 3 (marginal winner): {'✅ CORRECT' if result3 == 1 else '❌ WRONG'}")
    
    if result1 == 0 and result2 == 1:
        print(f"\n🚨 LOGIC IS COMPLETELY INVERTED!")
        print(f"   Winners are marked as losers")
        print(f"   Losers are marked as winners")
    elif result1 == 0 and result2 == 0:
        print(f"\n🚨 ALL TRADES MARKED AS LOSERS!")
        print(f"   Something is preventing wins")
    elif result1 == 1 and result2 == 1:
        print(f"\n🚨 ALL TRADES MARKED AS WINNERS!")
        print(f"   Something is preventing losses")

def create_obvious_short_winner():
    """Create data where short trade obviously wins"""
    return {
        'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(3)],
        'open': [1300.00, 1300.00, 1290.00],    # Entry at 1300
        'high': [1300.50, 1300.25, 1290.50],   # Doesn't hit stop (1308)
        'low': [1299.50, 1284.00, 1289.50],    # Hits target (1284) easily
        'close': [1300.00, 1285.00, 1290.00],
        'volume': [100, 200, 150]
    }

def create_obvious_short_loser():
    """Create data where short trade obviously loses"""
    return {
        'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(3)],
        'open': [1300.00, 1300.00, 1310.00],    # Entry at 1300
        'high': [1300.50, 1310.00, 1310.50],   # Hits stop (1308) easily
        'low': [1299.50, 1299.00, 1309.50],    # Doesn't hit target (1284)
        'close': [1300.00, 1309.00, 1310.00],
        'volume': [100, 200, 150]
    }

def create_marginal_case():
    """Create marginal case that should still win"""
    return {
        'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(3)],
        'open': [1300.00, 1300.00, 1285.00],    # Entry at 1300
        'high': [1300.50, 1302.00, 1285.50],   # Doesn't quite hit stop (1308)
        'low': [1299.50, 1284.00, 1284.50],    # Just hits target (1284)
        'close': [1300.00, 1285.00, 1285.00],
        'volume': [100, 200, 150]
    }

def test_scenario(data_dict, expected):
    """Test a single scenario"""
    
    df = pd.DataFrame(data_dict)
    mode = TRADING_MODES['normal_vol_short']  # 8 tick stop, 16 tick target
    calculator = LabelCalculator(mode, roll_detection_threshold=10.0)  # High threshold
    
    # Calculate expected values
    entry_price = data_dict['open'][1]  # Entry bar open
    target_price = entry_price - (16 * 0.25)  # 1300 - 4 = 1296
    stop_price = entry_price + (8 * 0.25)     # 1300 + 2 = 1302
    
    print(f"  Entry: {entry_price:.2f}")
    print(f"  Target: {target_price:.2f}, Stop: {stop_price:.2f}")
    print(f"  Entry bar: H={data_dict['high'][1]:.2f}, L={data_dict['low'][1]:.2f}")
    
    # Manual check
    target_hit = data_dict['low'][1] <= target_price
    stop_hit = data_dict['high'][1] >= stop_price
    print(f"  Target hit: {target_hit}, Stop hit: {stop_hit}")
    
    # Algorithm result
    labels, _, _ = calculator.calculate_labels(df)
    result = labels[0]
    
    print(f"  Expected: {expected}")
    print(f"  Algorithm result: {'WIN' if result == 1 else 'LOSS'}")
    
    return result

def investigate_specific_bug_types():
    """Investigate specific types of bugs"""
    
    print(f"\n🔍 SPECIFIC BUG TYPE INVESTIGATION")
    print("=" * 60)
    
    print("🧪 Testing if target/stop calculations are swapped:")
    
    # Create test where we know exact values
    mode = TRADING_MODES['normal_vol_short']
    calculator = LabelCalculator(mode)
    
    entry_price = 1300.00
    target_price, stop_price = calculator._calculate_target_stop_prices(entry_price)
    
    print(f"Entry: {entry_price:.2f}")
    print(f"Calculated target: {target_price:.2f}")
    print(f"Calculated stop: {stop_price:.2f}")
    print(f"Expected target: {entry_price - 16*0.25:.2f} (should go DOWN)")
    print(f"Expected stop: {entry_price + 8*0.25:.2f} (should go UP)")
    
    if target_price > entry_price:
        print("🚨 BUG: Target is ABOVE entry (should be below for short)")
    if stop_price < entry_price:
        print("🚨 BUG: Stop is BELOW entry (should be above for short)")
    
    print(f"\n🧪 Testing hit detection logic:")
    
    # Test hit detection
    bar_high = 1302.00  # Should hit stop (1302)
    bar_low = 1296.00   # Should hit target (1296)
    
    target_hit, stop_hit = calculator._check_price_hits(
        bar_high, bar_low, target_price, stop_price
    )
    
    print(f"Bar: H={bar_high:.2f}, L={bar_low:.2f}")
    print(f"Target hit: {target_hit} (should be True)")
    print(f"Stop hit: {stop_hit} (should be True)")
    
    if not target_hit:
        print("🚨 BUG: Target hit detection wrong")
    if not stop_hit:
        print("🚨 BUG: Stop hit detection wrong")

def main():
    """Main execution"""
    
    investigate_fundamental_issue()
    test_simple_scenarios()
    investigate_specific_bug_types()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. If simple tests show inverted logic → Fix the inversion")
    print("2. If target/stop calculations wrong → Fix the math")
    print("3. If hit detection wrong → Fix the comparisons")
    print("4. If all tests pass → Issue might be in real data")
    print()
    print("🚨 The 66% short win rates are definitely wrong!")
    print("   Need to find and fix the fundamental bug.")


if __name__ == "__main__":
    main()