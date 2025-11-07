#!/usr/bin/env python3
"""
Confirm Both Fixes Are in Place

This script confirms that both the rollover threshold fix and 
the correct short trade labeling logic are properly implemented.
"""

import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.weighted_labeling import TRADING_MODES, LabelCalculator

def confirm_rollover_fix():
    """Confirm rollover threshold is now 2.0 points"""
    
    print("🔧 ROLLOVER THRESHOLD FIX")
    print("-" * 40)
    
    # Create a calculator and check the threshold
    mode = TRADING_MODES['normal_vol_short']
    calculator = LabelCalculator(mode)
    
    threshold = calculator.roll_detection_threshold
    print(f"Current rollover threshold: {threshold} points")
    
    if threshold == 2.0:
        print("✅ FIXED: Threshold is now 2.0 points (was 20.0)")
        print("   This will properly detect ES rollover gaps")
        return True
    else:
        print(f"❌ NOT FIXED: Threshold is still {threshold} points")
        print("   Should be 2.0 points for ES data")
        return False

def confirm_short_labeling_logic():
    """Confirm short trade labeling logic is correct (not inverted)"""
    
    print("\n🎯 SHORT TRADE LABELING LOGIC")
    print("-" * 40)
    
    mode = TRADING_MODES['normal_vol_short']
    calculator = LabelCalculator(mode)
    
    # Test the logic with a sample entry
    entry_price = 4750.00
    target_price, stop_price = calculator._calculate_target_stop_prices(entry_price)
    
    print(f"Short trade example (Normal Vol):")
    print(f"  Entry price: {entry_price:.2f}")
    print(f"  Target price: {target_price:.2f} (entry - {mode.target_ticks} ticks)")
    print(f"  Stop price: {stop_price:.2f} (entry + {mode.stop_ticks} ticks)")
    
    # Verify the logic
    expected_target = entry_price - (mode.target_ticks * 0.25)
    expected_stop = entry_price + (mode.stop_ticks * 0.25)
    
    target_correct = abs(target_price - expected_target) < 0.01
    stop_correct = abs(stop_price - expected_stop) < 0.01
    
    if target_correct and stop_correct:
        print("✅ CORRECT: Short target goes DOWN, stop goes UP")
        
        # Test hit detection
        print(f"\n  Hit detection logic:")
        
        # Scenario 1: Target hit (price goes down)
        bar_high, bar_low = 4751.00, 4746.00  # Price went down to target
        target_hit, stop_hit = calculator._check_price_hits(bar_high, bar_low, target_price, stop_price)
        print(f"    Bar H={bar_high:.2f}, L={bar_low:.2f}")
        print(f"    Target hit: {target_hit} (low {bar_low:.2f} <= target {target_price:.2f})")
        print(f"    Stop hit: {stop_hit} (high {bar_high:.2f} >= stop {stop_price:.2f})")
        
        if target_hit and not stop_hit:
            print("    ✅ CORRECT: Target hit, stop not hit = WIN")
            return True
        else:
            print("    ❌ INCORRECT: Hit detection logic wrong")
            return False
    else:
        print("❌ INCORRECT: Target/stop calculation wrong")
        print(f"   Expected target: {expected_target:.2f}")
        print(f"   Expected stop: {expected_stop:.2f}")
        return False

def confirm_both_fixes():
    """Confirm both fixes are in place"""
    
    print("🔍 CONFIRMING ALL FIXES")
    print("=" * 60)
    
    rollover_fixed = confirm_rollover_fix()
    labeling_correct = confirm_short_labeling_logic()
    
    print("\n" + "=" * 60)
    print("📋 FINAL STATUS")
    
    if rollover_fixed and labeling_correct:
        print("✅ ALL FIXES CONFIRMED!")
        print("   1. ✅ Rollover threshold: 20.0 → 2.0 points")
        print("   2. ✅ Short labeling logic: Correct (not inverted)")
        print()
        print("🚀 READY TO REPROCESS:")
        print("   The high short win rates should now be resolved")
        print("   Contract rollover gaps will be properly detected")
        print("   Short trades will be labeled correctly")
        
        return True
    else:
        print("❌ FIXES NOT COMPLETE:")
        if not rollover_fixed:
            print("   - Rollover threshold still needs fixing")
        if not labeling_correct:
            print("   - Short labeling logic still incorrect")
        
        return False

def main():
    """Main execution"""
    success = confirm_both_fixes()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Delete old June 2011 results:")
        print("   aws s3 rm s3://es-1-second-data/processed-data/monthly/2011/06/ --recursive --region us-east-1")
        print()
        print("2. Reprocess with fixes:")
        print("   python3 process_monthly_chunks_fixed.py --test-month 2011-06")
        print()
        print("3. Validate results:")
        print("   Short win rates should be 30-45% (realistic)")
        print("   Rollover events should be detected")
    else:
        print("\n⚠️  Please fix the remaining issues before reprocessing")

if __name__ == "__main__":
    main()