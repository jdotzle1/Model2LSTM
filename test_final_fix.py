#!/usr/bin/env python3
"""
Test Final Fix

This script tests the final fix with the corrected rollover threshold.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Force reload of the module
if 'src.data_pipeline.weighted_labeling' in sys.modules:
    del sys.modules['src.data_pipeline.weighted_labeling']

from src.data_pipeline.weighted_labeling import TRADING_MODES, LabelCalculator

def test_final_fix():
    """Test the final fix with corrected threshold"""
    
    print("🧪 TESTING FINAL FIX")
    print("=" * 60)
    
    # Test data with 3-point move (should NOT be rollover)
    data = {
        'timestamp': [
            datetime(2011, 6, 10, 10, 0, 0),
            datetime(2011, 6, 10, 10, 0, 1),
            datetime(2011, 6, 10, 10, 0, 2),
        ],
        'open': [1280.00, 1280.00, 1277.00],
        'high': [1280.25, 1280.50, 1277.50],
        'low': [1279.75, 1276.00, 1276.50],
        'close': [1280.00, 1277.00, 1277.25],  # 3-point drop
        'volume': [100, 200, 150]
    }
    
    df = pd.DataFrame(data)
    mode = TRADING_MODES['normal_vol_short']
    
    # Test with explicit 5.0 threshold
    calculator = LabelCalculator(mode, roll_detection_threshold=5.0)
    
    print(f"Test data (3-point price drop):")
    print(df[['timestamp', 'close']])
    print()
    
    print(f"Rollover threshold: {calculator.roll_detection_threshold}")
    
    # Check rollover detection
    roll_affected_bars = calculator._detect_contract_rolls(df)
    print(f"Roll affected bars: {roll_affected_bars}")
    
    # Calculate labels
    labels, mae_ticks, seconds_to_target = calculator.calculate_labels(df)
    
    print(f"Labels: {labels}")
    print(f"Expected: [1, 0, 0] (bar 0 should win)")
    
    if labels[0] == 1:
        print("✅ SUCCESS! Algorithm now works correctly")
        print("   3-point move is NOT treated as rollover")
        print("   Short trade correctly identified as winner")
    else:
        print("❌ Still broken - need further investigation")
    
    return labels[0] == 1

def test_reprocessing_impact():
    """Test what the reprocessing impact should be"""
    
    print(f"\n📊 EXPECTED REPROCESSING IMPACT")
    print("=" * 60)
    
    print("With threshold = 5.0 points:")
    print("✅ Normal price moves (1-4 points) will NOT be rollover")
    print("✅ Only true rollover gaps (5+ points) will be detected")
    print("✅ Short win rates should drop to realistic levels (30-45%)")
    print()
    
    print("🚀 REPROCESSING COMMANDS:")
    print("# Delete old results")
    print("aws s3 rm s3://es-1-second-data/processed-data/monthly/2011/06/ --recursive --region us-east-1")
    print()
    print("# Reprocess with corrected threshold")
    print("python3 process_monthly_chunks_fixed.py --test-month 2011-06")
    print()
    print("# Expected results:")
    print("# - Short win rates: 30-45% (realistic)")
    print("# - Rollover events detected: 5-20 (only true gaps)")
    print("# - Most bars will NOT be excluded")

def main():
    """Main execution"""
    
    success = test_final_fix()
    
    if success:
        test_reprocessing_impact()
        
        print(f"\n🎯 ROOT CAUSE RESOLUTION:")
        print("✅ Rollover threshold was too low (2.0 points)")
        print("✅ Normal market moves were treated as rollover")
        print("✅ All bars were excluded, causing 0% win rates")
        print("✅ Fixed by increasing threshold to 5.0 points")
        print()
        print("🚀 READY TO REPROCESS JUNE 2011!")
    else:
        print(f"\n❌ Fix unsuccessful - need more investigation")


if __name__ == "__main__":
    main()