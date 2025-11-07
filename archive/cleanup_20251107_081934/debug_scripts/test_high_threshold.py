#!/usr/bin/env python3
"""
Test High Threshold

Test with a very high rollover threshold to eliminate false rollover detection.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Force reload
if 'src.data_pipeline.weighted_labeling' in sys.modules:
    del sys.modules['src.data_pipeline.weighted_labeling']

from src.data_pipeline.weighted_labeling import TRADING_MODES, LabelCalculator

def test_with_high_threshold():
    """Test with very high rollover threshold"""
    
    print("🧪 TESTING WITH HIGH ROLLOVER THRESHOLD")
    print("=" * 60)
    
    # Test data with large price moves (normal for 2011)
    data = {
        'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(3)],
        'open': [1300.00, 1300.00, 1290.00],    # Entry at 1300
        'high': [1300.50, 1300.25, 1290.50],   # Doesn't hit stop (1302)
        'low': [1299.50, 1284.00, 1289.50],    # Hits target (1296) easily
        'close': [1300.00, 1285.00, 1290.00],  # 15-point drop (normal in 2011)
        'volume': [100, 200, 150]
    }
    
    df = pd.DataFrame(data)
    mode = TRADING_MODES['normal_vol_short']
    
    # Test different thresholds
    thresholds = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold} points")
        calculator = LabelCalculator(mode, roll_detection_threshold=threshold)
        
        # Check rollover detection
        roll_affected_bars = calculator._detect_contract_rolls(df)
        print(f"  Roll affected bars: {roll_affected_bars}")
        
        # Calculate labels
        labels, _, _ = calculator.calculate_labels(df)
        print(f"  Labels: {labels}")
        print(f"  Bar 0 result: {'WIN' if labels[0] == 1 else 'LOSS'}")
        
        if not roll_affected_bars[0] and labels[0] == 1:
            print(f"  ✅ SUCCESS! Threshold {threshold} works correctly")
            return threshold
        elif roll_affected_bars[0]:
            print(f"  ❌ Still detecting rollover (15-point move)")
        else:
            print(f"  ❌ Not detecting rollover but still wrong result")
    
    return None

def test_realistic_2011_moves():
    """Test with realistic 2011 market moves"""
    
    print(f"\n📊 TESTING REALISTIC 2011 MARKET MOVES")
    print("=" * 60)
    
    print("2011 was extremely volatile. Common intraday moves:")
    print("- 5-15 points: Normal")
    print("- 15-30 points: Large but common")
    print("- 30-50 points: Very large but possible")
    print("- 50+ points: Likely rollover or major news")
    print()
    
    # Test with 100-point threshold (should only catch true rollovers)
    data = {
        'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(5)],
        'open': [1300.00, 1300.00, 1285.00, 1270.00, 1275.00],
        'high': [1300.50, 1300.25, 1285.50, 1270.50, 1275.50],
        'low': [1299.50, 1284.00, 1269.50, 1269.50, 1274.50],
        'close': [1300.00, 1285.00, 1270.00, 1275.00, 1275.25],  # Large moves but normal for 2011
        'volume': [100, 200, 150, 180, 160]
    }
    
    df = pd.DataFrame(data)
    mode = TRADING_MODES['normal_vol_short']
    calculator = LabelCalculator(mode, roll_detection_threshold=100.0)
    
    print("Test data with large moves:")
    for i, row in df.iterrows():
        if i > 0:
            change = row['close'] - df.iloc[i-1]['close']
            print(f"  Bar {i}: Close {row['close']:.2f} (change: {change:+.2f})")
        else:
            print(f"  Bar {i}: Close {row['close']:.2f}")
    
    # Check rollover detection
    roll_affected_bars = calculator._detect_contract_rolls(df)
    print(f"\nRollover affected (100-point threshold): {roll_affected_bars}")
    
    # Calculate labels
    labels, _, _ = calculator.calculate_labels(df)
    print(f"Labels: {labels}")
    
    # Count wins
    wins = labels.sum()
    total = len(labels) - 1  # Last bar can't be entry
    win_rate = wins / total if total > 0 else 0
    
    print(f"Win rate: {win_rate:.1%} ({wins}/{total})")
    
    if win_rate > 0 and win_rate < 0.8:
        print("✅ Realistic win rate achieved!")
    else:
        print("❌ Still unrealistic win rate")

def main():
    """Main execution"""
    
    working_threshold = test_with_high_threshold()
    
    if working_threshold:
        print(f"\n🎯 SOLUTION FOUND!")
        print(f"Working threshold: {working_threshold} points")
        print(f"This eliminates false rollover detection")
        
        test_realistic_2011_moves()
        
        print(f"\n🚀 FINAL FIX:")
        print(f"Set rollover threshold to {working_threshold}+ points")
        print(f"This allows normal 2011 volatility without false rollover detection")
    else:
        print(f"\n❌ No working threshold found")
        print(f"Need to investigate other causes")


if __name__ == "__main__":
    main()