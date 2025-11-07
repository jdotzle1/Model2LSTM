#!/usr/bin/env python3
"""
Debug Test Data Issue

This script investigates why the earlier test data produced
100% short wins and 0% long wins when the algorithm seems correct.
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

from src.data_pipeline.weighted_labeling import process_weighted_labeling

def analyze_problematic_test_data():
    """Analyze the test data that produced 100% short wins"""
    
    print("🔍 ANALYZING PROBLEMATIC TEST DATA")
    print("=" * 60)
    
    # Recreate the exact test data from debug_processing_environment.py
    data = {
        'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(10)],
        'open': [1300.00 + i*0.25 for i in range(10)],      # Steadily increasing
        'high': [1300.50 + i*0.25 for i in range(10)],     # Steadily increasing
        'low': [1299.50 + i*0.25 for i in range(10)],      # Steadily increasing
        'close': [1300.00 + i*0.25 for i in range(10)],    # Steadily increasing
        'volume': [100 + i*10 for i in range(10)]
    }
    
    df = pd.DataFrame(data)
    
    print("Problematic test data:")
    print(df[['timestamp', 'open', 'high', 'low', 'close']])
    print()
    
    print("Price analysis:")
    for i in range(len(df)):
        if i > 0:
            change = df.iloc[i]['close'] - df.iloc[i-1]['close']
            print(f"  Bar {i}: Close {df.iloc[i]['close']:.2f} (change: {change:+.2f})")
        else:
            print(f"  Bar {i}: Close {df.iloc[i]['close']:.2f}")
    
    print(f"\n🔍 ANALYSIS:")
    print("This data shows STEADILY RISING prices (+0.25 every bar)")
    print("In a rising market:")
    print("  - SHORT trades should LOSE (price going up)")
    print("  - LONG trades should WIN (price going up)")
    print()
    print("But the results showed:")
    print("  - SHORT trades: 100% wins ❌")
    print("  - LONG trades: 0% wins ❌")
    print()
    print("This suggests there IS a bug in the algorithm!")

def test_with_realistic_data():
    """Test with more realistic market data"""
    
    print(f"\n🧪 TESTING WITH REALISTIC DATA")
    print("=" * 60)
    
    # Create realistic data with mixed price movements
    data = {
        'timestamp': [datetime(2011, 6, 10, 9, 30, i) for i in range(20)],
        'open': [
            1300.00, 1300.25, 1299.75, 1300.50, 1301.00,  # Mixed movements
            1300.75, 1299.50, 1298.00, 1299.25, 1300.00,
            1301.50, 1302.00, 1301.25, 1300.75, 1299.00,
            1298.50, 1299.75, 1301.00, 1300.25, 1299.50
        ],
        'high': [o + 0.50 for o in [
            1300.00, 1300.25, 1299.75, 1300.50, 1301.00,
            1300.75, 1299.50, 1298.00, 1299.25, 1300.00,
            1301.50, 1302.00, 1301.25, 1300.75, 1299.00,
            1298.50, 1299.75, 1301.00, 1300.25, 1299.50
        ]],
        'low': [o - 0.50 for o in [
            1300.00, 1300.25, 1299.75, 1300.50, 1301.00,
            1300.75, 1299.50, 1298.00, 1299.25, 1300.00,
            1301.50, 1302.00, 1301.25, 1300.75, 1299.00,
            1298.50, 1299.75, 1301.00, 1300.25, 1299.50
        ]],
        'close': [
            1300.25, 1299.75, 1300.50, 1301.00, 1300.75,  # Mixed movements
            1299.50, 1298.00, 1299.25, 1300.00, 1301.50,
            1302.00, 1301.25, 1300.75, 1299.00, 1298.50,
            1299.75, 1301.00, 1300.25, 1299.50, 1300.00
        ],
        'volume': [100 + i*5 for i in range(20)]
    }
    
    df = pd.DataFrame(data)
    
    print("Realistic test data (first 10 bars):")
    for i in range(min(10, len(df))):
        if i > 0:
            change = df.iloc[i]['close'] - df.iloc[i-1]['close']
            print(f"  Bar {i}: Close {df.iloc[i]['close']:.2f} (change: {change:+.2f})")
        else:
            print(f"  Bar {i}: Close {df.iloc[i]['close']:.2f}")
    
    # Process with weighted labeling
    try:
        result = process_weighted_labeling(df)
        
        # Check results
        label_cols = [col for col in result.columns if col.startswith('label_')]
        print(f"\nResults with realistic data:")
        
        for col in label_cols:
            win_rate = result[col].mean()
            wins = result[col].sum()
            total = len(result)
            print(f"  {col}: {win_rate:.1%} ({wins}/{total})")
        
        # Check if results are more reasonable
        short_cols = [col for col in label_cols if 'short' in col]
        long_cols = [col for col in label_cols if 'long' in col]
        
        avg_short_rate = np.mean([result[col].mean() for col in short_cols])
        avg_long_rate = np.mean([result[col].mean() for col in long_cols])
        
        print(f"\nAverage win rates:")
        print(f"  Short trades: {avg_short_rate:.1%}")
        print(f"  Long trades: {avg_long_rate:.1%}")
        
        if 0.2 <= avg_short_rate <= 0.6 and 0.2 <= avg_long_rate <= 0.6:
            print("✅ REALISTIC: Both directions have reasonable win rates")
        else:
            print("❌ STILL UNREALISTIC: Win rates are extreme")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")

def main():
    """Main execution"""
    
    analyze_problematic_test_data()
    test_with_realistic_data()
    
    print(f"\n🎯 HYPOTHESIS:")
    print("The algorithm might be correct, but there could be an issue with:")
    print("1. How it handles monotonically increasing prices")
    print("2. Edge cases in very small datasets")
    print("3. RTH filtering or other preprocessing")
    print("4. The specific test data characteristics")


if __name__ == "__main__":
    main()