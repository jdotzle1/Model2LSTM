#!/usr/bin/env python3
"""
Debug Data Validation Issue

This script investigates if the data validation is causing issues
or if there's a problem with the WeightedLabelingEngine.
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

from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig, TRADING_MODES, LabelCalculator

def debug_data_validation():
    """Debug the data validation issues"""
    
    print("🔍 DEBUGGING DATA VALIDATION")
    print("=" * 60)
    
    # Create the problematic steadily rising data
    data = {
        'timestamp': [datetime(2011, 6, 10, 9, 30, i) for i in range(5)],  # RTH time
        'open': [1300.00, 1300.25, 1300.50, 1300.75, 1301.00],
        'high': [1300.50, 1300.75, 1301.00, 1301.25, 1301.50],  # High >= Open
        'low': [1299.50, 1299.75, 1300.00, 1300.25, 1300.50],   # Low <= Open
        'close': [1300.25, 1300.50, 1300.75, 1301.00, 1301.25], # Close between High/Low
        'volume': [100, 200, 150, 180, 160]
    }
    
    df = pd.DataFrame(data)
    
    print("Test data (RTH time, valid OHLC):")
    print(df)
    print()
    
    # Test data validation
    try:
        config = LabelingConfig()
        engine = WeightedLabelingEngine(config)
        
        print("✅ WeightedLabelingEngine created successfully")
        
        # Try to process
        result = engine.process_dataframe(df)
        
        print(f"✅ Processing successful: {len(result)} rows, {len(result.columns)} columns")
        
        # Check results
        label_cols = [col for col in result.columns if col.startswith('label_')]
        print(f"\nResults:")
        
        for col in label_cols:
            win_rate = result[col].mean()
            wins = result[col].sum()
            total = len(result)
            print(f"  {col}: {win_rate:.1%} ({wins}/{total})")
        
        # Check if results are inverted
        short_cols = [col for col in label_cols if 'short' in col]
        long_cols = [col for col in label_cols if 'long' in col]
        
        avg_short_rate = np.mean([result[col].mean() for col in short_cols])
        avg_long_rate = np.mean([result[col].mean() for col in long_cols])
        
        print(f"\nAverage win rates:")
        print(f"  Short trades: {avg_short_rate:.1%}")
        print(f"  Long trades: {avg_long_rate:.1%}")
        
        # In rising market, longs should win more than shorts
        if avg_long_rate > avg_short_rate:
            print("✅ CORRECT: Longs outperform shorts in rising market")
        else:
            print("❌ BUG: Shorts outperform longs in rising market")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

def compare_engine_vs_direct():
    """Compare WeightedLabelingEngine vs direct LabelCalculator"""
    
    print(f"\n🔄 COMPARING ENGINE VS DIRECT CALCULATOR")
    print("=" * 60)
    
    # Simple 3-bar test
    data = {
        'timestamp': [datetime(2011, 6, 10, 9, 30, i) for i in range(3)],
        'open': [1300.00, 1300.00, 1300.50],
        'high': [1300.25, 1300.50, 1300.75],
        'low': [1299.75, 1299.50, 1300.25],
        'close': [1300.00, 1300.25, 1300.50],  # Rising
        'volume': [100, 200, 150]
    }
    
    df = pd.DataFrame(data)
    
    print("Test data:")
    print(df[['open', 'high', 'low', 'close']])
    print()
    
    # Test 1: Direct LabelCalculator (we know this works)
    print("🔧 DIRECT LABEL CALCULATOR:")
    short_mode = TRADING_MODES['normal_vol_short']
    short_calc = LabelCalculator(short_mode, roll_detection_threshold=50.0)
    
    labels, _, _ = short_calc.calculate_labels(df)
    print(f"  Short labels: {labels}")
    print(f"  Win rate: {labels.mean():.1%}")
    
    # Test 2: WeightedLabelingEngine
    print(f"\n🏭 WEIGHTED LABELING ENGINE:")
    try:
        config = LabelingConfig()
        engine = WeightedLabelingEngine(config)
        
        result = engine.process_dataframe(df)
        
        short_label_col = 'label_normal_vol_short'
        if short_label_col in result.columns:
            engine_labels = result[short_label_col].values
            print(f"  Short labels: {engine_labels}")
            print(f"  Win rate: {result[short_label_col].mean():.1%}")
            
            # Compare
            if np.array_equal(labels, engine_labels):
                print("✅ MATCH: Engine and direct calculator give same results")
            else:
                print("❌ MISMATCH: Engine and direct calculator differ!")
                print(f"    Direct: {labels}")
                print(f"    Engine: {engine_labels}")
        else:
            print(f"❌ Column {short_label_col} not found in engine results")
            
    except Exception as e:
        print(f"❌ Engine failed: {e}")

def main():
    """Main execution"""
    
    debug_data_validation()
    compare_engine_vs_direct()
    
    print(f"\n🎯 NEXT STEPS:")
    print("If the engine and direct calculator give different results,")
    print("then there's a bug in the WeightedLabelingEngine implementation.")
    print("If they match, then the issue might be elsewhere.")


if __name__ == "__main__":
    main()