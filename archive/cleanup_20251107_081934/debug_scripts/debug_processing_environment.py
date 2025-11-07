#!/usr/bin/env python3
"""
Debug Processing Environment

This script replicates the exact environment and imports used by
the processing script to see what's happening.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Replicate the exact setup from process_monthly_chunks_fixed.py
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_processing_environment():
    """Debug the processing environment"""
    
    print("🔍 DEBUGGING PROCESSING ENVIRONMENT")
    print("=" * 60)
    
    print("Environment setup (same as process_monthly_chunks_fixed.py):")
    print(f"Project root: {project_root}")
    print(f"Python path[0]: {sys.path[0]}")
    
    # Replicate the exact imports from the processing script
    print(f"\n📦 REPLICATING PROCESSING SCRIPT IMPORTS:")
    
    try:
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, process_weighted_labeling
        print("✅ WeightedLabelingEngine imported successfully")
        
        from src.data_pipeline.weighted_labeling import LabelingConfig
        print("✅ LabelingConfig imported successfully")
        
        # Test the actual processing function
        print(f"\n🧪 TESTING PROCESS_WEIGHTED_LABELING FUNCTION:")
        
        # Create test data
        data = {
            'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(10)],
            'open': [1300.00 + i*0.25 for i in range(10)],
            'high': [1300.50 + i*0.25 for i in range(10)],
            'low': [1299.50 + i*0.25 for i in range(10)],
            'close': [1300.00 + i*0.25 for i in range(10)],
            'volume': [100 + i*10 for i in range(10)]
        }
        
        df = pd.DataFrame(data)
        print(f"Test data: {len(df)} rows")
        
        # Test the function
        result = process_weighted_labeling(df)
        print(f"Result: {len(result)} rows, {len(result.columns)} columns")
        
        # Check for label columns
        label_cols = [col for col in result.columns if col.startswith('label_')]
        print(f"Label columns: {label_cols}")
        
        if label_cols:
            for col in label_cols:
                win_rate = result[col].mean()
                print(f"  {col}: {win_rate:.1%}")
        
    except Exception as e:
        print(f"❌ Import/test failed: {e}")
        import traceback
        traceback.print_exc()

def test_weighted_labeling_engine():
    """Test the WeightedLabelingEngine directly"""
    
    print(f"\n🧪 TESTING WEIGHTED_LABELING_ENGINE:")
    
    try:
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig
        
        # Create test data with large price moves
        data = {
            'timestamp': [datetime(2011, 6, 10, 10, 0, i) for i in range(5)],
            'open': [1300.00, 1300.00, 1285.00, 1270.00, 1275.00],
            'high': [1300.50, 1300.25, 1285.50, 1270.50, 1275.50],
            'low': [1299.50, 1284.00, 1269.50, 1269.50, 1274.50],
            'close': [1300.00, 1285.00, 1270.00, 1275.00, 1275.25],  # Large moves
            'volume': [100, 200, 150, 180, 160]
        }
        
        df = pd.DataFrame(data)
        print(f"Test data with large moves:")
        for i, row in df.iterrows():
            if i > 0:
                change = row['close'] - df.iloc[i-1]['close']
                print(f"  Bar {i}: Close {row['close']:.2f} (change: {change:+.2f})")
            else:
                print(f"  Bar {i}: Close {row['close']:.2f}")
        
        # Create engine
        config = LabelingConfig()
        engine = WeightedLabelingEngine(config)
        
        # Process
        result = engine.process_dataframe(df)
        
        # Check results
        label_cols = [col for col in result.columns if col.startswith('label_')]
        print(f"\nResults:")
        for col in label_cols:
            win_rate = result[col].mean()
            wins = result[col].sum()
            total = len(result)
            print(f"  {col}: {win_rate:.1%} ({wins}/{total})")
        
        # Check if any bars were marked as rollover-affected
        print(f"\nChecking rollover detection:")
        # We can't directly access the rollover detection from here,
        # but we can infer from the results
        
    except Exception as e:
        print(f"❌ WeightedLabelingEngine test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution"""
    
    debug_processing_environment()
    test_weighted_labeling_engine()
    
    print(f"\n🎯 KEY QUESTIONS:")
    print("1. Are the win rates from this test realistic?")
    print("2. If yes, why are the S3 results still showing 66%?")
    print("3. If no, what's different about the processing environment?")


if __name__ == "__main__":
    main()