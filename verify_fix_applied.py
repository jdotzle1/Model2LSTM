#!/usr/bin/env python3
"""
Verify Fix Applied

This script verifies that our fix is actually being used by the processing system.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_fix_applied():
    """Verify that the fix is actually applied"""
    
    print("🔍 VERIFYING FIX IS APPLIED")
    print("=" * 60)
    
    # Force reload of the module
    if 'src.data_pipeline.weighted_labeling' in sys.modules:
        del sys.modules['src.data_pipeline.weighted_labeling']
    
    try:
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig
        
        # Create test data that should show the fix
        data = {
            'timestamp': [datetime(2011, 6, 10, 9, 30, i) for i in range(3)],
            'open': [1300.00, 1300.00, 1300.50],
            'high': [1300.25, 1300.50, 1300.75],
            'low': [1299.75, 1299.50, 1300.25],
            'close': [1300.00, 1300.25, 1300.50],  # Rising prices
            'volume': [100, 200, 150]
        }
        
        df = pd.DataFrame(data)
        
        print("Test data (rising prices):")
        print(df[['open', 'high', 'low', 'close']])
        print()
        
        # Test with WeightedLabelingEngine
        config = LabelingConfig()
        engine = WeightedLabelingEngine(config)
        
        result = engine.process_dataframe(df)
        
        # Check short labels
        short_cols = [col for col in result.columns if col.startswith('label_') and 'short' in col]
        
        print("Short trade results in rising market:")
        for col in short_cols:
            win_rate = result[col].mean()
            wins = result[col].sum()
            total = len(result)
            print(f"  {col}: {win_rate:.1%} ({wins}/{total})")
        
        # Check if fix is applied
        avg_short_rate = sum(result[col].mean() for col in short_cols) / len(short_cols)
        
        print(f"\nAverage short win rate: {avg_short_rate:.1%}")
        
        if avg_short_rate == 0.0:
            print("✅ FIX APPLIED: Short trades correctly lose in rising market")
            return True
        elif avg_short_rate == 1.0:
            print("❌ FIX NOT APPLIED: Short trades still inverted (100% wins)")
            return False
        else:
            print(f"⚠️  UNEXPECTED: Short win rate is {avg_short_rate:.1%}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing fix: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_file_modification():
    """Check when the weighted_labeling.py file was last modified"""
    
    print(f"\n📅 FILE MODIFICATION CHECK")
    print("-" * 40)
    
    file_path = "src/data_pipeline/weighted_labeling.py"
    
    if os.path.exists(file_path):
        mod_time = os.path.getmtime(file_path)
        mod_datetime = datetime.fromtimestamp(mod_time)
        
        print(f"File: {file_path}")
        print(f"Last modified: {mod_datetime}")
        
        # Check if modified recently (within last hour)
        now = datetime.now()
        time_diff = now - mod_datetime
        
        if time_diff.total_seconds() < 3600:  # Less than 1 hour
            print("✅ File was modified recently")
        else:
            print(f"⚠️  File was modified {time_diff} ago")
    else:
        print(f"❌ File not found: {file_path}")

def check_for_inversion_code():
    """Check if the inversion code is still in the file"""
    
    print(f"\n🔍 CHECKING FOR INVERSION CODE")
    print("-" * 40)
    
    file_path = "src/data_pipeline/weighted_labeling.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for the inversion code
        if "1 - labels" in content:
            print("❌ INVERSION CODE STILL PRESENT!")
            
            # Find the lines
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "1 - labels" in line:
                    print(f"  Line {i+1}: {line.strip()}")
            return False
        else:
            print("✅ Inversion code removed")
            return True
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    """Main execution"""
    
    fix_applied = verify_fix_applied()
    check_file_modification()
    inversion_removed = check_for_inversion_code()
    
    print(f"\n🎯 SUMMARY:")
    print(f"Fix applied in runtime: {'✅ YES' if fix_applied else '❌ NO'}")
    print(f"Inversion code removed: {'✅ YES' if inversion_removed else '❌ NO'}")
    
    if fix_applied and inversion_removed:
        print("\n✅ FIX IS PROPERLY APPLIED")
        print("The processing system should now produce correct results.")
    else:
        print("\n❌ FIX IS NOT WORKING")
        print("There may be caching issues or the wrong file is being used.")


if __name__ == "__main__":
    main()