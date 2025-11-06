#!/usr/bin/env python3
"""
Debug Import Path

This script checks which weighted_labeling module is actually being imported
and what the rollover threshold is set to.
"""

import sys
import os
from pathlib import Path

# Add project root to path (same as processing script)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_import_path():
    """Debug which weighted_labeling module is being imported"""
    
    print("🔍 DEBUGGING IMPORT PATH")
    print("=" * 60)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    print(f"\n📦 IMPORTING WEIGHTED_LABELING MODULE:")
    
    try:
        from src.data_pipeline.weighted_labeling import TRADING_MODES, LabelCalculator
        print("✅ Successfully imported from src.data_pipeline.weighted_labeling")
        
        # Check the module file location
        import src.data_pipeline.weighted_labeling as wl_module
        print(f"Module file: {wl_module.__file__}")
        
        # Check the threshold
        mode = TRADING_MODES['normal_vol_short']
        calculator = LabelCalculator(mode)
        print(f"Rollover threshold: {calculator.roll_detection_threshold}")
        
        # Check file modification time
        mod_time = os.path.getmtime(wl_module.__file__)
        from datetime import datetime
        print(f"File last modified: {datetime.fromtimestamp(mod_time)}")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    
    print(f"\n🔍 CHECKING ALL WEIGHTED_LABELING FILES:")
    
    # Check all weighted_labeling.py files
    files_to_check = [
        "src/data_pipeline/weighted_labeling.py",
        "deployment/ec2/ec2_deployment_package/project/project/data_pipeline/weighted_labeling.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            mod_time = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            print(f"  {file_path}:")
            print(f"    Size: {size:,} bytes")
            print(f"    Modified: {datetime.fromtimestamp(mod_time)}")
        else:
            print(f"  {file_path}: NOT FOUND")

def test_actual_threshold():
    """Test the actual threshold being used"""
    
    print(f"\n🧪 TESTING ACTUAL THRESHOLD:")
    
    try:
        from src.data_pipeline.weighted_labeling import TRADING_MODES, LabelCalculator
        
        # Create test data with known price change
        import pandas as pd
        from datetime import datetime
        
        data = {
            'timestamp': [datetime(2011, 6, 10, 10, 0, 0), datetime(2011, 6, 10, 10, 0, 1)],
            'open': [1300.00, 1300.00],
            'high': [1300.50, 1300.50],
            'low': [1299.50, 1299.50],
            'close': [1300.00, 1285.00],  # 15-point drop
            'volume': [100, 200]
        }
        
        df = pd.DataFrame(data)
        mode = TRADING_MODES['normal_vol_short']
        calculator = LabelCalculator(mode)
        
        print(f"Test data: 15-point price drop")
        print(f"Rollover threshold: {calculator.roll_detection_threshold}")
        
        # Check rollover detection
        roll_affected_bars = calculator._detect_contract_rolls(df)
        print(f"Roll affected bars: {roll_affected_bars}")
        
        if calculator.roll_detection_threshold == 20.0:
            if roll_affected_bars[0]:
                print("❌ 15-point move detected as rollover (threshold too low)")
            else:
                print("✅ 15-point move NOT detected as rollover (correct)")
        else:
            print(f"⚠️  Unexpected threshold: {calculator.roll_detection_threshold}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    """Main execution"""
    
    debug_import_path()
    test_actual_threshold()
    
    print(f"\n🎯 CONCLUSION:")
    print("If the threshold is 20.0 but 15-point moves are still detected as rollover,")
    print("then there might be a bug in the rollover detection logic itself.")
    print("If the threshold is not 20.0, then the wrong file is being imported.")


if __name__ == "__main__":
    main()