#!/usr/bin/env python3
"""
Debug Rollover Detection

This script investigates why the rollover detection fix didn't resolve
the high short win rates. It checks if rollover events are actually
being detected and excluded.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.weighted_labeling import TRADING_MODES, LabelCalculator

def analyze_rollover_detection_failure():
    """Analyze why rollover detection didn't fix the win rates"""
    
    print("🚨 ROLLOVER DETECTION FAILURE ANALYSIS")
    print("=" * 60)
    
    print("📊 CURRENT RESULTS (after 2.0 threshold fix):")
    print("   label_low_vol_short: 66.4%")
    print("   label_normal_vol_short: 68.4%") 
    print("   label_high_vol_short: 64.7%")
    print()
    print("🔍 These are still unrealistically high!")
    print("   Expected: 30-45% for realistic trading")
    print()
    
    # Possible reasons for failure
    reasons = [
        "1. 📉 ROLLOVER GAPS ARE UPWARD, NOT DOWNWARD",
        "   - We assumed rollover gaps help short trades",
        "   - But June 2011 gaps might be upward (backwardation)",
        "   - Upward gaps would help short trades hit stops (bad for shorts)",
        "   - Need to check actual gap direction",
        "",
        "2. 🔄 ROLLOVER DETECTION NOT WORKING",
        "   - 2.0 threshold might still be wrong",
        "   - ES gaps might be smaller (0.5-1.5 points)",
        "   - Detection logic might have bugs",
        "   - Need to verify detection is actually happening",
        "",
        "3. 📅 WRONG ROLLOVER DATES",
        "   - June 2011 rollover might be different dates",
        "   - Multiple rollover events not detected",
        "   - Need to check actual ES contract specifications",
        "",
        "4. 🐛 LABELING LOGIC STILL WRONG",
        "   - Despite our verification, logic might be inverted",
        "   - Edge case in the implementation",
        "   - Need to manually verify sample trades",
        "",
        "5. 📊 LEGITIMATE MARKET BEHAVIOR",
        "   - 2011 was extremely volatile",
        "   - High frequency mean reversion",
        "   - Short trades genuinely successful in that period"
    ]
    
    for reason in reasons:
        print(reason)
    
    print("\n🔍 INVESTIGATION PRIORITIES:")
    print("1. Check if rollover events are actually being detected")
    print("2. Verify the direction of rollover gaps (up vs down)")
    print("3. Manual verification of sample winning short trades")
    print("4. Test with even lower threshold (0.5 points)")

def create_rollover_verification_commands():
    """Create commands to verify rollover detection is working"""
    
    print(f"\n🛠️ ROLLOVER VERIFICATION COMMANDS")
    print("=" * 60)
    
    commands = [
        "# Download the newly processed June 2011 data",
        "aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/06/ /tmp/june_2011_debug/ --recursive --region us-east-1",
        "",
        "# Check if rollover statistics were collected",
        "python3 -c \"",
        "import pandas as pd",
        "import numpy as np",
        "from glob import glob",
        "",
        "# Find the parquet file",
        "parquet_files = glob('/tmp/june_2011_debug/*.parquet')",
        "if parquet_files:",
        "    df = pd.read_parquet(parquet_files[0])",
        "    print(f'Loaded {len(df):,} rows')",
        "    ",
        "    # Check for large price gaps",
        "    df['price_change'] = df['close'].diff()",
        "    df['price_change_abs'] = df['price_change'].abs()",
        "    ",
        "    # Count gaps at different thresholds",
        "    for threshold in [0.5, 1.0, 2.0, 5.0, 10.0]:",
        "        gaps = (df['price_change_abs'] > threshold).sum()",
        "        print(f'Gaps > {threshold} points: {gaps}')",
        "    ",
        "    # Show largest gaps",
        "    large_gaps = df[df['price_change_abs'] > 2.0].head(10)",
        "    print(f'\\\\nLargest gaps:')",
        "    for idx, row in large_gaps.iterrows():",
        "        print(f'{row[\\\"timestamp\\\"]}: {row[\\\"price_change\\\"]:+.2f} points')",
        "    ",
        "    # Check gap direction bias",
        "    large_gaps_df = df[df['price_change_abs'] > 2.0]",
        "    if len(large_gaps_df) > 0:",
        "        up_gaps = (large_gaps_df['price_change'] > 0).sum()",
        "        down_gaps = (large_gaps_df['price_change'] < 0).sum()",
        "        print(f'\\\\nGap direction:')",
        "        print(f'Up gaps: {up_gaps}')",
        "        print(f'Down gaps: {down_gaps}')",
        "        if up_gaps > down_gaps:",
        "            print('UPWARD BIAS - This would HURT short trades!')",
        "        elif down_gaps > up_gaps:",
        "            print('DOWNWARD BIAS - This would HELP short trades!')",
        "else:",
        "    print('No parquet files found')",
        "\"",
        "",
        "# Clean up",
        "rm -rf /tmp/june_2011_debug/"
    ]
    
    for cmd in commands:
        print(cmd)

def suggest_next_debugging_steps():
    """Suggest next debugging steps"""
    
    print(f"\n🎯 NEXT DEBUGGING STEPS")
    print("=" * 60)
    
    print("🔍 STEP 1: Verify Rollover Detection")
    print("   Run the verification commands above")
    print("   Check if gaps are actually being detected")
    print("   Verify gap direction (up vs down)")
    print()
    
    print("🔍 STEP 2: Manual Trade Verification")
    print("   Download sample data and manually verify trades:")
    print("   - Pick 10 winning short trades")
    print("   - Manually calculate target/stop prices")
    print("   - Check if targets were actually hit")
    print("   - Look for any obvious errors")
    print()
    
    print("🔍 STEP 3: Test Lower Threshold")
    print("   Try threshold = 0.5 points:")
    print("   - Modify weighted_labeling.py")
    print("   - Reprocess small sample")
    print("   - Check if more gaps are detected")
    print()
    
    print("🔍 STEP 4: Check Contract Specifications")
    print("   Research actual ES contract rollover for June 2011:")
    print("   - Exact rollover dates")
    print("   - Typical gap sizes")
    print("   - Market conditions during rollover")
    print()
    
    print("🔍 STEP 5: Consider Alternative Explanations")
    print("   If rollover isn't the cause, consider:")
    print("   - Data quality issues (bad ticks)")
    print("   - Timestamp problems")
    print("   - Fundamental labeling logic error")
    print("   - Legitimate market behavior (2011 was volatile)")

def create_manual_verification_script():
    """Create a script for manual trade verification"""
    
    script = '''#!/usr/bin/env python3
"""
Manual Trade Verification

This script manually verifies a sample of winning short trades
to check if they are actually legitimate wins.
"""

import pandas as pd
import numpy as np

def verify_sample_trades():
    """Manually verify sample winning short trades"""
    
    # This would need to be run with actual data
    print("Manual verification steps:")
    print("1. Load processed June 2011 data")
    print("2. Filter for winning short trades (label = 1)")
    print("3. For each trade:")
    print("   - Calculate entry price (next bar open)")
    print("   - Calculate target/stop prices")
    print("   - Check if target was actually hit")
    print("   - Look for any anomalies")
    print()
    print("Example verification:")
    print("Trade at 2011-06-10 10:30:15")
    print("  Entry: 1280.50 (next bar open)")
    print("  Target: 1276.50 (entry - 16 ticks)")
    print("  Stop: 1282.50 (entry + 8 ticks)")
    print("  Did price actually go to 1276.50? Check manually!")

if __name__ == "__main__":
    verify_sample_trades()
'''
    
    with open('manual_trade_verification.py', 'w') as f:
        f.write(script)
    
    print(f"\n📄 Created manual_trade_verification.py")
    print("   Use this template for manual verification")

def main():
    """Main execution"""
    
    analyze_rollover_detection_failure()
    create_rollover_verification_commands()
    suggest_next_debugging_steps()
    create_manual_verification_script()
    
    print(f"\n🚨 CRITICAL FINDING:")
    print("The rollover threshold fix did NOT resolve the issue!")
    print("This suggests:")
    print("1. Rollover detection isn't working as expected, OR")
    print("2. Rollover gaps aren't the root cause, OR") 
    print("3. There's a deeper issue with the labeling logic")
    print()
    print("🚀 IMMEDIATE ACTION:")
    print("Run the verification commands to check if rollover")
    print("events are actually being detected in the data.")


if __name__ == "__main__":
    main()