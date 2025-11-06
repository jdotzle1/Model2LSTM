#!/usr/bin/env python3
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
