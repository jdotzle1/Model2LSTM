#!/usr/bin/env python3
"""
Quick Trade Check

Let's quickly check a few trades to see if they're correctly labeled.
"""

import pandas as pd
import numpy as np

def quick_check():
    """Quick check of a few trades"""
    
    file_path = r"C:\Users\jdotzler\Desktop\monthly_2011-06_20251106_213935.parquet"
    
    print("🔍 QUICK TRADE CHECK")
    print("=" * 60)
    
    try:
        df = pd.read_parquet(file_path)
        
        # Get first 5 winning short trades
        winners = df[df['label_normal_vol_short'] == 1].head(5)
        
        print("Checking 5 'winning' normal vol short trades:")
        print("(Normal vol short: 8 tick stop, 16 tick target)")
        print()
        
        for i, (idx, row) in enumerate(winners.iterrows()):
            print(f"Trade {i+1}:")
            
            # Check if there's a next bar for entry
            if idx + 1 < len(df):
                entry_bar = df.iloc[idx + 1]
                entry_price = entry_bar['open']
                target_price = entry_price - (16 * 0.25)  # 4 points down
                stop_price = entry_price + (8 * 0.25)     # 2 points up
                
                print(f"  Entry: {entry_price:.2f}")
                print(f"  Target: {target_price:.2f}, Stop: {stop_price:.2f}")
                print(f"  Entry bar: H={entry_bar['high']:.2f}, L={entry_bar['low']:.2f}")
                
                # Quick check
                target_hit = entry_bar['low'] <= target_price
                stop_hit = entry_bar['high'] >= stop_price
                
                if target_hit and not stop_hit:
                    print(f"  ✅ VALID WIN: Target hit in entry bar")
                elif stop_hit:
                    print(f"  ❌ INVALID: Stop hit - should be loss")
                elif not target_hit and not stop_hit:
                    print(f"  ❓ UNCLEAR: Neither hit in entry bar (need to check forward)")
                    
                    # Quick forward check (just next 10 bars)
                    found_target = False
                    found_stop = False
                    
                    for j in range(idx + 2, min(idx + 12, len(df))):
                        check_bar = df.iloc[j]
                        if check_bar['low'] <= target_price:
                            found_target = True
                            print(f"    Target found at bar {j}")
                            break
                        if check_bar['high'] >= stop_price:
                            found_stop = True
                            print(f"    Stop found at bar {j}")
                            break
                    
                    if found_target:
                        print(f"    ✅ VALID WIN: Target found in forward bars")
                    elif found_stop:
                        print(f"    ❌ INVALID: Stop found - should be loss")
                    else:
                        print(f"    ❌ INVALID: Neither found - should be timeout loss")
            else:
                print(f"  ❌ No entry bar available")
            
            print()
        
        # Quick summary
        print("🎯 QUICK ASSESSMENT:")
        print("If most of these trades show 'INVALID', then there's definitely still a bug.")
        print("If most show 'VALID WIN', then the high win rates might be legitimate.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main execution"""
    quick_check()


if __name__ == "__main__":
    main()