#!/usr/bin/env python3
"""
Manual Trade Verification

Let's manually verify some of these "winning" short trades
to see why they're being labeled as winners when they shouldn't be.
"""

import pandas as pd
import numpy as np

def manual_verification():
    """Manually verify the trades from the parquet file"""
    
    file_path = r"C:\Users\jdotzler\Desktop\monthly_2011-06_20251106_213935.parquet"
    
    print("🔍 MANUAL TRADE VERIFICATION")
    print("=" * 60)
    
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df):,} rows")
        
        # Focus on normal_vol_short winners
        winners = df[df['label_normal_vol_short'] == 1].head(20)
        
        print(f"\n📊 ANALYZING 20 'WINNING' NORMAL VOL SHORT TRADES:")
        print("(Entry price is NEXT bar's open, not current bar)")
        print()
        
        suspicious_count = 0
        
        for i, (idx, row) in enumerate(winners.iterrows()):
            print(f"Trade {i+1} - Signal at index {idx}:")
            
            # Find the entry bar (next bar after signal)
            if idx + 1 < len(df):
                entry_bar = df.iloc[idx + 1]
                entry_price = entry_bar['open']
                
                # Calculate target/stop for normal vol short (8 tick stop, 16 tick target)
                target_price = entry_price - (16 * 0.25)  # 4 points down
                stop_price = entry_price + (8 * 0.25)     # 2 points up
                
                print(f"  Signal bar: {row['timestamp']}")
                print(f"  Entry bar: {entry_bar['timestamp']}")
                print(f"  Entry price: {entry_price:.2f}")
                print(f"  Target: {target_price:.2f} (need price to go DOWN)")
                print(f"  Stop: {stop_price:.2f} (price going UP)")
                
                # Check if target was hit in the entry bar
                entry_target_hit = entry_bar['low'] <= target_price
                entry_stop_hit = entry_bar['high'] >= stop_price
                
                print(f"  Entry bar range: H={entry_bar['high']:.2f}, L={entry_bar['low']:.2f}")
                print(f"  Target hit in entry bar: {entry_target_hit}")
                print(f"  Stop hit in entry bar: {entry_stop_hit}")
                
                if not entry_target_hit and not entry_stop_hit:
                    # Need to look forward for target/stop hits
                    print(f"  Neither hit in entry bar - checking forward...")
                    
                    target_found = False
                    stop_found = False
                    
                    # Look forward up to 900 seconds (15 minutes)
                    for j in range(idx + 1, min(idx + 901, len(df))):
                        check_bar = df.iloc[j]
                        
                        check_target_hit = check_bar['low'] <= target_price
                        check_stop_hit = check_bar['high'] >= stop_price
                        
                        if check_target_hit and not target_found:
                            print(f"    Target hit at bar {j}: {check_bar['timestamp']}")
                            print(f"      Low: {check_bar['low']:.2f} <= Target: {target_price:.2f}")
                            target_found = True
                            
                        if check_stop_hit and not stop_found:
                            print(f"    Stop hit at bar {j}: {check_bar['timestamp']}")
                            print(f"      High: {check_bar['high']:.2f} >= Stop: {stop_price:.2f}")
                            stop_found = True
                            
                        if target_found or stop_found:
                            break
                    
                    if target_found and not stop_found:
                        print(f"  ✅ LEGITIMATE WIN: Target hit before stop")
                    elif stop_found and not target_found:
                        print(f"  ❌ SHOULD BE LOSS: Stop hit before target")
                        suspicious_count += 1
                    elif target_found and stop_found:
                        print(f"  ❌ SHOULD BE LOSS: Both hit (conservative)")
                        suspicious_count += 1
                    else:
                        print(f"  ❌ SHOULD BE LOSS: Timeout (neither hit)")
                        suspicious_count += 1
                        
                elif entry_target_hit and not entry_stop_hit:
                    print(f"  ✅ LEGITIMATE WIN: Target hit in entry bar")
                elif entry_stop_hit:
                    print(f"  ❌ SHOULD BE LOSS: Stop hit in entry bar")
                    suspicious_count += 1
                    
            else:
                print(f"  ❌ No entry bar available")
                suspicious_count += 1
            
            print()
            
            if i >= 9:  # Only check first 10 for brevity
                break
        
        print(f"🎯 VERIFICATION SUMMARY:")
        print(f"Trades analyzed: {min(10, len(winners))}")
        print(f"Suspicious trades: {suspicious_count}")
        print(f"Suspicious rate: {suspicious_count/min(10, len(winners)):.1%}")
        
        if suspicious_count > 5:
            print(f"🚨 MAJOR BUG: Most 'winning' trades should actually be losses!")
        elif suspicious_count > 2:
            print(f"⚠️  MODERATE BUG: Some trades incorrectly labeled")
        else:
            print(f"✅ MOSTLY CORRECT: Few labeling errors")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution"""
    manual_verification()


if __name__ == "__main__":
    main()