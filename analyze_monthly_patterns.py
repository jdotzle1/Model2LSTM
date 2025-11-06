#!/usr/bin/env python3
"""
Analyze Short Win Rates Across Multiple Months
"""
import json
import os
from glob import glob

def analyze_monthly_patterns():
    """Analyze short win rate patterns across months"""
    
    months = ['2011-05', '2011-06', '2011-07', '2011-08']
    results = {}
    
    print("📊 SHORT WIN RATE ANALYSIS ACROSS MONTHS")
    print("=" * 60)
    
    for month in months:
        stats_dir = f"/tmp/stats_{month}/"
        if os.path.exists(stats_dir):
            stats_files = glob(f"{stats_dir}*.json")
            if stats_files:
                try:
                    with open(stats_files[0]) as f:
                        data = json.load(f)
                    
                    month_results = {}
                    print(f"\n📅 {month}:")
                    
                    for mode, stats in data.items():
                        if 'short' in mode and isinstance(stats, dict) and 'win_rate' in stats:
                            win_rate = stats['win_rate']
                            month_results[mode] = win_rate
                            
                            status = "🚨 HIGH" if win_rate > 0.5 else "✅ Normal"
                            print(f"  {mode}: {win_rate:.1%} {status}")
                    
                    results[month] = month_results
                    
                except Exception as e:
                    print(f"  ❌ Error reading {month}: {e}")
            else:
                print(f"  📝 No statistics files found for {month}")
        else:
            print(f"  📁 Directory not found: {stats_dir}")
    
    # Summary analysis
    print("\n" + "=" * 60)
    print("📋 PATTERN ANALYSIS:")
    
    if len(results) >= 2:
        # Check consistency across months
        short_modes = ['low_vol_short', 'normal_vol_short', 'high_vol_short']
        
        for mode in short_modes:
            mode_rates = []
            for month, month_data in results.items():
                if mode in month_data:
                    mode_rates.append(month_data[mode])
            
            if mode_rates:
                avg_rate = sum(mode_rates) / len(mode_rates)
                min_rate = min(mode_rates)
                max_rate = max(mode_rates)
                
                print(f"\n{mode}:")
                print(f"  Average: {avg_rate:.1%}")
                print(f"  Range: {min_rate:.1%} - {max_rate:.1%}")
                
                if avg_rate > 0.5:
                    print(f"  🚨 CONSISTENTLY HIGH across {len(mode_rates)} months")
                    if max_rate - min_rate < 0.1:
                        print(f"  📊 Pattern is CONSISTENT (low variance)")
                    else:
                        print(f"  📊 Pattern is VARIABLE (high variance)")
                else:
                    print(f"  ✅ Normal win rates")
        
        # Conclusion
        print("\n" + "=" * 60)
        print("🎯 CONCLUSION:")
        
        high_rate_months = sum(1 for month_data in results.values() 
                              for mode, rate in month_data.items() 
                              if 'short' in mode and rate > 0.5)
        total_short_measurements = sum(len([m for m in month_data.keys() if 'short' in m]) 
                                     for month_data in results.values())
        
        if high_rate_months > total_short_measurements * 0.5:
            print("🚨 HIGH SHORT WIN RATES ARE CONSISTENT ACROSS MONTHS")
            print("   This suggests:")
            print("   1. Market conditions in 2011 favored short trades")
            print("   2. High volatility period with mean reversion")
            print("   3. This might be CORRECT, not a bug!")
            print("   4. Consider this is post-financial crisis recovery period")
        else:
            print("✅ High short win rates appear to be month-specific")
            print("   This suggests:")
            print("   1. Data quality issues in specific months")
            print("   2. Contract rollover problems")
            print("   3. Need to investigate specific months with issues")
    
    else:
        print("❌ Insufficient data for pattern analysis")
        print("   Need at least 2 months of data to compare")

if __name__ == "__main__":
    analyze_monthly_patterns()
