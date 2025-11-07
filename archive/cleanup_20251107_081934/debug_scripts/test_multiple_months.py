#!/usr/bin/env python3
"""
Test Multiple Months for Short Win Rate Pattern

This script helps test whether the high short win rates are consistent
across multiple months or specific to June 2011.
"""

import subprocess
import json
import os
from datetime import datetime

class MultiMonthTester:
    """Test multiple months to validate short win rate patterns"""
    
    def __init__(self):
        self.test_months = [
            '2011-05',  # May 2011
            '2011-06',  # June 2011 (already processed)
            '2011-07',  # July 2011
            '2011-08',  # August 2011
        ]
        
    def generate_test_commands(self):
        """Generate commands to test multiple months"""
        
        print("🧪 MULTI-MONTH TESTING COMMANDS")
        print("=" * 60)
        print("Run these commands to test the short win rate pattern across multiple months:")
        print()
        
        for month in self.test_months:
            print(f"# Test {month}")
            print(f"python3 process_monthly_chunks_fixed.py --test-month {month}")
            print()
        
        print("# Compare statistics across months")
        print("aws s3 ls s3://es-1-second-data/processed-data/monthly/2011/ --recursive | grep statistics")
        print()
        
        print("# Download statistics for comparison")
        for month in self.test_months:
            month_num = month.split('-')[1]
            print(f"aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/{month_num}/statistics/ /tmp/stats_{month}/ --recursive --region us-east-1")
        print()
        
        print("# Quick comparison script")
        print('python3 -c "')
        print('import json')
        print('import os')
        print('from glob import glob')
        print('')
        print('months = ["2011-05", "2011-06", "2011-07", "2011-08"]')
        print('print("Short Win Rate Comparison Across Months:")')
        print('print("=" * 50)')
        print('')
        print('for month in months:')
        print('    stats_dir = f"/tmp/stats_{month}/"')
        print('    if os.path.exists(stats_dir):')
        print('        stats_files = glob(f"{stats_dir}*.json")')
        print('        if stats_files:')
        print('            with open(stats_files[0]) as f:')
        print('                data = json.load(f)')
        print('            print(f"\\n{month}:")')
        print('            for mode, stats in data.items():')
        print('                if "short" in mode and "win_rate" in stats:')
        print('                    win_rate = stats["win_rate"]')
        print('                    print(f"  {mode}: {win_rate:.1%}")')
        print('"')
        print()
        
    def create_analysis_script(self):
        """Create a script to analyze results across months"""
        
        analysis_script = '''#!/usr/bin/env python3
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
                    print(f"\\n📅 {month}:")
                    
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
    print("\\n" + "=" * 60)
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
                
                print(f"\\n{mode}:")
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
        print("\\n" + "=" * 60)
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
'''
        
        with open('analyze_monthly_patterns.py', 'w', encoding='utf-8') as f:
            f.write(analysis_script)
        
        print("📄 Created analyze_monthly_patterns.py")
        print("   Run this after downloading statistics from multiple months")
    
    def run_investigation_plan(self):
        """Present the complete investigation plan"""
        
        print("🔍 COMPLETE INVESTIGATION PLAN")
        print("=" * 60)
        
        print("\n📋 STEP 1: Process Additional Months")
        self.generate_test_commands()
        
        print("\n📋 STEP 2: Create Analysis Script")
        self.create_analysis_script()
        
        print("\n📋 STEP 3: Expected Outcomes")
        print()
        print("🎯 If HIGH SHORT WIN RATES are consistent across months:")
        print("   → This is likely CORRECT market behavior")
        print("   → 2011 was post-financial crisis with high volatility")
        print("   → Mean reversion favored short trades")
        print("   → Consider this a feature, not a bug!")
        print()
        print("🎯 If HIGH SHORT WIN RATES are month-specific:")
        print("   → Data quality issues in specific months")
        print("   → Contract rollover problems")
        print("   → Need targeted debugging of affected months")
        print()
        print("🎯 If ALL WIN RATES are unreasonable:")
        print("   → Systematic labeling logic error")
        print("   → Need to review algorithm implementation")
        print()
        
        print("📋 STEP 4: Historical Context Research")
        print("   Research 2011 ES market conditions:")
        print("   - Post-financial crisis recovery")
        print("   - European debt crisis")
        print("   - High volatility environment")
        print("   - Frequent intraday reversals")
        print()
        
        print("🚀 START HERE:")
        print("1. Run the test commands above")
        print("2. Execute analyze_monthly_patterns.py")
        print("3. Compare results with historical market context")
        print("4. Make decision: bug fix or accept as correct")


def main():
    """Main execution"""
    tester = MultiMonthTester()
    tester.run_investigation_plan()


if __name__ == "__main__":
    main()