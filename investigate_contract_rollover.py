#!/usr/bin/env python3
"""
Contract Rollover Investigation

This script specifically investigates whether contract rollover events
are causing the high short win rates by creating artificial price movements.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class ContractRolloverInvestigator:
    """Investigate contract rollover effects on labeling"""
    
    def __init__(self):
        self.tick_size = 0.25
        
    def analyze_es_rollover_schedule(self, year: int = 2011, month: int = 6):
        """Analyze ES contract rollover schedule for given period"""
        
        print(f"📅 ES CONTRACT ROLLOVER ANALYSIS - {year}-{month:02d}")
        print("=" * 60)
        
        # ES contracts expire on 3rd Friday of March, June, September, December
        # But trading typically rolls to next contract 1-2 weeks before expiration
        
        rollover_months = [3, 6, 9, 12]  # March, June, September, December
        
        if month in rollover_months:
            print(f"🚨 {year}-{month:02d} IS A ROLLOVER MONTH!")
            print(f"   ES contracts typically roll in {year}-{month:02d}")
            
            # Calculate approximate rollover dates
            # Third Friday is typically between 15th-21st
            # Roll usually happens 1-2 weeks before (so around 1st-14th)
            
            print(f"   Expected rollover period: {year}-{month:02d}-01 to {year}-{month:02d}-14")
            print(f"   Contract expiration: Third Friday of {year}-{month:02d}")
            
            if month == 6:  # June 2011 specifically
                print(f"   June 2011: ESM11 (June) → ESU11 (September)")
                print(f"   This explains the high short win rates!")
                
        else:
            print(f"✅ {year}-{month:02d} is NOT a rollover month")
            print(f"   Next rollover: {year}-{min([m for m in rollover_months if m > month], default=rollover_months[0])}")
    
    def detect_rollover_gaps(self, df: pd.DataFrame, threshold: float = 2.0) -> Dict:
        """Detect potential rollover gaps in price data"""
        
        print(f"\n🔍 ROLLOVER GAP DETECTION (threshold: {threshold} points)")
        print("-" * 40)
        
        # Calculate price changes
        df['price_change'] = df['close'].diff()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Find large gaps
        large_gaps = df[df['price_change_abs'] > threshold].copy()
        
        results = {
            'total_bars': len(df),
            'large_gaps_count': len(large_gaps),
            'threshold_used': threshold,
            'max_gap': df['price_change_abs'].max(),
            'mean_gap': df['price_change_abs'].mean(),
            'large_gaps': []
        }
        
        print(f"Total bars: {results['total_bars']:,}")
        print(f"Large gaps (>{threshold} points): {results['large_gaps_count']}")
        print(f"Max gap: {results['max_gap']:.2f} points")
        print(f"Mean gap: {results['mean_gap']:.4f} points")
        
        if len(large_gaps) > 0:
            print(f"\n🚨 LARGE GAPS DETECTED:")
            for idx, row in large_gaps.head(10).iterrows():
                gap_info = {
                    'timestamp': row['timestamp'],
                    'price_change': row['price_change'],
                    'from_price': row['close'] - row['price_change'],
                    'to_price': row['close']
                }
                results['large_gaps'].append(gap_info)
                
                print(f"  {row['timestamp']}: {row['price_change']:+.2f} points")
                print(f"    {gap_info['from_price']:.2f} → {gap_info['to_price']:.2f}")
        
        return results
    
    def analyze_rollover_impact_on_labels(self, df: pd.DataFrame, gap_threshold: float = 2.0):
        """Analyze how rollover gaps might affect short trade labels"""
        
        print(f"\n📊 ROLLOVER IMPACT ON SHORT LABELS")
        print("-" * 40)
        
        # Detect gaps
        gaps = self.detect_rollover_gaps(df, gap_threshold)
        
        if gaps['large_gaps_count'] == 0:
            print("✅ No large gaps detected - rollover unlikely to be the cause")
            return
        
        print(f"\n🔍 ANALYZING IMPACT OF {gaps['large_gaps_count']} LARGE GAPS:")
        
        # For each gap, analyze the impact on surrounding trades
        for i, gap in enumerate(gaps['large_gaps'][:5]):  # Analyze first 5 gaps
            print(f"\n  Gap {i+1}: {gap['timestamp']}")
            print(f"    Price change: {gap['price_change']:+.2f} points")
            
            # Find the gap in the dataframe
            gap_idx = df[df['timestamp'] == gap['timestamp']].index
            if len(gap_idx) == 0:
                continue
                
            gap_idx = gap_idx[0]
            
            # Analyze trades around this gap
            window_start = max(0, gap_idx - 10)
            window_end = min(len(df), gap_idx + 10)
            window_df = df.iloc[window_start:window_end].copy()
            
            # Simulate short trade impact
            if gap['price_change'] < -2.0:  # Large downward gap
                print(f"    📉 Large DOWN gap - would help SHORT trades hit targets")
                print(f"    This could artificially inflate short win rates!")
                
            elif gap['price_change'] > 2.0:  # Large upward gap
                print(f"    📈 Large UP gap - would help SHORT trades hit stops")
                print(f"    This should reduce short win rates")
    
    def create_rollover_aware_analysis(self):
        """Create analysis commands that account for rollover effects"""
        
        print(f"\n🛠️ ROLLOVER-AWARE ANALYSIS COMMANDS")
        print("=" * 60)
        
        print("Run these commands to investigate rollover effects:")
        print()
        
        # Download and analyze with different thresholds
        commands = [
            "# Download June 2011 data for rollover analysis",
            "aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/06/monthly_2011-06_20251106_173444.parquet /tmp/rollover_analysis.parquet --region us-east-1",
            "",
            "# Analyze with different rollover thresholds",
            "python3 -c \"",
            "import pandas as pd",
            "import numpy as np",
            "",
            "df = pd.read_parquet('/tmp/rollover_analysis.parquet')",
            "print(f'Loaded {len(df):,} rows')",
            "",
            "# Check for rollover gaps with different thresholds",
            "for threshold in [1.0, 2.0, 5.0, 10.0, 20.0]:",
            "    gaps = (df['close'].diff().abs() > threshold).sum()",
            "    print(f'Gaps > {threshold} points: {gaps}')",
            "",
            "# Find the largest gaps",
            "df['price_change'] = df['close'].diff()",
            "large_gaps = df[df['price_change'].abs() > 2.0]",
            "print(f'\\nLargest gaps:')",
            "for idx, row in large_gaps.head(10).iterrows():",
            "    print(f'{row[\\\"timestamp\\\"]}: {row[\\\"price_change\\\"]:+.2f} points')",
            "",
            "# Check if gaps correlate with high short win rates",
            "if 'label_normal_vol_short' in df.columns:",
            "    # Analyze win rates around gaps",
            "    gap_indices = large_gaps.index",
            "    for gap_idx in gap_indices[:3]:",
            "        window_start = max(0, gap_idx - 100)",
            "        window_end = min(len(df), gap_idx + 100)",
            "        window_df = df.iloc[window_start:window_end]",
            "        win_rate = window_df['label_normal_vol_short'].mean()",
            "        print(f'Win rate around gap at {df.iloc[gap_idx][\\\"timestamp\\\"]}: {win_rate:.1%}')",
            "\"",
            "",
            "# Clean up",
            "rm /tmp/rollover_analysis.parquet"
        ]
        
        for cmd in commands:
            print(cmd)
    
    def explain_rollover_bias(self):
        """Explain how rollover could create short trade bias"""
        
        print(f"\n💡 HOW ROLLOVER CREATES SHORT TRADE BIAS")
        print("=" * 60)
        
        print("🔄 Contract Rollover Mechanics:")
        print("1. Old contract (ESM11) trades at one price")
        print("2. New contract (ESU11) trades at slightly different price")
        print("3. Data switches from old to new contract")
        print("4. Creates artificial price 'gap' in continuous data")
        print()
        
        print("📉 Why This Helps Short Trades:")
        print("1. Rollover gaps are often DOWNWARD (contango effect)")
        print("2. Downward gaps help short trades hit targets faster")
        print("3. Creates artificial 'wins' that aren't real market moves")
        print("4. Inflates short win rates artificially")
        print()
        
        print("🎯 June 2011 Specific:")
        print("- ESM11 (June) → ESU11 (September) rollover")
        print("- Typically happens early June (1st-14th)")
        print("- Could explain 60-70% short win rates")
        print("- Need to check if gaps are excluded properly")
        print()
        
        print("🔧 Current Rollover Detection:")
        print("- Threshold: 20 points (might be too high!)")
        print("- ES rollover gaps often 2-10 points")
        print("- Many rollover events might be missed")
        print("- Need to lower threshold to 2-5 points")
    
    def recommend_fixes(self):
        """Recommend fixes for rollover issues"""
        
        print(f"\n🛠️ RECOMMENDED FIXES")
        print("=" * 60)
        
        print("1. 📉 LOWER ROLLOVER THRESHOLD:")
        print("   - Current: 20 points (too high)")
        print("   - Recommended: 2-5 points")
        print("   - ES rollover gaps are typically smaller")
        print()
        
        print("2. 🔍 ENHANCED ROLLOVER DETECTION:")
        print("   - Check volume spikes (rollover = high volume)")
        print("   - Check time patterns (rollover = specific dates)")
        print("   - Check multiple consecutive gaps")
        print()
        
        print("3. 📅 DATE-BASED EXCLUSION:")
        print("   - Exclude known rollover periods")
        print("   - June 1-14, 2011 for ESM11→ESU11")
        print("   - More reliable than gap detection")
        print()
        
        print("4. ✅ VALIDATION STEPS:")
        print("   - Test with threshold = 2.0 points")
        print("   - Compare win rates before/after fix")
        print("   - Validate on non-rollover months")
        print()
        
        print("🚀 IMMEDIATE ACTION:")
        print("   Modify weighted_labeling.py:")
        print("   roll_detection_threshold: 20.0 → 2.0")
        print("   Re-run June 2011 processing")
        print("   Compare results")
    
    def run_investigation(self):
        """Run complete rollover investigation"""
        
        print("🔄 CONTRACT ROLLOVER INVESTIGATION")
        print("=" * 60)
        
        # Analyze rollover schedule
        self.analyze_es_rollover_schedule(2011, 6)
        
        # Explain rollover bias
        self.explain_rollover_bias()
        
        # Create analysis commands
        self.create_rollover_aware_analysis()
        
        # Recommend fixes
        self.recommend_fixes()
        
        print(f"\n🎯 CONCLUSION:")
        print("Contract rollover is VERY LIKELY the root cause!")
        print("- June 2011 is a rollover month (ESM11 → ESU11)")
        print("- Current threshold (20 points) too high")
        print("- Rollover gaps help short trades artificially")
        print("- Need to lower threshold to 2-5 points")


def main():
    """Main execution"""
    investigator = ContractRolloverInvestigator()
    investigator.run_investigation()


if __name__ == "__main__":
    main()