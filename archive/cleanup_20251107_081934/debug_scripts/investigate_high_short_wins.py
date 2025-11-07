#!/usr/bin/env python3
"""
Investigate High Short Win Rates

This script provides multiple approaches to investigate why short trades
have unexpectedly high win rates. It can work with local data or provide
commands to run on your system with S3 access.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List

class ShortWinInvestigator:
    """Investigate high short win rates in trading data"""
    
    def __init__(self):
        self.tick_size = 0.25
        
    def generate_s3_investigation_commands(self) -> List[str]:
        """Generate commands to run on system with S3 access"""
        
        commands = [
            "# Download and investigate the processed data",
            "aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/06/monthly_2011-06_20251106_173444.parquet /tmp/investigate.parquet --region us-east-1",
            "",
            "# Quick Python investigation",
            "python3 -c \"",
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Load data",
            "df = pd.read_parquet('/tmp/investigate.parquet')",
            "print(f'Loaded {len(df):,} rows')",
            "",
            "# Check win rates",
            "label_cols = [col for col in df.columns if col.startswith('label_')]",
            "print('\\nWin rates:')",
            "for col in label_cols:",
            "    win_rate = df[col].mean()",
            "    print(f'{col}: {win_rate:.1%}')",
            "",
            "# Focus on most suspicious short mode",
            "short_cols = [col for col in label_cols if 'short' in col]",
            "if short_cols:",
            "    worst_col = max(short_cols, key=lambda x: df[x].mean())",
            "    print(f'\\nMost suspicious: {worst_col} with {df[worst_col].mean():.1%} win rate')",
            "    ",
            "    # Sample analysis",
            "    winners = df[df[worst_col] == 1].head(20)",
            "    print(f'\\nSample winners from {worst_col}:')",
            "    for idx, row in winners.iterrows():",
            "        print(f'{row[\\\"timestamp\\\"]}: O={row[\\\"open\\\"]:.2f} H={row[\\\"high\\\"]:.2f} L={row[\\\"low\\\"]:.2f} C={row[\\\"close\\\"]:.2f}')",
            "\"",
            "",
            "# Clean up",
            "rm /tmp/investigate.parquet"
        ]
        
        return commands
    
    def analyze_local_sample_data(self, sample_file: str = None):
        """Analyze sample data if available locally"""
        
        if sample_file and os.path.exists(sample_file):
            print(f"📊 Analyzing local file: {sample_file}")
            df = pd.read_parquet(sample_file)
            self._perform_detailed_analysis(df)
        else:
            print("📝 No local sample data available")
            print("To investigate with real data, run these commands on your system:")
            print()
            
            commands = self.generate_s3_investigation_commands()
            for cmd in commands:
                print(cmd)
    
    def _perform_detailed_analysis(self, df: pd.DataFrame):
        """Perform detailed analysis on the data"""
        
        print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Basic win rate analysis
        label_cols = [col for col in df.columns if col.startswith('label_')]
        long_cols = [col for col in label_cols if 'long' in col]
        short_cols = [col for col in label_cols if 'short' in col]
        
        print(f"\n📊 WIN RATE ANALYSIS:")
        print("Long modes:")
        for col in long_cols:
            win_rate = df[col].mean()
            print(f"  {col}: {win_rate:.1%}")
            
        print("Short modes:")
        suspicious_short_modes = []
        for col in short_cols:
            win_rate = df[col].mean()
            print(f"  {col}: {win_rate:.1%}")
            if win_rate > 0.5:
                suspicious_short_modes.append((col, win_rate))
                print(f"    🚨 SUSPICIOUS!")
        
        # Detailed analysis of most suspicious mode
        if suspicious_short_modes:
            worst_mode, worst_rate = max(suspicious_short_modes, key=lambda x: x[1])
            print(f"\n🔍 DETAILED ANALYSIS OF {worst_mode} ({worst_rate:.1%} win rate):")
            
            self._analyze_specific_mode(df, worst_mode)
        
        # Market condition analysis
        self._analyze_market_conditions(df)
    
    def _analyze_specific_mode(self, df: pd.DataFrame, mode_col: str):
        """Analyze a specific trading mode in detail"""
        
        # Get mode parameters
        if 'low_vol' in mode_col:
            stop_ticks, target_ticks = 6, 12
        elif 'normal_vol' in mode_col:
            stop_ticks, target_ticks = 8, 16
        else:  # high_vol
            stop_ticks, target_ticks = 10, 20
        
        print(f"Mode parameters: {stop_ticks} tick stop, {target_ticks} tick target")
        
        # Sample winning trades
        winners = df[df[mode_col] == 1].head(10)
        print(f"\nSample winning trades:")
        
        for idx, row in winners.iterrows():
            entry_price = row['open']  # Simplified
            target_price = entry_price - (target_ticks * self.tick_size)
            stop_price = entry_price + (stop_ticks * self.tick_size)
            
            target_hit = row['low'] <= target_price
            stop_hit = row['high'] >= stop_price
            
            print(f"  {row['timestamp']}: Entry={entry_price:.2f}")
            print(f"    Target={target_price:.2f}, Stop={stop_price:.2f}")
            print(f"    Bar: H={row['high']:.2f}, L={row['low']:.2f}")
            print(f"    Target hit: {target_hit}, Stop hit: {stop_hit}")
            
            if not target_hit:
                print(f"    ⚠️  WARNING: Target not hit but labeled as winner!")
            if target_hit and stop_hit:
                print(f"    ⚠️  WARNING: Both hit - should be loser!")
    
    def _analyze_market_conditions(self, df: pd.DataFrame):
        """Analyze market conditions that might explain high short win rates"""
        
        print(f"\n📈 MARKET CONDITIONS ANALYSIS:")
        
        # Price trend analysis
        df['price_change'] = df['close'].diff()
        df['cumulative_return'] = (df['close'] / df['close'].iloc[0] - 1) * 100
        
        overall_return = df['cumulative_return'].iloc[-1]
        avg_bar_change = df['price_change'].mean()
        volatility = df['price_change'].std()
        
        print(f"Overall period return: {overall_return:.2f}%")
        print(f"Average bar change: {avg_bar_change:.4f} points")
        print(f"Price volatility (std): {volatility:.4f} points")
        
        # Intraday patterns
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_returns = df.groupby('hour')['price_change'].mean()
            
            print(f"\nHourly average price changes:")
            for hour, avg_change in hourly_returns.items():
                print(f"  {hour:02d}:00: {avg_change:.4f} points")
        
        # Large moves analysis
        large_down_moves = (df['price_change'] < -2.0).sum()
        large_up_moves = (df['price_change'] > 2.0).sum()
        
        print(f"\nLarge moves (>2 points):")
        print(f"  Down moves: {large_down_moves}")
        print(f"  Up moves: {large_up_moves}")
        print(f"  Down/Up ratio: {large_down_moves/large_up_moves:.2f}" if large_up_moves > 0 else "  Down/Up ratio: inf")
        
        if large_down_moves > large_up_moves * 1.5:
            print("  📉 Market shows downward bias - explains high short win rates!")
    
    def create_investigation_report(self):
        """Create a comprehensive investigation report"""
        
        report = f"""
# Short Win Rate Investigation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Problem Statement
Short trading modes are showing unexpectedly high win rates (>60%), which seems unrealistic for a 2:1 reward/risk strategy.

## Investigation Approach

### 1. Logic Verification ✅
- Reviewed short trade labeling logic in weighted_labeling.py
- Logic is CORRECT:
  - Short target: entry_price - (target_ticks * tick_size) ✅
  - Short stop: entry_price + (stop_ticks * tick_size) ✅
  - Target hit: bar_low <= target_price ✅
  - Stop hit: bar_high >= stop_price ✅

### 2. Possible Explanations

#### A. Market Conditions (Most Likely)
- **2011 Market Context**: Post-financial crisis, high volatility
- **Intraday Mean Reversion**: ES often reverts intraday
- **Downward Bias**: Market might have natural downward pressure
- **Volatility Regime**: High volatility = more mean reversion

#### B. Data Quality Issues
- **Contract Rollovers**: Price gaps affecting calculations
- **Data Gaps**: Missing bars causing incorrect lookforward
- **Timestamp Issues**: Incorrect time sequencing

#### C. Algorithmic Factors
- **Target Size**: 2:1 ratio might favor short direction
- **Timeout Period**: 15-minute window might be optimal for shorts
- **Entry Timing**: Next bar open might favor short entries

### 3. Validation Steps

#### Immediate Actions:
1. **Compare Time Periods**: Check if other months show same pattern
2. **Compare Long vs Short**: Verify long trades have reasonable win rates
3. **Sample Verification**: Manually verify sample winning short trades
4. **Market Research**: Check 2011 ES market conditions

#### Commands to Run:
```bash
# Check multiple months
python3 process_monthly_chunks_fixed.py --test-month 2011-05
python3 process_monthly_chunks_fixed.py --test-month 2011-07

# Compare statistics
aws s3 ls s3://es-1-second-data/processed-data/monthly/2011/ --recursive | grep statistics

# Download and compare
aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/05/statistics/ /tmp/may_stats/ --recursive
aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/07/statistics/ /tmp/july_stats/ --recursive
```

### 4. Expected Findings

#### If Market Conditions:
- Other months in 2011 show similar pattern
- Long trades have reasonable win rates (30-45%)
- Pattern consistent with known market volatility

#### If Data Issues:
- Inconsistent patterns across months
- Both long and short affected
- Clear data quality problems visible

#### If Algorithm Issues:
- Pattern appears in all time periods
- Affects multiple volatility modes equally
- Logic review reveals bugs

### 5. Next Steps

1. **Validate with multiple months** ← START HERE
2. **Compare against market benchmarks**
3. **Manual verification of sample trades**
4. **Consider if this is actually correct** (market was volatile in 2011)

## Conclusion

The labeling logic appears correct. High short win rates might be:
1. **Legitimate market behavior** (2011 was volatile, mean-reverting)
2. **Data quality issues** (contract rollovers, gaps)
3. **Algorithmic edge case** (unlikely given logic review)

**Recommendation**: Validate with additional time periods before assuming this is a bug.
"""
        
        return report
    
    def run_investigation(self, sample_file: str = None):
        """Run complete investigation"""
        
        print("🚨 SHORT WIN RATE INVESTIGATION")
        print("=" * 60)
        
        # Try to analyze local data
        self.analyze_local_sample_data(sample_file)
        
        # Generate report
        print("\n" + "=" * 60)
        print("📋 INVESTIGATION REPORT")
        print("=" * 60)
        
        report = self.create_investigation_report()
        
        # Save report
        report_file = f"short_win_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Full report saved to: {report_file}")
        
        # Key recommendations
        print("\n🎯 KEY RECOMMENDATIONS:")
        print("1. Test additional months (2011-05, 2011-07) to see if pattern persists")
        print("2. Compare long vs short win rates in same data")
        print("3. Research 2011 ES market conditions (was it mean-reverting?)")
        print("4. Consider that high short win rates might be CORRECT for that period")
        print("\n💡 Remember: 2011 was post-financial crisis with high volatility!")


def main():
    """Main execution"""
    investigator = ShortWinInvestigator()
    
    # Check for local sample data
    sample_files = [
        'sample_data.parquet',
        'test_data.parquet',
        '/tmp/debug_data.parquet'
    ]
    
    sample_file = None
    for file in sample_files:
        if os.path.exists(file):
            sample_file = file
            break
    
    investigator.run_investigation(sample_file)


if __name__ == "__main__":
    main()