#!/usr/bin/env python3
"""
Debug Short Labeling Logic

This script simulates the short labeling logic to understand why
short win rates are so high. It creates synthetic data and tests
the labeling algorithm step by step.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ShortLabelingDebugger:
    """Debug short trade labeling logic"""
    
    def __init__(self):
        self.tick_size = 0.25
        
    def create_test_scenarios(self) -> pd.DataFrame:
        """Create test scenarios for short trades"""
        
        scenarios = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Scenario 1: Clear winner - price drops to target
        scenarios.append({
            'name': 'Clear Winner',
            'timestamp': base_time,
            'open': 4750.00,
            'high': 4750.50,  # Small move up
            'low': 4747.00,   # Big move down (hits target)
            'close': 4747.50,
            'expected_result': 'WIN'
        })
        
        # Scenario 2: Clear loser - price rises to stop
        scenarios.append({
            'name': 'Clear Loser',
            'timestamp': base_time + timedelta(minutes=1),
            'open': 4750.00,
            'high': 4752.50,  # Big move up (hits stop)
            'low': 4749.50,   # Small move down
            'close': 4751.00,
            'expected_result': 'LOSS'
        })
        
        # Scenario 3: Both hit same bar - should be loss
        scenarios.append({
            'name': 'Both Hit Same Bar',
            'timestamp': base_time + timedelta(minutes=2),
            'open': 4750.00,
            'high': 4752.50,  # Hits stop
            'low': 4747.00,   # Hits target
            'close': 4749.00,
            'expected_result': 'LOSS (conservative)'
        })
        
        # Scenario 4: Neither hit - timeout
        scenarios.append({
            'name': 'Timeout',
            'timestamp': base_time + timedelta(minutes=3),
            'open': 4750.00,
            'high': 4751.00,  # Small moves
            'low': 4749.00,   # Small moves
            'close': 4750.50,
            'expected_result': 'LOSS (timeout)'
        })
        
        return pd.DataFrame(scenarios)
    
    def test_short_logic(self, mode_name: str = 'normal_vol_short'):
        """Test short trade logic for different scenarios"""
        
        # Get mode parameters
        if 'low_vol' in mode_name:
            stop_ticks, target_ticks = 6, 12
        elif 'normal_vol' in mode_name:
            stop_ticks, target_ticks = 8, 16
        else:  # high_vol
            stop_ticks, target_ticks = 10, 20
            
        print(f"🔍 TESTING {mode_name.upper()}")
        print(f"   Stop: {stop_ticks} ticks ({stop_ticks * self.tick_size:.2f} points)")
        print(f"   Target: {target_ticks} ticks ({target_ticks * self.tick_size:.2f} points)")
        print("=" * 60)
        
        scenarios = self.create_test_scenarios()
        
        for idx, scenario in scenarios.iterrows():
            print(f"\n📊 Scenario: {scenario['name']}")
            
            # Entry price (simplified - using same bar's open)
            entry_price = scenario['open']
            
            # Calculate target and stop prices for SHORT trade
            target_price = entry_price - (target_ticks * self.tick_size)  # Price going DOWN
            stop_price = entry_price + (stop_ticks * self.tick_size)      # Price going UP
            
            print(f"   Entry: {entry_price:.2f}")
            print(f"   Target: {target_price:.2f} (need price to go DOWN)")
            print(f"   Stop: {stop_price:.2f} (price going UP)")
            print(f"   Bar: H={scenario['high']:.2f}, L={scenario['low']:.2f}")
            
            # Check hits
            target_hit = scenario['low'] <= target_price
            stop_hit = scenario['high'] >= stop_price
            
            print(f"   Target hit: {target_hit} (low {scenario['low']:.2f} <= target {target_price:.2f})")
            print(f"   Stop hit: {stop_hit} (high {scenario['high']:.2f} >= stop {stop_price:.2f})")
            
            # Determine result
            if target_hit and stop_hit:
                result = "LOSS (both hit - conservative)"
                label = 0
            elif target_hit:
                result = "WIN (target hit first)"
                label = 1
            elif stop_hit:
                result = "LOSS (stop hit)"
                label = 0
            else:
                result = "LOSS (timeout)"
                label = 0
                
            print(f"   Result: {result} (label = {label})")
            print(f"   Expected: {scenario['expected_result']}")
            
            # Check if result matches expectation
            if (label == 1 and 'WIN' in scenario['expected_result']) or \
               (label == 0 and 'LOSS' in scenario['expected_result']):
                print("   ✅ CORRECT")
            else:
                print("   ❌ INCORRECT!")
    
    def analyze_market_bias(self):
        """Analyze if market has natural bias that affects short trades"""
        print("\n🔍 MARKET BIAS ANALYSIS")
        print("=" * 60)
        
        print("Potential reasons for high short win rates:")
        print("1. 📈 BULL MARKET BIAS:")
        print("   - In trending up markets, pullbacks are common")
        print("   - Short trades might catch these pullbacks")
        print("   - Even in uptrends, intraday reversions happen")
        
        print("\n2. ⏰ INTRADAY MEAN REVERSION:")
        print("   - ES often reverts to mean intraday")
        print("   - Short trades benefit from this reversion")
        print("   - Especially during 2011 (post-crisis recovery)")
        
        print("\n3. 📊 VOLATILITY REGIME:")
        print("   - 2011 had high volatility periods")
        print("   - High volatility = more mean reversion")
        print("   - Short trades capture downward moves better")
        
        print("\n4. 🎯 TARGET/STOP RATIO:")
        print("   - 2:1 reward/risk ratio")
        print("   - Smaller targets easier to hit")
        print("   - Market might have natural downward bias intraday")
        
        print("\n5. ⚠️ POTENTIAL ISSUES:")
        print("   - Contract rollover effects")
        print("   - Data quality issues")
        print("   - Labeling logic bugs")
        print("   - Survivorship bias in data")
    
    def run_comprehensive_debug(self):
        """Run complete debugging analysis"""
        print("🚨 SHORT LABELING LOGIC DEBUGGER")
        print("=" * 60)
        
        # Test all three volatility modes
        for mode in ['low_vol_short', 'normal_vol_short', 'high_vol_short']:
            self.test_short_logic(mode)
            print("\n" + "-" * 40)
        
        self.analyze_market_bias()
        
        print("\n" + "=" * 60)
        print("📋 DEBUGGING RECOMMENDATIONS:")
        print("1. Check actual market data for 2011-06")
        print("2. Verify contract rollover handling")
        print("3. Compare with long trade win rates")
        print("4. Analyze different time periods")
        print("5. Check for data quality issues")
        print("6. Validate against known market conditions")


def main():
    """Main execution"""
    debugger = ShortLabelingDebugger()
    debugger.run_comprehensive_debug()


if __name__ == "__main__":
    main()