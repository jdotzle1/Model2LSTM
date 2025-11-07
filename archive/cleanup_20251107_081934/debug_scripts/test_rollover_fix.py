#!/usr/bin/env python3
"""
Test Rollover Threshold Fix

This script tests the impact of lowering the rollover detection threshold
from 20.0 to 2.0 points on the short win rates.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.weighted_labeling import TRADING_MODES, LabelCalculator

def test_rollover_thresholds():
    """Test different rollover thresholds on sample data"""
    
    print("🧪 TESTING ROLLOVER THRESHOLD FIX")
    print("=" * 60)
    
    # Create sample data with rollover gap
    sample_data = create_sample_data_with_rollover()
    
    print(f"📊 Sample data: {len(sample_data)} bars")
    print(f"   Includes artificial rollover gap at bar 50")
    
    # Test with different thresholds
    thresholds = [20.0, 10.0, 5.0, 2.0, 1.0]
    
    for threshold in thresholds:
        print(f"\n🔍 Testing threshold: {threshold} points")
        
        # Test short mode
        mode = TRADING_MODES['normal_vol_short']
        calculator = LabelCalculator(mode, roll_detection_threshold=threshold)
        
        labels, mae_ticks, seconds_to_target = calculator.calculate_labels(sample_data)
        
        win_rate = labels.mean()
        total_wins = labels.sum()
        
        print(f"   Win rate: {win_rate:.1%} ({total_wins} wins out of {len(labels)} trades)")
        
        # Check rollover detection
        if hasattr(calculator, '_rollover_stats'):
            stats = calculator._rollover_stats
            print(f"   Rollover events detected: {stats.get('roll_events_detected', 0)}")
            print(f"   Bars affected: {stats.get('bars_affected', 0)}")
    
    print(f"\n🎯 EXPECTED RESULT:")
    print(f"   Lower thresholds should detect more rollover events")
    print(f"   More rollover detection → lower short win rates")
    print(f"   Threshold 2.0 should be optimal for ES data")

def create_sample_data_with_rollover():
    """Create sample data with artificial rollover gap"""
    
    np.random.seed(42)  # Reproducible results
    
    # Create 100 bars of sample data
    n_bars = 100
    base_price = 4750.0
    
    timestamps = pd.date_range('2011-06-10 09:30:00', periods=n_bars, freq='1S')
    
    # Generate realistic price movement
    price_changes = np.random.normal(0, 0.5, n_bars)  # Small random moves
    
    # Add artificial rollover gap at bar 50
    price_changes[50] = -8.0  # 8-point downward gap (typical rollover)
    
    # Calculate prices
    prices = base_price + np.cumsum(price_changes)
    
    # Create OHLCV data
    data = []
    for i in range(n_bars):
        # Simple OHLC based on price
        open_price = prices[i-1] if i > 0 else base_price
        close_price = prices[i]
        
        # Add some intrabar movement
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.25))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.25))
        
        data.append({
            'timestamp': timestamps[i],
            'open': round(open_price / 0.25) * 0.25,  # Round to tick size
            'high': round(high_price / 0.25) * 0.25,
            'low': round(low_price / 0.25) * 0.25,
            'close': round(close_price / 0.25) * 0.25,
            'volume': np.random.randint(100, 1000)
        })
    
    df = pd.DataFrame(data)
    
    # Show the rollover gap
    print(f"\n📉 ARTIFICIAL ROLLOVER GAP:")
    gap_area = df.iloc[48:53]
    for idx, row in gap_area.iterrows():
        marker = " 🔄 ROLLOVER" if idx == 50 else ""
        print(f"   Bar {idx}: {row['close']:.2f}{marker}")
    
    return df

def generate_reprocessing_commands():
    """Generate commands to reprocess June 2011 with new threshold"""
    
    print(f"\n🚀 REPROCESSING COMMANDS")
    print("=" * 60)
    
    commands = [
        "# The rollover threshold has been changed from 20.0 to 2.0 points",
        "# Now reprocess June 2011 to see the impact",
        "",
        "# Delete existing June 2011 processed data",
        "aws s3 rm s3://es-1-second-data/processed-data/monthly/2011/06/ --recursive --region us-east-1",
        "",
        "# Reprocess with new rollover threshold",
        "python3 process_monthly_chunks_fixed.py --test-month 2011-06",
        "",
        "# Check the new statistics",
        "aws s3 ls s3://es-1-second-data/processed-data/monthly/2011/06/statistics/ --region us-east-1",
        "",
        "# Download and compare new results",
        "aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/06/statistics/ /tmp/new_stats/ --recursive --region us-east-1",
        "",
        "# Quick comparison",
        "python3 -c \"",
        "import json",
        "from glob import glob",
        "",
        "stats_files = glob('/tmp/new_stats/*.json')",
        "if stats_files:",
        "    with open(stats_files[0]) as f:",
        "        data = json.load(f)",
        "    print('NEW RESULTS (with 2.0 point threshold):') ",
        "    for mode, stats in data.items():",
        "        if 'short' in mode and 'win_rate' in stats:",
        "            print(f'{mode}: {stats[\\\"win_rate\\\"]:.1%}')",
        "else:",
        "    print('No statistics files found')",
        "\""
    ]
    
    for cmd in commands:
        print(cmd)

def main():
    """Main execution"""
    
    # Test with sample data
    test_rollover_thresholds()
    
    # Generate reprocessing commands
    generate_reprocessing_commands()
    
    print(f"\n🎯 SUMMARY:")
    print(f"✅ Rollover threshold changed: 20.0 → 2.0 points")
    print(f"🔄 This should detect ES rollover gaps properly")
    print(f"📉 Expected: Lower short win rates in June 2011")
    print(f"🚀 Next: Run reprocessing commands above")


if __name__ == "__main__":
    main()