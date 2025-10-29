#!/usr/bin/env python3
"""
Test script to verify contract roll detection is working correctly across multiple time periods
"""

import pandas as pd
import numpy as np
from project.data_pipeline.weighted_labeling import process_weighted_labeling, LabelCalculator, TRADING_MODES

def analyze_price_movement(df, period_name):
    """Analyze price movement characteristics of a dataset"""
    print(f"\n=== {period_name.upper()} PRICE ANALYSIS ===")
    
    # Basic stats
    price_range = df['close'].max() - df['close'].min()
    time_span = df['timestamp'].max() - df['timestamp'].min()
    minutes = time_span.total_seconds() / 60
    
    print(f"Time span: {minutes:.1f} minutes")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f} ({price_range:.2f} points)")
    
    # Movement analysis
    price_changes = df['close'].diff().dropna()
    up_moves = (price_changes > 0).sum()
    down_moves = (price_changes < 0).sum()
    flat_moves = (price_changes == 0).sum()
    
    print(f"Price movements: {up_moves} up ({up_moves/len(price_changes)*100:.1f}%), "
          f"{down_moves} down ({down_moves/len(price_changes)*100:.1f}%), "
          f"{flat_moves} flat ({flat_moves/len(price_changes)*100:.1f}%)")
    
    # Net movement
    net_change = df['close'].iloc[-1] - df['close'].iloc[0]
    trend = "UP" if net_change > 5 else "DOWN" if net_change < -5 else "SIDEWAYS"
    print(f"Net change: {net_change:.2f} points ({trend})")
    
    return {
        'price_range': price_range,
        'minutes': minutes,
        'net_change': net_change,
        'trend': trend
    }

def test_period(df, period_name, roll_threshold=20.0):
    """Test roll detection and labeling on a specific time period"""
    print(f"\n{'='*60}")
    print(f"TESTING {period_name.upper()}")
    print(f"{'='*60}")
    
    # Analyze price movement first
    movement_stats = analyze_price_movement(df, period_name)
    
    # Test roll detection
    mode = TRADING_MODES['low_vol_long']
    calculator = LabelCalculator(mode, roll_detection_threshold=roll_threshold)
    roll_bars = calculator._detect_contract_rolls(df)
    
    print(f"\nRoll detection (threshold={roll_threshold}):")
    print(f"  Bars affected by rolls: {roll_bars.sum()} ({roll_bars.sum()/len(df)*100:.1f}%)")
    
    # Show roll examples if any
    if roll_bars.sum() > 0:
        affected_indices = np.where(roll_bars)[0]
        print(f"  Roll events detected at bars: {affected_indices[:10].tolist()}{'...' if len(affected_indices) > 10 else ''}")
        
        # Show largest price jumps
        price_changes = df['close'].diff().abs()
        large_moves = price_changes[price_changes > roll_threshold]
        if len(large_moves) > 0:
            print(f"  Largest price jumps: {large_moves.nlargest(3).values}")
    
    # Test labeling with roll detection
    try:
        print(f"\nRunning weighted labeling...")
        result_df = process_weighted_labeling(df)
        
        print(f"\nWin rates with roll detection:")
        total_winners = 0
        for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                     'low_vol_short', 'normal_vol_short', 'high_vol_short']:
            label_col = f'label_{mode}'
            if label_col in result_df.columns:
                win_rate = result_df[label_col].mean()
                winners = result_df[label_col].sum()
                total_winners += winners
                print(f"  {mode}: {win_rate:.1%} ({winners} winners)")
        
        print(f"\nTotal winners across all modes: {total_winners}")
        
        # Calculate expected vs actual based on market conditions
        expected_performance = "Higher win rates expected" if movement_stats['trend'] != "SIDEWAYS" else "Lower win rates expected (sideways market)"
        print(f"Market assessment: {expected_performance}")
        
        return {
            'period': period_name,
            'total_winners': total_winners,
            'roll_affected_bars': roll_bars.sum(),
            'movement_stats': movement_stats,
            'win_rates': {mode: result_df[f'label_{mode}'].mean() for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 'low_vol_short', 'normal_vol_short', 'high_vol_short']}
        }
        
    except Exception as e:
        print(f"❌ Error during labeling: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_roll_detection():
    """Test the roll detection functionality across multiple time periods"""
    
    print("=== MULTI-PERIOD CONTRACT ROLL DETECTION TEST ===")
    
    results = []
    
    # Test Period 1: Original data (Sept 22, 2025 - known to have roll issues)
    print("\n" + "="*80)
    print("LOADING TEST DATASETS")
    print("="*80)
    
    try:
        # Load original problematic data
        df1 = pd.read_parquet('project/data/test/weighted_labeling_test_results.parquet')
        # Keep only OHLCV columns and clean
        ohlcv_cols = ['timestamp', 'rtype', 'publisher_id', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        df1_clean = df1[ohlcv_cols].copy()
        
        print(f"✅ Period 1 loaded: {len(df1_clean)} bars from Sept 22")
        
        # Load different day data
        df2 = pd.read_parquet('project/data/test/different_day_clean_1000.parquet')
        print(f"✅ Period 2 loaded: {len(df2)} bars from Sept 25")
        
        # Load October 14 high-activity data
        df3 = pd.read_parquet('project/data/test/oct14_high_activity_1000.parquet')
        print(f"✅ Period 3 loaded: {len(df3)} bars from Oct 14 (High Activity Day)")
        
        # Test all three periods
        result1 = test_period(df1_clean, "September 22 (Original - Roll Issues Expected)")
        result2 = test_period(df2, "September 25 (Different Day)")
        result3 = test_period(df3, "October 14 (High Activity Day)")
        
        if result1:
            results.append(result1)
        if result2:
            results.append(result2)
        if result3:
            results.append(result3)
        
        # Comparison analysis
        if len(results) >= 2:
            print(f"\n{'='*80}")
            print("COMPARATIVE ANALYSIS")
            print(f"{'='*80}")
            
            for i, result in enumerate(results, 1):
                print(f"\nPeriod {i}: {result['period']}")
                print(f"  Market trend: {result['movement_stats']['trend']}")
                print(f"  Price range: {result['movement_stats']['price_range']:.1f} points")
                print(f"  Roll-affected bars: {result['roll_affected_bars']} ({result['roll_affected_bars']/1000*100:.1f}%)")
                print(f"  Total winners: {result['total_winners']}")
                
                # Show best performing modes
                best_long = max(result['win_rates']['low_vol_long'], result['win_rates']['normal_vol_long'], result['win_rates']['high_vol_long'])
                best_short = max(result['win_rates']['low_vol_short'], result['win_rates']['normal_vol_short'], result['win_rates']['high_vol_short'])
                print(f"  Best long win rate: {best_long:.1%}")
                print(f"  Best short win rate: {best_short:.1%}")
            
            # Key insights
            print(f"\n🔍 KEY INSIGHTS:")
            
            # Compare all periods
            for i in range(1, len(results)):
                roll_diff = results[i]['roll_affected_bars'] - results[0]['roll_affected_bars']
                winner_diff = results[i]['total_winners'] - results[0]['total_winners']
                
                print(f"• Period {i+1} vs Period 1:")
                print(f"  - Roll detection difference: {roll_diff:+d} bars")
                print(f"  - Winner difference: {winner_diff:+d} total winners")
                
                if results[i]['movement_stats']['price_range'] > results[0]['movement_stats']['price_range']:
                    range_diff = results[i]['movement_stats']['price_range'] - results[0]['movement_stats']['price_range']
                    print(f"  - Price range: {range_diff:+.1f} points difference")
            
            # Overall assessment
            cleanest_period = min(results, key=lambda x: x['roll_affected_bars'])
            most_winners = max(results, key=lambda x: x['total_winners'])
            
            print(f"\n📊 SUMMARY:")
            print(f"• Cleanest data: {cleanest_period['period']} ({cleanest_period['roll_affected_bars']} roll-affected bars)")
            print(f"• Most opportunities: {most_winners['period']} ({most_winners['total_winners']} total winners)")
            
            if cleanest_period == most_winners:
                print("• ✅ Clean data correlates with more trading opportunities")
            else:
                print("• ⚠️ Different periods show varying opportunity levels despite roll filtering")
        
        print(f"\n✅ Multi-period roll detection test completed successfully")
        
    except Exception as e:
        print(f"❌ Error during multi-period test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_roll_detection()