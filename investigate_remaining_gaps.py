"""
Investigate the remaining 16 price gaps after contract filtering
"""
import pandas as pd
import pytz

def investigate_remaining_gaps():
    """Check what's causing the remaining gaps and extreme win rates"""
    
    print("=" * 80)
    print("INVESTIGATING REMAINING GAPS - July 2010")
    print("=" * 80)
    
    # Load filtered data
    filtered_path = r"C:\Users\jdotzler\Desktop\monthly_2010-07_contract_filtered.parquet"
    print(f"\n📖 Loading: {filtered_path}")
    
    df = pd.read_parquet(filtered_path)
    print(f"   Loaded: {len(df):,} rows")
    
    # Sort by timestamp
    df = df.sort_values('timestamp').copy()
    df['price_change'] = df['close'].diff()
    
    # Find remaining large gaps
    large_gaps = df[abs(df['price_change']) > 5.0].copy()
    
    print(f"\n🔍 REMAINING LARGE GAPS: {len(large_gaps)}")
    
    if len(large_gaps) > 0:
        print(f"\n   All {len(large_gaps)} gaps:")
        print(f"   {'Index':<8} {'Timestamp':<30} {'Price Change':>15} {'Close':>10}")
        print(f"   {'-'*8} {'-'*30} {'-'*15} {'-'*10}")
        
        for idx, row in large_gaps.iterrows():
            print(f"   {idx:<8} {str(row['timestamp']):<30} {row['price_change']:>15.2f} {row['close']:>10.2f}")
    
    # Check overall price movement
    print(f"\n📊 OVERALL PRICE ANALYSIS:")
    print(f"   Start price: {df['close'].iloc[0]:.2f}")
    print(f"   End price: {df['close'].iloc[-1]:.2f}")
    print(f"   Net change: {df['close'].iloc[-1] - df['close'].iloc[0]:.2f} points")
    print(f"   Min price: {df['close'].min():.2f}")
    print(f"   Max price: {df['close'].max():.2f}")
    print(f"   Range: {df['close'].max() - df['close'].min():.2f} points")
    
    # Check for systematic bias
    print(f"\n📈 DIRECTIONAL BIAS:")
    df['bar_direction'] = (df['close'] > df['open']).astype(int)  # 1 = up, 0 = down
    up_bars = (df['bar_direction'] == 1).sum()
    down_bars = (df['bar_direction'] == 0).sum()
    
    print(f"   Up bars: {up_bars:,} ({up_bars/len(df)*100:.1f}%)")
    print(f"   Down bars: {down_bars:,} ({down_bars/len(df)*100:.1f}%)")
    print(f"   Up/Down ratio: {up_bars/down_bars:.2f}")
    
    # Check high/low patterns
    print(f"\n🎯 HIGH/LOW ANALYSIS:")
    df['high_low_range'] = df['high'] - df['low']
    print(f"   Avg bar range: {df['high_low_range'].mean():.2f} points")
    print(f"   Median bar range: {df['high_low_range'].median():.2f} points")
    print(f"   Max bar range: {df['high_low_range'].max():.2f} points")
    
    # Sample some bars to manually verify
    print(f"\n📝 MANUAL VERIFICATION - Random Sample:")
    sample_indices = df[df.index < len(df) - 900].sample(min(3, len(df)-900)).index
    
    for idx in sample_indices:
        bar = df.loc[idx]
        next_900 = df.loc[idx:idx+900]
        
        print(f"\n   Bar {idx} - {bar['timestamp']}")
        print(f"      Entry: O={bar['open']:.2f}, H={bar['high']:.2f}, L={bar['low']:.2f}, C={bar['close']:.2f}")
        
        # Low vol long (6 tick stop, 12 tick target)
        entry = bar['close']
        long_target = entry + 3.0
        long_stop = entry - 1.5
        short_target = entry - 3.0
        short_stop = entry + 1.5
        
        # Check long trade
        long_target_hit = (next_900['high'] >= long_target).any()
        long_stop_hit = (next_900['low'] <= long_stop).any()
        
        if long_target_hit and long_stop_hit:
            target_bar = next_900[next_900['high'] >= long_target].index[0]
            stop_bar = next_900[next_900['low'] <= long_stop].index[0]
            long_result = "WIN" if target_bar < stop_bar else "LOSS"
        elif long_target_hit:
            long_result = "WIN"
        elif long_stop_hit:
            long_result = "LOSS"
        else:
            long_result = "TIMEOUT"
        
        # Check short trade
        short_target_hit = (next_900['low'] <= short_target).any()
        short_stop_hit = (next_900['high'] >= short_stop).any()
        
        if short_target_hit and short_stop_hit:
            target_bar = next_900[next_900['low'] <= short_target].index[0]
            stop_bar = next_900[next_900['high'] >= short_stop].index[0]
            short_result = "WIN" if target_bar < stop_bar else "LOSS"
        elif short_target_hit:
            short_result = "WIN"
        elif short_stop_hit:
            short_result = "LOSS"
        else:
            short_result = "TIMEOUT"
        
        print(f"      Long: Target={long_target:.2f}, Stop={long_stop:.2f} → {long_result}")
        print(f"      Short: Target={short_target:.2f}, Stop={short_stop:.2f} → {short_result}")
        
        # Show next few bars
        print(f"      Next 5 bars:")
        for i in range(min(5, len(next_900))):
            next_bar = next_900.iloc[i]
            print(f"         {i}: H={next_bar['high']:.2f}, L={next_bar['low']:.2f}, C={next_bar['close']:.2f}")
    
    print(f"\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    
    print(f"\n💡 HYPOTHESIS:")
    print(f"   If shorts are winning 87% and longs only 3%, this suggests:")
    print(f"   1. Market had strong downward bias in July 2010")
    print(f"   2. OR there's still a labeling logic issue")
    print(f"   3. OR the remaining gaps are still causing problems")
    
    return df, large_gaps

if __name__ == "__main__":
    df, gaps = investigate_remaining_gaps()
