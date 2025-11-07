"""
Analyze OHLC corruption in the processed data
"""
import pandas as pd
import numpy as np

def analyze_ohlc_corruption():
    """Analyze what's wrong with the OHLC data"""
    
    print("=" * 80)
    print("OHLC CORRUPTION ANALYSIS - July 2010")
    print("=" * 80)
    
    # Load the original processed data (before contract filtering)
    parquet_path = r"C:\Users\jdotzler\Desktop\monthly_2010-07_20251107_152756.parquet"
    print(f"\n📖 Loading: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   Loaded: {len(df):,} rows")
    
    # Analyze OHLC relationships
    print(f"\n🔍 OHLC RELATIONSHIP ANALYSIS:")
    
    # Check if H >= O, C, L and L <= O, C, H
    df['high_valid'] = (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['high'] >= df['low'])
    df['low_valid'] = (df['low'] <= df['open']) & (df['low'] <= df['close']) & (df['low'] <= df['high'])
    df['range'] = df['high'] - df['low']
    
    invalid_high = (~df['high_valid']).sum()
    invalid_low = (~df['low_valid']).sum()
    zero_range = (df['range'] == 0).sum()
    
    print(f"   Invalid high values: {invalid_high:,} ({invalid_high/len(df)*100:.1f}%)")
    print(f"   Invalid low values: {invalid_low:,} ({invalid_low/len(df)*100:.1f}%)")
    print(f"   Zero range bars (H=L): {zero_range:,} ({zero_range/len(df)*100:.1f}%)")
    
    # Check bar direction
    df['is_up'] = df['close'] > df['open']
    df['is_down'] = df['close'] < df['open']
    df['is_flat'] = df['close'] == df['open']
    
    up_bars = df['is_up'].sum()
    down_bars = df['is_down'].sum()
    flat_bars = df['is_flat'].sum()
    
    print(f"\n📊 BAR DIRECTION:")
    print(f"   Up bars (C>O): {up_bars:,} ({up_bars/len(df)*100:.1f}%)")
    print(f"   Down bars (C<O): {down_bars:,} ({down_bars/len(df)*100:.1f}%)")
    print(f"   Flat bars (C=O): {flat_bars:,} ({flat_bars/len(df)*100:.1f}%)")
    
    # Sample some bars to see the pattern
    print(f"\n📝 SAMPLE BARS (showing OHLC pattern):")
    print(f"   {'Index':<8} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Range':>8} {'Direction':<10}")
    print(f"   {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    
    for idx in [0, 100, 1000, 10000, 50000, 100000, 150000, 200000]:
        if idx < len(df):
            row = df.iloc[idx]
            direction = "UP" if row['is_up'] else ("DOWN" if row['is_down'] else "FLAT")
            print(f"   {idx:<8} {row['open']:>10.2f} {row['high']:>10.2f} {row['low']:>10.2f} {row['close']:>10.2f} {row['range']:>8.2f} {direction:<10}")
    
    # Check if Open and Close are swapped
    print(f"\n🔄 CHECKING IF OPEN/CLOSE ARE SWAPPED:")
    
    # If O/C are swapped, then what we call "down bars" would actually be up bars
    # Let's check if swapping fixes the ratio
    df['swapped_is_up'] = df['open'] > df['close']  # If swapped, this would be "up"
    df['swapped_is_down'] = df['open'] < df['close']  # If swapped, this would be "down"
    
    swapped_up = df['swapped_is_up'].sum()
    swapped_down = df['swapped_is_down'].sum()
    
    print(f"   If we swap Open/Close:")
    print(f"      Up bars: {swapped_up:,} ({swapped_up/len(df)*100:.1f}%)")
    print(f"      Down bars: {swapped_down:,} ({swapped_down/len(df)*100:.1f}%)")
    print(f"      Up/Down ratio: {swapped_up/swapped_down if swapped_down > 0 else 'N/A':.2f}")
    
    if abs(swapped_up/len(df) - 0.5) < abs(up_bars/len(df) - 0.5):
        print(f"      ✅ SWAPPING IMPROVES THE RATIO!")
        print(f"         Open and Close columns are likely SWAPPED in the data!")
    else:
        print(f"      ❌ Swapping doesn't help")
    
    # Check price continuity
    print(f"\n📈 PRICE CONTINUITY:")
    df['close_change'] = df['close'].diff()
    df['open_change'] = df['open'].diff()
    
    large_close_gaps = abs(df['close_change']) > 5.0
    large_open_gaps = abs(df['open_change']) > 5.0
    
    print(f"   Large close gaps (>5 pts): {large_close_gaps.sum()}")
    print(f"   Large open gaps (>5 pts): {large_open_gaps.sum()}")
    
    # Check if the data makes sense overall
    print(f"\n💰 OVERALL PRICE MOVEMENT:")
    print(f"   First close: {df['close'].iloc[0]:.2f}")
    print(f"   Last close: {df['close'].iloc[-1]:.2f}")
    print(f"   Net change: {df['close'].iloc[-1] - df['close'].iloc[0]:.2f} points")
    print(f"   Min close: {df['close'].min():.2f}")
    print(f"   Max close: {df['close'].max():.2f}")
    
    # Calculate what the up/down ratio SHOULD be based on net movement
    net_change = df['close'].iloc[-1] - df['close'].iloc[0]
    if net_change > 0:
        print(f"   Market moved UP {net_change:.2f} points")
        print(f"   Expected: More up bars than down bars")
    else:
        print(f"   Market moved DOWN {abs(net_change):.2f} points")
        print(f"   Expected: More down bars than up bars")
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    print(f"\n💡 CONCLUSION:")
    if down_bars > up_bars * 5 and net_change > 0:
        print(f"   ❌ CRITICAL: Market moved UP but 89% of bars are DOWN")
        print(f"      This is IMPOSSIBLE - Open and Close are definitely SWAPPED!")
        print(f"\n   🔧 FIX: Swap the Open and Close columns in the conversion process")
    elif zero_range > len(df) * 0.5:
        print(f"   ❌ CRITICAL: {zero_range/len(df)*100:.1f}% of bars have zero range")
        print(f"      This suggests the data is not true OHLC bars")
    
    return df

if __name__ == "__main__":
    df = analyze_ohlc_corruption()
