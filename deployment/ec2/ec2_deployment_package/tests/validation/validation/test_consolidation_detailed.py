"""
Detailed test for consolidation range features with specific scenarios
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.data_pipeline.features import add_consolidation_features


def create_consolidation_test_data():
    """Create test data with known consolidation patterns"""
    
    # Create a scenario with clear consolidation
    timestamps = pd.date_range('2025-01-15 09:30:00', periods=1000, freq='1S')
    
    # Create price pattern: consolidation then breakout
    prices = []
    base_price = 4750.0
    
    # First 400 bars: tight consolidation around 4750
    for i in range(400):
        noise = np.random.normal(0, 0.2)  # Tight range
        prices.append(base_price + noise)
    
    # Next 300 bars: wider consolidation 4748-4752
    for i in range(300):
        noise = np.random.uniform(-2, 2)  # Wider range
        prices.append(base_price + noise)
    
    # Last 300 bars: breakout and trend
    for i in range(300):
        trend = i * 0.01  # Upward trend
        noise = np.random.normal(0, 0.5)
        prices.append(base_price + trend + noise)
    
    # Create OHLCV data
    data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        # Add some intrabar range
        high = price + abs(np.random.normal(0, 0.1))
        low = price - abs(np.random.normal(0, 0.1))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(800, 1200)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_consolidation_scenarios():
    """Test consolidation features with specific scenarios"""
    print("Testing consolidation features with realistic scenarios...")
    
    df = create_consolidation_test_data()
    df_with_features = add_consolidation_features(df.copy())
    
    print(f"Created test data with {len(df)} bars")
    
    # Test different phases
    tight_phase = df_with_features.iloc[350:400]  # Tight consolidation phase
    wide_phase = df_with_features.iloc[600:650]   # Wider consolidation phase  
    trend_phase = df_with_features.iloc[900:950]  # Trending phase
    
    print("\n=== Tight Consolidation Phase (bars 350-400) ===")
    print(f"Avg short range size: {tight_phase['short_range_size'].mean():.2f}")
    print(f"Avg medium range size: {tight_phase['medium_range_size'].mean():.2f}")
    print(f"Avg compression ratio: {tight_phase['range_compression_ratio'].mean():.2f}")
    print(f"Avg short retouches: {tight_phase['short_range_retouches'].mean():.1f}")
    
    print("\n=== Wide Consolidation Phase (bars 600-650) ===")
    print(f"Avg short range size: {wide_phase['short_range_size'].mean():.2f}")
    print(f"Avg medium range size: {wide_phase['medium_range_size'].mean():.2f}")
    print(f"Avg compression ratio: {wide_phase['range_compression_ratio'].mean():.2f}")
    print(f"Avg short retouches: {wide_phase['short_range_retouches'].mean():.1f}")
    
    print("\n=== Trending Phase (bars 900-950) ===")
    print(f"Avg short range size: {trend_phase['short_range_size'].mean():.2f}")
    print(f"Avg medium range size: {trend_phase['medium_range_size'].mean():.2f}")
    print(f"Avg compression ratio: {trend_phase['range_compression_ratio'].mean():.2f}")
    print(f"Avg short retouches: {trend_phase['short_range_retouches'].mean():.1f}")
    
    # Test fade zone identification
    print("\n=== Fade Zone Analysis ===")
    fade_top = (df_with_features['position_in_short_range'] > 0.85).sum()
    fade_bottom = (df_with_features['position_in_short_range'] < 0.15).sum()
    middle_zone = ((df_with_features['position_in_short_range'] >= 0.15) & 
                   (df_with_features['position_in_short_range'] <= 0.85)).sum()
    
    total_valid = (~df_with_features['position_in_short_range'].isna()).sum()
    
    print(f"Top fade zone (>85%): {fade_top} bars ({fade_top/total_valid*100:.1f}%)")
    print(f"Bottom fade zone (<15%): {fade_bottom} bars ({fade_bottom/total_valid*100:.1f}%)")
    print(f"Middle zone (15-85%): {middle_zone} bars ({middle_zone/total_valid*100:.1f}%)")
    
    # Validate compression ratio behavior
    print("\n=== Compression Ratio Validation ===")
    high_compression = (df_with_features['range_compression_ratio'] < 0.5).sum()
    normal_compression = ((df_with_features['range_compression_ratio'] >= 0.5) & 
                         (df_with_features['range_compression_ratio'] <= 0.8)).sum()
    low_compression = (df_with_features['range_compression_ratio'] > 0.8).sum()
    
    total_ratios = (~df_with_features['range_compression_ratio'].isna()).sum()
    
    print(f"High compression (<0.5): {high_compression} bars ({high_compression/total_ratios*100:.1f}%)")
    print(f"Normal compression (0.5-0.8): {normal_compression} bars ({normal_compression/total_ratios*100:.1f}%)")
    print(f"Low compression (>0.8): {low_compression} bars ({low_compression/total_ratios*100:.1f}%)")
    
    print("\n✓ Consolidation features detailed test completed!")
    return df_with_features


if __name__ == "__main__":
    test_consolidation_scenarios()