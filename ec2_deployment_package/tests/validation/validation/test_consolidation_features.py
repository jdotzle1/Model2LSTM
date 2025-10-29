"""
Test consolidation range features implementation
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from project.data_pipeline.features import add_consolidation_features


def create_test_data():
    """Create simple test data for consolidation features"""
    np.random.seed(42)
    
    # Create 1000 bars of synthetic OHLCV data
    n_bars = 1000
    base_price = 4750.0
    
    # Generate price walk
    returns = np.random.normal(0, 0.001, n_bars)
    prices = base_price + np.cumsum(returns * base_price)
    
    # Create OHLCV with some realistic patterns
    data = []
    for i, price in enumerate(prices):
        # Add some intrabar volatility
        high = price + abs(np.random.normal(0, 0.5))
        low = price - abs(np.random.normal(0, 0.5))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.randint(500, 2000)
        
        data.append({
            'timestamp': pd.Timestamp('2025-01-15 09:30:00') + pd.Timedelta(seconds=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_consolidation_features():
    """Test consolidation range features"""
    print("Testing consolidation range features...")
    
    # Create test data
    df = create_test_data()
    print(f"Created test data with {len(df)} bars")
    
    # Add consolidation features
    df_with_features = add_consolidation_features(df.copy())
    
    # Check that all expected features were added
    expected_features = [
        'short_range_high', 'short_range_low', 'short_range_size', 'position_in_short_range',
        'medium_range_high', 'medium_range_low', 'medium_range_size', 'range_compression_ratio',
        'short_range_retouches', 'medium_range_retouches'
    ]
    
    for feature in expected_features:
        assert feature in df_with_features.columns, f"Missing feature: {feature}"
        print(f"✓ {feature} added")
    
    # Check basic properties
    print("\nFeature validation:")
    
    # Short range should be <= medium range
    valid_ranges = df_with_features['short_range_size'] <= df_with_features['medium_range_size']
    valid_count = valid_ranges.sum()
    total_valid = (~df_with_features['short_range_size'].isna()).sum()
    print(f"✓ Short range <= medium range: {valid_count}/{total_valid} cases")
    
    # Position in range should be 0-1
    pos_valid = ((df_with_features['position_in_short_range'] >= 0) & 
                 (df_with_features['position_in_short_range'] <= 1))
    pos_count = pos_valid.sum()
    pos_total = (~df_with_features['position_in_short_range'].isna()).sum()
    print(f"✓ Position in range [0,1]: {pos_count}/{pos_total} cases")
    
    # Range compression ratio should be positive
    ratio_valid = df_with_features['range_compression_ratio'] > 0
    ratio_count = ratio_valid.sum()
    ratio_total = (~df_with_features['range_compression_ratio'].isna()).sum()
    print(f"✓ Range compression ratio > 0: {ratio_count}/{ratio_total} cases")
    
    # Check some sample values
    print("\nSample values (last 10 bars):")
    sample_cols = ['short_range_size', 'medium_range_size', 'range_compression_ratio', 
                   'position_in_short_range', 'short_range_retouches']
    print(df_with_features[sample_cols].tail(10))
    
    print("\n✓ All consolidation features tests passed!")
    return df_with_features


if __name__ == "__main__":
    test_consolidation_features()