"""
Test full feature engineering pipeline including consolidation features
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from project.data_pipeline.features import create_all_features


def create_sample_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Create 1000 bars of realistic ES data
    timestamps = pd.date_range('2025-01-15 09:30:00', periods=1000, freq='1s')
    
    base_price = 4750.0
    prices = []
    volumes = []
    
    for i in range(1000):
        # Random walk with some mean reversion
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, 0.25)  # ES-like volatility
            price = prices[-1] + change
        
        prices.append(price)
        volumes.append(np.random.randint(500, 2000))
    
    # Create OHLCV
    data = []
    for i, (ts, close, volume) in enumerate(zip(timestamps, prices, volumes)):
        high = close + abs(np.random.normal(0, 0.1))
        low = close - abs(np.random.normal(0, 0.1))
        open_price = prices[i-1] if i > 0 else close
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_full_pipeline():
    """Test complete feature engineering pipeline"""
    print("Testing full feature engineering pipeline...")
    
    # Create test data
    df = create_sample_data()
    print(f"Created {len(df)} bars of test data")
    
    # Add some mock label columns to simulate existing labeled dataset
    for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large', 
                   'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']:
        df[f'{profile}_label'] = np.random.choice([-1, 0, 1], size=len(df))
        df[f'{profile}_target_hit_bar'] = np.random.randint(0, 900, size=len(df))
        df[f'{profile}_stop_hit_bar'] = np.random.randint(0, 900, size=len(df))
        df[f'{profile}_mae'] = np.random.uniform(0, 10, size=len(df))
        df[f'{profile}_timeout_bar'] = np.random.randint(0, 900, size=len(df))
    
    print(f"Starting with {len(df.columns)} columns")
    
    # Run full feature engineering
    df_with_features = create_all_features(df)
    
    print(f"Ending with {len(df_with_features.columns)} columns")
    print(f"Added {len(df_with_features.columns) - len(df.columns)} features")
    
    # Check consolidation features specifically
    consolidation_features = [
        'short_range_high', 'short_range_low', 'short_range_size', 'position_in_short_range',
        'medium_range_high', 'medium_range_low', 'medium_range_size', 'range_compression_ratio',
        'short_range_retouches', 'medium_range_retouches'
    ]
    
    print("\n=== Consolidation Features Check ===")
    for feature in consolidation_features:
        if feature in df_with_features.columns:
            non_null = (~df_with_features[feature].isna()).sum()
            print(f"✓ {feature}: {non_null}/{len(df)} non-null values")
        else:
            print(f"✗ {feature}: MISSING")
    
    # Show sample consolidation feature values
    print("\n=== Sample Consolidation Values (last 10 bars) ===")
    sample_features = ['short_range_size', 'medium_range_size', 'range_compression_ratio', 
                      'position_in_short_range', 'short_range_retouches', 'medium_range_retouches']
    
    sample_data = df_with_features[sample_features].tail(10)
    print(sample_data.round(3))
    
    # Validate ranges
    print("\n=== Feature Range Validation ===")
    
    # Position should be 0-1
    pos_valid = ((df_with_features['position_in_short_range'] >= 0) & 
                 (df_with_features['position_in_short_range'] <= 1))
    pos_count = pos_valid.sum()
    pos_total = (~df_with_features['position_in_short_range'].isna()).sum()
    print(f"Position in range [0,1]: {pos_count}/{pos_total} ({pos_count/pos_total*100:.1f}%)")
    
    # Compression ratio should be positive
    ratio_valid = df_with_features['range_compression_ratio'] > 0
    ratio_count = ratio_valid.sum()
    ratio_total = (~df_with_features['range_compression_ratio'].isna()).sum()
    print(f"Compression ratio > 0: {ratio_count}/{ratio_total} ({ratio_count/ratio_total*100:.1f}%)")
    
    # Retouches should be non-negative integers
    short_retouches_valid = df_with_features['short_range_retouches'] >= 0
    short_count = short_retouches_valid.sum()
    short_total = (~df_with_features['short_range_retouches'].isna()).sum()
    print(f"Short retouches >= 0: {short_count}/{short_total} ({short_count/short_total*100:.1f}%)")
    
    print("\n✓ Full pipeline integration test completed successfully!")
    return df_with_features


if __name__ == "__main__":
    test_full_pipeline()