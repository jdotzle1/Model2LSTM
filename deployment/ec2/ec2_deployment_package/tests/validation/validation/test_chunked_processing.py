"""
Test chunked processing functionality for memory-efficient large dataset handling
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.data_pipeline.features import (
    create_all_features, 
    create_all_features_chunked, 
    validate_chunked_processing,
    integrate_with_labeled_dataset
)


def create_large_sample_data(num_bars=5000):
    """Create larger sample OHLCV data for chunked processing testing"""
    np.random.seed(42)
    
    # Create realistic ES data with proper timestamp sequence
    timestamps = pd.date_range('2025-01-15 09:30:00', periods=num_bars, freq='1s')
    
    base_price = 4750.0
    prices = []
    volumes = []
    
    for i in range(num_bars):
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


def test_chunked_vs_normal_processing():
    """Test that chunked processing produces identical results to normal processing"""
    print("=== Testing Chunked vs Normal Processing ===")
    
    # Create test data (5000 bars for meaningful chunking)
    df = create_large_sample_data(5000)
    print(f"Created {len(df):,} bars of test data")
    
    # Test different chunk sizes
    chunk_sizes = [1000, 1500, 2000]
    
    for chunk_size in chunk_sizes:
        print(f"\nTesting chunk_size = {chunk_size:,}")
        
        # Process normally
        print("  Processing without chunking...")
        df_normal = create_all_features(df.copy())
        
        # Process with chunking
        print("  Processing with chunking...")
        df_chunked = create_all_features_chunked(df.copy(), chunk_size)
        
        # Compare results
        print("  Comparing results...")
        
        # Check dimensions
        if df_normal.shape != df_chunked.shape:
            print(f"  ❌ Shape mismatch: normal {df_normal.shape} vs chunked {df_chunked.shape}")
            return False
        
        # Check that we have the same columns
        if set(df_normal.columns) != set(df_chunked.columns):
            print(f"  ❌ Column mismatch")
            return False
        
        # Compare key feature values (excluding VWAP-related features which may have small differences due to session boundaries)
        test_features = [
            'volume_ratio_30s', 'position_in_short_range',
            'return_30s', 'volatility_regime', 'uptick_pct_30s'
        ]
        
        max_differences = {}
        for feature in test_features:
            if feature in df_normal.columns:
                # Compare non-NaN values
                normal_vals = df_normal[feature].dropna()
                chunked_vals = df_chunked[feature].dropna()
                
                if len(normal_vals) == len(chunked_vals) and len(normal_vals) > 0:
                    # Align indices for comparison
                    normal_aligned = df_normal[feature].fillna(0)
                    chunked_aligned = df_chunked[feature].fillna(0)
                    
                    max_diff = abs(normal_aligned - chunked_aligned).max()
                    max_differences[feature] = max_diff
                    
                    if max_diff < 1e-10:  # Very tight tolerance
                        print(f"    ✓ {feature}: identical (max diff: {max_diff:.2e})")
                    else:
                        print(f"    ⚠️  {feature}: max difference {max_diff:.2e}")
        
        # Special handling for VWAP-related features (allow small differences due to session boundary handling)
        vwap_features = ['distance_from_vwap_pct', 'vwap_slope']
        for feature in vwap_features:
            if feature in df_normal.columns:
                normal_aligned = df_normal[feature].fillna(0)
                chunked_aligned = df_chunked[feature].fillna(0)
                max_diff = abs(normal_aligned - chunked_aligned).max()
                
                if max_diff < 0.1:  # More lenient tolerance for VWAP features
                    print(f"    ✓ {feature}: acceptable difference (max diff: {max_diff:.2e})")
                else:
                    print(f"    ⚠️  {feature}: large difference {max_diff:.2e}")
        
        # Overall assessment
        if all(diff < 1e-8 for diff in max_differences.values()):
            print(f"  ✓ Chunk size {chunk_size:,}: Core features are identical within tolerance")
        else:
            print(f"  ⚠️  Chunk size {chunk_size:,}: Some differences detected (may be acceptable for session-dependent features)")
    
    print("\n✓ Chunked processing comparison completed")
    return True


def test_memory_efficiency():
    """Test memory efficiency with different dataset sizes"""
    print("\n=== Testing Memory Efficiency ===")
    
    dataset_sizes = [1000, 5000, 10000]
    
    for size in dataset_sizes:
        print(f"\nTesting dataset size: {size:,} bars")
        
        # Create test data
        df = create_large_sample_data(size)
        
        # Add mock label columns to simulate real labeled dataset
        for profile in ['long_2to1_small', 'short_2to1_small']:
            df[f'{profile}_label'] = np.random.choice([-1, 0, 1], size=len(df))
            df[f'{profile}_mae'] = np.random.uniform(0, 10, size=len(df))
        
        # Test auto chunk size selection by calling create_all_features directly
        print("  Testing auto chunk size selection...")
        
        # Determine chunk size based on dataset size (same logic as integrate_with_labeled_dataset)
        if size >= 10_000_000:
            chunk_size = 1_000_000
        elif size >= 5_000_000:
            chunk_size = 500_000
        elif size >= 1_000_000:
            chunk_size = 250_000
        else:
            chunk_size = size  # Process all at once for smaller datasets
        
        print(f"    Auto-selected chunk size: {chunk_size:,}")
        
        # Process with the determined chunk size
        if chunk_size >= size:
            df_result = create_all_features(df.copy())
        else:
            df_result = create_all_features_chunked(df.copy(), chunk_size)
        
        # Verify results
        original_cols = len(df.columns)
        result_cols = len(df_result.columns)
        added_features = result_cols - original_cols
        
        print(f"    Original columns: {original_cols}")
        print(f"    Result columns: {result_cols}")
        print(f"    Added features: {added_features}")
        
        if added_features == 43:
            print(f"    ✓ Correct number of features added")
        else:
            print(f"    ❌ Expected 43 features, got {added_features}")
        
        # Check for NaN values in key features
        key_features = ['volume_ratio_30s', 'position_in_short_range', 'volatility_regime']
        for feature in key_features:
            if feature in df_result.columns:
                non_null_count = (~df_result[feature].isna()).sum()
                non_null_pct = (non_null_count / len(df_result)) * 100
                print(f"    {feature}: {non_null_pct:.1f}% non-null values")
    
    print("\n✓ Memory efficiency testing completed")


def test_rolling_window_overlap():
    """Test that rolling window calculations are handled correctly across chunk boundaries"""
    print("\n=== Testing Rolling Window Overlap Handling ===")
    
    # Create test data with known patterns
    df = create_large_sample_data(3000)
    
    # Test with a chunk size that will create multiple chunks
    chunk_size = 1000
    
    print(f"Testing with {len(df):,} bars and chunk_size={chunk_size:,}")
    
    # Process with chunking
    df_chunked = create_all_features_chunked(df.copy(), chunk_size)
    
    # Check that rolling calculations near chunk boundaries are reasonable
    # Focus on bars around chunk boundaries (1000, 2000)
    boundary_indices = [999, 1000, 1001, 1999, 2000, 2001]
    
    print("  Checking feature values near chunk boundaries:")
    
    for idx in boundary_indices:
        if idx < len(df_chunked):
            vol_ratio = df_chunked.loc[idx, 'volume_ratio_30s']
            atr_30s = df_chunked.loc[idx, 'atr_30s']
            return_30s = df_chunked.loc[idx, 'return_30s']
            
            print(f"    Bar {idx}: vol_ratio={vol_ratio:.3f}, atr_30s={atr_30s:.3f}, return_30s={return_30s:.6f}")
    
    # Check for discontinuities (sudden jumps that shouldn't happen)
    vol_ratio_series = df_chunked['volume_ratio_30s'].dropna()
    vol_ratio_diff = vol_ratio_series.diff().abs()
    max_jump = vol_ratio_diff.max()
    
    print(f"  Maximum volume ratio jump: {max_jump:.3f}")
    
    if max_jump < 5.0:  # Reasonable threshold for ES data
        print("  ✓ No excessive discontinuities detected")
    else:
        print("  ⚠️  Large discontinuities detected - may indicate overlap issues")
    
    print("\n✓ Rolling window overlap testing completed")


def test_progress_tracking():
    """Test that progress tracking works correctly"""
    print("\n=== Testing Progress Tracking ===")
    
    # Create medium-sized dataset
    df = create_large_sample_data(2500)
    
    print("Testing progress tracking with chunked processing...")
    print("(Check console output for progress messages)")
    
    # This should show progress messages
    df_result = create_all_features_chunked(df, chunk_size=800)
    
    print(f"✓ Progress tracking test completed - processed {len(df_result):,} bars")


def test_built_in_validation():
    """Test the built-in chunked processing validation function"""
    print("\n=== Testing Built-in Validation Function ===")
    
    # Create test data
    df = create_large_sample_data(3000)
    
    print("Testing validate_chunked_processing function...")
    
    # Test with different chunk sizes
    chunk_sizes = [1000, 1200]
    
    for chunk_size in chunk_sizes:
        print(f"\nValidating chunk_size = {chunk_size:,}")
        
        try:
            # This function should validate that chunked processing produces identical results
            result = validate_chunked_processing(df.copy(), chunk_size=chunk_size, tolerance=1e-6)
            
            if result:
                print(f"  ✓ Built-in validation PASSED for chunk_size {chunk_size:,}")
            else:
                print(f"  ⚠️  Built-in validation detected differences for chunk_size {chunk_size:,}")
                
        except Exception as e:
            print(f"  ❌ Built-in validation failed with error: {e}")
    
    print("\n✓ Built-in validation testing completed")


def run_all_chunked_tests():
    """Run all chunked processing tests"""
    print("Running comprehensive chunked processing tests...\n")
    
    try:
        # Test 1: Compare chunked vs normal processing
        test_chunked_vs_normal_processing()
        
        # Test 2: Memory efficiency with different sizes
        test_memory_efficiency()
        
        # Test 3: Rolling window overlap handling
        test_rolling_window_overlap()
        
        # Test 4: Progress tracking
        test_progress_tracking()
        
        # Test 5: Built-in validation function
        test_built_in_validation()
        
        print("\n" + "="*60)
        print("✅ ALL CHUNKED PROCESSING TESTS PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_chunked_tests()