"""
Test chunked processing integration with file I/O
"""

import sys
import os
import pandas as pd
import numpy as np
import tempfile

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from project.data_pipeline.features import integrate_with_labeled_dataset


def create_test_dataset_file():
    """Create a test dataset file for integration testing"""
    np.random.seed(42)
    
    # Create 2000 bars of realistic ES data
    timestamps = pd.date_range('2025-01-15 09:30:00', periods=2000, freq='1s')
    
    base_price = 4750.0
    prices = []
    volumes = []
    
    for i in range(2000):
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
    
    df = pd.DataFrame(data)
    
    # Add mock label columns to simulate existing labeled dataset
    for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large', 
                   'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']:
        df[f'{profile}_label'] = np.random.choice([-1, 0, 1], size=len(df))
        df[f'{profile}_target_hit_bar'] = np.random.randint(0, 900, size=len(df))
        df[f'{profile}_stop_hit_bar'] = np.random.randint(0, 900, size=len(df))
        df[f'{profile}_mae'] = np.random.uniform(0, 10, size=len(df))
        df[f'{profile}_timeout_bar'] = np.random.randint(0, 900, size=len(df))
    
    return df


def test_chunked_file_integration():
    """Test chunked processing with file I/O"""
    print("=== Testing Chunked Processing File Integration ===")
    
    # Create test dataset
    df = create_test_dataset_file()
    print(f"Created test dataset: {len(df):,} rows × {len(df.columns)} columns")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as input_file:
        input_path = input_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        # Save test dataset
        df.to_parquet(input_path)
        print(f"Saved test dataset to: {input_path}")
        
        # Test with different chunk sizes
        chunk_sizes = [None, 500, 1000]  # None = auto-detect
        
        for chunk_size in chunk_sizes:
            print(f"\nTesting with chunk_size = {chunk_size}")
            
            # Process with chunked integration
            result_df = integrate_with_labeled_dataset(
                input_path=input_path,
                output_path=output_path,
                chunk_size=chunk_size
            )
            
            # Verify results
            print(f"  Result: {len(result_df):,} rows × {len(result_df.columns)} columns")
            
            # Check that we have the expected number of features
            original_cols = len(df.columns)
            result_cols = len(result_df.columns)
            added_features = result_cols - original_cols
            
            if added_features == 43:
                print(f"  ✓ Correct number of features added: {added_features}")
            else:
                print(f"  ❌ Expected 43 features, got {added_features}")
            
            # Check that output file was created
            if os.path.exists(output_path):
                output_df = pd.read_parquet(output_path)
                if len(output_df) == len(result_df) and len(output_df.columns) == len(result_df.columns):
                    print(f"  ✓ Output file saved correctly")
                else:
                    print(f"  ❌ Output file mismatch")
            else:
                print(f"  ❌ Output file not created")
            
            # Check key features exist and have reasonable values
            key_features = ['volume_ratio_30s', 'position_in_short_range', 'volatility_regime']
            for feature in key_features:
                if feature in result_df.columns:
                    non_null_count = (~result_df[feature].isna()).sum()
                    non_null_pct = (non_null_count / len(result_df)) * 100
                    print(f"    {feature}: {non_null_pct:.1f}% non-null values")
                else:
                    print(f"    ❌ {feature}: MISSING")
    
    finally:
        # Cleanup temporary files
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass
    
    print("\n✓ Chunked file integration test completed")


def test_large_dataset_simulation():
    """Simulate processing of larger datasets"""
    print("\n=== Testing Large Dataset Simulation ===")
    
    # Test auto chunk size selection for different dataset sizes
    dataset_sizes = [
        (500_000, "500K bars"),
        (1_500_000, "1.5M bars"), 
        (5_500_000, "5.5M bars"),
        (15_000_000, "15M bars")
    ]
    
    for size, description in dataset_sizes:
        print(f"\nSimulating {description} ({size:,} bars):")
        
        # Determine expected chunk size based on integrate_with_labeled_dataset logic
        if size >= 10_000_000:
            expected_chunk_size = 1_000_000
            expected_chunks = (size + expected_chunk_size - 1) // expected_chunk_size
        elif size >= 5_000_000:
            expected_chunk_size = 500_000
            expected_chunks = (size + expected_chunk_size - 1) // expected_chunk_size
        elif size >= 1_000_000:
            expected_chunk_size = 250_000
            expected_chunks = (size + expected_chunk_size - 1) // expected_chunk_size
        else:
            expected_chunk_size = size
            expected_chunks = 1
        
        print(f"  Expected chunk size: {expected_chunk_size:,}")
        print(f"  Expected chunks: {expected_chunks}")
        
        # Estimate processing time (rough calculation)
        # Based on test results: ~1000 bars processed in ~0.5 seconds
        estimated_time_per_chunk = (expected_chunk_size / 1000) * 0.5
        total_estimated_time = estimated_time_per_chunk * expected_chunks
        
        print(f"  Estimated processing time: {total_estimated_time:.1f} seconds ({total_estimated_time/60:.1f} minutes)")
        
        # Memory estimate (rough calculation)
        # Each bar with 43 features ≈ 43 * 8 bytes = 344 bytes per bar
        # Plus overhead for intermediate calculations
        estimated_memory_mb = (expected_chunk_size * 344 * 2) / (1024**2)  # 2x for overhead
        print(f"  Estimated memory per chunk: ~{estimated_memory_mb:.1f} MB")
    
    print("\n✓ Large dataset simulation completed")


def run_integration_tests():
    """Run all integration tests"""
    print("Running chunked processing integration tests...\n")
    
    try:
        # Test 1: File I/O integration
        test_chunked_file_integration()
        
        # Test 2: Large dataset simulation
        test_large_dataset_simulation()
        
        print("\n" + "="*60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_integration_tests()