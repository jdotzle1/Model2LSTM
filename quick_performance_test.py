#!/usr/bin/env python3
"""
Quick Performance Test for Feature Engineering System

Tests processing time on smaller samples to identify bottlenecks
and validate optimizations before running full 947K dataset.
"""

import pandas as pd
import numpy as np
import time
import os
import sys

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from project.data_pipeline.features import create_all_features, get_expected_feature_names


def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    else:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"


def test_sample_performance(sample_size=10000):
    """Test performance on a sample of the dataset"""
    print(f"Testing feature engineering on {sample_size:,} bars...")
    
    # Load full dataset
    dataset_path = 'project/data/processed/full_labeled_dataset.parquet'
    df_full = pd.read_parquet(dataset_path)
    
    # Take a sample
    df_sample = df_full.head(sample_size).copy()
    print(f"Sample: {len(df_sample):,} rows × {len(df_sample.columns)} columns")
    
    # Test processing time
    start_time = time.time()
    
    try:
        df_featured = create_all_features(df_sample)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate features per second
        features_per_second = sample_size / processing_time
        
        # Extrapolate to full dataset
        full_dataset_size = 947004
        estimated_full_time = full_dataset_size / features_per_second
        
        print(f"✓ Sample processing completed in {format_time(processing_time)}")
        print(f"  Processing rate: {features_per_second:,.0f} bars/second")
        print(f"  Estimated time for 947K bars: {format_time(estimated_full_time)}")
        
        # Check if estimated time is under 10 minutes
        if estimated_full_time <= 600:
            print(f"✅ PROJECTED PASS: Estimated time under 10 minutes")
        else:
            print(f"❌ PROJECTED FAIL: Estimated time over 10 minutes")
        
        # Validate features were created
        expected_features = get_expected_feature_names()
        actual_features = len(df_featured.columns) - len(df_sample.columns)
        
        if actual_features == len(expected_features):
            print(f"✅ Features: Added {actual_features} features as expected")
        else:
            print(f"❌ Features: Expected {len(expected_features)}, got {actual_features}")
        
        return processing_time, estimated_full_time, df_featured
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"❌ FAIL: Processing failed after {format_time(processing_time)}")
        print(f"Error: {str(e)}")
        return processing_time, None, None


def profile_feature_categories(sample_size=5000):
    """Profile individual feature categories to identify bottlenecks"""
    print(f"\nProfiling feature categories on {sample_size:,} bars...")
    
    # Load sample
    dataset_path = 'project/data/processed/full_labeled_dataset.parquet'
    df_full = pd.read_parquet(dataset_path)
    df_sample = df_full.head(sample_size).copy()
    
    # Import individual feature functions
    from project.data_pipeline.features import (
        add_volume_features, add_price_context_features, add_consolidation_features,
        add_return_features, add_volatility_features, add_microstructure_features, add_time_features
    )
    
    # Test each category
    categories = [
        ("Volume Features", add_volume_features),
        ("Price Context Features", add_price_context_features),
        ("Return Features", add_return_features),
        ("Volatility Features", add_volatility_features),
        ("Microstructure Features", add_microstructure_features),
        ("Time Features", add_time_features),
        ("Consolidation Features", add_consolidation_features),  # Test this last as it's likely slowest
    ]
    
    for category_name, feature_func in categories:
        df_test = df_sample.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in df_test.columns:
            df_test['timestamp'] = df_test.index
        
        start_time = time.time()
        
        try:
            df_result = feature_func(df_test)
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate rate
            rate = sample_size / processing_time if processing_time > 0 else float('inf')
            
            print(f"  {category_name:25s}: {format_time(processing_time):>12s} ({rate:>8,.0f} bars/sec)")
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"  {category_name:25s}: FAILED after {format_time(processing_time)} - {str(e)}")


def main():
    """Main performance testing function"""
    print("QUICK PERFORMANCE TEST FOR FEATURE ENGINEERING")
    print("=" * 60)
    
    # Test 1: Small sample (1K bars)
    print("\n1. Testing 1K bars (quick validation)...")
    test_sample_performance(1000)
    
    # Test 2: Medium sample (10K bars)
    print("\n2. Testing 10K bars (performance estimate)...")
    test_sample_performance(10000)
    
    # Test 3: Profile individual categories
    profile_feature_categories(5000)
    
    # Test 4: Larger sample if previous tests look good
    print("\n4. Testing 50K bars (final validation)...")
    processing_time, estimated_full_time, df_featured = test_sample_performance(50000)
    
    if estimated_full_time and estimated_full_time <= 600:
        print(f"\n✅ PERFORMANCE LOOKS GOOD - Ready for full dataset test")
        print(f"   Estimated full processing time: {format_time(estimated_full_time)}")
    else:
        print(f"\n❌ PERFORMANCE NEEDS OPTIMIZATION")
        if estimated_full_time:
            print(f"   Estimated full processing time: {format_time(estimated_full_time)}")
    
    return estimated_full_time


if __name__ == "__main__":
    main()