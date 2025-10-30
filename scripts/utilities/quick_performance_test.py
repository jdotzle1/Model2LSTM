#!/usr/bin/env python3
"""
Quick Performance Test - Optimized Feature Engineering

Tests a smaller subset to validate optimizations before running full validation
"""

import pandas as pd
import numpy as np
import time
import os
import sys

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.features import integrate_with_labeled_dataset


def test_performance_on_subset():
    """Test performance on a smaller subset first"""
    
    print("QUICK PERFORMANCE TEST")
    print("=" * 50)
    
    # Load full dataset
    dataset_path = 'project/data/processed/full_labeled_dataset.parquet'
    df = pd.read_parquet(dataset_path)
    
    # Test on different subset sizes
    test_sizes = [10000, 50000, 100000]
    
    for size in test_sizes:
        if size > len(df):
            continue
            
        print(f"\nTesting {size:,} rows...")
        df_subset = df.head(size).copy()
        
        start_time = time.time()
        
        try:
            # Process the subset directly
            from src.data_pipeline.features import create_all_features
            df_featured = create_all_features(df_subset)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate rate
            rows_per_second = size / processing_time
            
            print(f"  Time: {processing_time:.1f}s")
            print(f"  Rate: {rows_per_second:,.0f} rows/second")
            
            # Estimate full dataset time
            full_time_estimate = len(df) / rows_per_second / 60  # minutes
            print(f"  Estimated full dataset time: {full_time_estimate:.1f} minutes")
            
            if full_time_estimate <= 10:
                print(f"  ✅ Estimated time under 10 minutes")
            else:
                print(f"  ❌ Estimated time over 10 minutes")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")


if __name__ == "__main__":
    test_performance_on_subset()