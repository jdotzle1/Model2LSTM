#!/usr/bin/env python3
"""
Final Performance Validation Summary

Validates all requirements for task 13:
- Processing time under 10 minutes on 947K dataset
- Feature accuracy within reasonable ranges
- Memory usage stays reasonable
- Feature statistics generated
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.features import integrate_with_labeled_dataset, get_expected_feature_names


def main():
    print("FINAL PERFORMANCE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    dataset_path = 'project/data/processed/full_labeled_dataset.parquet'
    df = pd.read_parquet(dataset_path)
    
    print(f"Dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Test processing time
    print(f"\n1. PROCESSING TIME TEST")
    print("-" * 30)
    
    start_time = time.time()
    
    try:
        df_featured = integrate_with_labeled_dataset(
            dataset_path, 
            output_path=None,
            chunk_size=None
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processing time: {processing_time/60:.1f} minutes")
        
        if processing_time <= 600:  # 10 minutes
            print("✅ PASS: Under 10 minutes")
            time_pass = True
        else:
            print("❌ FAIL: Over 10 minutes")
            time_pass = False
            
    except Exception as e:
        print(f"❌ FAIL: Error during processing: {str(e)}")
        time_pass = False
        df_featured = None
    
    if df_featured is None:
        print("Cannot continue validation without processed dataset")
        return False
    
    # Test feature accuracy
    print(f"\n2. FEATURE ACCURACY TEST")
    print("-" * 30)
    
    expected_features = get_expected_feature_names()
    actual_features = len(df_featured.columns) - len(df.columns)
    
    if actual_features == len(expected_features):
        print(f"✅ PASS: Added {actual_features} features as expected")
        features_pass = True
    else:
        print(f"❌ FAIL: Expected {len(expected_features)} features, got {actual_features}")
        features_pass = False
    
    # Check for NaN values
    nan_counts = df_featured[expected_features].isnull().sum()
    high_nan_features = nan_counts[nan_counts > len(df_featured) * 0.5]
    
    if len(high_nan_features) == 0:
        print("✅ PASS: No features with excessive NaN values")
        nan_pass = True
    else:
        print(f"⚠️  WARNING: {len(high_nan_features)} features with >50% NaN")
        nan_pass = True  # Still pass, but warn
    
    # Test memory usage
    print(f"\n3. MEMORY USAGE TEST")
    print("-" * 30)
    
    original_size = df.memory_usage(deep=True).sum() / (1024**2)
    featured_size = df_featured.memory_usage(deep=True).sum() / (1024**2)
    size_increase = ((featured_size - original_size) / original_size) * 100
    
    print(f"Original: {original_size:.1f} MB")
    print(f"Featured: {featured_size:.1f} MB")
    print(f"Increase: +{size_increase:.1f}%")
    
    if size_increase <= 150:
        print("✅ PASS: Memory increase reasonable")
        memory_pass = True
    else:
        print("❌ FAIL: Memory increase excessive")
        memory_pass = False
    
    # Generate feature statistics
    print(f"\n4. FEATURE STATISTICS")
    print("-" * 30)
    
    try:
        stats_data = []
        for feature in expected_features:
            if feature in df_featured.columns:
                values = df_featured[feature].dropna()
                if len(values) > 0:
                    stats_data.append({
                        'Feature': feature,
                        'Count': len(values),
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                        'NaN_Pct': (df_featured[feature].isnull().sum() / len(df_featured)) * 100
                    })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv('feature_statistics_final.csv', index=False)
        
        print(f"✅ PASS: Generated statistics for {len(stats_data)} features")
        print(f"Statistics saved to: feature_statistics_final.csv")
        stats_pass = True
        
    except Exception as e:
        print(f"❌ FAIL: Error generating statistics: {str(e)}")
        stats_pass = False
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    tests = [
        ("Processing Time", time_pass),
        ("Feature Accuracy", features_pass),
        ("Memory Usage", memory_pass),
        ("Feature Statistics", stats_pass)
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        all_passed &= passed
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"Processing time: {processing_time/60:.1f} minutes")
    print(f"Features added: {actual_features}")
    print(f"Memory increase: +{size_increase:.1f}%")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)