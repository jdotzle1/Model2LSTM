import pandas as pd
import numpy as np
import time
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from project.data_pipeline.labeling import calculate_labels_for_all_profiles
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

def compare_results(df1, df2, profile_name):
    """
    Compare results between original and optimized versions for a single profile
    """
    print(f"\n--- Comparing {profile_name} ---")
    
    # Check if columns exist in both dataframes
    required_cols = [f'{profile_name}_outcome', f'{profile_name}_mae', 
                    f'{profile_name}_hold_time', f'{profile_name}_label']
    
    missing_cols = []
    for col in required_cols:
        if col not in df1.columns:
            missing_cols.append(f"df1 missing: {col}")
        if col not in df2.columns:
            missing_cols.append(f"df2 missing: {col}")
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    
    all_match = True
    
    # Compare outcomes
    outcome_match = (df1[f'{profile_name}_outcome'] == df2[f'{profile_name}_outcome']).all()
    print(f"Outcomes match: {outcome_match}")
    if not outcome_match:
        diff_count = (df1[f'{profile_name}_outcome'] != df2[f'{profile_name}_outcome']).sum()
        print(f"  Differences: {diff_count} out of {len(df1)}")
        all_match = False
    
    # Compare MAE (allowing for small floating point differences)
    mae1 = df1[f'{profile_name}_mae'].fillna(-999)
    mae2 = df2[f'{profile_name}_mae'].fillna(-999)
    mae_close = np.allclose(mae1, mae2, rtol=1e-5, atol=1e-8, equal_nan=True)
    print(f"MAE values match: {mae_close}")
    if not mae_close:
        mae_diff = ~np.isclose(mae1, mae2, rtol=1e-5, atol=1e-8, equal_nan=True)
        diff_count = mae_diff.sum()
        print(f"  MAE differences: {diff_count} out of {len(df1)}")
        if diff_count < 10:  # Show first few differences
            diff_indices = np.where(mae_diff)[0][:5]
            for idx in diff_indices:
                print(f"    Row {idx}: Original={mae1.iloc[idx]:.6f}, Optimized={mae2.iloc[idx]:.6f}")
        all_match = False
    
    # Compare hold times
    hold1 = df1[f'{profile_name}_hold_time'].fillna(-999)
    hold2 = df2[f'{profile_name}_hold_time'].fillna(-999)
    hold_match = np.allclose(hold1, hold2, rtol=1e-5, atol=1e-8, equal_nan=True)
    print(f"Hold times match: {hold_match}")
    if not hold_match:
        hold_diff = ~np.isclose(hold1, hold2, rtol=1e-5, atol=1e-8, equal_nan=True)
        diff_count = hold_diff.sum()
        print(f"  Hold time differences: {diff_count} out of {len(df1)}")
        all_match = False
    
    # Compare labels
    label1 = df1[f'{profile_name}_label'].fillna(-999)
    label2 = df2[f'{profile_name}_label'].fillna(-999)
    label_match = np.allclose(label1, label2, rtol=1e-5, atol=1e-8, equal_nan=True)
    print(f"Labels match: {label_match}")
    if not label_match:
        label_diff = ~np.isclose(label1, label2, rtol=1e-5, atol=1e-8, equal_nan=True)
        diff_count = label_diff.sum()
        print(f"  Label differences: {diff_count} out of {len(df1)}")
        
        # Show label distribution comparison
        print(f"  Original label distribution: {df1[f'{profile_name}_label'].value_counts().sort_index().to_dict()}")
        print(f"  Optimized label distribution: {df2[f'{profile_name}_label'].value_counts().sort_index().to_dict()}")
        all_match = False
    
    return all_match

def validate_optimization():
    """
    Run both original and optimized versions on the same data and compare results
    """
    print("=== VALIDATION: ORIGINAL vs OPTIMIZED ===")
    
    # Load a reasonable sample for testing
    print("Loading test dataset...")
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    
    # Use the same 1000 bars we tested before for direct comparison
    test_size = 1000
    df_test = df_full.head(test_size).copy()
    print(f"Testing on {test_size} bars")
    
    # Run original version
    print(f"\n--- Running ORIGINAL version ---")
    start_time = time.time()
    df_original = calculate_labels_for_all_profiles(df_test.copy())
    original_time = time.time() - start_time
    print(f"Original completed in {original_time:.2f} seconds")
    
    # Run optimized version
    print(f"\n--- Running OPTIMIZED version ---")
    start_time = time.time()
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test.copy())
    optimized_time = time.time() - start_time
    print(f"Optimized completed in {optimized_time:.2f} seconds")
    
    # Compare results for each profile
    print(f"\n=== DETAILED COMPARISON ===")
    
    profiles = ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']
    
    all_profiles_match = True
    
    for profile in profiles:
        profile_match = compare_results(df_original, df_optimized, profile)
        if not profile_match:
            all_profiles_match = False
    
    # Overall summary
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Original time: {original_time:.2f} seconds")
    print(f"Optimized time: {optimized_time:.2f} seconds")
    print(f"Speedup: {original_time/optimized_time:.1f}x")
    
    if all_profiles_match:
        print(f"✅ VALIDATION PASSED: Results are identical!")
        print(f"✅ Safe to use optimized version for full dataset")
        return True
    else:
        print(f"❌ VALIDATION FAILED: Results differ!")
        print(f"❌ Need to fix optimization before using on full dataset")
        return False

def debug_first_difference():
    """
    If validation fails, debug the first difference in detail
    """
    print("\n=== DEBUGGING FIRST DIFFERENCE ===")
    
    # Load small sample
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(100).copy()  # Very small for detailed debugging
    
    # Run both versions
    df_original = calculate_labels_for_all_profiles(df_test.copy())
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test.copy())
    
    # Find first difference
    profile = 'long_2to1_small'
    
    outcome_diff = df_original[f'{profile}_outcome'] != df_optimized[f'{profile}_outcome']
    if outcome_diff.any():
        first_diff_idx = outcome_diff.idxmax()
        print(f"First outcome difference at index {first_diff_idx}:")
        print(f"  Original: {df_original.iloc[first_diff_idx][f'{profile}_outcome']}")
        print(f"  Optimized: {df_optimized.iloc[first_diff_idx][f'{profile}_outcome']}")
        
        # Show the bar data
        print(f"  Bar data: {df_test.iloc[first_diff_idx][['open', 'high', 'low', 'close', 'volume']].to_dict()}")
        if first_diff_idx + 1 < len(df_test):
            print(f"  Next bar: {df_test.iloc[first_diff_idx + 1][['open', 'high', 'low', 'close', 'volume']].to_dict()}")

if __name__ == "__main__":
    # Run validation
    validation_passed = validate_optimization()
    
    if not validation_passed:
        # If validation failed, run debugging
        debug_first_difference()
    else:
        print(f"\n🎉 Ready to process full dataset with optimized version!")
        
        # Offer to run full dataset
        response = input(f"\nRun optimized version on full dataset now? (y/n): ")
        if response.lower() == 'y':
            print(f"\nProcessing full dataset with optimized version...")
            df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
            
            start_time = time.time()
            df_labeled = calculate_labels_for_all_profiles_optimized(df_full)
            elapsed = time.time() - start_time
            
            # Save result
            output_path = 'project/data/processed/full_labeled_dataset_optimized.parquet'
            df_labeled.to_parquet(output_path)
            
            print(f"\n✅ Full dataset completed in {elapsed/60:.1f} minutes")
            print(f"✅ Saved to: {output_path}")
            print(f"✅ Shape: {df_labeled.shape}")
        else:
            print(f"\nOptimization validated. Ready to run when needed.")