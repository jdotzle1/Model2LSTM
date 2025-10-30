import pandas as pd
import numpy as np
from src.data_pipeline.labeling import calculate_labels_for_all_profiles
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

def find_exact_differences():
    """
    Find the exact rows where labels differ between original and optimized
    """
    print("=== FINDING EXACT LABEL DIFFERENCES ===")
    
    # Use small sample for detailed analysis
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Run both versions
    print("Running original version...")
    df_original = calculate_labels_for_all_profiles(df_test.copy())
    
    print("Running optimized version...")
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test.copy())
    
    # Compare each profile
    profiles = ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']
    
    for profile in profiles:
        print(f"\n--- Analyzing {profile} ---")
        
        # Find rows where labels differ
        orig_labels = df_original[f'{profile}_label'].fillna(-999)
        opt_labels = df_optimized[f'{profile}_label'].fillna(-999)
        
        diff_mask = orig_labels != opt_labels
        diff_count = diff_mask.sum()
        
        if diff_count > 0:
            print(f"Found {diff_count} label differences")
            
            # Show first few differences
            diff_indices = np.where(diff_mask)[0][:5]
            
            for idx in diff_indices:
                print(f"\nRow {idx}:")
                print(f"  Outcome - Orig: {df_original.iloc[idx][f'{profile}_outcome']}, Opt: {df_optimized.iloc[idx][f'{profile}_outcome']}")
                print(f"  MAE - Orig: {df_original.iloc[idx][f'{profile}_mae']}, Opt: {df_optimized.iloc[idx][f'{profile}_mae']}")
                print(f"  Hold Time - Orig: {df_original.iloc[idx][f'{profile}_hold_time']}, Opt: {df_optimized.iloc[idx][f'{profile}_hold_time']}")
                print(f"  Label - Orig: {df_original.iloc[idx][f'{profile}_label']}, Opt: {df_optimized.iloc[idx][f'{profile}_label']}")
                
                # Check if this is a timeout vs loss issue
                if (df_original.iloc[idx][f'{profile}_outcome'] == 'timeout' and 
                    df_optimized.iloc[idx][f'{profile}_outcome'] == 'loss'):
                    print(f"  ❌ ISSUE: Original says timeout, Optimized says loss")
                elif (df_original.iloc[idx][f'{profile}_outcome'] == 'loss' and 
                      df_optimized.iloc[idx][f'{profile}_outcome'] == 'timeout'):
                    print(f"  ❌ ISSUE: Original says loss, Optimized says timeout")
                elif (pd.isna(df_original.iloc[idx][f'{profile}_label']) and 
                      df_optimized.iloc[idx][f'{profile}_label'] == -1):
                    print(f"  ❌ ISSUE: Original has NaN label, Optimized has -1 (loss)")
                elif (df_original.iloc[idx][f'{profile}_label'] == -1 and 
                      pd.isna(df_optimized.iloc[idx][f'{profile}_label'])):
                    print(f"  ❌ ISSUE: Original has -1 (loss), Optimized has NaN")
        else:
            print(f"✅ No differences found")

def check_timeout_handling():
    """
    Check if the issue is in how timeouts are handled
    """
    print("\n=== CHECKING TIMEOUT HANDLING ===")
    
    # Look at the last few bars where timeouts are most likely
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.tail(100).copy()  # Last 100 bars
    
    # Run both versions
    df_original = calculate_labels_for_all_profiles(df_test.copy())
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test.copy())
    
    profile = 'long_2to1_small'
    
    print(f"Original timeouts: {(df_original[f'{profile}_outcome'] == 'timeout').sum()}")
    print(f"Optimized timeouts: {(df_optimized[f'{profile}_outcome'] == 'timeout').sum()}")
    
    print(f"Original losses: {(df_original[f'{profile}_outcome'] == 'loss').sum()}")
    print(f"Optimized losses: {(df_optimized[f'{profile}_outcome'] == 'loss').sum()}")
    
    # Check the very last bars
    print(f"\nLast 10 bars outcomes:")
    print(f"Original: {df_original[f'{profile}_outcome'].tail(10).tolist()}")
    print(f"Optimized: {df_optimized[f'{profile}_outcome'].tail(10).tolist()}")

if __name__ == "__main__":
    find_exact_differences()
    check_timeout_handling()