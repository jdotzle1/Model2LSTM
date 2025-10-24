import pandas as pd
import numpy as np
from project.data_pipeline.labeling import calculate_labels_for_all_profiles
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

def debug_remaining_differences():
    """
    Debug the remaining differences after the fix
    """
    print("=== DEBUGGING REMAINING DIFFERENCES ===")
    
    # Load test data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Run both versions
    print("Running original version...")
    df_original = calculate_labels_for_all_profiles(df_test.copy())
    
    print("Running optimized version...")
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test.copy())
    
    # Focus on profiles that still have differences
    profiles_to_check = ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large', 'short_2to1_small']
    
    for profile in profiles_to_check:
        print(f"\n{'='*60}")
        print(f"ANALYZING {profile.upper()}")
        print(f"{'='*60}")
        
        # Find rows where labels differ
        orig_labels = df_original[f'{profile}_label'].fillna(-999)
        opt_labels = df_optimized[f'{profile}_label'].fillna(-999)
        
        diff_mask = orig_labels != opt_labels
        diff_indices = np.where(diff_mask)[0]
        
        if len(diff_indices) == 0:
            print("✅ No differences found!")
            continue
            
        print(f"Found {len(diff_indices)} differences at rows: {diff_indices.tolist()}")
        
        for i, idx in enumerate(diff_indices):
            if i >= 3:  # Limit to first 3 differences
                print(f"... and {len(diff_indices) - 3} more")
                break
                
            print(f"\n--- Difference {i+1}: Row {idx} ---")
            
            # Get all relevant data for this row
            orig_outcome = df_original.iloc[idx][f'{profile}_outcome']
            opt_outcome = df_optimized.iloc[idx][f'{profile}_outcome']
            
            orig_label = df_original.iloc[idx][f'{profile}_label']
            opt_label = df_optimized.iloc[idx][f'{profile}_label']
            
            print(f"Outcome: Original={orig_outcome:8s} | Optimized={opt_outcome:8s}")
            print(f"Label:   Original={orig_label:8.1f} | Optimized={opt_label:8.1f}")
            
            # Check if this is near the end of dataset
            remaining_bars = len(df_test) - idx - 1
            print(f"Remaining bars: {remaining_bars}")
            
            if remaining_bars < 900:
                print(f"⚠️  Near end of dataset - insufficient lookforward")
            
            # If optimized is marking as optimal but original isn't
            if opt_label == 1.0 and orig_label != 1.0:
                print(f"🚨 OPTIMIZED INCORRECTLY MARKING AS OPTIMAL")
                
                # Check what the outcome actually is
                if opt_outcome != 'win':
                    print(f"   ERROR: Marking non-winner as optimal!")
                    print(f"   Outcome is '{opt_outcome}' but label is 1 (optimal)")
                else:
                    print(f"   Outcome is 'win' but original doesn't mark as optimal")
                    print(f"   This suggests sequence grouping or MAE calculation difference")

def check_edge_case_handling():
    """
    Check how both versions handle edge cases near dataset end
    """
    print(f"\n{'='*60}")
    print(f"CHECKING EDGE CASE HANDLING")
    print(f"{'='*60}")
    
    # Load test data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Check last 20 bars
    print("Last 20 bars analysis:")
    for i in range(980, 1000):
        remaining = 1000 - i - 1
        print(f"Bar {i:3d}: {remaining:3d} bars remaining")
        
        if remaining < 900:
            print(f"         ⚠️  Insufficient lookforward (need 900, have {remaining})")

if __name__ == "__main__":
    debug_remaining_differences()
    check_edge_case_handling()