import pandas as pd
import numpy as np
from src.data_pipeline.labeling import calculate_labels_for_all_profiles
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

def debug_validation_discrepancy():
    """
    Debug why the validation shows differences but detailed analysis shows matches
    """
    print("=== DEBUGGING VALIDATION DISCREPANCY ===")
    
    # Load data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Run both versions
    print("Running both versions...")
    df_original = calculate_labels_for_all_profiles(df_test.copy())
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test.copy())
    
    # Check the specific row that was reported as different
    problem_row = 560
    profile = 'short_2to1_medium'
    
    print(f"\nChecking row {problem_row} for {profile}:")
    
    orig_label = df_original.iloc[problem_row][f'{profile}_label']
    opt_label = df_optimized.iloc[problem_row][f'{profile}_label']
    
    print(f"Original label: {orig_label}")
    print(f"Optimized label: {opt_label}")
    print(f"Match: {orig_label == opt_label}")
    
    # Check the timestamp
    timestamp = df_test.index[problem_row]
    print(f"Timestamp: {timestamp}")
    
    # Check if this timestamp appears multiple times
    timestamp_matches = df_test.index == timestamp
    matching_positions = np.where(timestamp_matches)[0]
    
    print(f"This timestamp appears at positions: {matching_positions.tolist()}")
    
    if len(matching_positions) > 1:
        print(f"DUPLICATE TIMESTAMP DETECTED!")
        
        print(f"\nLabels for all occurrences of this timestamp:")
        for pos in matching_positions:
            orig_lbl = df_original.iloc[pos][f'{profile}_label']
            opt_lbl = df_optimized.iloc[pos][f'{profile}_label']
            print(f"  Position {pos}: Original={orig_lbl}, Optimized={opt_lbl}")
    
    # Check overall label distributions
    print(f"\nOverall label distributions for {profile}:")
    orig_dist = df_original[f'{profile}_label'].value_counts().sort_index()
    opt_dist = df_optimized[f'{profile}_label'].value_counts().sort_index()
    
    print(f"Original:  {dict(orig_dist)}")
    print(f"Optimized: {dict(opt_dist)}")
    
    # Find ALL differences
    orig_labels = df_original[f'{profile}_label'].fillna(-999)
    opt_labels = df_optimized[f'{profile}_label'].fillna(-999)
    
    diff_mask = orig_labels != opt_labels
    diff_positions = np.where(diff_mask)[0]
    
    print(f"\nAll differences found: {len(diff_positions)} positions")
    for pos in diff_positions:
        orig_lbl = df_original.iloc[pos][f'{profile}_label']
        opt_lbl = df_optimized.iloc[pos][f'{profile}_label']
        ts = df_test.index[pos]
        print(f"  Position {pos} ({ts}): Original={orig_lbl}, Optimized={opt_lbl}")
    
    # Check if the issue is in the comparison logic
    print(f"\nDebugging comparison logic...")
    
    # Use the exact same comparison as in validate_optimization.py
    label1 = df_original[f'{profile}_label'].fillna(-999)
    label2 = df_optimized[f'{profile}_label'].fillna(-999)
    label_match = np.allclose(label1, label2, rtol=1e-5, atol=1e-8, equal_nan=True)
    
    print(f"np.allclose result: {label_match}")
    
    if not label_match:
        label_diff = ~np.isclose(label1, label2, rtol=1e-5, atol=1e-8, equal_nan=True)
        diff_count = label_diff.sum()
        print(f"Differences found by np.isclose: {diff_count}")
        
        diff_indices = np.where(label_diff)[0]
        for idx in diff_indices[:5]:  # Show first 5
            print(f"  Row {idx}: Original={label1.iloc[idx]}, Optimized={label2.iloc[idx]}")

if __name__ == "__main__":
    debug_validation_discrepancy()