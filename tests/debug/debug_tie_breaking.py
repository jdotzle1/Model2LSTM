import pandas as pd
import numpy as np
from project.data_pipeline.labeling import calculate_labels_for_all_profiles
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

def debug_tie_breaking():
    """
    Debug the tie-breaking logic differences between original and optimized versions
    """
    print("=== DEBUGGING TIE-BREAKING LOGIC ===")
    
    # Load data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Run both versions
    df_original = calculate_labels_for_all_profiles(df_test.copy())
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test.copy())
    
    # Focus on the problematic sequence in short_2to1_medium
    profile = 'short_2to1_medium'
    sequence_id = 2.0
    
    print(f"Analyzing sequence {sequence_id} in {profile}...")
    
    # Get sequence data from both versions
    orig_seq_mask = df_original[f'{profile}_sequence_id'] == sequence_id
    orig_seq_data = df_original[orig_seq_mask].copy()
    
    opt_seq_mask = df_optimized[f'{profile}_sequence_id'] == sequence_id
    opt_seq_data = df_optimized[opt_seq_mask].copy()
    
    print(f"Original sequence has {len(orig_seq_data)} bars")
    print(f"Optimized sequence has {len(opt_seq_data)} bars")
    
    # Find minimum MAE in both
    orig_min_mae = orig_seq_data[f'{profile}_mae'].min()
    opt_min_mae = opt_seq_data[f'{profile}_mae'].min()
    
    print(f"Minimum MAE - Original: {orig_min_mae}, Optimized: {opt_min_mae}")
    
    # Find all bars with minimum MAE
    orig_min_mae_bars = orig_seq_data[orig_seq_data[f'{profile}_mae'] == orig_min_mae]
    opt_min_mae_bars = opt_seq_data[opt_seq_data[f'{profile}_mae'] == opt_min_mae]
    
    print(f"Bars with minimum MAE - Original: {len(orig_min_mae_bars)}, Optimized: {len(opt_min_mae_bars)}")
    
    if len(orig_min_mae_bars) > 1 or len(opt_min_mae_bars) > 1:
        print(f"\nTIE DETECTED - Multiple bars with same minimum MAE!")
        
        print(f"\nOriginal version - bars with MAE = {orig_min_mae}:")
        for i, (idx, row) in enumerate(orig_min_mae_bars.iterrows()):
            hold_time = row[f'{profile}_hold_time']
            label = row[f'{profile}_label']
            marker = " <<<OPTIMAL" if label == 1 else ""
            print(f"  {i+1}. Index: {idx}, Hold time: {hold_time}, Label: {label}{marker}")
        
        print(f"\nOptimized version - bars with MAE = {opt_min_mae}:")
        for i, (idx, row) in enumerate(opt_min_mae_bars.iterrows()):
            hold_time = row[f'{profile}_hold_time']
            label = row[f'{profile}_label']
            marker = " <<<OPTIMAL" if label == 1 else ""
            print(f"  {i+1}. Index: {idx}, Hold time: {hold_time}, Label: {label}{marker}")
        
        # Check tie-breaking by hold time
        orig_min_hold = orig_min_mae_bars[f'{profile}_hold_time'].min()
        opt_min_hold = opt_min_mae_bars[f'{profile}_hold_time'].min()
        
        print(f"\nTie-breaking by hold time:")
        print(f"Original minimum hold time: {orig_min_hold}")
        print(f"Optimized minimum hold time: {opt_min_hold}")
        
        # Find bars with minimum hold time
        orig_min_hold_bars = orig_min_mae_bars[orig_min_mae_bars[f'{profile}_hold_time'] == orig_min_hold]
        opt_min_hold_bars = opt_min_mae_bars[opt_min_mae_bars[f'{profile}_hold_time'] == opt_min_hold]
        
        print(f"\nBars with minimum hold time:")
        print(f"Original ({len(orig_min_hold_bars)} bars):")
        for i, (idx, row) in enumerate(orig_min_hold_bars.iterrows()):
            label = row[f'{profile}_label']
            marker = " <<<OPTIMAL" if label == 1 else ""
            print(f"  {i+1}. Index: {idx}, Hold time: {row[f'{profile}_hold_time']}, Label: {label}{marker}")
        
        print(f"Optimized ({len(opt_min_hold_bars)} bars):")
        for i, (idx, row) in enumerate(opt_min_hold_bars.iterrows()):
            label = row[f'{profile}_label']
            marker = " <<<OPTIMAL" if label == 1 else ""
            print(f"  {i+1}. Index: {idx}, Hold time: {row[f'{profile}_hold_time']}, Label: {label}{marker}")
        
        # Check if there are still ties after hold time
        if len(orig_min_hold_bars) > 1 or len(opt_min_hold_bars) > 1:
            print(f"\nSTILL TIED after hold time tie-breaking!")
            print(f"This suggests different tie-breaking logic between versions")
            
            # Check the exact tie-breaking methods
            print(f"\nAnalyzing tie-breaking methods:")
            
            # Original method (from apply_mae_filter in original code)
            print(f"Original method uses: idxmin() on hold_time")
            orig_selected = orig_min_mae_bars[f'{profile}_hold_time'].idxmin()
            print(f"  Original selected index: {orig_selected}")
            
            # Optimized method 
            print(f"Optimized method uses: idxmin() on hold_time")
            opt_selected = opt_min_mae_bars[f'{profile}_hold_time'].idxmin()
            print(f"  Optimized selected index: {opt_selected}")
            
            if orig_selected != opt_selected:
                print(f"🚨 DIFFERENT SELECTIONS: Original={orig_selected}, Optimized={opt_selected}")
                
                # Check if this is due to duplicate timestamps
                print(f"\nChecking for duplicate timestamp issue...")
                print(f"Original selected timestamp: {orig_selected}")
                print(f"Optimized selected timestamp: {opt_selected}")
                
                # Check if these timestamps appear multiple times
                orig_ts_count = (df_test.index == orig_selected).sum()
                opt_ts_count = (df_test.index == opt_selected).sum()
                
                print(f"Original timestamp appears {orig_ts_count} times")
                print(f"Optimized timestamp appears {opt_ts_count} times")
                
                if orig_ts_count > 1 or opt_ts_count > 1:
                    print(f"🎯 FOUND THE ISSUE: Duplicate timestamps affecting tie-breaking!")

if __name__ == "__main__":
    debug_tie_breaking()