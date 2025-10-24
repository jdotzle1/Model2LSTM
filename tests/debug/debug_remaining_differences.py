import pandas as pd
import numpy as np
from project.data_pipeline.labeling import calculate_labels_for_all_profiles
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

def debug_remaining_differences():
    """
    Debug the exact differences in the 2 profiles that still don't match perfectly
    """
    print("=== DEBUGGING REMAINING DIFFERENCES ===")
    
    # Load data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Run both versions
    print("Running original version...")
    df_original = calculate_labels_for_all_profiles(df_test.copy())
    
    print("Running optimized version...")
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test.copy())
    
    # Focus on the 2 profiles with differences
    profiles_with_diffs = ['short_2to1_medium', 'short_2to1_large']
    
    for profile in profiles_with_diffs:
        print(f"\n{'='*60}")
        print(f"ANALYZING {profile.upper()}")
        print(f"{'='*60}")
        
        # Find rows where labels differ
        orig_labels = df_original[f'{profile}_label'].fillna(-999)
        opt_labels = df_optimized[f'{profile}_label'].fillna(-999)
        
        diff_mask = orig_labels != opt_labels
        diff_indices = np.where(diff_mask)[0]
        
        print(f"Found {len(diff_indices)} differences at rows: {diff_indices.tolist()}")
        
        # Show label distributions
        orig_dist = df_original[f'{profile}_label'].value_counts().sort_index()
        opt_dist = df_optimized[f'{profile}_label'].value_counts().sort_index()
        
        print(f"\nLabel distributions:")
        print(f"Original:  {dict(orig_dist)}")
        print(f"Optimized: {dict(opt_dist)}")
        
        # Analyze each difference
        for i, idx in enumerate(diff_indices):
            print(f"\n--- Difference {i+1}: Row {idx} ---")
            
            # Get all relevant data for this row
            orig_outcome = df_original.iloc[idx][f'{profile}_outcome']
            opt_outcome = df_optimized.iloc[idx][f'{profile}_outcome']
            
            orig_mae = df_original.iloc[idx][f'{profile}_mae']
            opt_mae = df_optimized.iloc[idx][f'{profile}_mae']
            
            orig_hold = df_original.iloc[idx][f'{profile}_hold_time']
            opt_hold = df_optimized.iloc[idx][f'{profile}_hold_time']
            
            orig_seq_id = df_original.iloc[idx][f'{profile}_sequence_id']
            opt_seq_id = df_optimized.iloc[idx][f'{profile}_sequence_id']
            
            orig_label = df_original.iloc[idx][f'{profile}_label']
            opt_label = df_optimized.iloc[idx][f'{profile}_label']
            
            print(f"Timestamp: {df_test.index[idx]}")
            print(f"Bar data: {df_test.iloc[idx][['open', 'high', 'low', 'close', 'volume']].to_dict()}")
            
            print(f"Outcome:    Original={orig_outcome:8s} | Optimized={opt_outcome:8s} | Match: {orig_outcome == opt_outcome}")
            print(f"MAE:        Original={orig_mae:8.2f} | Optimized={opt_mae:8.2f} | Match: {np.isclose(orig_mae, opt_mae, equal_nan=True)}")
            print(f"Hold Time:  Original={orig_hold:8.2f} | Optimized={opt_hold:8.2f} | Match: {np.isclose(orig_hold, opt_hold, equal_nan=True)}")
            print(f"Sequence:   Original={orig_seq_id:8.1f} | Optimized={opt_seq_id:8.1f} | Match: {np.isclose(orig_seq_id, opt_seq_id, equal_nan=True)}")
            print(f"LABEL:      Original={orig_label:8.1f} | Optimized={opt_label:8.1f} | Match: {np.isclose(orig_label, opt_label, equal_nan=True)}")
            
            # Analyze the type of difference
            if orig_outcome != opt_outcome:
                print(f"🚨 OUTCOME DIFFERENCE: This shouldn't happen!")
            elif not np.isclose(orig_mae, opt_mae, equal_nan=True):
                print(f"🚨 MAE DIFFERENCE: This shouldn't happen!")
            elif not np.isclose(orig_hold, opt_hold, equal_nan=True):
                print(f"🚨 HOLD TIME DIFFERENCE: This shouldn't happen!")
            elif not np.isclose(orig_seq_id, opt_seq_id, equal_nan=True):
                print(f"⚠️  SEQUENCE ID DIFFERENCE: Different sequence grouping")
                analyze_sequence_difference(df_original, df_optimized, profile, idx, orig_seq_id, opt_seq_id)
            else:
                print(f"🤔 OPTIMAL SELECTION DIFFERENCE: Same sequence, different optimal choice")
                analyze_optimal_selection_difference(df_original, df_optimized, profile, idx)

def analyze_sequence_difference(df_orig, df_opt, profile, idx, orig_seq_id, opt_seq_id):
    """
    Analyze why sequence IDs differ
    """
    print(f"    Analyzing sequence difference...")
    
    # Look at surrounding bars to understand sequence boundaries
    start_idx = max(0, idx - 5)
    end_idx = min(len(df_orig), idx + 6)
    
    print(f"    Surrounding bars ({start_idx} to {end_idx-1}):")
    for i in range(start_idx, end_idx):
        orig_out = df_orig.iloc[i][f'{profile}_outcome']
        opt_out = df_opt.iloc[i][f'{profile}_outcome']
        orig_seq = df_orig.iloc[i][f'{profile}_sequence_id']
        opt_seq = df_opt.iloc[i][f'{profile}_sequence_id']
        
        marker = " <<<" if i == idx else ""
        print(f"      Bar {i:3d}: Orig({orig_out:7s}, seq={orig_seq:4.1f}) | Opt({opt_out:7s}, seq={opt_seq:4.1f}){marker}")

def analyze_optimal_selection_difference(df_orig, df_opt, profile, idx):
    """
    Analyze why optimal selection differs within the same sequence
    """
    print(f"    Analyzing optimal selection difference...")
    
    # Get the sequence this bar belongs to
    orig_seq_id = df_orig.iloc[idx][f'{profile}_sequence_id']
    opt_seq_id = df_opt.iloc[idx][f'{profile}_sequence_id']
    
    # Find all bars in original sequence
    orig_seq_mask = df_orig[f'{profile}_sequence_id'] == orig_seq_id
    orig_seq_data = df_orig[orig_seq_mask]
    
    # Find all bars in optimized sequence  
    opt_seq_mask = df_opt[f'{profile}_sequence_id'] == opt_seq_id
    opt_seq_data = df_opt[opt_seq_mask]
    
    print(f"    Original sequence {orig_seq_id} has {len(orig_seq_data)} bars")
    print(f"    Optimized sequence {opt_seq_id} has {len(opt_seq_data)} bars")
    
    if len(orig_seq_data) <= 10:  # Show details for small sequences
        print(f"    Original sequence details:")
        for i, (seq_idx, row) in enumerate(orig_seq_data.iterrows()):
            mae = row[f'{profile}_mae']
            hold = row[f'{profile}_hold_time']
            label = row[f'{profile}_label']
            marker = " <<<OPTIMAL" if label == 1 else ""
            print(f"      Bar {seq_idx}: MAE={mae:5.1f}, Hold={hold:5.1f}, Label={label:3.1f}{marker}")
        
        print(f"    Optimized sequence details:")
        for i, (seq_idx, row) in enumerate(opt_seq_data.iterrows()):
            mae = row[f'{profile}_mae']
            hold = row[f'{profile}_hold_time']
            label = row[f'{profile}_label']
            marker = " <<<OPTIMAL" if label == 1 else ""
            print(f"      Bar {seq_idx}: MAE={mae:5.1f}, Hold={hold:5.1f}, Label={label:3.1f}{marker}")
        
        # Check for ties and tie-breaking
        orig_min_mae = orig_seq_data[f'{profile}_mae'].min()
        opt_min_mae = opt_seq_data[f'{profile}_mae'].min()
        
        print(f"    Minimum MAE - Original: {orig_min_mae}, Optimized: {opt_min_mae}")
        
        if orig_min_mae == opt_min_mae:
            orig_ties = orig_seq_data[orig_seq_data[f'{profile}_mae'] == orig_min_mae]
            opt_ties = opt_seq_data[opt_seq_data[f'{profile}_mae'] == opt_min_mae]
            
            if len(orig_ties) > 1 or len(opt_ties) > 1:
                print(f"    MAE tie detected - checking hold time tie-breaking...")
                print(f"    Original ties: {len(orig_ties)} bars with MAE={orig_min_mae}")
                print(f"    Optimized ties: {len(opt_ties)} bars with MAE={opt_min_mae}")
                
                if len(orig_ties) > 1:
                    orig_min_hold = orig_ties[f'{profile}_hold_time'].min()
                    print(f"    Original min hold time: {orig_min_hold}")
                
                if len(opt_ties) > 1:
                    opt_min_hold = opt_ties[f'{profile}_hold_time'].min()
                    print(f"    Optimized min hold time: {opt_min_hold}")

def check_duplicate_timestamp_impact():
    """
    Check if the remaining differences are due to duplicate timestamp handling
    """
    print(f"\n{'='*60}")
    print(f"CHECKING DUPLICATE TIMESTAMP IMPACT")
    print(f"{'='*60}")
    
    # Load data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Check for duplicates
    duplicate_mask = df_test.index.duplicated(keep=False)
    duplicates = df_test[duplicate_mask]
    
    print(f"Total duplicate timestamp rows: {len(duplicates)}")
    
    if len(duplicates) > 0:
        print(f"Duplicate timestamps:")
        for timestamp, group in duplicates.groupby(duplicates.index):
            positions = []
            for i, (idx, row) in enumerate(group.iterrows()):
                pos = df_test.index.get_loc(timestamp)
                if isinstance(pos, slice):
                    positions.extend(list(range(pos.start, pos.stop)))
                elif hasattr(pos, '__len__'):
                    positions.extend(pos.tolist())
                else:
                    positions.append(pos)
            
            print(f"  {timestamp}: appears at positions {sorted(set(positions))}")

if __name__ == "__main__":
    debug_remaining_differences()
    check_duplicate_timestamp_impact()