import pandas as pd
import numpy as np
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

def debug_mae_filter_bug():
    """
    Debug the exact issue in the MAE filter logic
    """
    print("=== DEBUGGING MAE FILTER BUG ===")
    
    # Load small sample and focus on the problematic rows
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Run optimized version with detailed debugging
    print("Running optimized version with debugging...")
    df_result = calculate_labels_for_all_profiles_optimized(df_test)
    
    # Focus on the problematic rows
    problem_rows = [985, 986, 998, 560]
    profile = 'long_2to1_small'
    
    print(f"\n=== EXAMINING PROBLEMATIC ROWS FOR {profile} ===")
    
    for row_idx in problem_rows:
        if row_idx < len(df_result):
            print(f"\n--- Row {row_idx} ---")
            
            outcome = df_result.iloc[row_idx][f'{profile}_outcome']
            mae = df_result.iloc[row_idx][f'{profile}_mae']
            hold_time = df_result.iloc[row_idx][f'{profile}_hold_time']
            seq_id = df_result.iloc[row_idx][f'{profile}_sequence_id']
            label = df_result.iloc[row_idx][f'{profile}_label']
            
            print(f"Outcome: {outcome}")
            print(f"MAE: {mae}")
            print(f"Hold Time: {hold_time}")
            print(f"Sequence ID: {seq_id}")
            print(f"Label: {label}")
            
            # Check if this should be a winner
            if outcome != 'win':
                print(f"🚨 BUG: Non-winner has label {label} (should be -1 for loss or NaN for timeout)")
                
                # Check if sequence ID is assigned to non-winner
                if not pd.isna(seq_id):
                    print(f"🚨 BUG: Non-winner has sequence ID {seq_id} (should be NaN)")
                
                # Check what the MAE filter is doing
                debug_mae_filter_for_row(df_result, profile, row_idx)

def debug_mae_filter_for_row(df, profile_name, row_idx):
    """
    Debug what the MAE filter is doing for a specific row
    """
    print(f"    Debugging MAE filter for row {row_idx}...")
    
    # Check if this row is being processed as a winner
    outcome = df.iloc[row_idx][f'{profile_name}_outcome']
    seq_id = df.iloc[row_idx][f'{profile_name}_sequence_id']
    
    print(f"    Row outcome: {outcome}")
    print(f"    Row sequence ID: {seq_id}")
    
    # Check if there are any sequences with this ID
    if not pd.isna(seq_id):
        seq_mask = df[f'{profile_name}_sequence_id'] == seq_id
        seq_data = df[seq_mask]
        
        print(f"    Sequence {seq_id} has {len(seq_data)} bars:")
        for i, (idx, row) in enumerate(seq_data.iterrows()):
            out = row[f'{profile_name}_outcome']
            mae = row[f'{profile_name}_mae']
            hold = row[f'{profile_name}_hold_time']
            lbl = row[f'{profile_name}_label']
            print(f"      Bar {idx}: outcome={out}, mae={mae}, hold={hold}, label={lbl}")
    
    # Check the winners dataframe
    winners = df[df[f'{profile_name}_outcome'] == 'win']
    print(f"    Total winners: {len(winners)}")
    
    if len(winners) > 0:
        unique_sequences = winners[f'{profile_name}_sequence_id'].dropna().unique()
        print(f"    Unique winning sequences: {unique_sequences}")
        
        # Check if our problematic sequence ID is in the winners
        if not pd.isna(seq_id) and seq_id in unique_sequences:
            print(f"🚨 BUG: Sequence {seq_id} is being processed as winning sequence!")

def trace_mae_filter_execution():
    """
    Trace the exact execution of the MAE filter to find where it goes wrong
    """
    print(f"\n=== TRACING MAE FILTER EXECUTION ===")
    
    # Create a minimal test case
    test_data = {
        'open': [100, 101, 102, 103, 104],
        'high': [100.5, 101.5, 102.5, 103.5, 104.5],
        'low': [99.5, 100.5, 101.5, 102.5, 103.5],
        'close': [100.2, 101.2, 102.2, 103.2, 104.2],
        'volume': [10, 20, 30, 40, 50]
    }
    
    df_test = pd.DataFrame(test_data)
    
    # Manually set outcomes to test the bug
    profile_name = 'test_profile'
    df_test[f'{profile_name}_outcome'] = ['win', 'win', 'loss', 'timeout', 'win']
    df_test[f'{profile_name}_mae'] = [1.0, 2.0, np.nan, np.nan, 1.5]
    df_test[f'{profile_name}_hold_time'] = [10, 20, np.nan, np.nan, 15]
    
    print("Test data:")
    print(df_test[[f'{profile_name}_outcome', f'{profile_name}_mae', f'{profile_name}_hold_time']])
    
    # Apply sequence identification
    from simple_optimized_labeling import identify_consecutive_winners_simple
    df_test = identify_consecutive_winners_simple(df_test, profile_name)
    
    print(f"\nAfter sequence identification:")
    print(df_test[[f'{profile_name}_outcome', f'{profile_name}_sequence_id']])
    
    # Apply MAE filter
    from simple_optimized_labeling import apply_mae_filter_simple
    apply_mae_filter_simple(df_test, profile_name)
    
    print(f"\nAfter MAE filter:")
    print(df_test[[f'{profile_name}_outcome', f'{profile_name}_sequence_id', f'{profile_name}_label']])
    
    # Check for bugs
    for idx, row in df_test.iterrows():
        outcome = row[f'{profile_name}_outcome']
        label = row[f'{profile_name}_label']
        
        if outcome == 'loss' and label != -1:
            print(f"🚨 BUG: Row {idx} is loss but has label {label}")
        elif outcome == 'timeout' and not pd.isna(label):
            print(f"🚨 BUG: Row {idx} is timeout but has label {label}")
        elif outcome == 'win' and label not in [0, 1]:
            print(f"🚨 BUG: Row {idx} is win but has label {label}")

if __name__ == "__main__":
    debug_mae_filter_bug()
    trace_mae_filter_execution()