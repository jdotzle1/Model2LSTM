import pandas as pd
import numpy as np

def debug_indexing_issue():
    """
    Debug the exact indexing issue in MAE filtering
    """
    print("=== DEBUGGING INDEXING ISSUE ===")
    
    # Load data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Check the DataFrame index
    print(f"DataFrame index type: {type(df_test.index)}")
    print(f"DataFrame index name: {df_test.index.name}")
    print(f"First few index values: {df_test.index[:5].tolist()}")
    print(f"Last few index values: {df_test.index[-5:].tolist()}")
    
    # Check problematic row indices
    problem_rows = [985, 986, 998]
    print(f"\nProblematic row indices in DataFrame:")
    for row_idx in problem_rows:
        if row_idx < len(df_test):
            actual_index = df_test.index[row_idx]
            print(f"  Row {row_idx}: DataFrame index = {actual_index}")
    
    # Now let's simulate the MAE filter bug
    print(f"\n=== SIMULATING MAE FILTER BUG ===")
    
    # Create a simple test case
    test_df = pd.DataFrame({
        'outcome': ['win', 'win', 'loss', 'timeout', 'win'],
        'mae': [1.0, 2.0, np.nan, np.nan, 1.5],
        'hold_time': [10, 20, np.nan, np.nan, 15],
        'sequence_id': [1.0, 1.0, np.nan, np.nan, 2.0]
    })
    
    # Set a DatetimeIndex like the real data
    test_df.index = pd.date_range('2025-01-01', periods=5, freq='1s')
    
    print(f"Test DataFrame:")
    print(test_df)
    print(f"Test DataFrame index: {test_df.index.tolist()}")
    
    # Apply MAE filter simulation
    print(f"\nApplying MAE filter simulation...")
    
    # Initialize labels
    test_df['label'] = np.nan
    
    # Mark losses
    test_df.loc[test_df['outcome'] == 'loss', 'label'] = -1
    print(f"After marking losses:")
    print(test_df[['outcome', 'label']])
    
    # Process winning sequences
    winners = test_df[test_df['outcome'] == 'win']
    print(f"\nWinners:")
    print(winners[['outcome', 'mae', 'hold_time', 'sequence_id']])
    
    unique_sequences = winners['sequence_id'].dropna().unique()
    print(f"Unique sequences: {unique_sequences}")
    
    for seq_id in unique_sequences:
        print(f"\nProcessing sequence {seq_id}:")
        
        seq_mask = test_df['sequence_id'] == seq_id
        seq_data = test_df[seq_mask]
        
        print(f"  Sequence data:")
        print(seq_data[['outcome', 'mae', 'hold_time', 'sequence_id']])
        print(f"  Sequence indices: {seq_data.index.tolist()}")
        
        # Find minimum MAE
        min_mae = seq_data['mae'].min()
        min_mae_candidates = seq_data[seq_data['mae'] == min_mae]
        
        print(f"  Min MAE: {min_mae}")
        print(f"  Min MAE candidates:")
        print(min_mae_candidates[['outcome', 'mae', 'hold_time']])
        print(f"  Min MAE candidate indices: {min_mae_candidates.index.tolist()}")
        
        if len(min_mae_candidates) == 1:
            optimal_idx = min_mae_candidates.index[0]
        else:
            optimal_idx = min_mae_candidates['hold_time'].idxmin()
        
        print(f"  Optimal index: {optimal_idx}")
        
        # Mark all in sequence as suboptimal
        print(f"  Before marking sequence:")
        print(test_df.loc[seq_mask, ['outcome', 'label']])
        
        test_df.loc[seq_mask, 'label'] = 0
        print(f"  After marking sequence as suboptimal:")
        print(test_df.loc[seq_mask, ['outcome', 'label']])
        
        # Mark optimal
        test_df.loc[optimal_idx, 'label'] = 1
        print(f"  After marking optimal:")
        print(test_df.loc[seq_mask, ['outcome', 'label']])
    
    print(f"\nFinal result:")
    print(test_df[['outcome', 'sequence_id', 'label']])
    
    # Check for bugs
    for idx, row in test_df.iterrows():
        outcome = row['outcome']
        label = row['label']
        
        if outcome == 'loss' and label != -1:
            print(f"🚨 BUG: Row {idx} is loss but has label {label}")
        elif outcome == 'timeout' and not pd.isna(label):
            print(f"🚨 BUG: Row {idx} is timeout but has label {label}")

if __name__ == "__main__":
    debug_indexing_issue()