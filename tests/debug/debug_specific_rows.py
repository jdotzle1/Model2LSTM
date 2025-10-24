import pandas as pd
import numpy as np
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

def debug_specific_problematic_rows():
    """
    Debug the exact state of problematic rows before and after MAE filtering
    """
    print("=== DEBUGGING SPECIFIC PROBLEMATIC ROWS ===")
    
    # Load data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Run just one profile to focus debugging
    profile = {'name': 'long_2to1_small', 'direction': 'long', 'target_ticks': 12, 'stop_ticks': 6}
    
    # Manually run the steps to debug
    from simple_optimized_labeling import calculate_profile_labels_optimized
    
    print("Running optimized labeling with step-by-step debugging...")
    
    # We'll modify the function to add debugging
    debug_calculate_profile_labels_optimized(df_test, profile)

def debug_calculate_profile_labels_optimized(df, profile):
    """
    Debug version of calculate_profile_labels_optimized with detailed logging
    """
    profile_name = profile['name']
    print(f"Processing {profile_name} with debugging...")
    
    n_bars = len(df)
    direction_is_long = profile['direction'] == 'long'
    target_points = profile['target_ticks'] * 0.25  # TICK_SIZE
    stop_points = profile['stop_ticks'] * 0.25
    
    # Pre-allocate result arrays
    outcomes = ['timeout'] * n_bars
    maes = [np.nan] * n_bars
    hold_times = [np.nan] * n_bars
    
    # Convert DataFrame to numpy arrays for faster access
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    print(f"Step 1: Processing outcomes...")
    
    # Process each bar (simplified version for debugging)
    for i in range(min(n_bars - 1, 1000)):  # Process up to 1000 bars
        # Check if we have a next bar to enter on
        if i + 1 >= n_bars:
            outcomes[i] = 'timeout'
            continue
            
        # Entry price is next bar's open
        entry_price = opens[i + 1]
        
        if direction_is_long:
            target_price = entry_price + target_points
            stop_price = entry_price - stop_points
        else:
            target_price = entry_price - target_points
            stop_price = entry_price + stop_points
        
        # Look forward from entry bar
        entry_idx = i + 1
        end_search_idx = min(entry_idx + 900, n_bars)  # 900 second lookforward
        
        worst_adverse = 0.0
        
        # Check each future bar
        for j in range(entry_idx, end_search_idx):
            if direction_is_long:
                target_hit = highs[j] >= target_price
                stop_hit = lows[j] <= stop_price
                adverse_move = entry_price - lows[j]
            else:
                target_hit = lows[j] <= target_price
                stop_hit = highs[j] >= stop_price
                adverse_move = highs[j] - entry_price
            
            worst_adverse = max(worst_adverse, adverse_move)
            
            if target_hit and stop_hit:
                outcomes[i] = 'loss'  # Conservative: assume stop hit first
                break
            elif target_hit:
                outcomes[i] = 'win'
                maes[i] = worst_adverse / 0.25  # Convert to ticks
                hold_times[i] = j - entry_idx
                break
            elif stop_hit:
                outcomes[i] = 'loss'
                break
    
    # Add results to dataframe
    df[f'{profile_name}_outcome'] = outcomes
    df[f'{profile_name}_mae'] = maes
    df[f'{profile_name}_hold_time'] = hold_times
    
    print(f"Step 2: Identifying sequences...")
    
    # Identify consecutive winners
    from simple_optimized_labeling import identify_consecutive_winners_simple
    df = identify_consecutive_winners_simple(df, profile_name)
    
    print(f"Step 3: Before MAE filter - checking problematic rows...")
    
    # Check problematic rows before MAE filter
    problem_rows = [985, 986, 998]
    for row_idx in problem_rows:
        if row_idx < len(df):
            outcome = df.iloc[row_idx][f'{profile_name}_outcome']
            seq_id = df.iloc[row_idx][f'{profile_name}_sequence_id']
            print(f"  Row {row_idx}: outcome={outcome}, seq_id={seq_id}")
    
    print(f"Step 4: Applying MAE filter...")
    
    # Apply MAE filter with debugging
    debug_apply_mae_filter_simple(df, profile_name)
    
    print(f"Step 5: After MAE filter - checking problematic rows...")
    
    # Check problematic rows after MAE filter
    for row_idx in problem_rows:
        if row_idx < len(df):
            outcome = df.iloc[row_idx][f'{profile_name}_outcome']
            seq_id = df.iloc[row_idx][f'{profile_name}_sequence_id']
            label = df.iloc[row_idx][f'{profile_name}_label']
            print(f"  Row {row_idx}: outcome={outcome}, seq_id={seq_id}, label={label}")
            
            if outcome in ['loss', 'timeout'] and label == 1.0:
                print(f"    🚨 BUG FOUND: Non-winner labeled as optimal!")

def debug_apply_mae_filter_simple(df, profile_name):
    """
    Debug version of apply_mae_filter_simple with detailed logging
    """
    print(f"    MAE Filter: Initializing all labels as NaN...")
    # Initialize all as NaN
    df[f'{profile_name}_label'] = np.nan
    
    print(f"    MAE Filter: Marking losses as -1...")
    # Mark losses
    loss_mask = df[f'{profile_name}_outcome'] == 'loss'
    df.loc[loss_mask, f'{profile_name}_label'] = -1
    print(f"    MAE Filter: Marked {loss_mask.sum()} losses")
    
    print(f"    MAE Filter: Processing winning sequences...")
    # Process winning sequences
    winners = df[df[f'{profile_name}_outcome'] == 'win']
    print(f"    MAE Filter: Found {len(winners)} winners")
    
    if len(winners) > 0:
        unique_sequences = winners[f'{profile_name}_sequence_id'].dropna().unique()
        print(f"    MAE Filter: Found {len(unique_sequences)} unique sequences: {unique_sequences}")
        
        for seq_id in unique_sequences:
            seq_mask = df[f'{profile_name}_sequence_id'] == seq_id
            seq_data = df[seq_mask]
            
            print(f"      Processing sequence {seq_id} with {len(seq_data)} bars")
            
            # Find minimum MAE
            min_mae = seq_data[f'{profile_name}_mae'].min()
            min_mae_candidates = seq_data[seq_data[f'{profile_name}_mae'] == min_mae]
            
            if len(min_mae_candidates) == 1:
                optimal_idx = min_mae_candidates.index[0]
            else:
                # Tie-breaker: shortest hold time
                optimal_idx = min_mae_candidates[f'{profile_name}_hold_time'].idxmin()
            
            print(f"        Optimal bar in sequence: {optimal_idx}")
            
            # Mark all in sequence as suboptimal
            df.loc[seq_mask, f'{profile_name}_label'] = 0
            
            # Mark optimal
            df.loc[optimal_idx, f'{profile_name}_label'] = 1
            
            print(f"        Marked {len(seq_data)} bars: 1 optimal, {len(seq_data)-1} suboptimal")
    
    print(f"    MAE Filter: Complete")

if __name__ == "__main__":
    debug_specific_problematic_rows()