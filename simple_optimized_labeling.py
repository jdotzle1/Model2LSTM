import pandas as pd
import numpy as np
import time
from project.data_pipeline.labeling import PROFILES, TICK_SIZE, LOOKFORWARD_SECONDS

def calculate_profile_labels_optimized(df, profile, lookforward_seconds=LOOKFORWARD_SECONDS):
    """
    Optimized version of profile labeling with progress tracking
    
    Key optimizations:
    1. Pre-calculate entry prices and target/stop levels
    2. Use vectorized operations where possible
    3. Progress tracking for long operations
    4. Memory-efficient processing
    """
    
    profile_name = profile['name']
    print(f"  Processing {profile_name}...")
    
    n_bars = len(df)
    direction_is_long = profile['direction'] == 'long'
    target_points = profile['target_ticks'] * TICK_SIZE
    stop_points = profile['stop_ticks'] * TICK_SIZE
    
    # Pre-allocate result arrays
    outcomes = ['timeout'] * n_bars
    maes = [np.nan] * n_bars
    hold_times = [np.nan] * n_bars
    
    # Convert DataFrame to numpy arrays for faster access
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    print(f"    Step 1/4: Checking targets and stops...")
    
    # Process in batches with progress updates
    batch_size = 1000
    n_batches = (n_bars + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_bars - 1)  # -1 because we need next bar
        
        if batch_idx % 10 == 0:  # Progress every 10 batches
            print(f"      Progress: {batch_idx + 1}/{n_batches} batches ({start_idx:,}/{n_bars:,} bars)")
        
        # Process this batch
        for i in range(start_idx, end_idx):
            # Check if we have a next bar to enter on (same as original logic)
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
            end_search_idx = min(entry_idx + lookforward_seconds, n_bars)
            
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
                    maes[i] = worst_adverse / TICK_SIZE
                    hold_times[i] = j - entry_idx
                    break
                elif stop_hit:
                    outcomes[i] = 'loss'
                    break
    
    print(f"    Step 2/4: Calculating sequences...")
    
    # Add results to dataframe
    df[f'{profile_name}_outcome'] = outcomes
    df[f'{profile_name}_mae'] = maes
    df[f'{profile_name}_hold_time'] = hold_times
    
    # Identify consecutive winners
    df = identify_consecutive_winners_simple(df, profile_name)
    
    print(f"    Step 3/4: Applying MAE filter...")
    apply_mae_filter_simple(df, profile_name)
    
    # Print statistics
    label_counts = df[f'{profile_name}_label'].value_counts()
    wins = (df[f'{profile_name}_outcome'] == 'win').sum()
    losses = (df[f'{profile_name}_outcome'] == 'loss').sum()
    timeouts = (df[f'{profile_name}_outcome'] == 'timeout').sum()
    
    print(f"    Results: Wins={wins}, Losses={losses}, Timeouts={timeouts}")
    print(f"    Labels: Optimal={label_counts.get(1.0, 0)}, Suboptimal={label_counts.get(0.0, 0)}, Loss={label_counts.get(-1.0, 0)}")
    
    return df

def identify_consecutive_winners_simple(df, profile_name):
    """
    Simple consecutive winner identification
    """
    # Create win flag
    is_win = (df[f'{profile_name}_outcome'] == 'win').astype(int)
    
    # Find sequence breaks
    sequence_breaks = (is_win != is_win.shift(1)).astype(int)
    sequence_ids = sequence_breaks.cumsum()
    
    # Only keep sequence IDs for winners
    df[f'{profile_name}_sequence_id'] = np.where(is_win == 1, sequence_ids, np.nan)
    
    return df

def apply_mae_filter_simple(df, profile_name):
    """
    Fixed MAE filter that properly handles edge cases
    """
    # Initialize all as NaN (this is correct for timeouts and insufficient data)
    df[f'{profile_name}_label'] = np.nan
    
    # Mark losses (only if outcome is explicitly 'loss')
    loss_mask = df[f'{profile_name}_outcome'] == 'loss'
    df.loc[loss_mask, f'{profile_name}_label'] = -1
    
    # Process winning sequences (only if outcome is explicitly 'win')
    win_mask = df[f'{profile_name}_outcome'] == 'win'
    winners = df[win_mask]
    
    if len(winners) > 0:
        # Get unique sequence IDs (excluding NaN)
        unique_sequences = winners[f'{profile_name}_sequence_id'].dropna().unique()
        
        for seq_id in unique_sequences:
            seq_mask = (df[f'{profile_name}_sequence_id'] == seq_id) & win_mask
            seq_data = df[seq_mask]
            
            if len(seq_data) == 0:
                continue
                
            # Mark all in sequence as suboptimal first
            df.loc[seq_mask, f'{profile_name}_label'] = 0
            
            # Find minimum MAE
            min_mae = seq_data[f'{profile_name}_mae'].min()
            min_mae_candidates = seq_data[seq_data[f'{profile_name}_mae'] == min_mae]
            
            if len(min_mae_candidates) == 1:
                optimal_idx = min_mae_candidates.index[0]
            else:
                # Tie-breaker: shortest hold time
                optimal_idx = min_mae_candidates[f'{profile_name}_hold_time'].idxmin()
            
            # Double-check: Mark optimal ONLY if it's actually a winner
            outcome_at_optimal = df.loc[optimal_idx, f'{profile_name}_outcome']
            if isinstance(outcome_at_optimal, pd.Series):
                outcome_at_optimal = outcome_at_optimal.iloc[0]
            
            if outcome_at_optimal == 'win':
                df.loc[optimal_idx, f'{profile_name}_label'] = 1
    
    # SAFETY CHECK: Ensure no non-winners are marked as optimal
    # This should never happen, but let's be extra safe
    non_win_mask = df[f'{profile_name}_outcome'] != 'win'
    incorrect_optimal_mask = (df[f'{profile_name}_label'] == 1) & non_win_mask
    
    if incorrect_optimal_mask.any():
        print(f"WARNING: Found {incorrect_optimal_mask.sum()} non-winners incorrectly marked as optimal!")
        print(f"Fixing by setting their labels to correct values...")
        
        # Fix losses
        loss_fix_mask = incorrect_optimal_mask & (df[f'{profile_name}_outcome'] == 'loss')
        df.loc[loss_fix_mask, f'{profile_name}_label'] = -1
        
        # Fix timeouts
        timeout_fix_mask = incorrect_optimal_mask & (df[f'{profile_name}_outcome'] == 'timeout')
        df.loc[timeout_fix_mask, f'{profile_name}_label'] = np.nan
    
    # Function modifies DataFrame in-place, no return needed
    
    return df[f'{profile_name}_label']

def calculate_labels_for_all_profiles_optimized(df, lookforward_seconds=LOOKFORWARD_SECONDS):
    """
    Optimized main labeling function
    """
    print(f"Calculating labels for {len(PROFILES)} profiles (OPTIMIZED)...")
    print(f"Dataset size: {len(df):,} bars")
    print(f"Lookforward window: {lookforward_seconds} seconds ({lookforward_seconds/60:.1f} minutes)")
    
    df_result = df.copy()
    total_start_time = time.time()
    
    for i, profile in enumerate(PROFILES, 1):
        print(f"\n[{i}/{len(PROFILES)}] {profile['name']}...")
        start_time = time.time()
        
        df_result = calculate_profile_labels_optimized(df_result, profile, lookforward_seconds)
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f} seconds")
        
        # Estimate remaining time
        if i < len(PROFILES):
            avg_time_per_profile = (time.time() - total_start_time) / i
            remaining_profiles = len(PROFILES) - i
            estimated_remaining = avg_time_per_profile * remaining_profiles
            print(f"  Estimated remaining time: {estimated_remaining/60:.1f} minutes")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n✓ All profiles completed in {total_elapsed/60:.1f} minutes!")
    
    return df_result

def test_optimization():
    """
    Test the optimization on progressively larger samples
    """
    print("=== TESTING OPTIMIZATION ===")
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_parquet('project/data/raw/test_sample.parquet')
    print(f"Full dataset: {len(df):,} bars")
    
    # Test on small sample first
    test_sizes = [1000, 5000, 10000]
    
    for test_size in test_sizes:
        if test_size > len(df):
            continue
            
        print(f"\n--- Testing on {test_size:,} bars ---")
        df_sample = df.head(test_size).copy()
        
        start_time = time.time()
        df_result = calculate_labels_for_all_profiles_optimized(df_sample)
        elapsed = time.time() - start_time
        
        print(f"Completed {test_size:,} bars in {elapsed:.1f} seconds")
        
        # Estimate full dataset time
        full_size = len(df)
        estimated_time = elapsed * (full_size / test_size)
        
        print(f"Estimated time for full dataset: {estimated_time/60:.1f} minutes ({estimated_time/3600:.1f} hours)")
        
        if estimated_time < 7200:  # Less than 2 hours
            print(f"✅ {test_size:,} bar test suggests full dataset is feasible")
            
            if test_size == max(test_sizes):
                return True, estimated_time
        else:
            print(f"⚠️  {test_size:,} bar test suggests full dataset would take {estimated_time/3600:.1f} hours")
    
    return False, None

if __name__ == "__main__":
    feasible, estimated_time = test_optimization()
    
    if feasible:
        print(f"\n🎉 Optimization successful!")
        print(f"Ready to process full dataset in approximately {estimated_time/60:.0f} minutes")
        
        # Optionally run full dataset
        response = input("\nProcess full dataset now? (y/n): ")
        if response.lower() == 'y':
            print("\nProcessing full dataset...")
            df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
            df_labeled = calculate_labels_for_all_profiles_optimized(df_full)
            
            # Save result
            output_path = 'project/data/processed/full_labeled_dataset.parquet'
            df_labeled.to_parquet(output_path)
            print(f"\n✅ Saved to: {output_path}")
    else:
        print(f"\n❌ Still too slow - need further optimization")