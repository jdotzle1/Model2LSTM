import pandas as pd
import numpy as np
from numba import jit
import time

# ============================================
# OPTIMIZED LABELING FUNCTIONS
# ============================================

TICK_SIZE = 0.25
LOOKFORWARD_SECONDS = 900

PROFILES = [
    {'name': 'long_2to1_small', 'direction': 'long', 'target_ticks': 12, 'stop_ticks': 6},
    {'name': 'long_2to1_medium', 'direction': 'long', 'target_ticks': 16, 'stop_ticks': 8},
    {'name': 'long_2to1_large', 'direction': 'long', 'target_ticks': 20, 'stop_ticks': 10},
    {'name': 'short_2to1_small', 'direction': 'short', 'target_ticks': 12, 'stop_ticks': 6},
    {'name': 'short_2to1_medium', 'direction': 'short', 'target_ticks': 16, 'stop_ticks': 8},
    {'name': 'short_2to1_large', 'direction': 'short', 'target_ticks': 20, 'stop_ticks': 10},
]

@jit(nopython=True)
def vectorized_target_stop_check(opens, highs, lows, target_ticks, stop_ticks, direction_is_long, lookforward):
    """
    Vectorized target/stop checking using Numba for speed
    
    Returns arrays of outcomes, MAEs, and hold times
    """
    n = len(opens)
    outcomes = np.full(n, -999, dtype=np.int8)  # -1=loss, 0=timeout, 1=win, -999=invalid
    maes = np.full(n, np.nan, dtype=np.float32)
    hold_times = np.full(n, np.nan, dtype=np.float32)
    
    target_points = target_ticks * TICK_SIZE
    stop_points = stop_ticks * TICK_SIZE
    
    # Process each potential entry point
    for i in range(n - 1):  # -1 because we need next bar for entry
        entry_price = opens[i + 1]  # Enter at next bar's open
        
        if direction_is_long:
            target_price = entry_price + target_points
            stop_price = entry_price - stop_points
        else:
            target_price = entry_price - target_points
            stop_price = entry_price + stop_points
        
        # Look forward from entry bar
        entry_idx = i + 1
        end_idx = min(entry_idx + lookforward, n)
        
        worst_adverse = 0.0
        
        for j in range(entry_idx, end_idx):
            if direction_is_long:
                # Long trade logic
                target_hit = highs[j] >= target_price
                stop_hit = lows[j] <= stop_price
                
                # Track MAE (adverse move)
                adverse_move = entry_price - lows[j]
                worst_adverse = max(worst_adverse, adverse_move)
                
                if target_hit and stop_hit:
                    # Both hit - conservative assumption: stop first
                    outcomes[i] = -1  # loss
                    break
                elif target_hit:
                    # Target hit first
                    outcomes[i] = 1  # win
                    maes[i] = worst_adverse / TICK_SIZE
                    hold_times[i] = j - entry_idx
                    break
                elif stop_hit:
                    # Stop hit first
                    outcomes[i] = -1  # loss
                    break
            else:
                # Short trade logic
                target_hit = lows[j] <= target_price
                stop_hit = highs[j] >= stop_price
                
                # Track MAE (adverse move)
                adverse_move = highs[j] - entry_price
                worst_adverse = max(worst_adverse, adverse_move)
                
                if target_hit and stop_hit:
                    # Both hit - conservative assumption: stop first
                    outcomes[i] = -1  # loss
                    break
                elif target_hit:
                    # Target hit first
                    outcomes[i] = 1  # win
                    maes[i] = worst_adverse / TICK_SIZE
                    hold_times[i] = j - entry_idx
                    break
                elif stop_hit:
                    # Stop hit first
                    outcomes[i] = -1  # loss
                    break
        else:
            # Neither hit within lookforward window
            outcomes[i] = 0  # timeout
    
    return outcomes, maes, hold_times

def identify_consecutive_sequences_vectorized(outcomes):
    """
    Vectorized identification of consecutive winning sequences
    """
    # Create win mask
    is_win = (outcomes == 1)
    
    # Find sequence breaks
    sequence_breaks = np.diff(np.concatenate(([False], is_win, [False]))).astype(int)
    
    # Get start and end indices of winning sequences
    starts = np.where(sequence_breaks == 1)[0]
    ends = np.where(sequence_breaks == -1)[0]
    
    # Assign sequence IDs
    sequence_ids = np.full(len(outcomes), np.nan)
    for seq_id, (start, end) in enumerate(zip(starts, ends), 1):
        sequence_ids[start:end] = seq_id
    
    return sequence_ids

def apply_mae_filter_vectorized(outcomes, maes, hold_times, sequence_ids):
    """
    Vectorized MAE filtering to select optimal entries
    """
    labels = np.full(len(outcomes), np.nan)
    
    # Handle losses
    labels[outcomes == -1] = -1
    
    # Handle timeouts (keep as NaN)
    
    # Handle winners - find optimal in each sequence
    unique_sequences = sequence_ids[~np.isnan(sequence_ids)]
    if len(unique_sequences) > 0:
        for seq_id in np.unique(unique_sequences):
            seq_mask = sequence_ids == seq_id
            seq_indices = np.where(seq_mask)[0]
            
            if len(seq_indices) > 0:
                # Find minimum MAE in sequence
                seq_maes = maes[seq_indices]
                min_mae = np.nanmin(seq_maes)
                min_mae_mask = seq_maes == min_mae
                min_mae_indices = seq_indices[min_mae_mask]
                
                if len(min_mae_indices) == 1:
                    # Single winner with lowest MAE
                    optimal_idx = min_mae_indices[0]
                else:
                    # Tie in MAE - pick shortest hold time
                    tie_hold_times = hold_times[min_mae_indices]
                    min_hold_idx = np.nanargmin(tie_hold_times)
                    optimal_idx = min_mae_indices[min_hold_idx]
                
                # Mark all in sequence
                labels[seq_indices] = 0  # suboptimal
                labels[optimal_idx] = 1  # optimal
    
    return labels

def calculate_labels_optimized(df, profile, lookforward_seconds=LOOKFORWARD_SECONDS):
    """
    Optimized labeling for a single profile using vectorized operations
    """
    print(f"  Processing {profile['name']} (optimized)...")
    
    # Extract arrays for vectorized processing
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    direction_is_long = profile['direction'] == 'long'
    
    # Vectorized target/stop checking
    print(f"    Step 1/4: Vectorized target/stop checking...")
    outcomes, maes, hold_times = vectorized_target_stop_check(
        opens, highs, lows, 
        profile['target_ticks'], profile['stop_ticks'], 
        direction_is_long, lookforward_seconds
    )
    
    # Convert outcomes to string format for compatibility
    outcome_map = {-1: 'loss', 0: 'timeout', 1: 'win', -999: 'timeout'}
    outcomes_str = np.array([outcome_map[x] for x in outcomes])
    
    print(f"    Step 2/4: Identifying consecutive sequences...")
    sequence_ids = identify_consecutive_sequences_vectorized(outcomes)
    
    print(f"    Step 3/4: Applying MAE filter...")
    labels = apply_mae_filter_vectorized(outcomes, maes, hold_times, sequence_ids)
    
    # Add results to dataframe
    profile_name = profile['name']
    df[f'{profile_name}_outcome'] = outcomes_str
    df[f'{profile_name}_mae'] = maes
    df[f'{profile_name}_hold_time'] = hold_times
    df[f'{profile_name}_sequence_id'] = sequence_ids
    df[f'{profile_name}_label'] = labels
    
    # Print statistics
    label_counts = pd.Series(labels).value_counts()
    print(f"    Results: Optimal={label_counts.get(1.0, 0)}, Suboptimal={label_counts.get(0.0, 0)}, Loss={label_counts.get(-1.0, 0)}, Timeout={pd.isna(labels).sum()}")
    
    return df

def calculate_labels_for_all_profiles_optimized(df, lookforward_seconds=LOOKFORWARD_SECONDS):
    """
    Optimized main labeling function
    """
    print(f"Calculating labels for {len(PROFILES)} profiles (OPTIMIZED)...")
    print(f"Lookforward window: {lookforward_seconds} seconds ({lookforward_seconds/60:.1f} minutes)")
    
    df_result = df.copy()
    
    for i, profile in enumerate(PROFILES, 1):
        print(f"\n[{i}/{len(PROFILES)}] Processing {profile['name']}...")
        start_time = time.time()
        
        df_result = calculate_labels_optimized(df_result, profile, lookforward_seconds)
        
        elapsed = time.time() - start_time
        print(f"    Completed in {elapsed:.1f} seconds")
    
    print("\n✓ All profiles labeled (OPTIMIZED)!")
    return df_result

# ============================================
# PERFORMANCE COMPARISON
# ============================================

def benchmark_optimization(df_sample):
    """
    Compare old vs new performance on a sample
    """
    print("=== PERFORMANCE BENCHMARK ===")
    
    # Test on small sample first
    sample_size = min(1000, len(df_sample))
    df_test = df_sample.head(sample_size).copy()
    
    print(f"Testing on {sample_size} bars...")
    
    # Optimized version
    print("\nTesting OPTIMIZED version...")
    start_time = time.time()
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test)
    optimized_time = time.time() - start_time
    
    print(f"\nOptimized version: {optimized_time:.2f} seconds")
    
    # Estimate full dataset time
    full_size = len(df_sample)
    estimated_full_time = optimized_time * (full_size / sample_size)
    
    print(f"\nEstimated time for full dataset ({full_size:,} bars):")
    print(f"  Optimized: {estimated_full_time/60:.1f} minutes")
    
    if estimated_full_time < 3600:  # Less than 1 hour
        print(f"✅ Optimized version should complete in reasonable time")
    else:
        print(f"⚠️  Still may take {estimated_full_time/3600:.1f} hours")
    
    return df_optimized, optimized_time

if __name__ == "__main__":
    # Load and test
    print("Loading dataset...")
    df = pd.read_parquet('project/data/raw/test_sample.parquet')
    print(f"Loaded {len(df):,} bars")
    
    # Run benchmark
    df_result, timing = benchmark_optimization(df)
    
    print(f"\n=== OPTIMIZATION SUMMARY ===")
    print(f"Successfully processed sample with optimized algorithm")
    print(f"Ready to scale to full dataset if performance is acceptable")