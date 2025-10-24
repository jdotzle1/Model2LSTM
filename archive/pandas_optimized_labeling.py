import pandas as pd
import numpy as np
import time

# ============================================
# PANDAS VECTORIZED LABELING
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

def calculate_labels_chunked(df, profile, chunk_size=10000, lookforward_seconds=LOOKFORWARD_SECONDS):
    """
    Process large dataset in chunks to manage memory and provide progress updates
    """
    print(f"  Processing {profile['name']} in chunks of {chunk_size:,}...")
    
    profile_name = profile['name']
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    # Initialize result columns
    df[f'{profile_name}_outcome'] = 'timeout'
    df[f'{profile_name}_mae'] = np.nan
    df[f'{profile_name}_hold_time'] = np.nan
    df[f'{profile_name}_sequence_id'] = np.nan
    df[f'{profile_name}_label'] = np.nan
    
    direction_is_long = profile['direction'] == 'long'
    target_points = profile['target_ticks'] * TICK_SIZE
    stop_points = profile['stop_ticks'] * TICK_SIZE
    
    total_processed = 0
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(df))
        
        print(f"    Chunk {chunk_idx + 1}/{n_chunks}: bars {start_idx:,} to {end_idx:,}")
        
        # Process this chunk
        for i in range(start_idx, min(end_idx, len(df) - 1)):  # -1 because we need next bar
            if i % 1000 == 0 and i > start_idx:
                print(f"      Progress: {i - start_idx:,}/{end_idx - start_idx:,} bars")
            
            # Entry price is next bar's open
            if i + 1 >= len(df):
                continue
                
            entry_price = df.iloc[i + 1]['open']
            
            if direction_is_long:
                target_price = entry_price + target_points
                stop_price = entry_price - stop_points
            else:
                target_price = entry_price - target_points
                stop_price = entry_price + stop_points
            
            # Look forward from entry bar
            entry_idx = i + 1
            end_search_idx = min(entry_idx + lookforward_seconds, len(df))
            
            worst_adverse = 0.0
            outcome = 'timeout'
            hold_time = np.nan
            
            # Check each future bar
            for j in range(entry_idx, end_search_idx):
                row = df.iloc[j]
                
                if direction_is_long:
                    target_hit = row['high'] >= target_price
                    stop_hit = row['low'] <= stop_price
                    adverse_move = entry_price - row['low']
                else:
                    target_hit = row['low'] <= target_price
                    stop_hit = row['high'] >= stop_price
                    adverse_move = row['high'] - entry_price
                
                worst_adverse = max(worst_adverse, adverse_move)
                
                if target_hit and stop_hit:
                    outcome = 'loss'  # Conservative: assume stop hit first
                    break
                elif target_hit:
                    outcome = 'win'
                    hold_time = j - entry_idx
                    break
                elif stop_hit:
                    outcome = 'loss'
                    break
            
            # Store results
            df.iloc[i, df.columns.get_loc(f'{profile_name}_outcome')] = outcome
            if outcome == 'win':
                df.iloc[i, df.columns.get_loc(f'{profile_name}_mae')] = worst_adverse / TICK_SIZE
                df.iloc[i, df.columns.get_loc(f'{profile_name}_hold_time')] = hold_time
        
        total_processed += (end_idx - start_idx)
        print(f"    Completed chunk {chunk_idx + 1}/{n_chunks}")
    
    # Now identify sequences and apply MAE filter
    print(f"    Identifying winning sequences...")
    df = identify_sequences_and_filter(df, profile_name)
    
    return df

def identify_sequences_and_filter(df, profile_name):
    """
    Identify consecutive winning sequences and apply MAE filter
    """
    # Create win mask
    is_win = df[f'{profile_name}_outcome'] == 'win'
    
    # Find sequence breaks
    sequence_breaks = is_win != is_win.shift(1)
    sequence_ids = sequence_breaks.cumsum()
    
    # Only assign IDs to winners
    df[f'{profile_name}_sequence_id'] = np.where(is_win, sequence_ids, np.nan)
    
    # Apply MAE filter
    labels = np.full(len(df), np.nan)
    
    # Mark losses
    labels[df[f'{profile_name}_outcome'] == 'loss'] = -1
    
    # Process each winning sequence
    winning_sequences = df[is_win][f'{profile_name}_sequence_id'].unique()
    
    for seq_id in winning_sequences:
        if pd.isna(seq_id):
            continue
            
        seq_mask = df[f'{profile_name}_sequence_id'] == seq_id
        seq_data = df[seq_mask]
        
        # Find minimum MAE
        min_mae = seq_data[f'{profile_name}_mae'].min()
        min_mae_mask = seq_data[f'{profile_name}_mae'] == min_mae
        min_mae_candidates = seq_data[min_mae_mask]
        
        if len(min_mae_candidates) == 1:
            optimal_idx = min_mae_candidates.index[0]
        else:
            # Tie-breaker: shortest hold time
            optimal_idx = min_mae_candidates[f'{profile_name}_hold_time'].idxmin()
        
        # Mark all in sequence as suboptimal, then mark optimal
        seq_indices = seq_data.index
        labels[seq_indices] = 0  # suboptimal
        labels[optimal_idx] = 1  # optimal
    
    df[f'{profile_name}_label'] = labels
    
    # Print statistics
    label_counts = pd.Series(labels).value_counts()
    print(f"    Results: Optimal={label_counts.get(1.0, 0)}, Suboptimal={label_counts.get(0.0, 0)}, Loss={label_counts.get(-1.0, 0)}, Timeout={pd.isna(labels).sum()}")
    
    return df

def calculate_labels_for_all_profiles_chunked(df, chunk_size=10000, lookforward_seconds=LOOKFORWARD_SECONDS):
    """
    Main function with chunked processing
    """
    print(f"Calculating labels for {len(PROFILES)} profiles (CHUNKED)...")
    print(f"Dataset size: {len(df):,} bars")
    print(f"Chunk size: {chunk_size:,} bars")
    print(f"Lookforward window: {lookforward_seconds} seconds ({lookforward_seconds/60:.1f} minutes)")
    
    df_result = df.copy()
    
    for i, profile in enumerate(PROFILES, 1):
        print(f"\n[{i}/{len(PROFILES)}] Processing {profile['name']}...")
        start_time = time.time()
        
        df_result = calculate_labels_chunked(df_result, profile, chunk_size, lookforward_seconds)
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    print("\n✓ All profiles labeled!")
    return df_result

def test_chunked_approach():
    """
    Test the chunked approach on sample data
    """
    print("=== TESTING CHUNKED APPROACH ===")
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_parquet('project/data/raw/test_sample.parquet')
    print(f"Loaded {len(df):,} bars")
    
    # Test on small sample first
    sample_size = 5000
    df_sample = df.head(sample_size).copy()
    
    print(f"\nTesting on {sample_size:,} bars...")
    start_time = time.time()
    
    df_result = calculate_labels_for_all_profiles_chunked(df_sample, chunk_size=1000)
    
    elapsed = time.time() - start_time
    print(f"\nSample completed in {elapsed:.1f} seconds")
    
    # Estimate full dataset time
    full_size = len(df)
    estimated_time = elapsed * (full_size / sample_size)
    
    print(f"\nEstimated time for full dataset ({full_size:,} bars):")
    print(f"  {estimated_time/60:.1f} minutes ({estimated_time/3600:.1f} hours)")
    
    if estimated_time < 7200:  # Less than 2 hours
        print(f"✅ Should complete in reasonable time")
        
        # Ask if user wants to proceed with full dataset
        print(f"\nReady to process full dataset? This will take approximately {estimated_time/60:.0f} minutes.")
        return True, df_result
    else:
        print(f"⚠️  Still quite slow - may need further optimization")
        return False, df_result

if __name__ == "__main__":
    ready, sample_result = test_chunked_approach()
    
    if ready:
        print("\nChunked approach looks promising for full dataset processing.")
    else:
        print("\nNeed further optimization before processing full dataset.")