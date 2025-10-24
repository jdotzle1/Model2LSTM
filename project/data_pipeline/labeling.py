import pandas as pd
import numpy as np

# ============================================
# PROFILE DEFINITIONS
# ============================================

PROFILES = [
    {'name': 'long_2to1_small', 'direction': 'long', 'target_ticks': 12, 'stop_ticks': 6},
    {'name': 'long_2to1_medium', 'direction': 'long', 'target_ticks': 16, 'stop_ticks': 8},
    {'name': 'long_2to1_large', 'direction': 'long', 'target_ticks': 20, 'stop_ticks': 10},
    {'name': 'short_2to1_small', 'direction': 'short', 'target_ticks': 12, 'stop_ticks': 6},
    {'name': 'short_2to1_medium', 'direction': 'short', 'target_ticks': 16, 'stop_ticks': 8},
    {'name': 'short_2to1_large', 'direction': 'short', 'target_ticks': 20, 'stop_ticks': 10},
]

TICK_SIZE = 0.25  # ES tick size in points
LOOKFORWARD_SECONDS = 900  # 15 minutes


# ============================================
# MAIN LABELING FUNCTION
# ============================================

def calculate_labels_for_all_profiles(df, lookforward_seconds=LOOKFORWARD_SECONDS):
    """
    Calculate win/loss/optimal labels for all 6 profiles
    
    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        lookforward_seconds: How many seconds to look forward for target/stop
        
    Returns:
        df: DataFrame with added label columns for each profile
    """
    
    print(f"Calculating labels for {len(PROFILES)} profiles...")
    print(f"Lookforward window: {lookforward_seconds} seconds ({lookforward_seconds/60:.1f} minutes)")
    
    for i, profile in enumerate(PROFILES, 1):
        print(f"\n[{i}/{len(PROFILES)}] Processing {profile['name']}...")
        df = calculate_profile_labels(df, profile, lookforward_seconds)
    
    print("\n✓ All profiles labeled!")
    return df


# ============================================
# PROFILE LABELING
# ============================================

def calculate_profile_labels(df, profile, lookforward_seconds):
    """
    Calculate labels for a single profile
    
    Steps:
    1. For each bar, look forward to see if target/stop hit
    2. Track MAE for all winners
    3. Find consecutive winning sequences
    4. Mark only lowest MAE in each sequence as optimal (+1)
    """
    
    profile_name = profile['name']
    
    # Step 1: Determine raw outcome (win/loss/timeout)
    print("  Step 1/4: Checking targets and stops...")
    outcomes = []
    for idx in range(len(df)):
        outcome = check_target_stop(df, idx, profile, lookforward_seconds)
        outcomes.append(outcome)
    df[f'{profile_name}_outcome'] = outcomes
    
    # Step 2: Calculate MAE for all winners
    print("  Step 2/4: Calculating MAE for winners...")
    maes = []
    for idx in range(len(df)):
        if outcomes[idx] == 'win':
            mae = calculate_mae(df, idx, profile, lookforward_seconds)
            maes.append(mae)
        else:
            maes.append(np.nan)
    df[f'{profile_name}_mae'] = maes
    
    # Step 3: Find consecutive winning sequences
    print("  Step 3/4: Identifying consecutive winners...")
    df = identify_consecutive_winners(df, profile_name)
    
    # Step 4: Mark optimal winners (lowest MAE in each sequence)
    print("  Step 4/4: Applying MAE filter...")
    df[f'{profile_name}_label'] = apply_mae_filter(df, profile_name)
    
    # Print statistics
    label_counts = df[f'{profile_name}_label'].value_counts()
    print(f"  Results: Optimal={label_counts.get(1, 0)}, Suboptimal={label_counts.get(0, 0)}, Loss={label_counts.get(-1, 0)}, Timeout={df[f'{profile_name}_label'].isna().sum()}")
    
    return df


# ============================================
# CHECK TARGET/STOP
# ============================================

def check_target_stop(df, current_idx, profile, lookforward_seconds):
    """
    Look forward from current bar to see if target/stop hit first
    
    Returns: 'win', 'loss', or 'timeout'
    """
    
    entry_price = df.iloc[current_idx]['close']
    direction = profile['direction']
    target_ticks = profile['target_ticks']
    stop_ticks = profile['stop_ticks']
    
    # Convert ticks to price levels
    target_points = target_ticks * TICK_SIZE
    stop_points = stop_ticks * TICK_SIZE
    
    if direction == 'long':
        target_price = entry_price + target_points
        stop_price = entry_price - stop_points
    else:  # short
        target_price = entry_price - target_points
        stop_price = entry_price + stop_points
    
    # Look forward up to lookforward_seconds bars
    end_idx = min(current_idx + lookforward_seconds + 1, len(df))
    
    for idx in range(current_idx + 1, end_idx):
        row = df.iloc[idx]
        
        if direction == 'long':
            # Check if target hit
            if row['high'] >= target_price:
                return 'win'
            # Check if stop hit
            if row['low'] <= stop_price:
                return 'loss'
        else:  # short
            # Check if target hit
            if row['low'] <= target_price:
                return 'win'
            # Check if stop hit
            if row['high'] >= stop_price:
                return 'loss'
    
    # Neither hit within window
    return 'timeout'


# ============================================
# CALCULATE MAE
# ============================================

def calculate_mae(df, current_idx, profile, lookforward_seconds):
    """
    Calculate Maximum Adverse Excursion (worst drawdown) for a winning trade
    
    Returns: MAE in ticks (always positive number)
    """
    
    entry_price = df.iloc[current_idx]['close']
    direction = profile['direction']
    target_ticks = profile['target_ticks']
    target_points = target_ticks * TICK_SIZE
    
    # Find where target was hit
    if direction == 'long':
        target_price = entry_price + target_points
    else:
        target_price = entry_price - target_points
    
    end_idx = min(current_idx + lookforward_seconds + 1, len(df))
    
    # Track worst adverse move before target hit
    worst_adverse_move = 0
    
    for idx in range(current_idx + 1, end_idx):
        row = df.iloc[idx]
        
        if direction == 'long':
            # For longs, adverse move is going down
            adverse_move = entry_price - row['low']
            worst_adverse_move = max(worst_adverse_move, adverse_move)
            
            # Stop when target hit
            if row['high'] >= target_price:
                break
        else:  # short
            # For shorts, adverse move is going up
            adverse_move = row['high'] - entry_price
            worst_adverse_move = max(worst_adverse_move, adverse_move)
            
            # Stop when target hit
            if row['low'] <= target_price:
                break
    
    # Convert to ticks
    mae_ticks = worst_adverse_move / TICK_SIZE
    
    return mae_ticks


# ============================================
# IDENTIFY CONSECUTIVE WINNERS
# ============================================

def identify_consecutive_winners(df, profile_name):
    """
    Find sequences of consecutive winning bars
    Mark each sequence with a unique ID
    """
    
    # Create win flag
    df[f'{profile_name}_is_win'] = (df[f'{profile_name}_outcome'] == 'win').astype(int)
    
    # Identify sequence breaks (when win streak ends)
    df[f'{profile_name}_sequence_break'] = (
        df[f'{profile_name}_is_win'] != df[f'{profile_name}_is_win'].shift(1)
    ).astype(int)
    
    # Assign sequence IDs
    df[f'{profile_name}_sequence_id'] = df[f'{profile_name}_sequence_break'].cumsum()
    
    # Only keep sequence IDs for winners
    df.loc[df[f'{profile_name}_is_win'] == 0, f'{profile_name}_sequence_id'] = np.nan
    
    return df


# ============================================
# APPLY MAE FILTER
# ============================================

def apply_mae_filter(df, profile_name):
    """
    Within each consecutive winning sequence, mark ONLY the bar with lowest MAE as optimal (+1)
    All other winners in sequence get 0 (suboptimal)
    All losers get -1
    """
    
    labels = []
    
    for idx in range(len(df)):
        outcome = df.iloc[idx][f'{profile_name}_outcome']
        
        if outcome == 'loss':
            labels.append(-1)
        elif outcome == 'timeout':
            labels.append(np.nan)
        elif outcome == 'win':
            sequence_id = df.iloc[idx][f'{profile_name}_sequence_id']
            mae = df.iloc[idx][f'{profile_name}_mae']
            
            # Find all bars in this sequence
            sequence_mask = df[f'{profile_name}_sequence_id'] == sequence_id
            sequence_df = df[sequence_mask]
            
            # Find the bar with minimum MAE in this sequence
            min_mae_idx = sequence_df[f'{profile_name}_mae'].idxmin()
            
            if df.index[idx] == min_mae_idx:
                labels.append(1)  # Optimal winner
            else:
                labels.append(0)  # Suboptimal winner
        else:
            labels.append(np.nan)
    
    return labels