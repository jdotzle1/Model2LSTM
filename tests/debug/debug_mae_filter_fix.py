import pandas as pd
import numpy as np
from src.data_pipeline.labeling import calculate_labels_for_all_profiles
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

def debug_specific_problematic_bars():
    """
    Debug the specific bars that are causing issues
    """
    print("=== DEBUGGING SPECIFIC PROBLEMATIC BARS ===")
    
    # Load test data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    # Run both versions
    print("Running original version...")
    df_original = calculate_labels_for_all_profiles(df_test.copy())
    
    print("Running optimized version...")
    df_optimized = calculate_labels_for_all_profiles_optimized(df_test.copy())
    
    # Focus on the problematic bars
    problematic_bars = [985, 986, 998, 560]
    
    for bar_idx in problematic_bars:
        print(f"\n{'='*50}")
        print(f"ANALYZING BAR {bar_idx}")
        print(f"{'='*50}")
        
        # Show bar data
        bar_data = df_test.iloc[bar_idx]
        print(f"Bar data: open={bar_data['open']}, high={bar_data['high']}, low={bar_data['low']}, close={bar_data['close']}, volume={bar_data['volume']}")
        
        # Check all profiles for this bar
        for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large', 'short_2to1_small']:
            orig_outcome = df_original.iloc[bar_idx][f'{profile}_outcome']
            opt_outcome = df_optimized.iloc[bar_idx][f'{profile}_outcome']
            
            orig_label = df_original.iloc[bar_idx][f'{profile}_label']
            opt_label = df_optimized.iloc[bar_idx][f'{profile}_label']
            
            if not np.isclose(orig_label, opt_label, equal_nan=True):
                print(f"\n  {profile}:")
                print(f"    Outcome: Original={orig_outcome:8s} | Optimized={opt_outcome:8s}")
                print(f"    Label:   Original={orig_label:8.1f} | Optimized={opt_label:8.1f}")
                
                # Check if this bar has enough lookforward data
                remaining_bars = len(df_test) - bar_idx - 1
                print(f"    Remaining bars after this one: {remaining_bars}")
                print(f"    Lookforward needed: 900 seconds")
                
                if remaining_bars < 900:
                    print(f"    ⚠️  INSUFFICIENT LOOKFORWARD DATA!")
                    print(f"    This bar is too close to end of dataset for proper labeling")

def create_fixed_mae_filter():
    """
    Create a fixed version of the MAE filter that properly handles edge cases
    """
    print(f"\n{'='*50}")
    print(f"CREATING FIXED MAE FILTER")
    print(f"{'='*50}")
    
    fixed_code = '''
def apply_mae_filter_fixed(df, profile_name):
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
            
            # Mark optimal (only if it's actually a winner)
            if df.loc[optimal_idx, f'{profile_name}_outcome'] == 'win':
                df.loc[optimal_idx, f'{profile_name}_label'] = 1
    
    return df[f'{profile_name}_label']
'''
    
    print("Fixed MAE filter code:")
    print(fixed_code)
    
    print(f"\nKey fixes:")
    print(f"1. Only process bars with explicit 'win' outcome")
    print(f"2. Only mark optimal if outcome is actually 'win'")
    print(f"3. Leave timeouts and insufficient data as NaN")
    print(f"4. Don't accidentally mark losses as optimal")

if __name__ == "__main__":
    debug_specific_problematic_bars()
    create_fixed_mae_filter()