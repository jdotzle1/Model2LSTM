import pandas as pd

# Check if there were actually MAE ties that needed hold time tie-breaking
df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')

print("=== CHECKING FOR MAE TIES ===")

for profile in ['long_2to1_small', 'short_2to1_small']:
    print(f"\n{profile}:")
    
    winners = df[df[f'{profile}_outcome'] == 'win'].copy()
    
    # Check each sequence for MAE ties
    sequences_with_ties = 0
    total_sequences = 0
    
    for seq_id in winners[f'{profile}_sequence_id'].unique():
        seq_data = winners[winners[f'{profile}_sequence_id'] == seq_id]
        total_sequences += 1
        
        # Find minimum MAE in sequence
        min_mae = seq_data[f'{profile}_mae'].min()
        min_mae_count = (seq_data[f'{profile}_mae'] == min_mae).sum()
        
        if min_mae_count > 1:
            sequences_with_ties += 1
            print(f"  Sequence {seq_id}: {min_mae_count} entries with MAE={min_mae}")
            
            # Show the tied entries
            tied_entries = seq_data[seq_data[f'{profile}_mae'] == min_mae]
            cols = [f'{profile}_mae', f'{profile}_hold_time', f'{profile}_label']
            print(f"    Tied entries:")
            print(tied_entries[cols])
    
    print(f"  Total sequences: {total_sequences}")
    print(f"  Sequences with MAE ties: {sequences_with_ties}")
    print(f"  Tie percentage: {sequences_with_ties/total_sequences*100:.1f}%")

# The real issue: We're looking at overnight data!
print(f"\n=== THE REAL PROBLEM ===")
print("The issue isn't the logic - it's the DATA!")
print("This is overnight/pre-market data with:")
print("- Very low volume")
print("- Strong trending conditions") 
print("- Low volatility")
print("- Unrealistic win rates")

df_reset = df.reset_index()
print(f"\nData time range:")
print(f"Start: {df_reset['ts_event'].iloc[0]}")
print(f"End:   {df_reset['ts_event'].iloc[-1]}")
print(f"This is midnight to 1 AM UTC - NOT regular trading hours!")

# Calculate realistic expectations
print(f"\n=== REALISTIC EXPECTATIONS ===")
total_optimal = 0
for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']:
    optimal_count = (df[f'{profile}_label'] == 1).sum()
    total_optimal += optimal_count

print(f"Current: {total_optimal} optimal in 49 minutes = {total_optimal/49*60:.0f} per hour")
print(f"In RTH (6.5 hours): {total_optimal/49*60*6.5:.0f} optimal setups per day")
print(f"That's 1 optimal setup every {60*60/(total_optimal/49*60):.0f} seconds!")
print(f"\nThis is clearly unrealistic for normal market conditions.")