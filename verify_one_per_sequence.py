import pandas as pd

# Load the updated results
df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')

print("=== VERIFICATION: ONE OPTIMAL PER SEQUENCE ===")

for profile in ['long_2to1_small', 'short_2to1_small']:  # Test two profiles
    print(f"\n{profile}:")
    
    # Get all winners
    winners = df[df[f'{profile}_outcome'] == 'win'].copy()
    optimal = df[df[f'{profile}_label'] == 1].copy()
    
    # Count sequences
    unique_sequences = winners[f'{profile}_sequence_id'].nunique()
    optimal_count = len(optimal)
    
    print(f"  Total winning sequences: {unique_sequences}")
    print(f"  Optimal entries: {optimal_count}")
    print(f"  Ratio: {optimal_count/unique_sequences:.2f} optimal per sequence")
    
    if optimal_count == unique_sequences:
        print("  ✅ CORRECT: Exactly 1 optimal entry per winning sequence")
    else:
        print("  ❌ ERROR: Not 1 optimal per sequence")
        
        # Debug: find sequences with wrong counts
        sequence_optimal_counts = optimal.groupby(f'{profile}_sequence_id').size()
        print(f"  Sequences with multiple optimal entries:")
        multiple = sequence_optimal_counts[sequence_optimal_counts > 1]
        if len(multiple) > 0:
            print(multiple)
        
        # Check if any sequences have 0 optimal entries
        all_sequence_ids = set(winners[f'{profile}_sequence_id'].unique())
        optimal_sequence_ids = set(optimal[f'{profile}_sequence_id'].unique())
        missing = all_sequence_ids - optimal_sequence_ids
        if missing:
            print(f"  Sequences with 0 optimal entries: {len(missing)}")

print(f"\n=== SAMPLE OPTIMAL ENTRIES WITH HOLD TIME ===")
profile = 'long_2to1_small'
optimal_sample = df[df[f'{profile}_label'] == 1].copy()
if len(optimal_sample) > 0:
    cols = ['open', 'close', f'{profile}_mae', f'{profile}_hold_time', f'{profile}_sequence_id']
    print(optimal_sample[cols].head(10))