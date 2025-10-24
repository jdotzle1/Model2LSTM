import pandas as pd

# Load both old and new results to compare
print("=== DEBUGGING SEQUENCE LOGIC ===")

df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')

# Focus on long_2to1_small for detailed analysis
profile = 'long_2to1_small'
winners = df[df[f'{profile}_outcome'] == 'win'].copy()
optimal = df[df[f'{profile}_label'] == 1].copy()

print(f"Profile: {profile}")
print(f"Total winners: {len(winners)}")
print(f"Optimal entries: {len(optimal)}")

# Check sequence distribution
print(f"\n=== SEQUENCE ANALYSIS ===")
sequence_counts = winners.groupby(f'{profile}_sequence_id').size().sort_values(ascending=False)
print(f"Number of sequences: {len(sequence_counts)}")
print(f"Sequence sizes:")
print(sequence_counts.head(10))

# Check if we have the right number of optimal entries
print(f"\nExpected optimal entries (1 per sequence): {len(sequence_counts)}")
print(f"Actual optimal entries: {len(optimal)}")

if len(optimal) != len(sequence_counts):
    print("❌ MISMATCH! Not 1 optimal per sequence")
else:
    print("✅ Correct count")

# Let's examine a large sequence in detail
largest_seq_id = sequence_counts.index[0]
largest_seq_size = sequence_counts.iloc[0]

print(f"\n=== EXAMINING LARGEST SEQUENCE ===")
print(f"Sequence ID: {largest_seq_id}, Size: {largest_seq_size}")

seq_data = winners[winners[f'{profile}_sequence_id'] == largest_seq_id].copy()
print(f"Sequence data:")
cols = ['open', 'close', f'{profile}_mae', f'{profile}_hold_time', f'{profile}_label']
print(seq_data[cols].head(10))

# Count optimal in this sequence
optimal_in_seq = (seq_data[f'{profile}_label'] == 1).sum()
print(f"Optimal entries in this sequence: {optimal_in_seq}")

if optimal_in_seq != 1:
    print("❌ ERROR: Should be exactly 1 optimal per sequence!")
    
    # Show all optimal entries in this sequence
    optimal_in_this_seq = seq_data[seq_data[f'{profile}_label'] == 1]
    print("All optimal entries in this sequence:")
    print(optimal_in_this_seq[cols])
    
    # Check MAE values
    print(f"\nMAE distribution in sequence:")
    print(seq_data[f'{profile}_mae'].value_counts().sort_index())
    
    min_mae = seq_data[f'{profile}_mae'].min()
    print(f"Minimum MAE in sequence: {min_mae}")
    
    min_mae_entries = seq_data[seq_data[f'{profile}_mae'] == min_mae]
    print(f"Entries with minimum MAE ({len(min_mae_entries)}):")
    print(min_mae_entries[cols])

# Check a few more sequences
print(f"\n=== CHECKING OTHER SEQUENCES ===")
for i in range(1, min(4, len(sequence_counts))):
    seq_id = sequence_counts.index[i]
    seq_size = sequence_counts.iloc[i]
    seq_data = winners[winners[f'{profile}_sequence_id'] == seq_id]
    optimal_count = (seq_data[f'{profile}_label'] == 1).sum()
    print(f"Sequence {seq_id} (size {seq_size}): {optimal_count} optimal entries")