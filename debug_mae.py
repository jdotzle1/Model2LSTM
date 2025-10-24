import pandas as pd
import numpy as np

# Load and examine the MAE filtering
df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')

print("=== MAE FILTERING ANALYSIS ===")

# Focus on long_2to1_small for detailed analysis
profile = 'long_2to1_small'
winners = df[df[f'{profile}_outcome'] == 'win'].copy()
optimal = df[df[f'{profile}_label'] == 1].copy()

print(f"Total winners: {len(winners)}")
print(f"Optimal winners: {len(optimal)}")
print(f"Optimal percentage of winners: {len(optimal)/len(winners)*100:.1f}%")

print(f"\n=== SEQUENCE ANALYSIS ===")
# Look at sequence lengths
sequence_lengths = winners.groupby(f'{profile}_sequence_id').size()
print(f"Number of winning sequences: {len(sequence_lengths)}")
print(f"Average sequence length: {sequence_lengths.mean():.1f}")
print(f"Sequence length distribution:")
print(sequence_lengths.value_counts().sort_index())

print(f"\n=== MAE DISTRIBUTION ===")
print(f"MAE statistics for all winners:")
print(winners[f'{profile}_mae'].describe())

print(f"\nMAE statistics for optimal entries:")
print(optimal[f'{profile}_mae'].describe())

print(f"\n=== SAMPLE SEQUENCES ===")
# Show a few sequences in detail
for seq_id in sequence_lengths.head(3).index:
    seq_data = winners[winners[f'{profile}_sequence_id'] == seq_id]
    print(f"\nSequence {seq_id} (length {len(seq_data)}):")
    cols = ['open', 'close', f'{profile}_mae', f'{profile}_label']
    print(seq_data[cols])

print(f"\n=== PROBLEM DIAGNOSIS ===")
single_bar_sequences = (sequence_lengths == 1).sum()
print(f"Single-bar sequences: {single_bar_sequences} out of {len(sequence_lengths)}")
print(f"Single-bar sequence percentage: {single_bar_sequences/len(sequence_lengths)*100:.1f}%")

if single_bar_sequences > len(sequence_lengths) * 0.5:
    print("🚨 PROBLEM: Too many single-bar sequences!")
    print("   Every isolated winner becomes 'optimal' by default")
    print("   This inflates the optimal count artificially")