import pandas as pd

# Load the data and analyze sequences in detail
df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')

print("=== CONSECUTIVE WINNING SEQUENCE ANALYSIS ===")
print(f"Total bars analyzed: {len(df)}")
print(f"Time span: ~{len(df)/60:.1f} minutes")

for profile in ['long_2to1_small', 'short_2to1_small']:
    print(f"\n{profile.upper()}:")
    
    # Get all outcomes
    outcomes = df[f'{profile}_outcome']
    winners = df[df[f'{profile}_outcome'] == 'win']
    
    print(f"Total bars: {len(df)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(df)*100:.1f}%)")
    print(f"Losses: {(outcomes == 'loss').sum()}")
    print(f"Timeouts: {outcomes.isna().sum()}")
    
    # Analyze sequences
    if len(winners) > 0:
        sequence_counts = winners.groupby(f'{profile}_sequence_id').size().sort_values(ascending=False)
        print(f"\nNumber of consecutive winning sequences: {len(sequence_counts)}")
        print(f"Sequence sizes (largest first):")
        for i, (seq_id, size) in enumerate(sequence_counts.head(10).items()):
            print(f"  Sequence {seq_id}: {size} consecutive winners")
        
        print(f"\nSequence size distribution:")
        size_dist = sequence_counts.value_counts().sort_index()
        for size, count in size_dist.items():
            print(f"  {count} sequences of length {size}")
        
        # This is the key question: How can we have so many long sequences?
        total_winner_bars = sequence_counts.sum()
        print(f"\nTotal winner bars: {total_winner_bars}")
        print(f"Average sequence length: {sequence_counts.mean():.1f}")
        
        # Check if sequences make sense
        largest_seq = sequence_counts.iloc[0]
        print(f"\nLargest sequence has {largest_seq} consecutive winners")
        print(f"That means {largest_seq} bars in a row ALL won their trades!")
        print(f"In a {len(df)/60:.1f} minute period, that's {largest_seq/(len(df)/60):.1f} winners per minute")
        
        if largest_seq > 100:
            print(f"🚨 SUSPICIOUS: {largest_seq} consecutive winners is extremely unlikely!")
            print(f"   This suggests a bug in sequence identification or unrealistic market conditions")

# Let's examine the sequence identification logic by looking at raw outcomes
print(f"\n=== RAW OUTCOME PATTERN ANALYSIS ===")
profile = 'long_2to1_small'
outcomes = df[f'{profile}_outcome'].values

# Count consecutive wins manually
consecutive_wins = []
current_streak = 0

for outcome in outcomes:
    if outcome == 'win':
        current_streak += 1
    else:
        if current_streak > 0:
            consecutive_wins.append(current_streak)
        current_streak = 0

# Don't forget the last streak if it ends with wins
if current_streak > 0:
    consecutive_wins.append(current_streak)

print(f"Manual count of consecutive win streaks: {len(consecutive_wins)}")
print(f"Longest manual streak: {max(consecutive_wins) if consecutive_wins else 0}")
print(f"All streaks: {sorted(consecutive_wins, reverse=True)[:10]}")

# Compare with our sequence logic
winners = df[df[f'{profile}_outcome'] == 'win']
if len(winners) > 0:
    sequence_counts = winners.groupby(f'{profile}_sequence_id').size()
    print(f"Our sequence logic found: {len(sequence_counts)} sequences")
    print(f"Largest sequence: {sequence_counts.max()}")
    
    if len(consecutive_wins) != len(sequence_counts):
        print(f"🚨 MISMATCH: Manual count ({len(consecutive_wins)}) != Sequence logic ({len(sequence_counts)})")
    else:
        print(f"✅ Counts match")
        
    if max(consecutive_wins) != sequence_counts.max():
        print(f"🚨 MISMATCH: Manual max ({max(consecutive_wins)}) != Sequence max ({sequence_counts.max()})")
    else:
        print(f"✅ Max lengths match")