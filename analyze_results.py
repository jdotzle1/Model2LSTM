import pandas as pd

# Load the results and analyze the frequency
df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')

print("=== ANALYSIS OF OPTIMAL ENTRY FREQUENCY ===")
print(f"Total bars: {len(df)}")
print(f"Time span: ~{len(df)/60:.1f} minutes ({len(df)} seconds)")

# Count optimal entries across all profiles
total_optimal = 0
for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']:
    optimal_count = (df[f'{profile}_label'] == 1).sum()
    total_optimal += optimal_count
    print(f"{profile}: {optimal_count} optimal entries")

print(f"\nTotal optimal entries: {total_optimal}")
print(f"Optimal entries per minute: {total_optimal / (len(df)/60):.1f}")
print(f"Optimal entry frequency: {total_optimal / len(df) * 100:.1f}% of all bars")

# This means in live trading:
print(f"\n=== LIVE TRADING IMPLICATIONS ===")
print(f"At this rate, we'd have:")
print(f"- {total_optimal * 60:.0f} optimal setups per hour")
print(f"- {total_optimal * 60 * 6.5:.0f} optimal setups per trading day (6.5 hours)")
print(f"- That's an optimal setup every {60*60/total_optimal:.0f} seconds!")

# Let's look at the time distribution
print(f"\n=== TIME DISTRIBUTION OF OPTIMAL ENTRIES ===")
df_reset = df.reset_index()
optimal_mask = df_reset[f'long_2to1_small_label'] == 1
for profile in ['long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']:
    optimal_mask |= (df_reset[f'{profile}_label'] == 1)

optimal_times = df_reset[optimal_mask]['ts_event']
print(f"First optimal: {optimal_times.iloc[0]}")
print(f"Last optimal:  {optimal_times.iloc[-1]}")
print(f"Spread over:   {(optimal_times.iloc[-1] - optimal_times.iloc[0]).total_seconds()/60:.1f} minutes")

# Check if they're clustered
print(f"\n=== CLUSTERING ANALYSIS ===")
time_diffs = optimal_times.diff().dt.total_seconds().dropna()
print(f"Average time between optimal entries: {time_diffs.mean():.1f} seconds")
print(f"Median time between optimal entries: {time_diffs.median():.1f} seconds")
print(f"Min time between optimal entries: {time_diffs.min():.1f} seconds")
print(f"Max time between optimal entries: {time_diffs.max():.1f} seconds")

# Show some examples
print(f"\n=== SAMPLE OPTIMAL ENTRIES ===")
for profile in ['long_2to1_small', 'short_2to1_small']:
    optimal_df = df_reset[df[f'{profile}_label'] == 1]
    if len(optimal_df) > 0:
        print(f"\n{profile} optimal entries:")
        cols = ['ts_event', 'open', 'high', 'low', 'close', f'{profile}_mae']
        print(optimal_df[cols].head())