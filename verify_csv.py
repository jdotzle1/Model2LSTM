import pandas as pd

# Load the CSV to verify timestamp is included
print("Verifying CSV with timestamp...")
df_csv = pd.read_csv('project/data/test/test_labeled_1000_with_timestamp.csv')

print(f"CSV Shape: {df_csv.shape}")
print(f"\nFirst 5 columns:")
print(df_csv.columns[:5].tolist())

print(f"\nFirst few rows with timestamp and OHLCV:")
cols_to_show = ['ts_event', 'open', 'high', 'low', 'close', 'volume']
print(df_csv[cols_to_show].head(10))

print(f"\nTimestamp range:")
print(f"First: {df_csv['ts_event'].iloc[0]}")
print(f"Last:  {df_csv['ts_event'].iloc[-1]}")

print(f"\nSample optimal entries (long_2to1_small_label = 1):")
optimal_mask = df_csv['long_2to1_small_label'] == 1
if optimal_mask.any():
    optimal_cols = ['ts_event', 'close', 'long_2to1_small_outcome', 'long_2to1_small_mae', 'long_2to1_small_label']
    print(df_csv[optimal_mask][optimal_cols].head())
else:
    print("No optimal entries found")