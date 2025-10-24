import pandas as pd

# Load the labeled test sample and examine structure
print("Loading labeled test sample...")
df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')

print(f"Shape: {df.shape}")
print(f"\nAll columns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"{i:2d}: {col}")

print(f"\nFirst few rows of core data:")
core_cols = [col for col in df.columns if col in ['timestamp', 'ts_event', 'ts_recv', 'open', 'high', 'low', 'close', 'volume']]
if core_cols:
    print(df[core_cols].head())
else:
    print("No timestamp columns found! Showing first 5 columns:")
    print(df.iloc[:, :5].head())

print(f"\nData types:")
print(df.dtypes.head(10))

# Check if there's an index that might be the timestamp
print(f"\nIndex info:")
print(f"Index name: {df.index.name}")
print(f"Index type: {type(df.index)}")
print(f"First few index values: {df.index[:5].tolist()}")