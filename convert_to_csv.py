import pandas as pd

# Load the labeled test sample
print("Loading labeled test sample...")
df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')
print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

# Reset index to make timestamp a column
df_with_timestamp = df.reset_index()
print(f"After adding timestamp column: {df_with_timestamp.shape}")

# Convert to CSV with timestamp included
output_path = 'project/data/test/test_labeled_1000_with_timestamp.csv'
df_with_timestamp.to_csv(output_path, index=False)
print(f"✓ Saved to {output_path}")

# Show basic info
print(f"\nDataset info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Show label distributions for quick verification
print(f"\nLabel distributions:")
for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']:
    if f'{profile}_label' in df.columns:
        counts = df[f'{profile}_label'].value_counts().sort_index()
        print(f"{profile}: {dict(counts)}")