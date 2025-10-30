import pandas as pd
from src.data_pipeline.labeling import calculate_labels_for_all_profiles

# Load test sample
print("Loading test sample...")
df = pd.read_parquet('project/data/raw/test_sample.parquet')
print(f"Loaded {len(df):,} bars")

# Test on first 1000 bars only (quick test)
df_test = df.head(1000).copy()

# Calculate labels
print("\nCalculating labels...")
df_labeled = calculate_labels_for_all_profiles(df_test)

# Show results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']:
    print(f"\n{profile}:")
    print(df_labeled[f'{profile}_label'].value_counts())

# Save
df_labeled.to_parquet('project/data/test/test_labeled_1000.parquet')
print("\n✓ Saved to project/data/test/test_labeled_1000.parquet")