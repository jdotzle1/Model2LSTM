import pandas as pd
from project.data_pipeline.features import create_all_features

# Load labeled test sample
print("Loading labeled test sample...")
df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')
print(f"Loaded {len(df):,} bars")

# Create features
print("\nCreating features...")
df_featured = create_all_features(df)

print("\n" + "="*60)
print("FEATURE VALIDATION")
print("="*60)
print(f"Total columns: {len(df_featured.columns)}")
print(f"NaN counts:")
print(df_featured.isnull().sum().sort_values(ascending=False).head(20))

print("\nSample features:")
print(df_featured[['close', 'return_5s', 'vwap', 'distance_from_vwap', 'atr_30s', 'volume_ratio_30s']].head(20))

# Save
df_featured.to_parquet('project/data/test/test_featured_1000.parquet')
print("\n✓ Saved to project/data/test/test_featured_1000.parquet")