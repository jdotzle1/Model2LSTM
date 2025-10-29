import sys
import os
import pandas as pd

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

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
sample_cols = ['close', 'vwap', 'distance_from_vwap_pct', 'atr_30s', 'volume_ratio_30s', 'return_30s']
available_cols = [col for col in sample_cols if col in df_featured.columns]
print(df_featured[available_cols].head(20))

# Save
df_featured.to_parquet('project/data/test/test_featured_1000.parquet')
print("\n✓ Saved to project/data/test/test_featured_1000.parquet")