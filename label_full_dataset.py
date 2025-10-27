import pandas as pd
import time
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized

# Load the full dataset
print("Loading full dataset...")
df = pd.read_parquet('project/data/raw/test_sample.parquet')
print(f"Loaded {len(df):,} bars")

# Show basic info about the dataset
print(f"\nDataset info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check time range
df_reset = df.reset_index()
if 'ts_event' in df_reset.columns:
    print(f"Time range:")
    print(f"  Start: {df_reset['ts_event'].iloc[0]}")
    print(f"  End:   {df_reset['ts_event'].iloc[-1]}")
    time_span = df_reset['ts_event'].iloc[-1] - df_reset['ts_event'].iloc[0]
    print(f"  Duration: {time_span}")
    print(f"  Duration in hours: {time_span.total_seconds() / 3600:.1f}")

# Show price range
print(f"\nPrice info:")
print(f"  Open range: {df['open'].min():.2f} - {df['open'].max():.2f}")
print(f"  Close range: {df['close'].min():.2f} - {df['close'].max():.2f}")
print(f"  Volume range: {df['volume'].min():,} - {df['volume'].max():,}")

# Calculate labels for full dataset
print(f"\n{'='*60}")
print("CALCULATING LABELS FOR FULL DATASET (OPTIMIZED)")
print(f"{'='*60}")

start_time = time.time()
df_labeled = calculate_labels_for_all_profiles_optimized(df)
end_time = time.time()

processing_time = end_time - start_time
print(f"\n⏱️  Processing completed in {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")

# Show results
print(f"\n{'='*60}")
print("FULL DATASET RESULTS")
print(f"{'='*60}")

for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']:
    print(f"\n{profile}:")
    label_counts = df_labeled[f'{profile}_label'].value_counts().sort_index()
    total_non_timeout = label_counts.sum()
    
    optimal = label_counts.get(1.0, 0)
    suboptimal = label_counts.get(0.0, 0)
    losses = label_counts.get(-1.0, 0)
    timeouts = df_labeled[f'{profile}_label'].isna().sum()
    
    print(f"  Optimal (+1): {optimal:,} ({optimal/total_non_timeout*100:.1f}%)")
    print(f"  Suboptimal (0): {suboptimal:,} ({suboptimal/total_non_timeout*100:.1f}%)")
    print(f"  Loss (-1): {losses:,} ({losses/total_non_timeout*100:.1f}%)")
    print(f"  Timeout (NaN): {timeouts:,}")
    print(f"  Win rate: {(optimal + suboptimal)/total_non_timeout*100:.1f}%")

# Calculate frequency statistics
total_optimal = sum(df_labeled[f'{profile}_label'].eq(1).sum() 
                   for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                                  'short_2to1_small', 'short_2to1_medium', 'short_2to1_large'])

time_span_hours = time_span.total_seconds() / 3600
optimal_per_hour = total_optimal / time_span_hours

print(f"\n{'='*60}")
print("FREQUENCY ANALYSIS")
print(f"{'='*60}")
print(f"Total optimal entries across all profiles: {total_optimal:,}")
print(f"Time span: {time_span_hours:.1f} hours")
print(f"Optimal entries per hour: {optimal_per_hour:.1f}")
print(f"Optimal entries per trading day (6.5 hours): {optimal_per_hour * 6.5:.0f}")

if optimal_per_hour > 10:
    print(f"⚠️  HIGH FREQUENCY: {optimal_per_hour:.1f} per hour may be unrealistic for normal market conditions")
else:
    print(f"✅ REASONABLE FREQUENCY: {optimal_per_hour:.1f} per hour seems realistic")

# Save the labeled dataset
output_path = 'project/data/processed/full_labeled_dataset.parquet'
df_labeled.to_parquet(output_path)
print(f"\n✅ Saved labeled dataset to: {output_path}")
print(f"   Shape: {df_labeled.shape}")
print(f"   Size: {len(df_labeled):,} bars")