import pandas as pd
import os

# Check the size of the full dataset
file_path = 'project/data/raw/test_sample.parquet'

print("=== DATASET SIZE ANALYSIS ===")

# File size
file_size = os.path.getsize(file_path)
print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

# Load just the shape info without processing
print("\nLoading dataset info...")
df = pd.read_parquet(file_path)
print(f"Shape: {df.shape}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# Time range
df_reset = df.reset_index()
if 'ts_event' in df_reset.columns:
    print(f"\nTime range:")
    print(f"Start: {df_reset['ts_event'].iloc[0]}")
    print(f"End:   {df_reset['ts_event'].iloc[-1]}")
    duration = df_reset['ts_event'].iloc[-1] - df_reset['ts_event'].iloc[0]
    print(f"Duration: {duration}")
    print(f"Hours: {duration.total_seconds() / 3600:.1f}")

# Estimate processing time
print(f"\n=== PERFORMANCE ESTIMATE ===")
rows = len(df)
print(f"Total bars to process: {rows:,}")

# Our labeling does nested loops:
# - For each bar (N bars)
# - For each profile (6 profiles) 
# - Look forward up to 900 bars for target/stop
# - Calculate MAE (another forward loop up to 900 bars)

operations_per_bar = 6 * 900 * 2  # 6 profiles * 900 lookforward * 2 operations (target check + MAE)
total_operations = rows * operations_per_bar

print(f"Estimated operations: {total_operations:,}")
print(f"Operations per bar: {operations_per_bar:,}")

if total_operations > 1e9:  # 1 billion operations
    print(f"🚨 PERFORMANCE WARNING: {total_operations/1e9:.1f} billion operations!")
    print(f"   This could take hours on a laptop")
    
    # Estimate time (rough)
    ops_per_second = 100000  # Conservative estimate for Python loops
    estimated_seconds = total_operations / ops_per_second
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60
    
    print(f"   Estimated time: {estimated_hours:.1f} hours")
    
    print(f"\n💡 SOLUTIONS:")
    print(f"   1. Test on smaller sample (10,000 bars)")
    print(f"   2. Optimize the nested loops")
    print(f"   3. Use vectorized operations")
    print(f"   4. Process in chunks")

else:
    print(f"✅ Should be manageable: {total_operations/1e6:.1f} million operations")

# Memory usage
memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
print(f"\nMemory usage: {memory_mb:.1f} MB")

if memory_mb > 1000:  # > 1GB
    print(f"🚨 MEMORY WARNING: Large dataset may cause issues")
else:
    print(f"✅ Memory usage looks reasonable")