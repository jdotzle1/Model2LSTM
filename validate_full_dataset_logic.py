#!/usr/bin/env python3
"""
Validate the full dataset processing logic on a small sample before running the full 20-hour job
This tests the exact same code path but with a tiny dataset
"""
import sys
import os
import time
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_full_dataset_logic():
    """Test the exact same logic as the full dataset script but on small data"""
    print("🧪 VALIDATING FULL DATASET PROCESSING LOGIC")
    print("=" * 60)
    print("Testing the exact same code path on small sample...")
    
    # Use the existing 30-day RTH data as our test input
    test_file = Path("project/data/processed/es_30day_rth.parquet")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        print("Run the 30-day pipeline first")
        return False
    
    try:
        # Load test data (this simulates the DBN conversion output)
        print("📖 Loading test data...")
        df = pd.read_parquet(test_file)
        
        # Convert timestamp index to column (simulate DBN conversion output)
        if 'timestamp' not in df.columns:
            df = df.reset_index()
        
        print(f"✅ Loaded {len(df):,} rows")
        print(f"   Columns: {df.columns.tolist()}")
        
        # EXACT SAME LOGIC AS FULL DATASET SCRIPT
        
        # Step 2: Filter to RTH hours (using exact logic from working step2b)
        print("\n🕐 STEP 2: Filtering to RTH hours...")
        rth_filter_start = time.time()
        
        import pytz
        from datetime import time as dt_time
        
        # Convert timestamp to Central Time (exact logic from step2b_filter_rth.py)
        central_tz = pytz.timezone('US/Central')
        
        # Handle timezone conversion properly
        timestamps = pd.to_datetime(df['timestamp'])
        if timestamps.dt.tz is None:
            # Assume UTC if no timezone
            utc_tz = pytz.UTC
            timestamps = timestamps.dt.tz_localize(utc_tz)
        
        # Convert timezone - for Series, always use .dt accessor
        central_timestamps = timestamps.dt.tz_convert(central_tz)
        
        # Extract time component - for Series, always use .dt accessor
        df_time = central_timestamps.dt.time
        
        # Filter to RTH (07:30-15:00 Central) - exact logic from step2b
        rth_start_time = dt_time(7, 30)
        rth_end_time = dt_time(15, 0)
        
        rth_mask = (df_time >= rth_start_time) & (df_time < rth_end_time)
        df_rth = df[rth_mask].copy()
        
        # Convert timestamps back to UTC for consistency
        df_rth['timestamp'] = timestamps[rth_mask].dt.tz_convert(pytz.UTC)
        
        rth_time = time.time() - rth_filter_start
        print(f"✅ RTH filtering complete!")
        print(f"   Original rows: {len(df):,}")
        print(f"   RTH rows: {len(df_rth):,}")
        print(f"   Filtered out: {len(df) - len(df_rth):,} ({(len(df) - len(df_rth))/len(df)*100:.1f}%)")
        print(f"   RTH filtering time: {rth_time:.1f} seconds")
        
        # Clear original dataframe to save memory
        del df
        
        # Step 3: Weighted Labeling (exact logic from working test)
        print("\n🏷️  STEP 3: Applying weighted labeling...")
        labeling_start = time.time()
        
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
        engine = WeightedLabelingEngine()
        
        # Process without performance validation (exact logic from working test)
        try:
            df_labeled = engine.process_dataframe(df_rth, validate_performance=False)
        except TypeError:
            # Fallback if validate_performance parameter doesn't exist
            from src.data_pipeline.weighted_labeling import process_weighted_labeling
            df_labeled = process_weighted_labeling(df_rth)
        
        labeling_time = time.time() - labeling_start
        print(f"✅ Weighted labeling complete!")
        print(f"   Processing time: {labeling_time:.1f} seconds")
        print(f"   Processing rate: {len(df_labeled)/labeling_time:.0f} rows/second")
        
        # Check for the 12 expected labeling columns (6 labels + 6 weights)
        label_cols = [col for col in df_labeled.columns if col.startswith('label_')]
        weight_cols = [col for col in df_labeled.columns if col.startswith('weight_')]
        
        print(f"   Label columns: {len(label_cols)} - {label_cols}")
        print(f"   Weight columns: {len(weight_cols)} - {weight_cols}")
        
        if len(label_cols) != 6 or len(weight_cols) != 6:
            print(f"❌ Expected 6 label + 6 weight columns, got {len(label_cols)} + {len(weight_cols)}")
            return False
        
        # Clear RTH dataframe to save memory
        del df_rth
        
        # Step 4: Feature Engineering
        print("\n🔧 STEP 4: Generating features...")
        features_start = time.time()
        
        from src.data_pipeline.features import create_all_features
        df_final = create_all_features(df_labeled)
        
        features_time = time.time() - features_start
        print(f"✅ Feature engineering complete!")
        print(f"   Processing time: {features_time:.1f} seconds")
        print(f"   Final columns: {len(df_final.columns)}")
        
        # Check for expected feature count
        feature_cols = [col for col in df_final.columns if col not in df_labeled.columns]
        print(f"   New feature columns: {len(feature_cols)}")
        
        if len(feature_cols) != 43:
            print(f"⚠️  Expected 43 features, got {len(feature_cols)}")
        
        # Clear labeled dataframe to save memory
        del df_labeled
        
        # Step 5: Save test results
        print("\n💾 STEP 5: Saving validation results...")
        output_file = Path("/tmp/validation_test_output.parquet")
        
        df_final.to_parquet(output_file, index=False)
        
        file_size_mb = output_file.stat().st_size / (1024**2)
        total_time = time.time() - rth_filter_start
        
        print(f"🎉 VALIDATION COMPLETE!")
        print(f"   Output file: {output_file}")
        print(f"   Final rows: {len(df_final):,}")
        print(f"   Final columns: {len(df_final.columns)}")
        print(f"   File size: {file_size_mb:.1f} MB")
        print(f"   Total processing time: {total_time:.1f} seconds")
        
        # Validate final format
        expected_columns = 65  # 10 original + 12 labeling + 43 features
        if len(df_final.columns) == expected_columns:
            print("✅ Perfect! Column count matches expected format")
        else:
            print(f"⚠️  Got {len(df_final.columns)} columns, expected {expected_columns}")
        
        # Check for NaN values
        nan_counts = df_final.isnull().sum()
        problematic_cols = nan_counts[nan_counts > 0]
        if len(problematic_cols) > 0:
            print(f"⚠️  Found NaN values in {len(problematic_cols)} columns:")
            for col, count in problematic_cols.head(5).items():
                print(f"     {col}: {count:,} NaN values")
        else:
            print("✅ No NaN values found")
        
        print(f"\n📊 FULL DATASET PROJECTIONS:")
        sample_rows = len(df_final)
        full_dataset_rows = 54_000_000  # Estimated
        
        scaling_factor = full_dataset_rows / sample_rows
        estimated_full_time = total_time * scaling_factor
        estimated_full_size = file_size_mb * scaling_factor / 1024  # GB
        
        print(f"   Estimated full dataset rows: {full_dataset_rows:,}")
        print(f"   Estimated processing time: {estimated_full_time/3600:.1f} hours")
        print(f"   Estimated output size: {estimated_full_size:.1f} GB")
        
        print(f"\n✅ VALIDATION SUCCESSFUL!")
        print(f"The full dataset processing logic is ready to run.")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_dataset_logic()
    if success:
        print("\n🚀 READY TO RUN FULL DATASET PROCESSING!")
        print("Command: nohup python3 run_full_dataset_processing.py > full_processing.log 2>&1 &")
    else:
        print("\n💥 VALIDATION FAILED - DO NOT RUN FULL DATASET YET")
        print("Fix the issues above first")
        sys.exit(1)