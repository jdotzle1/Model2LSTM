#!/usr/bin/env python3
"""
Test the weighted labeling and feature engineering pipeline on 30-day RTH sample
"""
import sys
import os
import time
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_30day_pipeline():
    """Test the complete pipeline on 30-day RTH sample"""
    print("🚀 TESTING WEIGHTED LABELING & FEATURE ENGINEERING")
    print("=" * 60)
    
    # Input file
    input_file = project_root / "project/data/processed/es_30day_rth.parquet"
    output_file = project_root / "project/data/processed/es_30day_labeled_features.parquet"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Run: cp /tmp/es_30day_processing/es_data_rth.parquet project/data/processed/es_30day_rth.parquet")
        return False
    
    print(f"📥 Input: {input_file}")
    print(f"📤 Output: {output_file}")
    
    # Load the RTH data
    print("\n📖 Loading RTH data...")
    start_time = time.time()
    df = pd.read_parquet(input_file)
    
    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Index: {type(df.index)} with timezone {df.index.tz}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Check if we have the required columns for ES data
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    
    print(f"✅ All required OHLCV columns present")
    
    # Convert timestamp index to column (required by weighted labeling system)
    print(f"\n🕐 Converting timestamp index to column...")
    df = df.reset_index()  # This moves the timestamp index to a 'timestamp' column
    print(f"   New columns: {df.columns.tolist()}")
    
    # Sample data check
    print(f"\n📋 Sample data:")
    print(df.head(3))
    
    # Check timezone of timestamp column
    print(f"\n🕐 Timestamp analysis:")
    print(f"   Timestamp dtype: {df['timestamp'].dtype}")
    print(f"   Has timezone: {hasattr(df['timestamp'].dt, 'tz') and df['timestamp'].dt.tz is not None}")
    if hasattr(df['timestamp'].dt, 'tz') and df['timestamp'].dt.tz is not None:
        print(f"   Timezone: {df['timestamp'].dt.tz}")
        
        # Convert to Central Time to see what the weighted labeling system should see
        import pytz
        central_times = df['timestamp'].dt.tz_convert(pytz.timezone('US/Central'))
        sample_times = central_times.dt.time.head(10)
        print(f"   Sample Central times: {sample_times.tolist()}")
        
        # Check RTH compliance in Central Time
        from datetime import time as dt_time
        rth_start = dt_time(7, 30)
        rth_end = dt_time(15, 0)
        central_time_only = central_times.dt.time
        rth_mask = (central_time_only >= rth_start) & (central_time_only <= rth_end)
        non_rth_count = (~rth_mask).sum()
        print(f"   Non-RTH bars in Central Time: {non_rth_count:,}")
        
        if non_rth_count > 0:
            print("   ⚠️  Data contains non-RTH times - need to fix timezone handling")
        else:
            print("   ✅ All data is within RTH when converted to Central Time")
    
    try:
        # Step 1: Weighted Labeling (6 volatility modes)
        print(f"\n🏷️  STEP 1: WEIGHTED LABELING")
        print("-" * 40)
        
        # Import your weighted labeling system
        try:
            from src.data_pipeline.weighted_labeling import process_weighted_labeling
            print("✅ Weighted labeling module imported")
        except ImportError as e:
            print(f"❌ Could not import weighted labeling: {e}")
            print("Available modules:")
            if (project_root / "src").exists():
                for item in (project_root / "src").rglob("*.py"):
                    print(f"   {item.relative_to(project_root)}")
            return False
        
        # Run weighted labeling (disable performance checks for testing)
        labeling_start = time.time()
        
        # Import and configure for testing
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
        engine = WeightedLabelingEngine()
        
        # Process without performance validation
        try:
            df_labeled = engine.process_dataframe(df, validate_performance=False)
        except TypeError:
            # Fallback if validate_performance parameter doesn't exist
            df_labeled = process_weighted_labeling(df)
        
        labeling_time = time.time() - labeling_start
        
        print(f"✅ Weighted labeling complete in {labeling_time:.1f} seconds")
        print(f"   Input rows: {len(df):,}")
        print(f"   Output rows: {len(df_labeled):,}")
        print(f"   New columns: {len(df_labeled.columns) - len(df.columns)}")
        
        # Check for the 12 expected labeling columns (6 labels + 6 weights)
        label_cols = [col for col in df_labeled.columns if col.startswith('label_')]
        weight_cols = [col for col in df_labeled.columns if col.startswith('weight_')]
        
        print(f"   Label columns: {len(label_cols)} - {label_cols}")
        print(f"   Weight columns: {len(weight_cols)} - {weight_cols}")
        
        if len(label_cols) != 6 or len(weight_cols) != 6:
            print(f"⚠️  Expected 6 label + 6 weight columns, got {len(label_cols)} + {len(weight_cols)}")
        
        # Step 2: Feature Engineering (43 features)
        print(f"\n🔧 STEP 2: FEATURE ENGINEERING")
        print("-" * 40)
        
        try:
            from src.data_pipeline.features import create_all_features
            print("✅ Feature engineering module imported")
        except ImportError as e:
            print(f"❌ Could not import feature engineering: {e}")
            return False
        
        # Run feature engineering
        features_start = time.time()
        df_final = create_all_features(df_labeled)
        features_time = time.time() - features_start
        
        print(f"✅ Feature engineering complete in {features_time:.1f} seconds")
        print(f"   Input rows: {len(df_labeled):,}")
        print(f"   Output rows: {len(df_final):,}")
        print(f"   Total columns: {len(df_final.columns)}")
        
        # Check for expected feature categories
        feature_cols = [col for col in df_final.columns if col not in df.columns and not col.startswith(('label_', 'weight_'))]
        print(f"   New feature columns: {len(feature_cols)}")
        
        if len(feature_cols) != 43:
            print(f"⚠️  Expected 43 features, got {len(feature_cols)}")
        
        # Step 3: Save results
        print(f"\n💾 STEP 3: SAVING RESULTS")
        print("-" * 40)
        
        df_final.to_parquet(output_file, index=True)
        output_size_mb = output_file.stat().st_size / (1024**2)
        
        print(f"✅ Results saved to: {output_file}")
        print(f"   File size: {output_size_mb:.1f} MB")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎉 PIPELINE TEST COMPLETE!")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Final dataset: {len(df_final):,} rows × {len(df_final.columns)} columns")
        print(f"   Expected format: 6 original + 12 labeling + 43 features = 61 columns")
        
        if len(df_final.columns) == 61:
            print("✅ Perfect! Ready for XGBoost training")
        else:
            print(f"⚠️  Got {len(df_final.columns)} columns instead of 61")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_30day_pipeline()
    if success:
        print("\n🚀 Ready for XGBoost model training!")
    else:
        print("\n💥 Pipeline test failed - check errors above")
        sys.exit(1)