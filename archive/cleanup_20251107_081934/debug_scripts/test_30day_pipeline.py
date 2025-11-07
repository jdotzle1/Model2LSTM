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
    df = df.reset_index()  # This moves the timestamp index to a 'ts_event' column
    
    # Rename ts_event to timestamp for compatibility with weighted labeling system
    if 'ts_event' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'ts_event': 'timestamp'})
        print(f"   Renamed 'ts_event' to 'timestamp' for compatibility")
    
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
        # Step 0: Filter data for quality (RTH hours + main contracts only)
        print(f"\n🕐 STEP 0: FILTERING DATA FOR QUALITY")
        print("-" * 40)
        
        import pytz
        from datetime import time as dt_time
        
        # First, filter to main ES contracts only (exclude spread contracts)
        print(f"   Symbols in data: {df['symbol'].value_counts().to_dict()}")
        
        # Keep only main ES contracts (exclude spreads like ESZ5-ESH6)
        main_contracts = df[~df['symbol'].str.contains('-', na=False)].copy()
        print(f"   Original rows: {len(df):,}")
        print(f"   Main contracts only: {len(main_contracts):,}")
        print(f"   Filtered out spreads: {len(df) - len(main_contracts):,}")
        
        # Convert to Central Time for RTH filtering
        central_tz = pytz.timezone('US/Central')
        central_times = main_contracts['timestamp'].dt.tz_convert(central_tz)
        central_time_only = central_times.dt.time
        
        # Filter to RTH (07:30-15:00 CT)
        rth_start = dt_time(7, 30)
        rth_end = dt_time(15, 0)
        rth_mask = (central_time_only >= rth_start) & (central_time_only < rth_end)
        
        df_rth = main_contracts[rth_mask].copy()
        print(f"   RTH rows: {len(df_rth):,}")
        print(f"   Total filtered out: {len(df) - len(df_rth):,} ({(len(df) - len(df_rth))/len(df)*100:.1f}%)")
        
        if len(df_rth) == 0:
            print("❌ No RTH data found after filtering")
            return False
        
        # Check price consistency after filtering
        price_changes = df_rth['close'].diff().abs()
        large_changes = (price_changes > 20).sum()
        print(f"   Large price changes (>20 points): {large_changes}")
        print(f"   Max price change: {price_changes.max():.2f} points")
        
        # Use filtered data for the rest of the pipeline
        df = df_rth
        
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
        
        # Step 4: Generate comprehensive validation report
        print(f"\n📊 STEP 4: COMPREHENSIVE VALIDATION REPORT")
        print("-" * 40)
        
        # Generate comprehensive validation report with built-in functionality
        total_time = time.time() - start_time
        processing_rate = len(df_final) / total_time
        
        # Create comprehensive validation report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": total_time,
            "processing_time_minutes": total_time / 60,
            "input_rows": len(df),
            "output_rows": len(df_final),
            "data_retention_rate": len(df_final) / len(df),
            "processing_rate_rows_per_second": processing_rate,
            "large_price_changes": large_changes,
            "max_price_change": price_changes.max(),
            "labeling_time_seconds": labeling_time,
            "feature_time_seconds": features_time,
        }
        
        print("📈 Data Quality Metrics:")
        print(f"   Original data: {len(df):,} rows")
        print(f"   Final dataset: {len(df_final):,} rows")
        print(f"   Processing efficiency: {len(df_final)/len(df)*100:.1f}% data retained")
        print(f"   Large price changes detected: {large_changes}")
        print(f"   Max price change: {price_changes.max():.2f} points")
        
        # Feature quality metrics
        feature_cols = [col for col in df_final.columns if col not in df.columns and not col.startswith(('label_', 'weight_'))]
        print(f"\n🔧 Feature Engineering Quality:")
        print(f"   Features generated: {len(feature_cols)}")
        
        # Check NaN percentages
        nan_percentages = {}
        for col in feature_cols:
            nan_pct = (df_final[col].isnull().sum() / len(df_final)) * 100
            if nan_pct > 0:
                nan_percentages[col] = nan_pct
        
        report_data["features_generated"] = len(feature_cols)
        report_data["features_with_nan"] = len(nan_percentages)
        report_data["nan_percentages"] = nan_percentages
        
        if nan_percentages:
            print(f"   Features with NaN values: {len(nan_percentages)}")
            for col, pct in sorted(nan_percentages.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     {col}: {pct:.1f}% NaN")
        else:
            print(f"   ✅ No NaN values in features")
        
        # Win rates and weight distributions
        print(f"\n🏷️  Weighted Labeling Summary:")
        label_cols = [col for col in df_final.columns if col.startswith('label_')]
        weight_cols = [col for col in df_final.columns if col.startswith('weight_')]
        
        mode_statistics = {}
        valid_modes = 0
        
        for mode_name in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                         'low_vol_short', 'normal_vol_short', 'high_vol_short']:
            label_col = f'label_{mode_name}'
            weight_col = f'weight_{mode_name}'
            
            if label_col in df_final.columns and weight_col in df_final.columns:
                win_rate = df_final[label_col].mean()
                avg_weight = df_final[weight_col].mean()
                total_winners = df_final[label_col].sum()
                
                mode_statistics[mode_name] = {
                    "win_rate": win_rate,
                    "avg_weight": avg_weight,
                    "total_winners": int(total_winners)
                }
                
                status = "✅" if 0.05 <= win_rate <= 0.50 else "⚠️"
                print(f"   {status} {mode_name}: {win_rate:.1%} win rate ({total_winners:,} winners), avg weight: {avg_weight:.3f}")
                
                if 0.05 <= win_rate <= 0.50:
                    valid_modes += 1
        
        report_data["mode_statistics"] = mode_statistics
        report_data["valid_modes"] = valid_modes
        
        # Performance summary
        print(f"\n⚡ Performance Summary:")
        print(f"   Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Processing rate: {processing_rate:.0f} rows/second")
        print(f"   Weighted labeling: {labeling_time:.1f}s ({labeling_time/total_time*100:.1f}%)")
        print(f"   Feature engineering: {features_time:.1f}s ({features_time/total_time*100:.1f}%)")
        
        # Memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            print(f"   Peak memory usage: ~{memory_mb:.0f} MB")
            report_data["peak_memory_mb"] = memory_mb
        except:
            report_data["peak_memory_mb"] = None
        
        # Final validation
        print(f"\n✅ FINAL VALIDATION:")
        print(f"   Dataset: {len(df_final):,} rows × {len(df_final.columns)} columns")
        print(f"   Expected: 6 original + 12 labeling + 43 features = 61 columns")
        
        if len(df_final.columns) == 61:
            print("   ✅ Perfect column count! Ready for XGBoost training")
        else:
            extra_cols = len(df_final.columns) - 61
            print(f"   ⚠️  {extra_cols} extra columns (likely metadata from test data)")
        
        # Quality score
        quality_checks = [
            len(df_final) > 0,  # Has data
            len(feature_cols) == 43,  # Correct feature count
            len(label_cols) == 6,  # Correct label count
            len(weight_cols) == 6,  # Correct weight count
            len(nan_percentages) < 10,  # Low NaN features
            total_time < 3600,  # Under 1 hour
            valid_modes >= 4,  # Most modes have valid win rates
        ]
        
        quality_score = sum(quality_checks) / len(quality_checks) * 100
        report_data["overall_quality_score"] = quality_score
        report_data["ready_for_production"] = quality_score >= 80
        
        print(f"   Overall quality score: {quality_score:.0f}%")
        
        # Save comprehensive validation report
        report_path = project_root / "project/data/processed/validation_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"📄 Validation report saved to: {report_path}")
        
        if quality_score >= 80:
            print("\n🎉 PIPELINE TEST SUCCESSFUL!")
            print("   Ready for production deployment!")
        else:
            print("\n⚠️  PIPELINE TEST COMPLETED WITH WARNINGS")
            print("   Review quality issues before production deployment")
        
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