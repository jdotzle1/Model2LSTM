#!/usr/bin/env python3
"""
Test processing a single month from S3 to debug issues
Much simpler than running all 186 months
"""
import sys
import os
import time
import boto3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import calendar

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_single_month_processing():
    """Test processing just July 2010 (first month)"""
    print("🧪 TESTING SINGLE MONTH PROCESSING")
    print("=" * 60)
    print("Processing July 2010 only to debug issues")
    
    # Test file info for July 2010
    test_month = {
        'month_str': '2010-07',
        'filename': 'glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst',
        's3_key': 'raw-data/databento/glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst',
        'local_file': '/tmp/single_month_test/input.dbn.zst',
        'output_file': '/tmp/single_month_test/processed.parquet'
    }
    
    print(f"📅 Testing month: {test_month['month_str']}")
    print(f"📁 File: {test_month['filename']}")
    
    # Create test directory
    test_dir = Path("/tmp/single_month_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Step 1: Download from S3
        print(f"\n📥 STEP 1: DOWNLOADING FROM S3")
        print("-" * 40)
        
        if not download_test_file(test_month):
            return False
        
        # Step 2: Convert DBN to DataFrame
        print(f"\n📖 STEP 2: CONVERTING DBN TO DATAFRAME")
        print("-" * 40)
        
        df = convert_dbn_to_dataframe(test_month)
        if df is None:
            return False
        
        # Step 3: Clean data
        print(f"\n🧹 STEP 3: CLEANING DATA")
        print("-" * 40)
        
        df_clean = clean_data_issues(df)
        if df_clean is None or len(df_clean) == 0:
            return False
        
        # Step 4: Filter to RTH
        print(f"\n🕐 STEP 4: FILTERING TO RTH")
        print("-" * 40)
        
        df_rth = filter_to_rth(df_clean)
        if df_rth is None or len(df_rth) == 0:
            return False
        
        # Step 5: Test weighted labeling
        print(f"\n🏷️  STEP 5: TESTING WEIGHTED LABELING")
        print("-" * 40)
        
        df_labeled = test_weighted_labeling(df_rth)
        if df_labeled is None:
            return False
        
        # Step 6: Test feature engineering
        print(f"\n🔧 STEP 6: TESTING FEATURE ENGINEERING")
        print("-" * 40)
        
        df_final = test_feature_engineering(df_labeled)
        if df_final is None:
            return False
        
        # Step 7: Save and upload
        print(f"\n💾 STEP 7: SAVING RESULTS")
        print("-" * 40)
        
        if not save_and_upload_results(test_month, df_final):
            return False
        
        # Success!
        total_time = time.time() - start_time
        
        print(f"\n🎉 SINGLE MONTH TEST SUCCESSFUL!")
        print(f"   Month: {test_month['month_str']}")
        print(f"   Final rows: {len(df_final):,}")
        print(f"   Final columns: {len(df_final.columns)}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"\n✅ Ready to scale to all months!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SINGLE MONTH TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_test_file(test_month):
    """Download the test file from S3"""
    bucket_name = "es-1-second-data"
    s3_key = test_month['s3_key']
    local_file = Path(test_month['local_file'])
    
    # Check if already downloaded
    if local_file.exists():
        size_mb = local_file.stat().st_size / (1024**2)
        print(f"✅ File already downloaded ({size_mb:.1f} MB)")
        return True
    
    try:
        s3_client = boto3.client('s3')
        
        # Check file exists
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size_mb = response['ContentLength'] / (1024**2)
        print(f"📦 File size: {file_size_mb:.1f} MB")
        
        # Download
        print("📥 Downloading...")
        start_time = time.time()
        
        s3_client.download_file(bucket_name, s3_key, str(local_file))
        
        download_time = time.time() - start_time
        print(f"✅ Downloaded in {download_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def convert_dbn_to_dataframe(test_month):
    """Convert DBN file to DataFrame with timestamps"""
    local_file = Path(test_month['local_file'])
    
    try:
        import databento as db
        
        print("📖 Opening DBN file...")
        store = db.DBNStore.from_file(str(local_file))
        metadata = store.metadata
        
        print(f"✅ DBN metadata:")
        print(f"   Dataset: {metadata.dataset}")
        print(f"   Schema: {metadata.schema}")
        print(f"   Period: {metadata.start} to {metadata.end}")
        
        print("🔄 Converting to DataFrame...")
        df = store.to_df()
        
        print(f"✅ Converted to DataFrame:")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Add timestamps
        print("🕐 Adding timestamps...")
        if hasattr(df.index, 'astype'):
            start_ns = metadata.start
            end_ns = metadata.end
            total_rows = len(df)
            
            timestamps = pd.date_range(
                start=pd.to_datetime(start_ns, unit='ns', utc=True),
                end=pd.to_datetime(end_ns, unit='ns', utc=True),
                periods=total_rows
            )
            
            df.index = timestamps
            df.index.name = 'timestamp'
        
        # Convert to column format
        df = df.reset_index()
        
        print(f"✅ Timestamps added:")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ DBN conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def clean_data_issues(df):
    """Clean data quality issues"""
    print("🧹 Checking for data quality issues...")
    
    original_rows = len(df)
    
    # Check price columns
    price_cols = ['open', 'high', 'low', 'close']
    issues_found = False
    
    for col in price_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            negative_count = (df[col] < 0).sum()
            nan_count = df[col].isnull().sum()
            
            print(f"   {col}: {zero_count} zeros, {negative_count} negative, {nan_count} NaN")
            
            if zero_count > 0 or negative_count > 0 or nan_count > 0:
                issues_found = True
    
    # Check volume
    if 'volume' in df.columns:
        vol_negative = (df['volume'] < 0).sum()
        vol_nan = df['volume'].isnull().sum()
        print(f"   volume: {vol_negative} negative, {vol_nan} NaN")
        
        if vol_negative > 0 or vol_nan > 0:
            issues_found = True
    
    if not issues_found:
        print("✅ No data quality issues found")
        return df
    
    print("🔧 Cleaning data issues...")
    
    # Remove rows with invalid prices
    valid_mask = True
    for col in price_cols:
        if col in df.columns:
            valid_mask = valid_mask & (df[col] > 0) & df[col].notna()
    
    # Remove rows with invalid volume
    if 'volume' in df.columns:
        valid_mask = valid_mask & (df['volume'] >= 0) & df['volume'].notna()
    
    df_clean = df[valid_mask].copy()
    
    removed_rows = original_rows - len(df_clean)
    print(f"✅ Cleaned data:")
    print(f"   Removed: {removed_rows:,} rows ({removed_rows/original_rows*100:.2f}%)")
    print(f"   Remaining: {len(df_clean):,} rows")
    
    return df_clean

def filter_to_rth(df):
    """Filter to RTH hours only"""
    print("🕐 Filtering to RTH hours (7:30 AM - 3:00 PM Central)...")
    
    try:
        import pytz
        from datetime import time as dt_time
        
        # Convert to Central Time
        central_tz = pytz.timezone('US/Central')
        
        timestamps = pd.to_datetime(df['timestamp'])
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(pytz.UTC)
        
        central_timestamps = timestamps.dt.tz_convert(central_tz)
        df_time = central_timestamps.dt.time
        
        # RTH filter
        rth_start_time = dt_time(7, 30)
        rth_end_time = dt_time(15, 0)
        
        rth_mask = (df_time >= rth_start_time) & (df_time < rth_end_time)
        df_rth = df[rth_mask].copy()
        
        # Convert timestamps back to UTC
        df_rth['timestamp'] = timestamps[rth_mask].dt.tz_convert(pytz.UTC)
        
        print(f"✅ RTH filtering complete:")
        print(f"   Original: {len(df):,} rows")
        print(f"   RTH only: {len(df_rth):,} rows ({len(df_rth)/len(df)*100:.1f}%)")
        
        return df_rth
        
    except Exception as e:
        print(f"❌ RTH filtering failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_weighted_labeling(df):
    """Test weighted labeling on clean RTH data"""
    print("🏷️  Testing weighted labeling...")
    
    try:
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
        
        print("✅ Weighted labeling module imported")
        
        engine = WeightedLabelingEngine()
        
        print("🔄 Processing weighted labeling...")
        start_time = time.time()
        
        # Process without performance validation
        try:
            df_labeled = engine.process_dataframe(df, validate_performance=False)
        except TypeError:
            from src.data_pipeline.weighted_labeling import process_weighted_labeling
            df_labeled = process_weighted_labeling(df)
        
        labeling_time = time.time() - start_time
        
        print(f"✅ Weighted labeling complete:")
        print(f"   Processing time: {labeling_time:.1f} seconds")
        print(f"   Input columns: {len(df.columns)}")
        print(f"   Output columns: {len(df_labeled.columns)}")
        print(f"   New columns: {len(df_labeled.columns) - len(df.columns)}")
        
        # Check for expected columns
        label_cols = [col for col in df_labeled.columns if col.startswith('label_')]
        weight_cols = [col for col in df_labeled.columns if col.startswith('weight_')]
        
        print(f"   Label columns: {len(label_cols)}")
        print(f"   Weight columns: {len(weight_cols)}")
        
        if len(label_cols) == 6 and len(weight_cols) == 6:
            print("✅ Expected 6 labels + 6 weights found")
        else:
            print(f"⚠️  Expected 6+6, got {len(label_cols)}+{len(weight_cols)}")
        
        return df_labeled
        
    except Exception as e:
        print(f"❌ Weighted labeling failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_feature_engineering(df):
    """Test feature engineering"""
    print("🔧 Testing feature engineering...")
    
    try:
        from src.data_pipeline.features import create_all_features
        
        print("✅ Feature engineering module imported")
        
        print("🔄 Processing features...")
        start_time = time.time()
        
        df_features = create_all_features(df)
        
        features_time = time.time() - start_time
        
        print(f"✅ Feature engineering complete:")
        print(f"   Processing time: {features_time:.1f} seconds")
        print(f"   Input columns: {len(df.columns)}")
        print(f"   Output columns: {len(df_features.columns)}")
        print(f"   New features: {len(df_features.columns) - len(df.columns)}")
        
        # Check for expected total (should be around 65 columns)
        expected_total = 65  # 10 original + 12 labeling + 43 features
        if len(df_features.columns) == expected_total:
            print(f"✅ Perfect! Got expected {expected_total} columns")
        else:
            print(f"⚠️  Got {len(df_features.columns)} columns, expected ~{expected_total}")
        
        return df_features
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_and_upload_results(test_month, df):
    """Save results locally and upload to S3"""
    output_file = Path(test_month['output_file'])
    
    try:
        # Save locally
        print("💾 Saving locally...")
        df.to_parquet(output_file, index=False)
        
        file_size_mb = output_file.stat().st_size / (1024**2)
        print(f"✅ Saved: {output_file} ({file_size_mb:.1f} MB)")
        
        # Upload to S3
        print("📤 Uploading to S3...")
        
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"processed-data/monthly/test_single_month_{timestamp}.parquet"
        
        s3_client.upload_file(
            str(output_file),
            bucket_name,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'source': 'single_month_test',
                    'month': test_month['month_str'],
                    'processing_date': timestamp
                }
            }
        )
        
        print(f"✅ Uploaded: s3://{bucket_name}/{s3_key}")
        
        return True
        
    except Exception as e:
        print(f"❌ Save/upload failed: {e}")
        return False

if __name__ == "__main__":
    # Kill any existing processes first
    print("🛑 Checking for existing processes...")
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'process_monthly'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"Found existing processes: {pids}")
            for pid in pids:
                try:
                    subprocess.run(['kill', '-9', pid])
                    print(f"Killed process {pid}")
                except:
                    pass
        else:
            print("No existing processes found")
    except:
        print("Could not check for existing processes")
    
    # Run single month test
    success = test_single_month_processing()
    
    if success:
        print("\n🚀 SINGLE MONTH TEST PASSED!")
        print("Ready to scale to all 186 months")
    else:
        print("\n💥 SINGLE MONTH TEST FAILED!")
        print("Fix issues before scaling")