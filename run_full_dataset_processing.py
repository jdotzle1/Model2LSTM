#!/usr/bin/env python3
"""
Non-interactive version for running the full 15-year dataset processing with nohup
This script runs the complete pipeline without user prompts
"""
import sys
import os
import time
import boto3
import pandas as pd
from pathlib import Path
from datetime import datetime
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def log_progress(message):
    """Log progress with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # Also write to log file
    log_file = Path("/tmp/es_full_processing/processing.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(log_msg + "\n")
        f.flush()

def download_full_dataset():
    """Download the full 15-year dataset from S3"""
    log_progress("📥 STARTING DOWNLOAD OF FULL 15-YEAR DATASET")
    
    bucket_name = "es-1-second-data"
    s3_key = "raw-data/databento/glbx-mdp3-20100606-20251021.ohlcv-1s.dbn.zst"
    local_file = Path("/tmp/es_full_processing/es_15year_data.dbn.zst")
    
    log_progress(f"S3 source: s3://{bucket_name}/{s3_key}")
    log_progress(f"Local destination: {local_file}")
    
    # Check if already downloaded
    if local_file.exists():
        size_gb = local_file.stat().st_size / (1024**3)
        log_progress(f"✅ File already exists ({size_gb:.2f} GB)")
        return str(local_file)
    
    try:
        s3_client = boto3.client('s3')
        
        # Get file info
        log_progress("📊 Getting file information...")
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = response['ContentLength']
        file_size_gb = file_size / (1024**3)
        
        log_progress(f"📦 File size: {file_size_gb:.2f} GB")
        log_progress("⏳ Starting download...")
        
        start_time = time.time()
        
        # Download with progress tracking
        def progress_callback(bytes_transferred):
            if bytes_transferred % (100 * 1024 * 1024) == 0:  # Log every 100MB
                percent = (bytes_transferred / file_size) * 100
                elapsed = time.time() - start_time
                if elapsed > 0:
                    speed_mbps = (bytes_transferred / (1024**2)) / elapsed
                    log_progress(f"   Download progress: {percent:.1f}% ({speed_mbps:.1f} MB/s)")
        
        s3_client.download_file(
            bucket_name, 
            s3_key, 
            str(local_file),
            Callback=progress_callback
        )
        
        download_time = time.time() - start_time
        log_progress(f"✅ Download complete in {download_time/60:.1f} minutes")
        
        return str(local_file)
        
    except Exception as e:
        log_progress(f"❌ Download failed: {e}")
        return None

def process_full_dataset(input_file):
    """Process the full dataset with weighted labeling and feature engineering"""
    log_progress("🔄 STARTING FULL DATASET PROCESSING")
    
    dbn_file = Path(input_file)
    if not dbn_file.exists():
        log_progress(f"❌ Input file not found: {dbn_file}")
        return None
    
    try:
        # Import processing modules
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
        from src.data_pipeline.features import create_all_features
        import databento as db
        
        log_progress("✅ Processing modules imported")
        
        # Step 1: Convert DBN to DataFrame
        log_progress("📖 STEP 1: Converting DBN to DataFrame...")
        conversion_start = time.time()
        
        store = db.DBNStore.from_file(str(dbn_file))
        metadata = store.metadata
        
        log_progress(f"📊 Dataset: {metadata.dataset}")
        log_progress(f"📅 Period: {metadata.start} to {metadata.end}")
        
        # Convert to DataFrame
        df = store.to_df()
        
        # Add proper timestamps
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
        
        conversion_time = time.time() - conversion_start
        log_progress(f"✅ DBN conversion complete!")
        log_progress(f"   Rows: {len(df):,}")
        log_progress(f"   Columns: {df.columns.tolist()}")
        log_progress(f"   Conversion time: {conversion_time/60:.1f} minutes")
        
        # Step 2: Filter to RTH hours (using exact logic from working step2b)
        log_progress("🕐 STEP 2: Filtering to RTH hours...")
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
        log_progress(f"✅ RTH filtering complete!")
        log_progress(f"   Original rows: {len(df):,}")
        log_progress(f"   RTH rows: {len(df_rth):,}")
        log_progress(f"   Filtered out: {len(df) - len(df_rth):,} ({(len(df) - len(df_rth))/len(df)*100:.1f}%)")
        log_progress(f"   RTH filtering time: {rth_time/60:.1f} minutes")
        
        # Clear original dataframe to save memory
        del df
        
        # Step 3: Weighted Labeling (exact logic from working test)
        log_progress("🏷️  STEP 3: Applying weighted labeling...")
        labeling_start = time.time()
        
        engine = WeightedLabelingEngine()
        
        # Process without performance validation (exact logic from working test)
        try:
            df_labeled = engine.process_dataframe(df_rth, validate_performance=False)
        except TypeError:
            # Fallback if validate_performance parameter doesn't exist
            from src.data_pipeline.weighted_labeling import process_weighted_labeling
            df_labeled = process_weighted_labeling(df_rth)
        
        labeling_time = time.time() - labeling_start
        log_progress(f"✅ Weighted labeling complete!")
        log_progress(f"   Processing time: {labeling_time/60:.1f} minutes")
        log_progress(f"   Processing rate: {len(df_labeled)/labeling_time:.0f} rows/second")
        
        # Clear RTH dataframe to save memory
        del df_rth
        
        # Step 4: Feature Engineering
        log_progress("🔧 STEP 4: Generating features...")
        features_start = time.time()
        
        df_final = create_all_features(df_labeled)
        
        features_time = time.time() - features_start
        log_progress(f"✅ Feature engineering complete!")
        log_progress(f"   Processing time: {features_time/60:.1f} minutes")
        log_progress(f"   Final columns: {len(df_final.columns)}")
        
        # Clear labeled dataframe to save memory
        del df_labeled
        
        # Step 5: Save results
        log_progress("💾 STEP 5: Saving final dataset...")
        output_file = Path("/tmp/es_full_processing/es_15year_labeled_features.parquet")
        
        df_final.to_parquet(output_file, index=False)
        
        file_size_gb = output_file.stat().st_size / (1024**3)
        total_time = time.time() - conversion_start
        
        log_progress(f"🎉 PROCESSING COMPLETE!")
        log_progress(f"   Output file: {output_file}")
        log_progress(f"   Final rows: {len(df_final):,}")
        log_progress(f"   Final columns: {len(df_final.columns)}")
        log_progress(f"   File size: {file_size_gb:.1f} GB")
        log_progress(f"   Total processing time: {total_time/3600:.1f} hours")
        
        return str(output_file)
        
    except Exception as e:
        log_progress(f"❌ Processing failed: {e}")
        import traceback
        log_progress("Full error traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                log_progress(f"   {line}")
        return None

def upload_results(processed_file):
    """Upload the processed results to S3"""
    log_progress("📤 UPLOADING RESULTS TO S3")
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"processed-data/es_15year_labeled_features_{timestamp}.parquet"
        
        file_size_gb = Path(processed_file).stat().st_size / (1024**3)
        
        log_progress(f"📥 Local: {processed_file}")
        log_progress(f"📤 S3: s3://{bucket_name}/{s3_key}")
        log_progress(f"📊 Size: {file_size_gb:.1f} GB")
        log_progress("⏳ Starting upload...")
        
        start_time = time.time()
        
        s3_client.upload_file(
            processed_file,
            bucket_name,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'source': 'full_15year_processing',
                    'date_range': '2010-06-06_to_2025-10-21',
                    'processing_date': timestamp,
                    'features': '43',
                    'labels': '6',
                    'weights': '6'
                }
            }
        )
        
        upload_time = time.time() - start_time
        
        log_progress(f"✅ Upload complete!")
        log_progress(f"   Upload time: {upload_time/60:.1f} minutes")
        log_progress(f"   S3 URL: s3://{bucket_name}/{s3_key}")
        
        return True
        
    except Exception as e:
        log_progress(f"❌ Upload failed: {e}")
        return False

def main():
    """Main processing pipeline - non-interactive"""
    log_progress("🚀 STARTING FULL 15-YEAR DATASET PROCESSING")
    log_progress("=" * 60)
    log_progress("📊 Target: ~54 million rows, 15+ years of ES data")
    log_progress("⏱️  Estimated time: 20+ hours")
    log_progress("💾 Peak memory usage: ~40 GB")
    log_progress("📁 Log file: /tmp/es_full_processing/processing.log")
    
    # Check system resources
    memory = psutil.virtual_memory()
    log_progress(f"💾 Available memory: {memory.available / (1024**3):.1f} GB")
    log_progress(f"🖥️  CPU cores: {psutil.cpu_count()}")
    log_progress(f"💽 Free disk: {psutil.disk_usage('/').free / (1024**3):.1f} GB")
    
    start_time = time.time()
    
    try:
        # Step 1: Download
        input_file = download_full_dataset()
        if not input_file:
            log_progress("❌ FAILED: Could not download dataset")
            return 1
        
        # Step 2: Process
        processed_file = process_full_dataset(input_file)
        if not processed_file:
            log_progress("❌ FAILED: Could not process dataset")
            return 1
        
        # Step 3: Upload
        if upload_results(processed_file):
            log_progress("✅ SUCCESS: Full pipeline complete!")
        else:
            log_progress("⚠️  WARNING: Processing complete but upload failed")
            log_progress(f"Manual upload needed: {processed_file}")
        
        total_time = time.time() - start_time
        log_progress(f"🎉 TOTAL PIPELINE TIME: {total_time/3600:.1f} hours")
        log_progress("🚀 Ready for XGBoost model training!")
        
        return 0
        
    except Exception as e:
        log_progress(f"❌ PIPELINE FAILED: {e}")
        import traceback
        log_progress("Full error traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                log_progress(f"   {line}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)