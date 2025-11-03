#!/usr/bin/env python3
"""
Process 15 years of ES data in monthly chunks with data quality fixes built-in
"""
import sys
import os
import time
import boto3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import calendar

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def log_progress(message):
    """Log progress with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # Also write to log file
    log_file = Path("/tmp/monthly_processing.log")
    with open(log_file, "a") as f:
        f.write(log_msg + "\n")
        f.flush()

def clean_price_data(df):
    """Clean price data to remove invalid values - built into main script"""
    log_progress("   🧹 Cleaning price data...")
    
    original_rows = len(df)
    
    # Check for invalid prices
    price_cols = ['open', 'high', 'low', 'close']
    
    issues_found = False
    for col in price_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            negative_count = (df[col] < 0).sum()
            
            if zero_count > 0 or negative_count > 0:
                log_progress(f"     ⚠️  {col}: {zero_count} zeros, {negative_count} negative values")
                issues_found = True
    
    if not issues_found:
        log_progress("     ✅ No price issues found")
        return df
    
    # Remove rows with any invalid prices
    valid_mask = True
    for col in price_cols:
        if col in df.columns:
            valid_mask = valid_mask & (df[col] > 0)
    
    df_clean = df[valid_mask].copy()
    
    # Remove negative volume
    if 'volume' in df_clean.columns:
        negative_volume = (df_clean['volume'] < 0).sum()
        if negative_volume > 0:
            log_progress(f"     ⚠️  Found {negative_volume} negative volume values")
            df_clean = df_clean[df_clean['volume'] >= 0].copy()
    
    removed_rows = original_rows - len(df_clean)
    log_progress(f"     🗑️  Removed {removed_rows:,} rows with invalid data ({removed_rows/original_rows*100:.2f}%)")
    
    return df_clean

def generate_monthly_file_list():
    """Generate list of monthly DBN files to process"""
    log_progress("📅 GENERATING MONTHLY FILE LIST")
    
    monthly_files = []
    
    # Generate for 15 years: 2010-2025
    start_year = 2010
    end_year = 2025
    end_month = 10  # October 2025
    
    for year in range(start_year, end_year + 1):
        start_month = 7 if year == 2010 else 1  # Start from July 2010
        last_month = end_month if year == end_year else 12
        
        for month in range(start_month, last_month + 1):
            # Calculate first and last day of month
            first_day = 1
            last_day = calendar.monthrange(year, month)[1]
            
            # Generate date strings
            start_date = f"{year:04d}{month:02d}{first_day:02d}"
            end_date = f"{year:04d}{month:02d}{last_day:02d}"
            
            # Generate file info
            month_str = f"{year:04d}-{month:02d}"
            filename = f"glbx-mdp3-{start_date}-{end_date}.ohlcv-1s.dbn.zst"
            file_key = f"raw-data/databento/{filename}"
            
            monthly_files.append({
                'year': year,
                'month': month,
                'month_str': month_str,
                'filename': filename,
                's3_key': file_key,
                'local_file': f"/tmp/monthly_processing/{month_str}/input.dbn.zst",
                'output_file': f"/tmp/monthly_processing/{month_str}/processed.parquet"
            })
    
    log_progress(f"📊 Generated {len(monthly_files)} monthly files to process")
    return monthly_files

def check_existing_processed_files(monthly_files):
    """Check which files are already processed in S3"""
    log_progress("🔍 CHECKING EXISTING PROCESSED FILES")
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix="processed-data/monthly/"
        )
        
        existing_files = set()
        if 'Contents' in response:
            for obj in response['Contents']:
                filename = obj['Key'].split('/')[-1]
                if 'monthly_' in filename:
                    month_part = filename.split('monthly_')[1].split('_')[0]
                    existing_files.add(month_part)
        
        to_process = []
        already_done = []
        
        for file_info in monthly_files:
            if file_info['month_str'] in existing_files:
                already_done.append(file_info['month_str'])
            else:
                to_process.append(file_info)
        
        log_progress(f"✅ Already processed: {len(already_done)} months")
        log_progress(f"🔄 Need to process: {len(to_process)} months")
        
        return to_process
        
    except Exception as e:
        log_progress(f"⚠️  Could not check existing files: {e}")
        return monthly_files

def process_single_month(file_info):
    """Process a single month of data"""
    month_str = file_info['month_str']
    log_progress(f"🔄 PROCESSING {month_str}")
    
    # Create month directory
    month_dir = Path(f"/tmp/monthly_processing/{month_str}")
    month_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Step 1: Download
        if not download_monthly_file(file_info):
            return None
        
        # Step 2: Process
        processed_file = process_monthly_data(file_info)
        if not processed_file:
            return None
        
        # Step 3: Upload
        if not upload_monthly_results(file_info, processed_file):
            return None
        
        # Step 4: Cleanup
        cleanup_monthly_files(file_info)
        
        total_time = time.time() - start_time
        log_progress(f"✅ {month_str} complete in {total_time/60:.1f} minutes")
        
        return processed_file
        
    except Exception as e:
        log_progress(f"❌ {month_str} failed: {e}")
        return None

def download_monthly_file(file_info):
    """Download a single monthly file"""
    bucket_name = "es-1-second-data"
    s3_key = file_info['s3_key']
    local_file = Path(file_info['local_file'])
    
    if local_file.exists():
        log_progress(f"   ✅ Already downloaded")
        return True
    
    try:
        s3_client = boto3.client('s3')
        
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            file_size_mb = response['ContentLength'] / (1024**2)
            log_progress(f"   📦 Size: {file_size_mb:.1f} MB")
        except:
            log_progress(f"   ❌ File not found in S3: {s3_key}")
            return False
        
        s3_client.download_file(bucket_name, s3_key, str(local_file))
        log_progress(f"   ✅ Downloaded successfully")
        return True
        
    except Exception as e:
        log_progress(f"   ❌ Download failed: {e}")
        return False

def process_monthly_data(file_info):
    """Process monthly data with weighted labeling and features"""
    local_file = Path(file_info['local_file'])
    output_file = Path(file_info['output_file'])
    
    try:
        # Import processing modules
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
        from src.data_pipeline.features import create_all_features
        import databento as db
        import pytz
        from datetime import time as dt_time
        
        # Step 1: Convert DBN to DataFrame
        log_progress(f"   📖 Converting DBN...")
        store = db.DBNStore.from_file(str(local_file))
        metadata = store.metadata
        
        df = store.to_df()
        
        # Add timestamps
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
        
        df = df.reset_index()
        log_progress(f"   ✅ Converted {len(df):,} rows")
        
        # Clean data quality issues (built-in function)
        df = clean_price_data(df)
        
        if len(df) == 0:
            log_progress(f"   ❌ No valid data after cleaning")
            return None
        
        # Step 2: Filter to RTH
        log_progress(f"   🕐 Filtering to RTH...")
        central_tz = pytz.timezone('US/Central')
        
        timestamps = pd.to_datetime(df['timestamp'])
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(pytz.UTC)
        
        central_timestamps = timestamps.dt.tz_convert(central_tz)
        df_time = central_timestamps.dt.time
        
        rth_start_time = dt_time(7, 30)
        rth_end_time = dt_time(15, 0)
        
        rth_mask = (df_time >= rth_start_time) & (df_time < rth_end_time)
        df_rth = df[rth_mask].copy()
        df_rth['timestamp'] = timestamps[rth_mask].dt.tz_convert(pytz.UTC)
        
        log_progress(f"   ✅ RTH filtered: {len(df_rth):,} rows ({len(df_rth)/len(df)*100:.1f}%)")
        del df
        
        if len(df_rth) == 0:
            log_progress(f"   ❌ No RTH data found")
            return None
        
        # Step 3: Weighted Labeling
        log_progress(f"   🏷️  Weighted labeling...")
        engine = WeightedLabelingEngine()
        
        try:
            df_labeled = engine.process_dataframe(df_rth, validate_performance=False)
        except TypeError:
            from src.data_pipeline.weighted_labeling import process_weighted_labeling
            df_labeled = process_weighted_labeling(df_rth)
        
        log_progress(f"   ✅ Labeled: {len(df_labeled.columns)} columns")
        del df_rth
        
        # Step 4: Feature Engineering
        log_progress(f"   🔧 Features...")
        df_final = create_all_features(df_labeled)
        
        log_progress(f"   ✅ Features: {len(df_final.columns)} columns")
        del df_labeled
        
        # Step 5: Save
        df_final.to_parquet(output_file, index=False)
        
        file_size_mb = output_file.stat().st_size / (1024**2)
        log_progress(f"   💾 Saved: {file_size_mb:.1f} MB")
        
        return str(output_file)
        
    except Exception as e:
        log_progress(f"   ❌ Processing failed: {e}")
        import traceback
        log_progress(f"   Error details: {traceback.format_exc()}")
        return None

def upload_monthly_results(file_info, processed_file):
    """Upload monthly results to S3"""
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"processed-data/monthly/monthly_{file_info['month_str']}_{timestamp}.parquet"
        
        s3_client.upload_file(
            processed_file,
            bucket_name,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'source': 'monthly_processing',
                    'month': file_info['month_str'],
                    'processing_date': timestamp
                }
            }
        )
        
        log_progress(f"   ✅ Uploaded: s3://{bucket_name}/{s3_key}")
        return True
        
    except Exception as e:
        log_progress(f"   ❌ Upload failed: {e}")
        return False

def cleanup_monthly_files(file_info):
    """Clean up temporary files for the month"""
    month_dir = Path(f"/tmp/monthly_processing/{file_info['month_str']}")
    
    try:
        for file_path in month_dir.glob("*"):
            file_path.unlink()
        month_dir.rmdir()
        log_progress(f"   🧹 Cleaned up temporary files")
    except Exception as e:
        log_progress(f"   ⚠️  Cleanup warning: {e}")

def main():
    """Main monthly processing pipeline"""
    log_progress("🚀 MONTHLY ES DATA PROCESSING PIPELINE (WITH DATA CLEANING)")
    log_progress("Processing 15 years of data in monthly chunks")
    
    # Generate file list
    monthly_files = generate_monthly_file_list()
    
    # Check existing files
    to_process = check_existing_processed_files(monthly_files)
    
    if not to_process:
        log_progress("✅ All months already processed!")
        return
    
    log_progress(f"🎯 PROCESSING PLAN")
    log_progress(f"   Total months to process: {len(to_process)}")
    log_progress(f"   Estimated time per month: 10-30 minutes")
    log_progress(f"   Total estimated time: {len(to_process) * 20 / 60:.1f} hours")
    
    # Process each month
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, file_info in enumerate(to_process, 1):
        log_progress(f"📊 PROGRESS: {i}/{len(to_process)} months")
        
        result = process_single_month(file_info)
        if result:
            successful += 1
        else:
            failed += 1
        
        # Progress summary
        elapsed = time.time() - start_time
        if i > 0:
            avg_time = elapsed / i
            remaining = (len(to_process) - i) * avg_time
            
            log_progress(f"   ✅ Successful: {successful}")
            log_progress(f"   ❌ Failed: {failed}")
            log_progress(f"   ⏱️  Remaining: {remaining/3600:.1f} hours")
    
    total_time = time.time() - start_time
    
    log_progress(f"🎉 MONTHLY PROCESSING COMPLETE!")
    log_progress(f"   Successful: {successful}/{len(to_process)} months")
    log_progress(f"   Failed: {failed}/{len(to_process)} months")
    log_progress(f"   Total time: {total_time/3600:.1f} hours")

if __name__ == "__main__":
    main()