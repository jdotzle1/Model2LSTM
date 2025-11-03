#!/usr/bin/env python3
"""
Process 15 years of ES data in monthly chunks
Much more reliable than processing the entire dataset at once
"""
import sys
import os
import time
import boto3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import calendar

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def generate_monthly_file_list():
    """Generate list of monthly DBN files to process"""
    print("📅 GENERATING MONTHLY FILE LIST")
    print("=" * 50)
    
    # Databento monthly files pattern:
    # glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1s.dbn.zst
    
    monthly_files = []
    
    # Generate for 15 years: 2010-2025
    start_year = 2010
    end_year = 2025
    end_month = 10  # October 2025
    
    for year in range(start_year, end_year + 1):
        start_month = 7 if year == 2010 else 1  # Start from July 2010 (based on your example)
        last_month = end_month if year == end_year else 12
        
        for month in range(start_month, last_month + 1):
            # Calculate first and last day of month
            import calendar
            first_day = 1
            last_day = calendar.monthrange(year, month)[1]
            
            # Generate date strings
            start_date = f"{year:04d}{month:02d}{first_day:02d}"
            end_date = f"{year:04d}{month:02d}{last_day:02d}"
            
            # Generate file info
            month_str = f"{year:04d}-{month:02d}"
            filename = f"glbx-mdp3-{start_date}-{end_date}.ohlcv-1s.dbn.zst"
            file_key = f"raw-data/databento/{filename}"  # Assuming they're in raw-data/databento/
            
            monthly_files.append({
                'year': year,
                'month': month,
                'month_str': month_str,
                'filename': filename,
                's3_key': file_key,
                'local_file': f"/tmp/monthly_processing/{month_str}/input.dbn.zst",
                'output_file': f"/tmp/monthly_processing/{month_str}/processed.parquet"
            })
    
    print(f"📊 Generated {len(monthly_files)} monthly files to process")
    print(f"   Period: {monthly_files[0]['month_str']} to {monthly_files[-1]['month_str']}")
    print(f"   Example files:")
    for i in range(min(3, len(monthly_files))):
        print(f"     {monthly_files[i]['filename']}")
    
    return monthly_files

def check_existing_processed_files(monthly_files):
    """Check which files are already processed in S3"""
    print("\n🔍 CHECKING EXISTING PROCESSED FILES")
    print("=" * 50)
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        
        # List existing processed files
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix="processed-data/monthly/"
        )
        
        existing_files = set()
        if 'Contents' in response:
            for obj in response['Contents']:
                # Extract month from filename
                filename = obj['Key'].split('/')[-1]
                if 'monthly_' in filename:
                    month_part = filename.split('monthly_')[1].split('_')[0]
                    existing_files.add(month_part)
        
        # Mark which files need processing
        to_process = []
        already_done = []
        
        for file_info in monthly_files:
            if file_info['month_str'] in existing_files:
                already_done.append(file_info['month_str'])
            else:
                to_process.append(file_info)
        
        print(f"✅ Already processed: {len(already_done)} months")
        print(f"🔄 Need to process: {len(to_process)} months")
        
        if already_done:
            print(f"   Existing: {already_done[:5]}{'...' if len(already_done) > 5 else ''}")
        
        return to_process
        
    except Exception as e:
        print(f"⚠️  Could not check existing files: {e}")
        print("   Will process all files")
        return monthly_files

def process_single_month(file_info):
    """Process a single month of data"""
    month_str = file_info['month_str']
    print(f"\n🔄 PROCESSING {month_str}")
    print("=" * 40)
    
    # Create month directory
    month_dir = Path(f"/tmp/monthly_processing/{month_str}")
    month_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Step 1: Download monthly file
        print(f"📥 Downloading {month_str}...")
        if not download_monthly_file(file_info):
            return None
        
        # Step 2: Process the month
        print(f"🔄 Processing {month_str}...")
        processed_file = process_monthly_data(file_info)
        if not processed_file:
            return None
        
        # Step 3: Upload results
        print(f"📤 Uploading {month_str}...")
        if not upload_monthly_results(file_info, processed_file):
            return None
        
        # Step 4: Cleanup
        cleanup_monthly_files(file_info)
        
        total_time = time.time() - start_time
        print(f"✅ {month_str} complete in {total_time/60:.1f} minutes")
        
        return processed_file
        
    except Exception as e:
        print(f"❌ {month_str} failed: {e}")
        return None

def download_monthly_file(file_info):
    """Download a single monthly file"""
    bucket_name = "es-1-second-data"
    s3_key = file_info['s3_key']
    local_file = Path(file_info['local_file'])
    
    # Check if already downloaded
    if local_file.exists():
        print(f"   ✅ Already downloaded")
        return True
    
    try:
        s3_client = boto3.client('s3')
        
        # Check if file exists in S3
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            file_size_mb = response['ContentLength'] / (1024**2)
            print(f"   📦 Size: {file_size_mb:.1f} MB")
        except:
            print(f"   ❌ File not found in S3: {s3_key}")
            return False
        
        # Download
        s3_client.download_file(bucket_name, s3_key, str(local_file))
        print(f"   ✅ Downloaded successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
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
        print(f"   📖 Converting DBN...")
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
        print(f"   ✅ Converted {len(df):,} rows")
        
        # Step 2: Filter to RTH
        print(f"   🕐 Filtering to RTH...")
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
        
        print(f"   ✅ RTH filtered: {len(df_rth):,} rows ({len(df_rth)/len(df)*100:.1f}%)")
        del df
        
        # Step 3: Weighted Labeling
        print(f"   🏷️  Weighted labeling...")
        engine = WeightedLabelingEngine()
        
        try:
            df_labeled = engine.process_dataframe(df_rth, validate_performance=False)
        except TypeError:
            from src.data_pipeline.weighted_labeling import process_weighted_labeling
            df_labeled = process_weighted_labeling(df_rth)
        
        print(f"   ✅ Labeled: {len(df_labeled.columns)} columns")
        del df_rth
        
        # Step 4: Feature Engineering
        print(f"   🔧 Features...")
        df_final = create_all_features(df_labeled)
        
        print(f"   ✅ Features: {len(df_final.columns)} columns")
        del df_labeled
        
        # Step 5: Save
        df_final.to_parquet(output_file, index=False)
        
        file_size_mb = output_file.stat().st_size / (1024**2)
        print(f"   💾 Saved: {file_size_mb:.1f} MB")
        
        return str(output_file)
        
    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
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
        
        print(f"   ✅ Uploaded: s3://{bucket_name}/{s3_key}")
        return True
        
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return False

def cleanup_monthly_files(file_info):
    """Clean up temporary files for the month"""
    month_dir = Path(f"/tmp/monthly_processing/{file_info['month_str']}")
    
    try:
        # Remove all files in the month directory
        for file_path in month_dir.glob("*"):
            file_path.unlink()
        month_dir.rmdir()
        print(f"   🧹 Cleaned up temporary files")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")

def main():
    """Main monthly processing pipeline"""
    print("🚀 MONTHLY ES DATA PROCESSING PIPELINE")
    print("=" * 60)
    print("Processing 15 years of data in monthly chunks")
    print("Much more reliable than processing entire dataset at once")
    
    # Generate file list
    monthly_files = generate_monthly_file_list()
    
    # Check existing files
    to_process = check_existing_processed_files(monthly_files)
    
    if not to_process:
        print("\n✅ All months already processed!")
        return
    
    print(f"\n🎯 PROCESSING PLAN")
    print(f"   Total months to process: {len(to_process)}")
    print(f"   Estimated time per month: 10-30 minutes")
    print(f"   Total estimated time: {len(to_process) * 20 / 60:.1f} hours")
    
    # Process each month
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, file_info in enumerate(to_process, 1):
        print(f"\n📊 PROGRESS: {i}/{len(to_process)} months")
        
        result = process_single_month(file_info)
        if result:
            successful += 1
        else:
            failed += 1
        
        # Progress summary
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = (len(to_process) - i) * avg_time
        
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️  Remaining: {remaining/3600:.1f} hours")
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 MONTHLY PROCESSING COMPLETE!")
    print(f"   Successful: {successful}/{len(to_process)} months")
    print(f"   Failed: {failed}/{len(to_process)} months")
    print(f"   Total time: {total_time/3600:.1f} hours")
    
    if successful > 0:
        print(f"\n📊 Results available in:")
        print(f"   s3://es-1-second-data/processed-data/monthly/")

if __name__ == "__main__":
    main()