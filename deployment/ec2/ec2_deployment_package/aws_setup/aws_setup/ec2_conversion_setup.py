#!/usr/bin/env python3
"""
EC2 Setup Script for DBN to Parquet Conversion

This script sets up an EC2 instance to:
1. Download compressed DBN files from S3
2. Convert to Parquet format
3. Upload Parquet files back to S3
4. Clean up local storage

Usage:
    python ec2_conversion_setup.py
"""

import boto3
import pandas as pd
import databento as db
import os
from pathlib import Path
import time

# Configuration
S3_BUCKET = "your-es-data-bucket"
S3_DBN_PREFIX = "raw/dbn/"  # Where your .dbn.zst files are stored
S3_PARQUET_PREFIX = "raw/parquet/"  # Where to store converted files
LOCAL_TEMP_DIR = "/tmp/conversion"  # Local processing directory

def setup_environment():
    """Install required packages and create directories"""
    print("Setting up environment...")
    
    # Create local temp directory
    Path(LOCAL_TEMP_DIR).mkdir(parents=True, exist_ok=True)
    
    # Install required packages (run this manually on EC2)
    print("""
    Run these commands on your EC2 instance:
    
    # Update system
    sudo yum update -y
    
    # Install Python 3.9+
    sudo yum install python3 python3-pip -y
    
    # Install required packages
    pip3 install --user databento pandas pyarrow boto3
    
    # Configure AWS credentials (use IAM role or aws configure)
    aws configure
    """)

def list_dbn_files():
    """List all DBN files in S3"""
    s3 = boto3.client('s3')
    
    print(f"Listing DBN files in s3://{S3_BUCKET}/{S3_DBN_PREFIX}")
    
    paginator = s3.get_paginator('list_objects_v2')
    dbn_files = []
    
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_DBN_PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.dbn.zst'):
                    size_mb = obj['Size'] / (1024 * 1024)
                    dbn_files.append({
                        'key': obj['Key'],
                        'size_mb': size_mb,
                        'filename': os.path.basename(obj['Key'])
                    })
    
    print(f"Found {len(dbn_files)} DBN files:")
    total_size_gb = sum(f['size_mb'] for f in dbn_files) / 1024
    
    for f in dbn_files:
        print(f"  {f['filename']} ({f['size_mb']:.1f} MB)")
    
    print(f"Total size: {total_size_gb:.1f} GB")
    return dbn_files

def filter_rth_data(df):
    """
    Filter data to only include Regular Trading Hours (RTH)
    RTH: 07:30 CT to 15:00 CT (includes pre-market + RTH + close)
    
    This removes:
    - ETH (Extended Trading Hours): 15:00-07:30 CT
    - Overnight sessions
    - Weekend gaps
    """
    print(f"  Filtering to RTH only (07:30-15:00 CT)...")
    
    original_count = len(df)
    
    try:
        import pytz
        
        # Ensure we have timestamp column
        if 'timestamp' not in df.columns:
            if df.index.name == 'ts_event' or pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.reset_index()
                df.rename(columns={'ts_event': 'timestamp'}, inplace=True)
            else:
                raise ValueError("No timestamp found")
        
        # Convert to Central Time
        central_tz = pytz.timezone('US/Central')
        
        if df['timestamp'].dt.tz is None:
            # Assume UTC if no timezone
            ct_time = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(central_tz)
        else:
            ct_time = df['timestamp'].dt.tz_convert(central_tz)
        
        # Calculate decimal hours (e.g., 7:30 = 7.5, 15:00 = 15.0)
        ct_decimal = ct_time.dt.hour + ct_time.dt.minute / 60.0
        
        # RTH filter: 07:30 CT (7.5) to 15:00 CT (15.0)
        rth_mask = (ct_decimal >= 7.5) & (ct_decimal < 15.0)
        
        # Apply filter
        df_rth = df[rth_mask].copy()
        
        # Reset index to be clean
        df_rth = df_rth.reset_index(drop=True)
        
        # Set timestamp back as index if it was originally
        if 'ts_event' in df.columns or df.index.name == 'ts_event':
            df_rth = df_rth.set_index('timestamp')
            df_rth.index.name = 'ts_event'
        
        filtered_count = len(df_rth)
        removed_count = original_count - filtered_count
        removed_pct = (removed_count / original_count) * 100
        
        print(f"    Original: {original_count:,} bars")
        print(f"    RTH only: {filtered_count:,} bars")
        print(f"    Removed: {removed_count:,} bars ({removed_pct:.1f}%)")
        
        # Verify we have reasonable data
        if filtered_count == 0:
            raise ValueError("No RTH data found - check timezone conversion")
        
        if removed_pct < 50:
            print(f"    Warning: Expected to remove ~65% of data, only removed {removed_pct:.1f}%")
        
        return df_rth
        
    except Exception as e:
        print(f"    Error in RTH filtering: {str(e)}")
        print(f"    Returning original data without filtering")
        return df

def convert_single_file(s3_key, local_dbn_path, local_parquet_path):
    """Convert a single DBN file to Parquet with RTH filtering"""
    print(f"Converting {os.path.basename(s3_key)}...")
    
    try:
        # Load DBN file
        print(f"  Loading DBN data...")
        store = db.DBNStore.from_file(local_dbn_path)
        df = store.to_df()
        
        print(f"  Loaded {len(df):,} bars")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        # Filter to RTH only
        df_rth = filter_rth_data(df)
        
        # Save as Parquet with compression
        print(f"  Saving RTH data as Parquet...")
        df_rth.to_parquet(local_parquet_path, compression='snappy')
        
        # Verify the conversion
        df_verify = pd.read_parquet(local_parquet_path)
        if len(df_verify) != len(df_rth):
            raise ValueError(f"Verification failed: {len(df_rth)} != {len(df_verify)}")
        
        print(f"  ✓ Conversion successful: {len(df_rth):,} RTH bars")
        return True
        
    except Exception as e:
        print(f"  ❌ Conversion failed: {str(e)}")
        return False

def process_all_files():
    """Main processing function"""
    s3 = boto3.client('s3')
    dbn_files = list_dbn_files()
    
    if not dbn_files:
        print("No DBN files found!")
        return
    
    print(f"\nStarting conversion of {len(dbn_files)} files...")
    
    successful = 0
    failed = 0
    
    for i, file_info in enumerate(dbn_files, 1):
        s3_key = file_info['key']
        filename = file_info['filename']
        
        print(f"\n[{i}/{len(dbn_files)}] Processing {filename}")
        
        # Define local paths
        local_dbn_path = os.path.join(LOCAL_TEMP_DIR, filename)
        parquet_filename = filename.replace('.dbn.zst', '.parquet')
        local_parquet_path = os.path.join(LOCAL_TEMP_DIR, parquet_filename)
        
        try:
            # Download DBN file
            print(f"  Downloading from S3...")
            s3.download_file(S3_BUCKET, s3_key, local_dbn_path)
            
            # Convert to Parquet
            if convert_single_file(s3_key, local_dbn_path, local_parquet_path):
                # Upload Parquet file
                print(f"  Uploading Parquet to S3...")
                s3_parquet_key = f"{S3_PARQUET_PREFIX}{parquet_filename}"
                s3.upload_file(local_parquet_path, S3_BUCKET, s3_parquet_key)
                print(f"  ✓ Uploaded to s3://{S3_BUCKET}/{s3_parquet_key}")
                successful += 1
            else:
                failed += 1
            
        except Exception as e:
            print(f"  ❌ Failed to process {filename}: {str(e)}")
            failed += 1
        
        finally:
            # Clean up local files
            for path in [local_dbn_path, local_parquet_path]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"  Cleaned up {os.path.basename(path)}")
        
        # Progress update
        print(f"  Progress: {successful} successful, {failed} failed")
    
    print(f"\n🎉 Conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Parquet files available at: s3://{S3_BUCKET}/{S3_PARQUET_PREFIX}")

if __name__ == "__main__":
    print("ES Data Conversion: DBN → Parquet")
    print("=" * 50)
    
    # Check if running on EC2
    try:
        import requests
        response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
        instance_id = response.text
        print(f"Running on EC2 instance: {instance_id}")
    except:
        print("Not running on EC2 - make sure to run this on your EC2 instance")
    
    # Run conversion
    process_all_files()