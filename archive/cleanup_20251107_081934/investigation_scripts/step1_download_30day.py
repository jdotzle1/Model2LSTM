#!/usr/bin/env python3
"""
Step 1: Download 30-day ES data from new S3 bucket
"""
import boto3
import os
from pathlib import Path
import time

def download_30day_data():
    """Download the 30-day ES data from S3"""
    bucket_name = "es-1-second-30-days"
    s3_key = "raw-data/databento/"  # Will need to update with actual filename
    
    # Create work directory
    work_dir = Path('/tmp/es_30day_processing')
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 STEP 1: DOWNLOADING 30-DAY ES DATA FROM S3")
    print(f"Bucket: s3://{bucket_name}")
    print(f"Key prefix: {s3_key}")
    print(f"Local: {work_dir}")
    
    start_time = time.time()
    
    try:
        s3_client = boto3.client('s3')
        
        # List objects in bucket to find the actual file
        print("📋 Listing files in bucket...")
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_key)
        
        if 'Contents' not in response:
            print(f"❌ No files found in s3://{bucket_name}/{s3_key}")
            return None
        
        # Find DBN file
        dbn_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.dbn.zst')]
        
        if not dbn_files:
            print(f"❌ No DBN.ZST files found")
            return None
        
        # Download the first DBN file found
        s3_key_full = dbn_files[0]['Key']
        file_size = dbn_files[0]['Size']
        file_size_mb = file_size / (1024 * 1024)
        
        local_file = work_dir / "es_data_30day.dbn.zst"
        
        print(f"📥 Downloading: {s3_key_full}")
        print(f"File size: {file_size_mb:.1f} MB")
        print("Downloading...")
        
        # Download file
        s3_client.download_file(bucket_name, s3_key_full, str(local_file))
        
        # Verify download
        if local_file.exists():
            actual_size = local_file.stat().st_size
            if actual_size == file_size:
                elapsed = time.time() - start_time
                print(f"✅ Download complete!")
                print(f"Time: {elapsed:.1f} seconds")
                print(f"File: {local_file}")
                print(f"Size: {actual_size:,} bytes")
                return str(local_file)
            else:
                print(f"❌ Size mismatch: expected {file_size}, got {actual_size}")
                return None
        else:
            print(f"❌ File not found after download")
            return None
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

if __name__ == "__main__":
    result = download_30day_data()
    if result:
        print(f"\n🎉 STEP 1 COMPLETE: {result}")
    else:
        print(f"\n💥 STEP 1 FAILED")
        exit(1)