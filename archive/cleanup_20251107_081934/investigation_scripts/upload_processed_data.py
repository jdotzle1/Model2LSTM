#!/usr/bin/env python3
"""
Upload the processed 30-day dataset to S3 for safekeeping and future use
"""
import boto3
import os
import time
from pathlib import Path
from datetime import datetime

def upload_processed_data():
    """Upload the processed dataset to S3"""
    print("🚀 UPLOADING PROCESSED DATA TO S3")
    print("=" * 50)
    
    # File paths
    local_file = Path("project/data/processed/es_30day_labeled_features.parquet")
    
    if not local_file.exists():
        print(f"❌ Processed file not found: {local_file}")
        print("Run the pipeline test first: python3 test_30day_pipeline.py")
        return False
    
    # S3 configuration
    bucket_name = "es-1-second-30-days"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"processed-data/es_30day_labeled_features_{timestamp}.parquet"
    
    file_size_mb = local_file.stat().st_size / (1024**2)
    
    print(f"📥 Local file: {local_file}")
    print(f"📤 S3 destination: s3://{bucket_name}/{s3_key}")
    print(f"📊 File size: {file_size_mb:.1f} MB")
    
    try:
        # Initialize S3 client
        print("\n🔗 Connecting to S3...")
        s3_client = boto3.client('s3')
        
        # Upload file with progress
        print("📤 Uploading to S3...")
        start_time = time.time()
        
        s3_client.upload_file(
            str(local_file),
            bucket_name,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'source': 'ec2_processing',
                    'rows': '295939',
                    'columns': '65',
                    'features': '43',
                    'labels': '6',
                    'weights': '6',
                    'date_range': '2025-09-22_to_2025-10-21',
                    'processing_date': timestamp
                }
            }
        )
        
        upload_time = time.time() - start_time
        upload_speed_mbps = file_size_mb / upload_time
        
        print(f"✅ Upload complete!")
        print(f"   Time: {upload_time:.1f} seconds")
        print(f"   Speed: {upload_speed_mbps:.1f} MB/s")
        print(f"   S3 URL: s3://{bucket_name}/{s3_key}")
        
        # Verify upload
        print("\n🔍 Verifying upload...")
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        s3_size = response['ContentLength']
        local_size = local_file.stat().st_size
        
        if s3_size == local_size:
            print(f"✅ Verification passed: {s3_size:,} bytes")
        else:
            print(f"❌ Size mismatch: S3={s3_size:,}, Local={local_size:,}")
            return False
        
        # Also upload the raw RTH data for reference
        print("\n📤 Uploading RTH-filtered data...")
        rth_file = Path("project/data/processed/es_30day_rth.parquet")
        if rth_file.exists():
            rth_s3_key = f"processed-data/es_30day_rth_{timestamp}.parquet"
            s3_client.upload_file(
                str(rth_file),
                bucket_name,
                rth_s3_key,
                ExtraArgs={
                    'Metadata': {
                        'source': 'ec2_processing',
                        'rows': '295939',
                        'columns': '10',
                        'stage': 'rth_filtered',
                        'date_range': '2025-09-22_to_2025-10-21',
                        'processing_date': timestamp
                    }
                }
            )
            print(f"✅ RTH data uploaded: s3://{bucket_name}/{rth_s3_key}")
        
        print(f"\n🎉 UPLOAD COMPLETE!")
        print(f"📋 Files uploaded:")
        print(f"   1. Processed dataset: s3://{bucket_name}/{s3_key}")
        if rth_file.exists():
            print(f"   2. RTH-filtered data: s3://{bucket_name}/{rth_s3_key}")
        
        print(f"\n📋 Dataset summary:")
        print(f"   • 295,939 rows of RTH-only ES data")
        print(f"   • 30-day period: Sept 22 - Oct 21, 2025")
        print(f"   • 6 volatility modes with labels & weights")
        print(f"   • 43 engineered features")
        print(f"   • Ready for XGBoost training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False

def list_s3_processed_data():
    """List all processed datasets in S3"""
    print("\n🔍 LISTING PROCESSED DATA IN S3")
    print("-" * 40)
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-30-days"
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix="processed-data/"
        )
        
        if 'Contents' not in response:
            print("No processed data found in S3")
            return
        
        print(f"Found {len(response['Contents'])} files:")
        for obj in response['Contents']:
            key = obj['Key']
            size_mb = obj['Size'] / (1024**2)
            modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"   {key} ({size_mb:.1f} MB, {modified})")
            
    except Exception as e:
        print(f"❌ Failed to list S3 data: {e}")

if __name__ == "__main__":
    success = upload_processed_data()
    if success:
        list_s3_processed_data()
        print("\n🚀 Ready for XGBoost model training!")
    else:
        print("\n💥 Upload failed")
        exit(1)