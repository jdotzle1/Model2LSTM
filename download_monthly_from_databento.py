#!/usr/bin/env python3
"""
Download monthly ES data files from Databento FTP and upload to S3
This handles the 15 years of monthly 1-second OHLCV data
"""
import ftplib
import boto3
import os
import time
from pathlib import Path
from datetime import datetime
import calendar

def log_progress(message):
    """Log progress with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def connect_to_databento_ftp():
    """Connect to Databento FTP server"""
    log_progress("🔗 Connecting to Databento FTP...")
    
    try:
        ftp = ftplib.FTP()
        ftp.connect("ftp.databento.com")
        ftp.login("josh.dotzler@gmail.com", "Alexi$R0ck$1109")
        
        log_progress("✅ Connected to Databento FTP")
        
        # Navigate to the data directory
        ftp.cwd("/EREGVYEC/GLBX-20251103-S8RYURC9BV")
        
        log_progress("📁 Navigated to data directory")
        
        return ftp
        
    except Exception as e:
        log_progress(f"❌ FTP connection failed: {e}")
        return None

def list_available_files(ftp):
    """List all available monthly files on FTP"""
    log_progress("📋 Listing available files on FTP...")
    
    try:
        files = []
        ftp.retrlines('LIST', files.append)
        
        # Filter for monthly DBN files
        monthly_files = []
        for file_line in files:
            # Parse FTP LIST output (usually: permissions size date time filename)
            parts = file_line.split()
            if len(parts) >= 9:
                filename = parts[-1]
                if filename.endswith('.dbn.zst') and 'glbx-mdp3-' in filename:
                    # Extract file size (approximate, FTP LIST format varies)
                    try:
                        size_bytes = int(parts[4])
                        size_mb = size_bytes / (1024**2)
                    except:
                        size_mb = 0
                    
                    monthly_files.append({
                        'filename': filename,
                        'size_mb': size_mb,
                        'ftp_line': file_line
                    })
        
        log_progress(f"📊 Found {len(monthly_files)} monthly files")
        
        # Show first few files
        for i, file_info in enumerate(monthly_files[:5]):
            log_progress(f"   {file_info['filename']} ({file_info['size_mb']:.1f} MB)")
        
        if len(monthly_files) > 5:
            log_progress(f"   ... and {len(monthly_files) - 5} more files")
        
        return monthly_files
        
    except Exception as e:
        log_progress(f"❌ Failed to list files: {e}")
        return []

def check_existing_s3_files():
    """Check which files are already uploaded to S3"""
    log_progress("🔍 Checking existing files in S3...")
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix="raw-data/databento/"
        )
        
        existing_files = set()
        if 'Contents' in response:
            for obj in response['Contents']:
                filename = obj['Key'].split('/')[-1]
                existing_files.add(filename)
        
        log_progress(f"📊 Found {len(existing_files)} existing files in S3")
        
        return existing_files
        
    except Exception as e:
        log_progress(f"⚠️  Could not check S3: {e}")
        return set()

def download_and_upload_file(ftp, file_info, existing_files):
    """Download a file from FTP and upload to S3"""
    filename = file_info['filename']
    
    # Check if already exists
    if filename in existing_files:
        log_progress(f"⏭️  Skipping {filename} (already in S3)")
        return True
    
    log_progress(f"📥 Processing {filename} ({file_info['size_mb']:.1f} MB)")
    
    # Local temporary file
    local_file = Path(f"/tmp/databento_download/{filename}")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download from FTP
        log_progress(f"   📥 Downloading from FTP...")
        start_time = time.time()
        
        with open(local_file, 'wb') as f:
            ftp.retrbinary(f'RETR {filename}', f.write)
        
        download_time = time.time() - start_time
        actual_size_mb = local_file.stat().st_size / (1024**2)
        download_speed = actual_size_mb / download_time if download_time > 0 else 0
        
        log_progress(f"   ✅ Downloaded in {download_time:.1f}s ({download_speed:.1f} MB/s)")
        
        # Upload to S3
        log_progress(f"   📤 Uploading to S3...")
        start_time = time.time()
        
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        s3_key = f"raw-data/databento/{filename}"
        
        s3_client.upload_file(
            str(local_file),
            bucket_name,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'source': 'databento_ftp',
                    'upload_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'original_size_mb': str(actual_size_mb)
                }
            }
        )
        
        upload_time = time.time() - start_time
        upload_speed = actual_size_mb / upload_time if upload_time > 0 else 0
        
        log_progress(f"   ✅ Uploaded in {upload_time:.1f}s ({upload_speed:.1f} MB/s)")
        log_progress(f"   🔗 S3: s3://{bucket_name}/{s3_key}")
        
        # Cleanup local file
        local_file.unlink()
        
        return True
        
    except Exception as e:
        log_progress(f"   ❌ Failed: {e}")
        
        # Cleanup on failure
        if local_file.exists():
            local_file.unlink()
        
        return False

def main():
    """Main download and upload pipeline"""
    log_progress("🚀 DATABENTO MONTHLY FILES DOWNLOAD & UPLOAD")
    log_progress("=" * 60)
    log_progress("Downloading 15 years of monthly ES data from Databento FTP")
    log_progress("Uploading to S3 for processing")
    
    # Connect to FTP
    ftp = connect_to_databento_ftp()
    if not ftp:
        log_progress("❌ Cannot proceed without FTP connection")
        return
    
    try:
        # List available files
        monthly_files = list_available_files(ftp)
        if not monthly_files:
            log_progress("❌ No monthly files found on FTP")
            return
        
        # Check existing S3 files
        existing_files = check_existing_s3_files()
        
        # Filter files that need to be downloaded
        to_download = [f for f in monthly_files if f['filename'] not in existing_files]
        
        log_progress(f"\n📊 DOWNLOAD PLAN:")
        log_progress(f"   Total files on FTP: {len(monthly_files)}")
        log_progress(f"   Already in S3: {len(existing_files)}")
        log_progress(f"   Need to download: {len(to_download)}")
        
        if not to_download:
            log_progress("✅ All files already uploaded to S3!")
            return
        
        # Calculate total size
        total_size_mb = sum(f['size_mb'] for f in to_download)
        log_progress(f"   Total download size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
        
        # Estimate time (assuming 10 MB/s average)
        estimated_minutes = (total_size_mb / 10) / 60
        log_progress(f"   Estimated time: {estimated_minutes:.1f} minutes")
        
        # Confirm before proceeding
        print(f"\nProceed with downloading {len(to_download)} files? (yes/no): ", end="")
        response = input().strip().lower()
        
        if response != 'yes':
            log_progress("❌ Download cancelled")
            return
        
        # Download and upload each file
        log_progress(f"\n🔄 STARTING DOWNLOAD & UPLOAD PROCESS")
        log_progress("=" * 50)
        
        successful = 0
        failed = 0
        start_time = time.time()
        
        for i, file_info in enumerate(to_download, 1):
            log_progress(f"\n📊 Progress: {i}/{len(to_download)} files")
            
            if download_and_upload_file(ftp, file_info, existing_files):
                successful += 1
            else:
                failed += 1
            
            # Progress update
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / i
                remaining_time = (len(to_download) - i) * avg_time
                log_progress(f"   ✅ Successful: {successful}, ❌ Failed: {failed}")
                log_progress(f"   ⏱️  Remaining: {remaining_time/60:.1f} minutes")
        
        # Final summary
        total_time = time.time() - start_time
        
        log_progress(f"\n🎉 DOWNLOAD & UPLOAD COMPLETE!")
        log_progress(f"   Successful: {successful}/{len(to_download)} files")
        log_progress(f"   Failed: {failed}/{len(to_download)} files")
        log_progress(f"   Total time: {total_time/60:.1f} minutes")
        
        if successful > 0:
            log_progress(f"\n📊 Files available in S3:")
            log_progress(f"   s3://es-1-second-data/raw-data/databento/")
            log_progress(f"\n🚀 Ready for monthly processing!")
            log_progress(f"   Next: python3 test_monthly_processing.py")
        
    finally:
        # Close FTP connection
        try:
            ftp.quit()
            log_progress("🔗 FTP connection closed")
        except:
            pass

if __name__ == "__main__":
    main()