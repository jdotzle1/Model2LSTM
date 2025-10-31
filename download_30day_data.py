#!/usr/bin/env python3
"""
Download 30-day ES data from Databento FTP and upload to S3
"""
import os
import sys
import time
import boto3
from pathlib import Path
from datetime import datetime
import subprocess

def log_progress(message):
    """Log progress with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def download_from_ftp():
    """Download the 30-day dataset from Databento FTP"""
    
    # FTP details
    ftp_host = "ftp.databento.com"
    username = "josh.dotzler@gmail.com"
    password = "Alexi$R0ck$1109"
    remote_path = "/EREGVYEC/GLBX-20251023-G5FBXRA73N"
    
    # Local paths
    work_dir = Path('/tmp/es_30day_processing')
    work_dir.mkdir(exist_ok=True)
    
    log_progress("🚀 DOWNLOADING 30-DAY ES DATA FROM DATABENTO FTP")
    log_progress(f"FTP Host: {ftp_host}")
    log_progress(f"Remote path: {remote_path}")
    log_progress(f"Local directory: {work_dir}")
    
    try:
        # Use wget to download (more reliable than Python FTP)
        log_progress("📥 Starting FTP download with wget...")
        
        # Build wget command
        wget_cmd = [
            'wget',
            '--recursive',
            '--no-parent',
            '--no-host-directories',
            '--cut-dirs=2',
            '--reject=index.html*',
            '--user=' + username,
            '--password=' + password,
            f'ftp://{ftp_host}{remote_path}/',
            '--directory-prefix=' + str(work_dir)
        ]
        
        log_progress(f"Command: {' '.join(wget_cmd[:6])} [credentials hidden]")
        
        # Run wget
        result = subprocess.run(wget_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            log_progress("✅ FTP download completed successfully")
            
            # List downloaded files
            downloaded_files = list(work_dir.rglob('*'))
            log_progress(f"Downloaded {len(downloaded_files)} files:")
            for file in downloaded_files:
                if file.is_file():
                    size_mb = file.stat().st_size / (1024**2)
                    log_progress(f"  {file.name}: {size_mb:.1f} MB")
            
            return work_dir
        else:
            log_progress(f"❌ wget failed: {result.stderr}")
            return None
            
    except Exception as e:
        log_progress(f"❌ Download failed: {e}")
        return None

def upload_to_s3(local_dir):
    """Upload downloaded files to S3"""
    
    bucket_name = "es-1-second-30-days"
    s3_prefix = "raw-data/databento/"
    
    log_progress("🚀 UPLOADING TO S3")
    log_progress(f"Bucket: s3://{bucket_name}")
    log_progress(f"Prefix: {s3_prefix}")
    
    try:
        s3_client = boto3.client('s3')
        
        # Find all files to upload
        files_to_upload = []
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                files_to_upload.append(file_path)
        
        log_progress(f"Found {len(files_to_upload)} files to upload")
        
        # Upload each file
        for i, file_path in enumerate(files_to_upload, 1):
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}{relative_path}"
            
            log_progress(f"[{i}/{len(files_to_upload)}] Uploading {file_path.name}...")
            
            # Upload file
            s3_client.upload_file(str(file_path), bucket_name, s3_key)
            
            # Verify upload
            try:
                response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                uploaded_size = response['ContentLength']
                local_size = file_path.stat().st_size
                
                if uploaded_size == local_size:
                    log_progress(f"  ✅ Upload verified: {uploaded_size:,} bytes")
                else:
                    log_progress(f"  ⚠️  Size mismatch: local={local_size}, s3={uploaded_size}")
            except Exception as e:
                log_progress(f"  ❌ Verification failed: {e}")
        
        log_progress("✅ S3 upload completed successfully")
        log_progress(f"Files available at: s3://{bucket_name}/{s3_prefix}")
        
        return True
        
    except Exception as e:
        log_progress(f"❌ S3 upload failed: {e}")
        return False

def main():
    """Main function"""
    log_progress("🚀 STARTING 30-DAY ES DATA DOWNLOAD AND UPLOAD")
    
    # Step 1: Download from FTP
    local_dir = download_from_ftp()
    if not local_dir:
        log_progress("💥 Download failed - aborting")
        sys.exit(1)
    
    # Step 2: Upload to S3
    success = upload_to_s3(local_dir)
    if not success:
        log_progress("💥 S3 upload failed")
        sys.exit(1)
    
    log_progress("🎉 COMPLETE! 30-day ES data is now in S3")
    log_progress("Next step: Update your processing scripts to use the new bucket")

if __name__ == "__main__":
    main()