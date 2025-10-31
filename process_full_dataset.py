#!/usr/bin/env python3
"""
Process the full 15-year ES dataset with weighted labeling and feature engineering
This is the production version designed for large-scale processing
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

def check_system_requirements():
    """Check if system has enough resources for full dataset processing"""
    print("🔍 CHECKING SYSTEM REQUIREMENTS")
    print("=" * 50)
    
    # Check memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    
    print(f"💾 Memory: {memory_gb:.1f} GB total, {available_gb:.1f} GB available")
    
    if memory_gb < 32:
        print("⚠️  Warning: Less than 32 GB RAM. Consider upgrading instance.")
    if available_gb < 16:
        print("❌ Error: Less than 16 GB available. Close other processes.")
        return False
    
    # Check disk space
    disk = psutil.disk_usage('/')
    disk_gb = disk.total / (1024**3)
    free_gb = disk.free / (1024**3)
    
    print(f"💽 Disk: {disk_gb:.1f} GB total, {free_gb:.1f} GB free")
    
    if free_gb < 100:
        print("❌ Error: Less than 100 GB free space needed.")
        return False
    
    # Check CPU
    cpu_count = psutil.cpu_count()
    print(f"🖥️  CPU: {cpu_count} cores")
    
    print("✅ System requirements check passed")
    return True

def download_full_dataset():
    """Download the full 15-year dataset from S3"""
    print("\n📥 DOWNLOADING FULL DATASET")
    print("=" * 50)
    
    # S3 configuration for full dataset
    bucket_name = "your-full-dataset-bucket"  # Update with actual bucket
    s3_key = "raw-data/es_15year_data.dbn.zst"  # Update with actual key
    local_file = Path("/tmp/es_full_processing/es_15year_data.dbn.zst")
    
    # Create work directory
    work_dir = local_file.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📤 S3: s3://{bucket_name}/{s3_key}")
    print(f"📥 Local: {local_file}")
    
    if local_file.exists():
        size_gb = local_file.stat().st_size / (1024**3)
        print(f"✅ File already exists ({size_gb:.1f} GB)")
        return str(local_file)
    
    try:
        s3_client = boto3.client('s3')
        
        # Get file size first
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = response['ContentLength']
        file_size_gb = file_size / (1024**3)
        
        print(f"📊 File size: {file_size_gb:.1f} GB")
        print("📥 Starting download (this will take 30-60 minutes)...")
        
        start_time = time.time()
        
        # Download with progress callback
        def progress_callback(bytes_transferred):
            percent = (bytes_transferred / file_size) * 100
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed_mbps = (bytes_transferred / (1024**2)) / elapsed
                print(f"   Progress: {percent:.1f}% ({speed_mbps:.1f} MB/s)", end='\r')
        
        s3_client.download_file(
            bucket_name, 
            s3_key, 
            str(local_file),
            Callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Download complete in {elapsed/60:.1f} minutes")
        
        return str(local_file)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def process_full_dataset_chunked():
    """Process the full dataset in chunks to manage memory"""
    print("\n🔄 PROCESSING FULL DATASET")
    print("=" * 50)
    
    # File paths
    input_file = Path("/tmp/es_full_processing/es_15year_data.dbn.zst")
    output_dir = Path("/tmp/es_full_processing/processed_chunks")
    final_output = Path("/tmp/es_full_processing/es_15year_labeled_features.parquet")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return None
    
    try:
        # Import processing modules
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
        from src.data_pipeline.features import create_all_features
        
        print("✅ Processing modules imported")
        
        # Initialize processing engine
        engine = WeightedLabelingEngine()
        
        # Process in chunks (adjust chunk size based on available memory)
        chunk_size = 1_000_000  # 1M rows per chunk
        chunk_files = []
        
        print(f"📊 Processing in chunks of {chunk_size:,} rows")
        print("⏳ This will take several hours for the full dataset...")
        
        # Open DBN store
        import databento as db
        store = db.DBNStore.from_file(str(input_file))
        
        # Get total rows estimate
        metadata = store.metadata
        print(f"📅 Date range: {metadata.start} to {metadata.end}")
        
        # Process chunks
        chunk_num = 0
        total_processed = 0
        start_time = time.time()
        
        # This is a simplified version - you'd need to implement proper chunked reading
        # For now, let's process the full dataset if memory allows
        print("🔄 Converting DBN to DataFrame...")
        df = store.to_df()
        
        # Add timestamps
        if hasattr(df.index, 'astype'):
            df.index = pd.to_datetime(df.index, unit='ns', utc=True)
            df.index.name = 'timestamp'
        
        # Convert to column format
        df = df.reset_index()
        
        total_rows = len(df)
        print(f"📊 Total rows: {total_rows:,}")
        
        # Process in chunks
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            chunk_num += 1
            print(f"\n🔄 Processing chunk {chunk_num} (rows {start_idx:,}-{end_idx:,})")
            
            # Apply weighted labeling
            print("   🏷️  Weighted labeling...")
            labeled_df = engine.process_dataframe(chunk_df, validate_performance=False)
            
            # Apply feature engineering
            print("   🔧 Feature engineering...")
            final_df = create_all_features(labeled_df)
            
            # Save chunk
            chunk_file = output_dir / f"chunk_{chunk_num:04d}.parquet"
            final_df.to_parquet(chunk_file, index=False)
            chunk_files.append(chunk_file)
            
            total_processed += len(final_df)
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            
            print(f"   ✅ Chunk {chunk_num} complete ({len(final_df):,} rows)")
            print(f"   📊 Progress: {total_processed:,}/{total_rows:,} ({total_processed/total_rows*100:.1f}%)")
            print(f"   ⏱️  Rate: {rate:.0f} rows/second")
            
            # Memory cleanup
            del chunk_df, labeled_df, final_df
            
        # Combine all chunks
        print(f"\n🔗 COMBINING {len(chunk_files)} CHUNKS")
        print("-" * 40)
        
        combined_dfs = []
        for chunk_file in chunk_files:
            chunk_df = pd.read_parquet(chunk_file)
            combined_dfs.append(chunk_df)
            print(f"   Loaded {chunk_file.name}: {len(chunk_df):,} rows")
        
        # Concatenate all chunks
        print("🔗 Concatenating chunks...")
        final_df = pd.concat(combined_dfs, ignore_index=True)
        
        # Save final result
        print("💾 Saving final dataset...")
        final_df.to_parquet(final_output, index=False)
        
        # Cleanup chunk files
        print("🧹 Cleaning up chunk files...")
        for chunk_file in chunk_files:
            chunk_file.unlink()
        output_dir.rmdir()
        
        total_time = time.time() - start_time
        file_size_gb = final_output.stat().st_size / (1024**3)
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"   Output: {final_output}")
        print(f"   Rows: {len(final_df):,}")
        print(f"   Columns: {len(final_df.columns)}")
        print(f"   File size: {file_size_gb:.1f} GB")
        print(f"   Total time: {total_time/3600:.1f} hours")
        
        return str(final_output)
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def upload_full_dataset_results(processed_file):
    """Upload the processed full dataset to S3"""
    print("\n📤 UPLOADING RESULTS TO S3")
    print("=" * 50)
    
    if not processed_file or not Path(processed_file).exists():
        print("❌ No processed file to upload")
        return False
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "your-results-bucket"  # Update with actual bucket
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"processed-data/es_15year_labeled_features_{timestamp}.parquet"
        
        file_size_gb = Path(processed_file).stat().st_size / (1024**3)
        
        print(f"📥 Local: {processed_file}")
        print(f"📤 S3: s3://{bucket_name}/{s3_key}")
        print(f"📊 Size: {file_size_gb:.1f} GB")
        print("⏳ Upload will take 30-60 minutes...")
        
        start_time = time.time()
        
        # Upload with multipart for large files
        s3_client.upload_file(
            processed_file,
            bucket_name,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'source': 'full_dataset_processing',
                    'processing_date': timestamp,
                    'dataset': '15_year_es_data'
                }
            }
        )
        
        upload_time = time.time() - start_time
        
        print(f"✅ Upload complete in {upload_time/60:.1f} minutes")
        print(f"📤 S3 URL: s3://{bucket_name}/{s3_key}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main processing pipeline for full dataset"""
    print("🚀 FULL DATASET PROCESSING PIPELINE")
    print("=" * 60)
    print("⚠️  This will process the complete 15-year ES dataset")
    print("⚠️  Estimated time: 6-12 hours")
    print("⚠️  Estimated output size: 50-100 GB")
    print()
    
    # Confirm before proceeding
    response = input("Continue with full dataset processing? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Processing cancelled")
        return
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        return
    
    # Download full dataset
    input_file = download_full_dataset()
    if not input_file:
        print("❌ Failed to download dataset")
        return
    
    # Process full dataset
    processed_file = process_full_dataset_chunked()
    if not processed_file:
        print("❌ Failed to process dataset")
        return
    
    # Upload results
    if upload_full_dataset_results(processed_file):
        print("\n🎉 FULL DATASET PROCESSING COMPLETE!")
        print("🚀 Ready for XGBoost model training on 15 years of data!")
    else:
        print("\n⚠️  Processing complete but upload failed")
        print(f"Manual upload needed: {processed_file}")

if __name__ == "__main__":
    main()