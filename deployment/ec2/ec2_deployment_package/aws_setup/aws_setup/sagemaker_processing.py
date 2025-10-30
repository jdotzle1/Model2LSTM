#!/usr/bin/env python3
"""
SageMaker Processing Script for ES Data Labeling + Feature Engineering

This script runs on SageMaker to:
1. Download Parquet files from S3
2. Apply labeling algorithm (6 trading profiles)
3. Add 43 engineered features
4. Upload final dataset to S3 (ready for 6 XGBoost models)

Designed for SageMaker Processing Jobs with spot instances.
Output dataset will be used to train 6 specialized XGBoost models.
"""

import pandas as pd
import numpy as np
import boto3
import os
import sys
import time
from pathlib import Path
import argparse

# Add project modules to path
sys.path.append('/opt/ml/code')

# Import our labeling and feature modules
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized
from src.data_pipeline.features import integrate_with_labeled_dataset

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'your-es-data-bucket')
S3_PARQUET_PREFIX = os.environ.get('S3_PARQUET_PREFIX', 'raw/parquet/')
S3_OUTPUT_PREFIX = os.environ.get('S3_OUTPUT_PREFIX', 'processed/')
LOCAL_INPUT_DIR = '/opt/ml/processing/input'
LOCAL_OUTPUT_DIR = '/opt/ml/processing/output'

def setup_directories():
    """Create necessary directories"""
    for dir_path in [LOCAL_INPUT_DIR, LOCAL_OUTPUT_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def download_parquet_files():
    """Download all Parquet files from S3"""
    s3 = boto3.client('s3')
    
    print("Downloading Parquet files from S3...")
    
    # List all Parquet files
    paginator = s3.get_paginator('list_objects_v2')
    parquet_files = []
    
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PARQUET_PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.parquet'):
                    parquet_files.append(obj['Key'])
    
    print(f"Found {len(parquet_files)} Parquet files")
    
    # Download each file
    downloaded_files = []
    for i, s3_key in enumerate(parquet_files, 1):
        filename = os.path.basename(s3_key)
        local_path = os.path.join(LOCAL_INPUT_DIR, filename)
        
        print(f"  [{i}/{len(parquet_files)}] Downloading {filename}...")
        s3.download_file(S3_BUCKET, s3_key, local_path)
        downloaded_files.append(local_path)
    
    print(f"✓ Downloaded {len(downloaded_files)} files")
    return downloaded_files

def verify_rth_only(df):
    """
    Verify that dataset contains only RTH data (07:30-15:00 CT)
    This is a safety check since filtering should happen during conversion
    """
    print("  Verifying RTH-only data...")
    
    try:
        import pytz
        
        # Get timestamp column
        if 'timestamp' not in df.columns:
            if df.index.name == 'ts_event' or pd.api.types.is_datetime64_any_dtype(df.index):
                timestamps = df.index
            else:
                print("    Warning: No timestamp found for RTH verification")
                return df
        else:
            timestamps = df['timestamp']
        
        # Convert to Central Time
        central_tz = pytz.timezone('US/Central')
        
        if timestamps.tz is None:
            ct_time = timestamps.tz_localize('UTC').tz_convert(central_tz)
        else:
            ct_time = timestamps.tz_convert(central_tz)
        
        # Check time ranges
        ct_decimal = ct_time.hour + ct_time.minute / 60.0
        
        min_hour = ct_decimal.min()
        max_hour = ct_decimal.max()
        
        print(f"    Time range: {min_hour:.2f} to {max_hour:.2f} CT")
        
        # Check if we have any ETH data (should be none)
        eth_mask = (ct_decimal < 7.5) | (ct_decimal >= 15.0)
        eth_count = eth_mask.sum()
        
        if eth_count > 0:
            eth_pct = (eth_count / len(df)) * 100
            print(f"    ⚠️  Found {eth_count:,} ETH bars ({eth_pct:.1f}%) - filtering them out")
            
            # Filter out ETH data
            rth_mask = ~eth_mask
            df_filtered = df[rth_mask].copy().reset_index(drop=True)
            
            print(f"    Removed {eth_count:,} ETH bars, kept {len(df_filtered):,} RTH bars")
            return df_filtered
        else:
            print(f"    ✓ All data is RTH (07:30-15:00 CT)")
            return df
            
    except Exception as e:
        print(f"    Error in RTH verification: {str(e)}")
        print(f"    Proceeding without verification")
        return df

def combine_parquet_files(file_paths):
    """Combine multiple Parquet files into single DataFrame with RTH verification"""
    print("Combining Parquet files...")
    
    dfs = []
    total_rows = 0
    
    for i, file_path in enumerate(file_paths, 1):
        filename = os.path.basename(file_path)
        print(f"  [{i}/{len(file_paths)}] Loading {filename}...")
        
        df = pd.read_parquet(file_path)
        print(f"    {len(df):,} rows")
        
        dfs.append(df)
        total_rows += len(df)
    
    # Combine all DataFrames
    print("  Concatenating DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp to ensure proper order
    if 'timestamp' in combined_df.columns:
        print("  Sorting by timestamp...")
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    elif combined_df.index.name == 'ts_event':
        print("  Sorting by index timestamp...")
        combined_df = combined_df.sort_index()
    
    print(f"✓ Combined dataset: {len(combined_df):,} rows")
    print(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    # Verify RTH-only data (safety check)
    combined_df = verify_rth_only(combined_df)
    
    return combined_df

def process_dataset(df, chunk_size=250_000):
    """Apply labeling and feature engineering"""
    print(f"Processing dataset with {len(df):,} rows...")
    
    # Step 1: Apply labeling
    print("\n=== STEP 1: LABELING ===")
    start_time = time.time()
    
    df_labeled = calculate_labels_for_all_profiles_optimized(df)
    
    labeling_time = time.time() - start_time
    print(f"✓ Labeling completed in {labeling_time/60:.1f} minutes")
    
    # Save intermediate result (checkpoint)
    labeled_path = os.path.join(LOCAL_OUTPUT_DIR, 'labeled_dataset.parquet')
    df_labeled.to_parquet(labeled_path)
    print(f"✓ Saved labeled dataset checkpoint: {labeled_path}")
    
    # Step 2: Add features
    print("\n=== STEP 2: FEATURE ENGINEERING ===")
    start_time = time.time()
    
    # Use chunked processing for memory efficiency
    if len(df_labeled) > chunk_size:
        print(f"Using chunked processing (chunk_size={chunk_size:,})")
        df_final = add_features_chunked(df_labeled, chunk_size)
    else:
        print("Using single-pass processing")
        from src.data_pipeline.features import create_all_features
        df_final = create_all_features(df_labeled)
    
    feature_time = time.time() - start_time
    print(f"✓ Feature engineering completed in {feature_time/60:.1f} minutes")
    
    # Final statistics
    print(f"\n=== FINAL DATASET STATISTICS ===")
    print(f"Total rows: {len(df_final):,}")
    print(f"Total columns: {len(df_final.columns)}")
    print(f"Total processing time: {(labeling_time + feature_time)/60:.1f} minutes")
    
    # Show label distribution for first profile
    first_profile = 'long_2to1_small_label'
    if first_profile in df_final.columns:
        label_counts = df_final[first_profile].value_counts()
        print(f"\nLabel distribution ({first_profile}):")
        for label, count in label_counts.items():
            pct = count / len(df_final) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    return df_final

def add_features_chunked(df, chunk_size):
    """Memory-efficient feature engineering"""
    from src.data_pipeline.features import create_all_features
    
    print(f"Processing {len(df):,} rows in chunks of {chunk_size:,}...")
    
    # Calculate overlap needed for rolling calculations
    overlap_size = 1000
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk_num = (i // chunk_size) + 1
        chunk_start = max(0, i - overlap_size) if i > 0 else 0
        chunk_end = min(i + chunk_size, len(df))
        
        print(f"  Processing chunk {chunk_num}/{total_chunks} (rows {i:,}-{chunk_end:,})...")
        
        # Extract chunk with overlap
        chunk_df = df.iloc[chunk_start:chunk_end].copy()
        
        # Process chunk
        chunk_processed = create_all_features(chunk_df)
        
        # Remove overlap (keep only new data)
        if i > 0:
            overlap_rows = i - chunk_start
            chunk_processed = chunk_processed.iloc[overlap_rows:]
        
        chunks.append(chunk_processed)
        
        # Progress update
        rows_processed = sum(len(chunk) for chunk in chunks)
        progress_pct = (rows_processed / len(df)) * 100
        print(f"    Progress: {rows_processed:,}/{len(df):,} rows ({progress_pct:.1f}%)")
    
    # Combine chunks
    print("  Combining processed chunks...")
    result_df = pd.concat(chunks, ignore_index=True)
    
    if len(result_df) != len(df):
        raise ValueError(f"Chunk processing error: expected {len(df)} rows, got {len(result_df)}")
    
    return result_df

def upload_results(df, filename='final_es_dataset.parquet'):
    """Upload final dataset to S3"""
    s3 = boto3.client('s3')
    
    # Save locally first
    local_path = os.path.join(LOCAL_OUTPUT_DIR, filename)
    print(f"Saving final dataset locally: {local_path}")
    df.to_parquet(local_path, compression='snappy')
    
    # Upload to S3
    s3_key = f"{S3_OUTPUT_PREFIX}{filename}"
    print(f"Uploading to S3: s3://{S3_BUCKET}/{s3_key}")
    
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"✓ Upload complete!")
    
    # Also save metadata
    metadata = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'processing_date': pd.Timestamp.now().isoformat(),
        's3_location': f"s3://{S3_BUCKET}/{s3_key}"
    }
    
    metadata_path = os.path.join(LOCAL_OUTPUT_DIR, 'dataset_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    s3_metadata_key = f"{S3_OUTPUT_PREFIX}dataset_metadata.json"
    s3.upload_file(metadata_path, S3_BUCKET, s3_metadata_key)
    
    return f"s3://{S3_BUCKET}/{s3_key}"

def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(description='Process ES data on SageMaker')
    parser.add_argument('--chunk-size', type=int, default=250000, 
                       help='Chunk size for processing (default: 250000)')
    args = parser.parse_args()
    
    print("ES Data Processing on SageMaker")
    print("=" * 50)
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"Input Prefix: {S3_PARQUET_PREFIX}")
    print(f"Output Prefix: {S3_OUTPUT_PREFIX}")
    print(f"Chunk Size: {args.chunk_size:,}")
    
    try:
        # Setup
        setup_directories()
        
        # Download data
        parquet_files = download_parquet_files()
        
        # Combine files
        df = combine_parquet_files(parquet_files)
        
        # Process (label + features)
        df_final = process_dataset(df, chunk_size=args.chunk_size)
        
        # Upload results
        s3_location = upload_results(df_final)
        
        print(f"\n🎉 Processing complete!")
        print(f"Final dataset available at: {s3_location}")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()