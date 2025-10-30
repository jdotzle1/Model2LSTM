#!/usr/bin/env python3
"""
EC2 Integrated Pipeline for ES Data Processing (Updated for Weighted Labeling)

This script runs the complete pipeline on a single EC2 instance using the new weighted labeling system:
1. Download DBN files from S3
2. Convert DBN → Parquet with RTH filtering
3. Apply weighted labeling system (6 volatility-based modes)
4. Engineer 43 features
5. Train 6 XGBoost models with weighted samples
6. Upload results to S3

Key Updates for Weighted Labeling System:
- Volatility-based trading modes (low/normal/high vol) instead of size-based
- Quality weights based on MAE performance
- Velocity weights based on speed to target
- Time decay weights for data recency
- Weighted XGBoost training for better model performance

Designed for c5.4xlarge EC2 instance (16 vCPU, 32 GB RAM).
Estimated runtime: 6-8 hours for 15 years of data.
"""

import pandas as pd
import numpy as np
import boto3
import os
import sys
import time
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# Import the unified pipeline
from src.data_pipeline.pipeline import (
    process_complete_pipeline,
    PipelineConfig
)

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'your-es-data-bucket')
S3_DBN_PREFIX = os.environ.get('S3_DBN_PREFIX', 'raw/dbn/')
S3_OUTPUT_PREFIX = os.environ.get('S3_OUTPUT_PREFIX', 'processed/weighted_labeling/')
LOCAL_WORK_DIR = '/tmp/es_weighted_pipeline'

def main():
    """Main pipeline execution using unified pipeline"""
    parser = argparse.ArgumentParser(description='ES Weighted Labeling Pipeline on EC2')
    parser.add_argument('--bucket', default=S3_BUCKET, help='S3 bucket name')
    parser.add_argument('--input-path', required=True, help='Path to input Parquet file with OHLCV data')
    parser.add_argument('--output-path', help='Path to save final dataset (optional)')
    parser.add_argument('--chunk-size', type=int, default=500_000, help='Chunk size for processing')
    
    args = parser.parse_args()
    
    print("ES Weighted Labeling Pipeline on EC2 (Unified)")
    print("=" * 50)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path or 'S3 only'}")
    print(f"S3 Bucket: {args.bucket}")
    print(f"Chunk Size: {args.chunk_size:,} rows")
    print(f"Pipeline Version: Weighted Labeling v2.0")
    
    start_time = time.time()
    
    try:
        # Configure pipeline for EC2
        config = PipelineConfig(
            chunk_size=args.chunk_size,
            enable_performance_monitoring=True,
            enable_progress_tracking=True,
            enable_memory_optimization=True,
            output_dir=LOCAL_WORK_DIR
        )
        
        # Run complete pipeline
        df_result = process_complete_pipeline(
            input_path=args.input_path,
            output_path=args.output_path,
            config=config
        )
        
        # Upload to S3 if bucket specified
        if args.bucket != 'your-es-data-bucket':
            print(f"\nUploading results to S3...")
            s3 = boto3.client('s3')
            
            if args.output_path:
                s3_key = f"{S3_OUTPUT_PREFIX}weighted_labeled_dataset.parquet"
                s3.upload_file(args.output_path, args.bucket, s3_key)
                print(f"  ✓ Dataset uploaded to s3://{args.bucket}/{s3_key}")
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"Total runtime: {total_time/3600:.1f} hours")
        print(f"Final dataset: {len(df_result):,} rows × {len(df_result.columns)} columns")
        print(f"Pipeline version: Weighted Labeling v2.0")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()