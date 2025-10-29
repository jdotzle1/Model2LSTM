#!/usr/bin/env python3
"""
EC2 Complete Pipeline for ES Data Processing with Weighted Labeling System

This script runs the complete pipeline on a single EC2 instance using the new weighted labeling system:
1. Download DBN files from S3
2. Convert DBN → Parquet with RTH filtering
3. Apply weighted labeling system (6 volatility-based modes)
4. Engineer 43 features
5. Train 6 XGBoost models with weighted samples
6. Upload results to S3

Key Updates for Weighted Labeling System:
- Uses volatility-based trading modes instead of size-based
- Generates 12 columns (6 labels + 6 weights) for weighted XGBoost training
- Improved performance monitoring and validation
- Compatible with existing EC2 infrastructure

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
import json
import joblib

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Import pipeline components
from project.data_pipeline.pipeline import (
    process_complete_pipeline,
    process_labeling_and_features, 
    train_xgboost_models,
    validate_pipeline_output,
    create_pipeline_summary,
    PipelineConfig
)

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'your-es-data-bucket')
S3_DBN_PREFIX = os.environ.get('S3_DBN_PREFIX', 'raw/dbn/')
S3_OUTPUT_PREFIX = os.environ.get('S3_OUTPUT_PREFIX', 'processed/weighted_labeling/')
LOCAL_WORK_DIR = '/tmp/es_weighted_pipeline'

def setup_environment():
    """Setup local working directory and validate dependencies"""
    print("Setting up environment for weighted labeling pipeline...")
    
    # Create working directory
    Path(LOCAL_WORK_DIR).mkdir(parents=True, exist_ok=True)
    
    # Check required packages
    required_packages = ['databento', 'pandas', 'numpy', 'xgboost', 'scikit-learn', 'pyarrow', 'boto3', 'pytz']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        sys.exit(1)
    
    print("✓ Environment setup complete")

def download_dbn_files():
    """Download all DBN files from S3"""
    print("\n=== STEP 1: DOWNLOADING DBN FILES ===")
    
    s3 = boto3.client('s3')
    
    # List DBN files
    paginator = s3.get_paginator('list_objects_v2')
    dbn_files = []
    
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_DBN_PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.dbn.zst'):
                    dbn_files.append(obj['Key'])
    
    print(f"Found {len(dbn_files)} DBN files")
    
    # Download each file
    downloaded_files = []
    for i, s3_key in enumerate(dbn_files, 1):
        filename = os.path.basename(s3_key)
        local_path = os.path.join(LOCAL_WORK_DIR, 'dbn', filename)
        
        print(f"  [{i}/{len(dbn_files)}] Downloading {filename}...")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        s3.download_file(S3_BUCKET, s3_key, local_path)
        downloaded_files.append(local_path)
    
    print(f"✓ Downloaded {len(downloaded_files)} files")
    return downloaded_files

def convert_dbn_to_parquet(dbn_files):
    """Convert DBN files to Parquet with RTH filtering"""
    print("\n=== STEP 2: CONVERTING DBN → PARQUET (RTH ONLY) ===")
    
    import databento as db
    import pytz
    
    parquet_files = []
    central_tz = pytz.timezone('US/Central')
    
    for i, dbn_file in enumerate(dbn_files, 1):
        filename = os.path.basename(dbn_file)
        parquet_filename = filename.replace('.dbn.zst', '.parquet')
        parquet_path = os.path.join(LOCAL_WORK_DIR, 'parquet', parquet_filename)
        
        print(f"  [{i}/{len(dbn_files)}] Converting {filename}...")
        
        try:
            # Load DBN file
            store = db.DBNStore.from_file(dbn_file)
            df = store.to_df()
            
            print(f"    Loaded {len(df):,} bars")
            
            # Filter to RTH only (07:30-15:00 CT)
            if 'timestamp' not in df.columns:
                df = df.reset_index()
                df.rename(columns={'ts_event': 'timestamp'}, inplace=True)
            
            # Convert to Central Time
            if df['timestamp'].dt.tz is None:
                ct_time = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(central_tz)
            else:
                ct_time = df['timestamp'].dt.tz_convert(central_tz)
            
            # RTH filter: 07:30-15:00 CT
            ct_decimal = ct_time.dt.hour + ct_time.dt.minute / 60.0
            rth_mask = (ct_decimal >= 7.5) & (ct_decimal < 15.0)
            
            df_rth = df[rth_mask].copy().reset_index(drop=True)
            
            # Set timestamp back as index
            df_rth = df_rth.set_index('timestamp')
            df_rth.index.name = 'ts_event'
            
            print(f"    RTH filtered: {len(df_rth):,} bars ({len(df_rth)/len(df)*100:.1f}%)")
            
            # Save as Parquet
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
            df_rth.to_parquet(parquet_path, compression='snappy')
            
            parquet_files.append(parquet_path)
            print(f"    ✓ Saved: {parquet_filename}")
            
        except Exception as e:
            print(f"    ❌ Error converting {filename}: {str(e)}")
            continue
        
        finally:
            # Clean up DBN file to save space
            os.remove(dbn_file)
    
    print(f"✓ Converted {len(parquet_files)} files to RTH-only Parquet")
    return parquet_files

def combine_parquet_files(parquet_files):
    """Combine all Parquet files into single dataset"""
    print("\n=== STEP 3: COMBINING PARQUET FILES ===")
    
    dfs = []
    total_rows = 0
    
    for i, file_path in enumerate(parquet_files, 1):
        filename = os.path.basename(file_path)
        print(f"  [{i}/{len(parquet_files)}] Loading {filename}...")
        
        df = pd.read_parquet(file_path)
        print(f"    {len(df):,} rows")
        
        dfs.append(df)
        total_rows += len(df)
        
        # Clean up individual file to save space
        os.remove(file_path)
    
    # Combine all DataFrames
    print("  Concatenating DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp
    print("  Sorting by timestamp...")
    combined_df = combined_df.sort_index()
    
    print(f"✓ Combined dataset: {len(combined_df):,} rows")
    print(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    return combined_df

def process_weighted_labeling_and_features(df):
    """Apply weighted labeling system and feature engineering"""
    print("\n=== STEP 4: WEIGHTED LABELING + FEATURE ENGINEERING ===")
    
    # Configure pipeline for EC2 processing
    config = PipelineConfig(
        chunk_size=500_000,  # 500K rows per chunk for EC2
        enable_performance_monitoring=True,
        enable_progress_tracking=True,
        enable_memory_optimization=True,
        output_dir=LOCAL_WORK_DIR
    )
    
    print("Processing with weighted labeling system:")
    print("  - 6 volatility-based trading modes")
    print("  - Quality weights based on MAE performance")
    print("  - Velocity weights based on speed to target")
    print("  - Time decay weights for recency")
    print("  - 43 engineered features")
    
    # Apply labeling and feature engineering
    df_processed = process_labeling_and_features(df, config)
    
    # Validate results
    validation_results = validate_pipeline_output(df_processed)
    
    if not validation_results['valid']:
        raise ValueError(f"Pipeline validation failed: {validation_results['errors']}")
    
    if validation_results['warnings']:
        print("  Validation warnings:")
        for warning in validation_results['warnings']:
            print(f"    - {warning}")
    
    # Print statistics
    stats = validation_results['statistics']
    print(f"✓ Processing complete:")
    print(f"  Dataset: {stats['total_rows']:,} rows × {stats['total_columns']} columns")
    print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print("  Win rates by mode:")
    for mode, win_rate in stats['win_rates'].items():
        print(f"    {mode}: {win_rate:.1%}")
    
    return df_processed

def train_weighted_xgboost_models(df):
    """Train 6 XGBoost models using weighted binary classification"""
    print("\n=== STEP 5: TRAINING WEIGHTED XGBOOST MODELS ===")
    
    config = PipelineConfig(
        enable_progress_tracking=True,
        enable_performance_monitoring=True,
        output_dir=os.path.join(LOCAL_WORK_DIR, 'models')
    )
    
    print("Training XGBoost models with weighted samples:")
    print("  - Binary classification for each volatility mode")
    print("  - Sample weights based on quality, velocity, and recency")
    print("  - Chronological train/test split (80%/20%)")
    print("  - Early stopping with weighted validation")
    
    # Train models
    models = train_xgboost_models(df, config, save_models=True)
    
    # Print model performance summary
    print(f"\n✓ Trained {len(models)} weighted XGBoost models")
    print("  Model Performance Summary:")
    for mode, model_info in models.items():
        print(f"    {mode}:")
        print(f"      - Weighted Test AUC: {model_info['test_auc']:.4f}")
        print(f"      - Win Rate: {model_info['win_rate']:.1%}")
        print(f"      - Training Samples: {model_info['training_samples']:,}")
        print(f"      - Avg Winner Weight: {model_info['avg_winner_weight']:.3f}")
        print(f"      - Avg Loser Weight: {model_info['avg_loser_weight']:.3f}")
    
    return models

def save_results_to_s3(df, models):
    """Save final dataset and models to S3"""
    print("\n=== STEP 6: SAVING RESULTS TO S3 ===")
    
    s3 = boto3.client('s3')
    
    # Save final dataset
    dataset_path = os.path.join(LOCAL_WORK_DIR, 'weighted_labeled_dataset.parquet')
    df.to_parquet(dataset_path, compression='snappy')
    
    s3_dataset_key = f"{S3_OUTPUT_PREFIX}weighted_labeled_es_dataset.parquet"
    print(f"  Uploading dataset to s3://{S3_BUCKET}/{s3_dataset_key}")
    s3.upload_file(dataset_path, S3_BUCKET, s3_dataset_key)
    
    # Save models and metadata
    model_metadata = {}
    
    for mode, model_info in models.items():
        # Save model
        model_path = os.path.join(LOCAL_WORK_DIR, 'models', f'{mode}_weighted_model.pkl')
        joblib.dump(model_info['model'], model_path)
        
        s3_model_key = f"{S3_OUTPUT_PREFIX}models/{mode}_weighted_model.pkl"
        print(f"  Uploading {mode} model to s3://{S3_BUCKET}/{s3_model_key}")
        s3.upload_file(model_path, S3_BUCKET, s3_model_key)
        
        # Save model metadata
        model_metadata[mode] = {
            'test_auc': model_info['test_auc'],
            'win_rate': model_info['win_rate'],
            'training_samples': model_info['training_samples'],
            'avg_winner_weight': model_info['avg_winner_weight'],
            'avg_loser_weight': model_info['avg_loser_weight'],
            's3_location': f"s3://{S3_BUCKET}/{s3_model_key}",
            'xgb_params': model_info['xgb_params']
        }
        
        # Save feature importance
        importance_path = os.path.join(LOCAL_WORK_DIR, 'models', f'{mode}_feature_importance.json')
        with open(importance_path, 'w') as f:
            json.dump(model_info['feature_importance'], f, indent=2)
        
        s3_importance_key = f"{S3_OUTPUT_PREFIX}models/{mode}_feature_importance.json"
        s3.upload_file(importance_path, S3_BUCKET, s3_importance_key)
    
    # Create comprehensive pipeline summary
    pipeline_summary = create_pipeline_summary(df, models)
    
    # Add weighted labeling specific metadata
    pipeline_summary.update({
        'weighted_labeling_system': {
            'version': '2.0',
            'trading_modes': [
                {'name': 'low_vol_long', 'stop_ticks': 6, 'target_ticks': 12, 'direction': 'long'},
                {'name': 'normal_vol_long', 'stop_ticks': 8, 'target_ticks': 16, 'direction': 'long'},
                {'name': 'high_vol_long', 'stop_ticks': 10, 'target_ticks': 20, 'direction': 'long'},
                {'name': 'low_vol_short', 'stop_ticks': 6, 'target_ticks': 12, 'direction': 'short'},
                {'name': 'normal_vol_short', 'stop_ticks': 8, 'target_ticks': 16, 'direction': 'short'},
                {'name': 'high_vol_short', 'stop_ticks': 10, 'target_ticks': 20, 'direction': 'short'}
            ],
            'weight_components': ['quality_weight', 'velocity_weight', 'time_decay'],
            'timeout_seconds': 900,
            'tick_size': 0.25
        },
        'models': model_metadata,
        'processing_info': {
            'date': datetime.now().isoformat(),
            'instance_type': 'c5.4xlarge',
            'pipeline_version': 'weighted_labeling_v2.0',
            'pipeline_steps': [
                'dbn_download',
                'dbn_to_parquet_conversion',
                'parquet_combination',
                'weighted_labeling_and_features',
                'weighted_xgboost_training',
                'results_upload'
            ]
        },
        's3_locations': {
            'dataset': f"s3://{S3_BUCKET}/{s3_dataset_key}",
            'models': f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}models/",
            'feature_importance': f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}models/*_feature_importance.json"
        }
    })
    
    # Save comprehensive metadata
    metadata_path = os.path.join(LOCAL_WORK_DIR, 'weighted_pipeline_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(pipeline_summary, f, indent=2)
    
    s3_metadata_key = f"{S3_OUTPUT_PREFIX}weighted_pipeline_metadata.json"
    s3.upload_file(metadata_path, S3_BUCKET, s3_metadata_key)
    
    print(f"✓ Results saved to S3")
    print(f"  Dataset: s3://{S3_BUCKET}/{s3_dataset_key}")
    print(f"  Models: s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}models/")
    print(f"  Metadata: s3://{S3_BUCKET}/{s3_metadata_key}")
    
    return pipeline_summary

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='ES Weighted Labeling Pipeline on EC2')
    parser.add_argument('--bucket', default=S3_BUCKET, help='S3 bucket name')
    parser.add_argument('--dry-run', action='store_true', help='Validate setup only')
    parser.add_argument('--skip-download', action='store_true', help='Skip DBN download (use existing files)')
    parser.add_argument('--test-mode', action='store_true', help='Process only first 2 DBN files for testing')
    
    args = parser.parse_args()
    
    print("ES Weighted Labeling Pipeline on EC2")
    print("=" * 50)
    print(f"S3 Bucket: {args.bucket}")
    print(f"Instance Type: c5.4xlarge (recommended)")
    print(f"Estimated Runtime: 6-8 hours")
    print(f"Pipeline Version: Weighted Labeling v2.0")
    
    if args.dry_run:
        print("Dry run mode - validating setup only")
        setup_environment()
        print("✓ Setup validation complete")
        return
    
    start_time = time.time()
    
    try:
        # Setup
        setup_environment()
        
        # Step 1: Download DBN files (unless skipped)
        if args.skip_download:
            print("\n=== STEP 1: SKIPPING DBN DOWNLOAD ===")
            # Look for existing parquet files
            parquet_dir = os.path.join(LOCAL_WORK_DIR, 'parquet')
            if os.path.exists(parquet_dir):
                parquet_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
                print(f"Found {len(parquet_files)} existing parquet files")
            else:
                raise ValueError("No existing parquet files found. Remove --skip-download flag.")
        else:
            dbn_files = download_dbn_files()
            
            # Test mode: process only first 2 files
            if args.test_mode:
                print(f"Test mode: processing only first 2 DBN files")
                dbn_files = dbn_files[:2]
            
            # Step 2: Convert to Parquet with RTH filtering
            parquet_files = convert_dbn_to_parquet(dbn_files)
        
        # Step 3: Combine into single dataset
        df = combine_parquet_files(parquet_files)
        
        # Step 4: Apply weighted labeling and feature engineering
        df_processed = process_weighted_labeling_and_features(df)
        
        # Step 5: Train weighted XGBoost models
        models = train_weighted_xgboost_models(df_processed)
        
        # Step 6: Save results to S3
        pipeline_summary = save_results_to_s3(df_processed, models)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎉 WEIGHTED LABELING PIPELINE COMPLETE!")
        print(f"Total runtime: {total_time/3600:.1f} hours")
        print(f"Final dataset: {len(df_processed):,} rows × {len(df_processed.columns)} columns")
        print(f"Models trained: {len(models)} (weighted XGBoost)")
        print(f"Results available at: s3://{args.bucket}/{S3_OUTPUT_PREFIX}")
        print(f"Pipeline version: {pipeline_summary['weighted_labeling_system']['version']}")
        
        # Print model performance summary
        print(f"\nModel Performance Summary:")
        for mode, metadata in pipeline_summary['models'].items():
            print(f"  {mode}: AUC={metadata['test_auc']:.4f}, WinRate={metadata['win_rate']:.1%}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()