#!/usr/bin/env python3
"""
EC2 Complete Pipeline for ES Data Processing

This script runs the complete pipeline on a single EC2 instance:
1. Download DBN files from S3
2. Convert DBN → Parquet with RTH filtering
3. Apply weighted labeling system (6 modes)
4. Engineer 43 features
5. Train 6 XGBoost models
6. Upload results to S3

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
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'your-es-data-bucket')
S3_DBN_PREFIX = os.environ.get('S3_DBN_PREFIX', 'raw/dbn/')
S3_OUTPUT_PREFIX = os.environ.get('S3_OUTPUT_PREFIX', 'processed/')
LOCAL_WORK_DIR = '/tmp/es_pipeline'

def setup_environment():
    """Setup local working directory and validate dependencies"""
    print("Setting up environment...")
    
    # Create working directory
    Path(LOCAL_WORK_DIR).mkdir(parents=True, exist_ok=True)
    
    # Check required packages
    required_packages = ['databento', 'pandas', 'numpy', 'xgboost', 'scikit-learn', 'pyarrow', 'boto3']
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

def apply_weighted_labeling(df):
    """Apply weighted labeling system"""
    print("\n=== STEP 4: APPLYING WEIGHTED LABELING ===")
    
    # Import the new weighted labeling system
    print("  Loading weighted labeling system...")
    
    # Add project root to path for imports
    import sys
    import os
    project_root = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, project_root)
    
    from project.data_pipeline.weighted_labeling import process_weighted_labeling, LabelingConfig
    
    print("  Processing 6 volatility-based trading modes...")
    print("    - low_vol_long (6 stop / 12 target ticks)")
    print("    - normal_vol_long (8 stop / 16 target ticks)") 
    print("    - high_vol_long (10 stop / 20 target ticks)")
    print("    - low_vol_short (6 stop / 12 target ticks)")
    print("    - normal_vol_short (8 stop / 16 target ticks)")
    print("    - high_vol_short (10 stop / 20 target ticks)")
    
    # Configure for EC2 processing (enable performance monitoring)
    config = LabelingConfig(
        chunk_size=500_000,  # 500K rows per chunk for EC2
        enable_performance_monitoring=True,
        enable_progress_tracking=True,
        enable_memory_optimization=True
    )
    
    # Apply weighted labeling
    df_labeled = process_weighted_labeling(df, config)
    
    # Validate results
    expected_columns = []
    for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                 'low_vol_short', 'normal_vol_short', 'high_vol_short']:
        expected_columns.extend([f'label_{mode}', f'weight_{mode}'])
    
    missing_columns = set(expected_columns) - set(df_labeled.columns)
    if missing_columns:
        raise ProcessingError(f"Missing expected columns: {missing_columns}")
    
    # Print labeling statistics
    print(f"✓ Added 12 new columns (6 labels + 6 weights)")
    print(f"  Dataset size: {len(df_labeled):,} rows × {len(df_labeled.columns)} columns")
    
    # Show win rates for each mode
    print("  Win rates by mode:")
    for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                 'low_vol_short', 'normal_vol_short', 'high_vol_short']:
        label_col = f'label_{mode}'
        weight_col = f'weight_{mode}'
        win_rate = df_labeled[label_col].mean()
        avg_weight = df_labeled[weight_col].mean()
        print(f"    {mode}: {win_rate:.1%} win rate, avg weight: {avg_weight:.3f}")
    
    return df_labeled

def engineer_features(df):
    """Engineer 43 features"""
    print("\n=== STEP 5: ENGINEERING FEATURES ===")
    
    # Import existing feature engineering
    print("  Loading feature engineering system...")
    
    # Add project root to path for imports
    import sys
    import os
    project_root = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, project_root)
    
    from project.data_pipeline.features import create_all_features, get_expected_feature_names
    
    print("  Adding 43 engineered features...")
    print("    - Volume features (4)")
    print("    - Price context features (5)")
    print("    - Consolidation features (10)")
    print("    - Return features (5)")
    print("    - Volatility features (6)")
    print("    - Microstructure features (6)")
    print("    - Time features (7)")
    
    # Apply feature engineering (handles chunking automatically for large datasets)
    df_featured = create_all_features(df)
    
    # Validate feature engineering results
    expected_features = get_expected_feature_names()
    expected_feature_count = len(expected_features)
    
    # Count original columns (OHLCV + timestamp if present)
    original_columns = ['open', 'high', 'low', 'close', 'volume']
    if 'timestamp' in df.columns:
        original_columns.append('timestamp')
    
    # Count label/weight columns (new weighted labeling system)
    label_weight_columns = [col for col in df.columns if col.startswith(('label_', 'weight_'))]
    
    expected_total_columns = len(original_columns) + len(label_weight_columns) + expected_feature_count
    
    # Check for missing features
    missing_features = set(expected_features) - set(df_featured.columns)
    if missing_features:
        print(f"  Warning: Missing expected features: {missing_features}")
    
    # Validate column count
    actual_columns = len(df_featured.columns)
    if actual_columns != expected_total_columns:
        print(f"  Column count: Expected {expected_total_columns}, got {actual_columns}")
        print(f"    Original OHLCV: {len(original_columns)}")
        print(f"    Labels/Weights: {len(label_weight_columns)}")
        print(f"    Features: {actual_columns - len(original_columns) - len(label_weight_columns)}")
    else:
        print(f"  ✓ Column validation passed: {actual_columns} columns")
    
    # Validate that new weighted labeling columns are preserved
    preserved_label_weight_cols = [col for col in df_featured.columns if col.startswith(('label_', 'weight_'))]
    if len(preserved_label_weight_cols) != len(label_weight_columns):
        missing_lw_cols = set(label_weight_columns) - set(preserved_label_weight_cols)
        print(f"  Warning: Missing label/weight columns: {missing_lw_cols}")
    
    print(f"✓ Feature engineering complete")
    print(f"  Final dataset: {len(df_featured):,} rows × {len(df_featured.columns)} columns")
    print(f"  Ready for XGBoost training with weighted labels")
    
    return df_featured

def train_xgboost_models(df):
    """Train 6 XGBoost models using weighted binary classification"""
    print("\n=== STEP 6: TRAINING XGBOOST MODELS ===")
    
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, classification_report
    
    # Feature columns (exclude OHLCV, timestamp, and label/weight columns)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    exclude_cols.extend([col for col in df.columns if col.startswith(('label_', 'weight_'))])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"  Training with {len(feature_cols)} features")
    print(f"  Dataset size: {len(df):,} rows")
    
    models = {}
    trading_modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                    'low_vol_short', 'normal_vol_short', 'high_vol_short']
    
    # Chronological split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    
    for i, mode in enumerate(trading_modes, 1):
        print(f"  [{i}/6] Training {mode} model...")
        
        # Prepare data for this mode
        X = df[feature_cols]
        y = df[f'label_{mode}']
        sample_weights = df[f'weight_{mode}']
        
        # Chronological split (no shuffling for time series)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        w_train, w_test = sample_weights.iloc[:split_idx], sample_weights.iloc[split_idx:]
        
        # Validate data quality
        if y_train.isna().any() or w_train.isna().any():
            print(f"    Warning: NaN values found in {mode} labels or weights")
            # Remove NaN rows
            valid_mask = ~(y_train.isna() | w_train.isna())
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            w_train = w_train[valid_mask]
        
        # Check class balance
        win_rate = y_train.mean()
        total_samples = len(y_train)
        winners = (y_train == 1).sum()
        losers = (y_train == 0).sum()
        
        print(f"    Data: {total_samples:,} samples, {winners:,} winners ({win_rate:.1%}), {losers:,} losers")
        
        # XGBoost parameters optimized for trading data
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',  # Fast for large datasets
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eta': 0.1,
            'n_estimators': 1000,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Train XGBoost model with sample weights
        model = xgb.XGBClassifier(**xgb_params)
        
        model.fit(
            X_train, y_train,
            sample_weight=w_train,  # Use weighted training
            eval_set=[(X_test, y_test)],
            eval_sample_weight=[w_test],  # Weighted validation
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate model performance
        test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred, sample_weight=w_test)
        
        # Weight statistics
        avg_winner_weight = w_train[y_train == 1].mean()
        avg_loser_weight = w_train[y_train == 0].mean()
        
        print(f"    Weighted Test AUC: {test_auc:.4f}")
        print(f"    Win Rate: {win_rate:.3f}")
        print(f"    Avg Winner Weight: {avg_winner_weight:.3f}")
        print(f"    Avg Loser Weight: {avg_loser_weight:.3f}")
        
        # Store model with metadata
        models[mode] = {
            'model': model,
            'test_auc': test_auc,
            'win_rate': win_rate,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
            'training_samples': total_samples,
            'avg_winner_weight': avg_winner_weight,
            'avg_loser_weight': avg_loser_weight
        }
    
    print(f"✓ Trained {len(models)} XGBoost models")
    
    # Print summary statistics
    print(f"\n  Model Performance Summary:")
    for mode, model_info in models.items():
        print(f"    {mode}: AUC={model_info['test_auc']:.4f}, WinRate={model_info['win_rate']:.3f}")
    
    return models

def save_results(df, models):
    """Save final dataset and models to S3"""
    print("\n=== STEP 7: SAVING RESULTS ===")
    
    s3 = boto3.client('s3')
    
    # Save final dataset
    dataset_path = os.path.join(LOCAL_WORK_DIR, 'final_dataset.parquet')
    df.to_parquet(dataset_path, compression='snappy')
    
    s3_dataset_key = f"{S3_OUTPUT_PREFIX}final_es_dataset.parquet"
    print(f"  Uploading dataset to s3://{S3_BUCKET}/{s3_dataset_key}")
    s3.upload_file(dataset_path, S3_BUCKET, s3_dataset_key)
    
    # Save models and metadata
    import joblib
    import json
    
    model_metadata = {}
    
    for mode, model_info in models.items():
        # Save model
        model_path = os.path.join(LOCAL_WORK_DIR, f'{mode}_model.pkl')
        joblib.dump(model_info['model'], model_path)
        
        s3_model_key = f"{S3_OUTPUT_PREFIX}models/{mode}_model.pkl"
        print(f"  Uploading {mode} model to s3://{S3_BUCKET}/{s3_model_key}")
        s3.upload_file(model_path, S3_BUCKET, s3_model_key)
        
        # Save model metadata
        model_metadata[mode] = {
            'test_auc': model_info['test_auc'],
            'win_rate': model_info['win_rate'],
            'training_samples': model_info['training_samples'],
            'avg_winner_weight': model_info['avg_winner_weight'],
            'avg_loser_weight': model_info['avg_loser_weight'],
            's3_location': f"s3://{S3_BUCKET}/{s3_model_key}"
        }
        
        # Save feature importance
        importance_path = os.path.join(LOCAL_WORK_DIR, f'{mode}_feature_importance.json')
        with open(importance_path, 'w') as f:
            json.dump(model_info['feature_importance'], f, indent=2)
        
        s3_importance_key = f"{S3_OUTPUT_PREFIX}models/{mode}_feature_importance.json"
        s3.upload_file(importance_path, S3_BUCKET, s3_importance_key)
    
    # Save comprehensive metadata
    metadata = {
        'pipeline_version': '2.0_weighted_labeling',
        'dataset_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'date_range': {
                'start': str(df.index.min()) if hasattr(df.index, 'min') else 'unknown',
                'end': str(df.index.max()) if hasattr(df.index, 'max') else 'unknown'
            },
            'labeling_system': 'weighted_volatility_based',
            'feature_count': len([col for col in df.columns if col not in 
                                ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and 
                                not col.startswith(('label_', 'weight_'))])
        },
        'models': model_metadata,
        'processing_info': {
            'date': datetime.now().isoformat(),
            'instance_type': 'c5.4xlarge',
            'pipeline_steps': [
                'dbn_download',
                'dbn_to_parquet_conversion',
                'parquet_combination',
                'weighted_labeling',
                'feature_engineering',
                'xgboost_training',
                'results_upload'
            ]
        },
        's3_locations': {
            'dataset': f"s3://{S3_BUCKET}/{s3_dataset_key}",
            'models': f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}models/",
            'feature_importance': f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}models/*_feature_importance.json"
        }
    }
    
    metadata_path = os.path.join(LOCAL_WORK_DIR, 'pipeline_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    s3_metadata_key = f"{S3_OUTPUT_PREFIX}pipeline_metadata.json"
    s3.upload_file(metadata_path, S3_BUCKET, s3_metadata_key)
    
    print(f"✓ Results saved to S3")
    print(f"  Dataset: s3://{S3_BUCKET}/{s3_dataset_key}")
    print(f"  Models: s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}models/")
    print(f"  Metadata: s3://{S3_BUCKET}/{s3_metadata_key}")
    
    return metadata

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='ES Complete Pipeline on EC2')
    parser.add_argument('--bucket', default=S3_BUCKET, help='S3 bucket name')
    parser.add_argument('--dry-run', action='store_true', help='Validate setup only')
    
    args = parser.parse_args()
    
    print("ES Complete Pipeline on EC2")
    print("=" * 50)
    print(f"S3 Bucket: {args.bucket}")
    print(f"Instance Type: c5.4xlarge (recommended)")
    print(f"Estimated Runtime: 6-8 hours")
    
    if args.dry_run:
        print("Dry run mode - validating setup only")
        setup_environment()
        print("✓ Setup validation complete")
        return
    
    start_time = time.time()
    
    try:
        # Setup
        setup_environment()
        
        # Step 1: Download DBN files
        dbn_files = download_dbn_files()
        
        # Step 2: Convert to Parquet with RTH filtering
        parquet_files = convert_dbn_to_parquet(dbn_files)
        
        # Step 3: Combine into single dataset
        df = combine_parquet_files(parquet_files)
        
        # Step 4: Apply weighted labeling
        df_labeled = apply_weighted_labeling(df)
        
        # Step 5: Engineer features
        df_featured = engineer_features(df_labeled)
        
        # Step 6: Train XGBoost models
        models = train_xgboost_models(df_featured)
        
        # Step 7: Save results
        metadata = save_results(df_featured, models)
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"Total runtime: {total_time/3600:.1f} hours")
        print(f"Final dataset: {len(df_featured):,} rows × {len(df_featured.columns)} columns")
        print(f"Models trained: {len(models)}")
        print(f"Results available at: s3://{args.bucket}/{S3_OUTPUT_PREFIX}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()