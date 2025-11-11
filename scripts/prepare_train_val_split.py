#!/usr/bin/env python3
"""
Prepare Train/Validation Split for XGBoost Training

Downloads processed data from S3 and creates chronological split:
- Training: July 2010 - December 2023 (80%)
- Validation: January 2024 - October 2025 (20%)

Usage:
    python scripts/prepare_train_val_split.py --bucket es-1-second-data
    python scripts/prepare_train_val_split.py --bucket es-1-second-data --output-dir data/xgboost
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
from datetime import datetime
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_from_s3(bucket: str, local_dir: Path):
    """Download all processed parquet files from S3"""
    print("=" * 80)
    print("DOWNLOADING PROCESSED DATA FROM S3")
    print("=" * 80)
    
    s3_path = f"s3://{bucket}/processed-data/monthly/"
    
    print(f"Source: {s3_path}")
    print(f"Destination: {local_dir}")
    print()
    
    # Create local directory
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using AWS CLI
    cmd = [
        "aws", "s3", "sync",
        s3_path,
        str(local_dir),
        "--exclude", "*",
        "--include", "*.parquet"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Download failed: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    print("✓ Download complete")


def create_train_val_split(data_dir: Path, output_dir: Path, split_date: datetime):
    """Create train/validation split based on chronological date"""
    print("\n" + "=" * 80)
    print("CREATING TRAIN/VALIDATION SPLIT")
    print("=" * 80)
    print(f"Split date: {split_date.strftime('%Y-%m-%d')}")
    print(f"Training: Before {split_date.strftime('%Y-%m-%d')}")
    print(f"Validation: {split_date.strftime('%Y-%m-%d')} and after")
    print()
    
    # Find all parquet files
    all_files = sorted(data_dir.glob('**/*.parquet'))
    
    if not all_files:
        print(f"❌ No parquet files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(all_files)} parquet files")
    
    # Split files by date
    train_files = []
    val_files = []
    
    for file in all_files:
        # Extract year-month from path or filename
        # Path format: YYYY/MM/monthly_YYYY-MM_timestamp.parquet
        parts = file.parts
        
        # Try to find year and month in path
        year = None
        month = None
        
        for part in parts:
            if part.isdigit() and len(part) == 4:
                year = int(part)
            elif part.isdigit() and len(part) == 2 and year:
                month = int(part)
                break
        
        # Fallback: extract from filename
        if not (year and month):
            # Format: monthly_2010-07_timestamp.parquet
            stem = file.stem
            if 'monthly_' in stem:
                year_month = stem.split('_')[1]
                year, month = map(int, year_month.split('-'))
        
        if not (year and month):
            print(f"⚠️  Skipping file (can't parse date): {file}")
            continue
        
        file_date = datetime(year, month, 1)
        
        if file_date < split_date:
            train_files.append(file)
        else:
            val_files.append(file)
    
    print(f"\nTraining files: {len(train_files)} months")
    print(f"Validation files: {len(val_files)} months")
    
    if not train_files or not val_files:
        print("❌ Error: Need files in both train and validation sets")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine training data
    print("\n📊 Combining training data...")
    train_dfs = []
    for i, file in enumerate(train_files, 1):
        if i % 10 == 0 or i == len(train_files):
            print(f"  Loading {i}/{len(train_files)}: {file.name}")
        df = pd.read_parquet(file)
        train_dfs.append(df)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    train_output = output_dir / 'train_data.parquet'
    train_df.to_parquet(train_output, index=False)
    print(f"✓ Training data saved: {train_output}")
    print(f"  Rows: {len(train_df):,}")
    print(f"  Columns: {len(train_df.columns)}")
    print(f"  Size: {train_output.stat().st_size / (1024**3):.2f} GB")
    
    # Combine validation data
    print("\n📊 Combining validation data...")
    val_dfs = []
    for i, file in enumerate(val_files, 1):
        if i % 10 == 0 or i == len(val_files):
            print(f"  Loading {i}/{len(val_files)}: {file.name}")
        df = pd.read_parquet(file)
        val_dfs.append(df)
    
    val_df = pd.concat(val_dfs, ignore_index=True)
    
    val_output = output_dir / 'val_data.parquet'
    val_df.to_parquet(val_output, index=False)
    print(f"✓ Validation data saved: {val_output}")
    print(f"  Rows: {len(val_df):,}")
    print(f"  Columns: {len(val_df.columns)}")
    print(f"  Size: {val_output.stat().st_size / (1024**3):.2f} GB")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    print(f"Training:")
    print(f"  Files: {len(train_files)} months")
    print(f"  Rows: {len(train_df):,}")
    print(f"  Percentage: {len(train_df)/(len(train_df)+len(val_df))*100:.1f}%")
    print(f"\nValidation:")
    print(f"  Files: {len(val_files)} months")
    print(f"  Rows: {len(val_df):,}")
    print(f"  Percentage: {len(val_df)/(len(train_df)+len(val_df))*100:.1f}%")
    print(f"\nOutput files:")
    print(f"  {train_output}")
    print(f"  {val_output}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare train/validation split for XGBoost training"
    )
    
    parser.add_argument(
        "--bucket", default="es-1-second-data",
        help="S3 bucket name (default: es-1-second-data)"
    )
    parser.add_argument(
        "--output-dir", default="data/xgboost",
        help="Output directory for train/val files (default: data/xgboost)"
    )
    parser.add_argument(
        "--split-date", default="2024-01-01",
        help="Split date YYYY-MM-DD (default: 2024-01-01)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip S3 download (use existing local files)"
    )
    parser.add_argument(
        "--local-data-dir", default="data/processed_monthly",
        help="Local directory with downloaded parquet files (default: data/processed_monthly)"
    )
    
    args = parser.parse_args()
    
    # Parse split date
    split_date = datetime.strptime(args.split_date, "%Y-%m-%d")
    
    # Setup paths
    local_data_dir = Path(args.local_data_dir)
    output_dir = Path(args.output_dir)
    
    # Download from S3 if needed
    if not args.skip_download:
        download_from_s3(args.bucket, local_data_dir)
    else:
        print(f"Skipping download, using local files in {local_data_dir}")
    
    # Create split
    create_train_val_split(local_data_dir, output_dir, split_date)
    
    print("\n✅ Train/validation split complete!")
    print(f"\nNext step: Train XGBoost models")
    print(f"  python scripts/train_xgboost_models.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
