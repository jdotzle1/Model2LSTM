#!/usr/bin/env python3
"""
Debug script to download and analyze labeling results from S3
"""
import pandas as pd
import numpy as np
import boto3
import os

def analyze_s3_labeling():
    """Download and analyze the labeling results"""
    
    s3_path = "s3://es-1-second-data/processed-data/monthly/2011/06/monthly_2011-06_20251106_173444.parquet"
    local_file = "/tmp/debug_data.parquet"
    
    try:
        # Download the file
        print(f"📥 Downloading {s3_path}...")
        os.system(f"aws s3 cp {s3_path} {local_file} --region us-east-1")
        
        if not os.path.exists(local_file):
            print("❌ Failed to download file")
            return
        
        # Read the data
        print(f"📊 Reading parquet file...")
        df = pd.read_parquet(local_file)
        print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Check label columns
        label_cols = [col for col in df.columns if col.startswith('label_')]
        print(f"\n📋 Found {len(label_cols)} label columns: {label_cols}")
        
        print(f"\n📊 DETAILED LABEL ANALYSIS:")
        for col in label_cols:
            unique_vals = sorted(df[col].unique())
            win_rate = df[col].mean()
            wins = df[col].sum()
            total = len(df)
            
            print(f"\n  {col}:")
            print(f"    Unique values: {unique_vals}")
            print(f"    Win rate: {win_rate:.1%} ({wins:,} wins out of {total:,} trades)")
            
            # Check for issues
            if not set(unique_vals).issubset({0, 1}):
                print(f"    ⚠️  WARNING: Non-binary values!")
            
            if win_rate > 0.6:
                print(f"    🚨 SUSPICIOUS: Win rate too high!")
        
        # Compare long vs short
        long_cols = [col for col in label_cols if 'long' in col]
        short_cols = [col for col in label_cols if 'short' in col]
        
        print(f"\n🔍 LONG vs SHORT COMPARISON:")
        print(f"Long columns: {long_cols}")
        print(f"Short columns: {short_cols}")
        
        if long_cols and short_cols:
            print(f"\nWin rates:")
            for col in long_cols:
                print(f"  {col}: {df[col].mean():.1%}")
            for col in short_cols:
                print(f"  {col}: {df[col].mean():.1%}")
        
        # Check sample data
        print(f"\n🔍 SAMPLE DATA (first 5 rows):")
        sample_cols = ['timestamp', 'open', 'high', 'low', 'close'] + label_cols
        print(df[sample_cols].head(5))
        
        # Check if short labels are inverted
        print(f"\n🔍 INVERSION CHECK:")
        for short_col in short_cols:
            inverted_rate = 1 - df[short_col].mean()
            print(f"  {short_col}: Current {df[short_col].mean():.1%}, Inverted would be {inverted_rate:.1%}")
            
            if inverted_rate < 0.4:  # More reasonable win rate
                print(f"    🚨 LIKELY INVERTED: {short_col} should probably be inverted!")
        
        # Clean up
        os.remove(local_file)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_s3_labeling()