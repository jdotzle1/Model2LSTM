#!/usr/bin/env python3
"""
Debug script to check labeling results
"""
import pandas as pd
import numpy as np

def analyze_labeling_results():
    """Analyze the actual labeling results to find the issue"""
    
    # Try to read from the most recent processed file
    try:
        # Check what files exist
        import os
        from pathlib import Path
        
        # Look for processed files
        processed_files = []
        for root, dirs, files in os.walk('/tmp/monthly_processing'):
            for file in files:
                if file.endswith('.parquet') and 'processed' in file:
                    processed_files.append(os.path.join(root, file))
        
        if not processed_files:
            print("❌ No processed parquet files found")
            return
        
        # Use the most recent file
        latest_file = max(processed_files, key=os.path.getmtime)
        print(f"📊 Analyzing: {latest_file}")
        
        # Read the data
        df = pd.read_parquet(latest_file)
        print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Check label columns
        label_cols = [col for col in df.columns if col.startswith('label_')]
        weight_cols = [col for col in df.columns if col.startswith('weight_')]
        
        print(f"\n📋 Found {len(label_cols)} label columns:")
        for col in label_cols:
            print(f"  - {col}")
        
        print(f"\n📊 Label Analysis:")
        for col in label_cols:
            unique_vals = df[col].unique()
            win_rate = df[col].mean()
            print(f"  {col}:")
            print(f"    Unique values: {sorted(unique_vals)}")
            print(f"    Win rate: {win_rate:.1%}")
            print(f"    Total wins: {df[col].sum():,}")
            print(f"    Total trades: {len(df):,}")
            
            # Check for any non-binary values
            if not set(unique_vals).issubset({0, 1, np.nan}):
                print(f"    ⚠️  WARNING: Non-binary values found!")
        
        # Check a sample of the data
        print(f"\n🔍 Sample Data (first 10 rows):")
        sample_cols = ['timestamp', 'open', 'high', 'low', 'close'] + label_cols[:3]
        print(df[sample_cols].head(10))
        
        # Check for any obvious issues
        print(f"\n🔍 Data Quality Checks:")
        
        # Check if all labels are identical (would indicate a bug)
        long_labels = [col for col in label_cols if 'long' in col]
        short_labels = [col for col in label_cols if 'short' in col]
        
        print(f"Long label correlations:")
        if len(long_labels) > 1:
            long_corr = df[long_labels].corr()
            print(long_corr)
        
        print(f"\nShort label correlations:")
        if len(short_labels) > 1:
            short_corr = df[short_labels].corr()
            print(short_corr)
        
        # Check if short labels are just inverted long labels (would be wrong)
        if len(long_labels) > 0 and len(short_labels) > 0:
            for long_col in long_labels:
                for short_col in short_labels:
                    correlation = df[long_col].corr(df[short_col])
                    print(f"Correlation {long_col} vs {short_col}: {correlation:.3f}")
                    
                    # Check if they're perfectly negatively correlated (would be wrong)
                    if correlation < -0.9:
                        print(f"  ⚠️  WARNING: {short_col} might be inverted {long_col}!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_labeling_results()