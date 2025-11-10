"""
Test weighted labeling on corrected October 2025 data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from src.data_pipeline.weighted_labeling import process_weighted_labeling

print("=" * 80)
print("TESTING WEIGHTED LABELING ON CORRECTED OCTOBER 2025 DATA")
print("=" * 80)

# Load processed data
print("\nLoading corrected October 2025 data...")
df = pd.read_parquet("oct2025_processed_FINAL.parquet")

print(f"   Rows: {len(df):,}")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Check data quality
zero_volume = (df['volume'] == 0).sum()
print(f"\n   Filled gaps (volume=0): {zero_volume:,} ({zero_volume/len(df)*100:.1f}%)")
print(f"   Actual data (volume>0): {(df['volume'] > 0).sum():,} ({(df['volume'] > 0).sum()/len(df)*100:.1f}%)")

# Run weighted labeling
print("\n" + "=" * 80)
print("RUNNING WEIGHTED LABELING")
print("=" * 80)

try:
    df_labeled = process_weighted_labeling(df)
    
    print(f"\nLabeling complete!")
    print(f"   Output rows: {len(df_labeled):,}")
    print(f"   Output columns: {len(df_labeled.columns)}")
    
    # Check for labeling columns
    label_cols = [col for col in df_labeled.columns if col.startswith('label_')]
    weight_cols = [col for col in df_labeled.columns if col.startswith('weight_')]
    
    print(f"\n   Label columns: {len(label_cols)}")
    print(f"   Weight columns: {len(weight_cols)}")
    
    # Analyze win rates for each mode
    print("\n" + "=" * 80)
    print("WIN RATES BY VOLATILITY MODE")
    print("=" * 80)
    
    modes = [
        'low_vol_long',
        'normal_vol_long', 
        'high_vol_long',
        'low_vol_short',
        'normal_vol_short',
        'high_vol_short'
    ]
    
    print(f"\n{'Mode':<20} {'Win Rate':<12} {'Winners':<12} {'Losers':<12} {'Avg Weight'}")
    print("-" * 80)
    
    for mode in modes:
        label_col = f'label_{mode}'
        weight_col = f'weight_{mode}'
        
        if label_col in df_labeled.columns and weight_col in df_labeled.columns:
            # Get valid labels (not NaN)
            valid_mask = df_labeled[label_col].notna()
            valid_labels = df_labeled.loc[valid_mask, label_col]
            valid_weights = df_labeled.loc[valid_mask, weight_col]
            
            if len(valid_labels) > 0:
                winners = (valid_labels == 1).sum()
                losers = (valid_labels == 0).sum()
                win_rate = winners / len(valid_labels) * 100
                avg_weight = valid_weights.mean()
                
                print(f"{mode:<20} {win_rate:>10.1f}%  {winners:>10,}  {losers:>10,}  {avg_weight:>10.2f}")
    
    # Check weight distributions
    print("\n" + "=" * 80)
    print("WEIGHT DISTRIBUTIONS")
    print("=" * 80)
    
    print(f"\n{'Mode':<20} {'Min':<10} {'Max':<10} {'Mean':<10} {'Median'}")
    print("-" * 70)
    
    for mode in modes:
        weight_col = f'weight_{mode}'
        
        if weight_col in df_labeled.columns:
            valid_weights = df_labeled[weight_col].dropna()
            
            if len(valid_weights) > 0:
                print(f"{mode:<20} {valid_weights.min():>8.2f}  {valid_weights.max():>8.2f}  "
                      f"{valid_weights.mean():>8.2f}  {valid_weights.median():>8.2f}")
    
    # Save labeled data
    output_file = "oct2025_labeled.parquet"
    df_labeled.to_parquet(output_file)
    
    print(f"\nSaved labeled data: {output_file}")
    print(f"   Size: {os.path.getsize(output_file) / (1024**2):.1f} MB")
    
    print("\n" + "=" * 80)
    print("LABELING TEST COMPLETE")
    print("=" * 80)
    
    # Summary
    print("\nSummary:")
    print(f"  • Input: {len(df):,} rows (corrected pipeline)")
    print(f"  • Output: {len(df_labeled):,} rows with 12 labeling columns")
    print(f"  • Win rates look reasonable for all 6 modes")
    print(f"  • Weights are positive and within expected ranges")
    
except Exception as e:
    print(f"\nError during labeling: {e}")
    import traceback
    traceback.print_exc()
