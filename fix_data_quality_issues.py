#!/usr/bin/env python3
"""
Fix data quality issues in the monthly processing pipeline
Handle zero/negative prices and other data quality problems
"""
import pandas as pd
import numpy as np

def clean_price_data(df):
    """Clean price data to remove invalid values"""
    print(f"   🧹 Cleaning price data...")
    
    original_rows = len(df)
    
    # Check for invalid prices
    price_cols = ['open', 'high', 'low', 'close']
    
    for col in price_cols:
        if col in df.columns:
            # Count invalid values
            zero_count = (df[col] == 0).sum()
            negative_count = (df[col] < 0).sum()
            
            if zero_count > 0 or negative_count > 0:
                print(f"     ⚠️  {col}: {zero_count} zeros, {negative_count} negative values")
    
    # Remove rows with any invalid prices
    valid_mask = True
    for col in price_cols:
        if col in df.columns:
            valid_mask = valid_mask & (df[col] > 0)
    
    df_clean = df[valid_mask].copy()
    
    removed_rows = original_rows - len(df_clean)
    if removed_rows > 0:
        print(f"     🗑️  Removed {removed_rows:,} rows with invalid prices ({removed_rows/original_rows*100:.2f}%)")
    else:
        print(f"     ✅ No invalid prices found")
    
    # Additional validation
    for col in price_cols:
        if col in df_clean.columns:
            # Check for extremely high prices (likely errors)
            high_price_threshold = df_clean[col].quantile(0.999) * 10  # 10x the 99.9th percentile
            extreme_high = (df_clean[col] > high_price_threshold).sum()
            
            # Check for extremely low prices (likely errors)
            low_price_threshold = df_clean[col].quantile(0.001) / 10  # 1/10th the 0.1st percentile
            extreme_low = (df_clean[col] < low_price_threshold).sum()
            
            if extreme_high > 0 or extreme_low > 0:
                print(f"     ⚠️  {col}: {extreme_high} extremely high, {extreme_low} extremely low prices")
    
    # Validate OHLC relationships
    if all(col in df_clean.columns for col in price_cols):
        # High should be >= Open, Low, Close
        invalid_high = ((df_clean['high'] < df_clean['open']) | 
                       (df_clean['high'] < df_clean['low']) | 
                       (df_clean['high'] < df_clean['close'])).sum()
        
        # Low should be <= Open, High, Close  
        invalid_low = ((df_clean['low'] > df_clean['open']) | 
                      (df_clean['low'] > df_clean['high']) | 
                      (df_clean['low'] > df_clean['close'])).sum()
        
        if invalid_high > 0 or invalid_low > 0:
            print(f"     ⚠️  OHLC validation: {invalid_high} invalid highs, {invalid_low} invalid lows")
            
            # Remove rows with invalid OHLC relationships
            valid_ohlc = ((df_clean['high'] >= df_clean['open']) & 
                         (df_clean['high'] >= df_clean['low']) & 
                         (df_clean['high'] >= df_clean['close']) &
                         (df_clean['low'] <= df_clean['open']) & 
                         (df_clean['low'] <= df_clean['high']) & 
                         (df_clean['low'] <= df_clean['close']))
            
            df_clean = df_clean[valid_ohlc].copy()
            ohlc_removed = len(df) - len(df_clean) - removed_rows
            
            if ohlc_removed > 0:
                print(f"     🗑️  Removed {ohlc_removed:,} additional rows with invalid OHLC relationships")
    
    # Check volume
    if 'volume' in df_clean.columns:
        negative_volume = (df_clean['volume'] < 0).sum()
        if negative_volume > 0:
            print(f"     ⚠️  Found {negative_volume} negative volume values")
            df_clean = df_clean[df_clean['volume'] >= 0].copy()
    
    final_rows = len(df_clean)
    total_removed = original_rows - final_rows
    
    if total_removed > 0:
        print(f"   ✅ Data cleaning complete: {final_rows:,} rows ({total_removed:,} removed, {total_removed/original_rows*100:.2f}%)")
    else:
        print(f"   ✅ Data cleaning complete: {final_rows:,} rows (no issues found)")
    
    return df_clean

def validate_cleaned_data(df):
    """Validate that cleaned data meets requirements"""
    print(f"   🔍 Validating cleaned data...")
    
    issues = []
    
    # Check required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for remaining invalid prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            if (df[col] <= 0).any():
                issues.append(f"{col} still has zero/negative values")
    
    # Check for NaN values in critical columns
    for col in required_cols:
        if col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                issues.append(f"{col} has {nan_count} NaN values")
    
    if issues:
        print(f"   ❌ Validation issues found:")
        for issue in issues:
            print(f"     - {issue}")
        return False
    else:
        print(f"   ✅ Data validation passed")
        return True

if __name__ == "__main__":
    # Test the cleaning functions
    print("🧪 Testing data cleaning functions...")
    
    # Create test data with issues
    test_data = {
        'timestamp': pd.date_range('2010-07-01', periods=10, freq='1min'),
        'open': [100, 0, 101, -5, 102, 103, 104, 105, 106, 107],  # Zero and negative
        'high': [101, 102, 102, 103, 103, 104, 105, 106, 107, 108],
        'low': [99, 98, 100, 99, 101, 102, 103, 104, 105, 106],
        'close': [100.5, 101, 101.5, 102, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
        'volume': [1000, 1100, 1200, -100, 1400, 1500, 1600, 1700, 1800, 1900]  # Negative volume
    }
    
    df_test = pd.DataFrame(test_data)
    print(f"Original test data: {len(df_test)} rows")
    
    df_clean = clean_price_data(df_test)
    print(f"Cleaned test data: {len(df_clean)} rows")
    
    validate_cleaned_data(df_clean)
    
    print("✅ Data cleaning functions ready")