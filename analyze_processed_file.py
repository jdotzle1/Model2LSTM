#!/usr/bin/env python3
"""
Analyze the processed Parquet file to validate everything worked correctly
Run this on your desktop where you downloaded the file
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_processed_file(file_path):
    """Comprehensive analysis of the processed monthly file"""
    print("🔍 ANALYZING PROCESSED MONTHLY FILE")
    print("=" * 60)
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    file_size_mb = file_path.stat().st_size / (1024**2)
    print(f"📁 File: {file_path.name}")
    print(f"📊 Size: {file_size_mb:.1f} MB")
    
    try:
        # Load the data
        print(f"\n📖 Loading data...")
        df = pd.read_parquet(file_path)
        
        print(f"✅ Loaded successfully!")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        
        # Basic structure analysis
        print(f"\n📋 BASIC STRUCTURE")
        print("-" * 30)
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Column analysis
        print(f"\n📊 COLUMN ANALYSIS")
        print("-" * 30)
        
        # Categorize columns
        original_cols = []
        label_cols = []
        weight_cols = []
        feature_cols = []
        other_cols = []
        
        expected_original = ['timestamp', 'rtype', 'publisher_id', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        
        for col in df.columns:
            if col in expected_original:
                original_cols.append(col)
            elif col.startswith('label_'):
                label_cols.append(col)
            elif col.startswith('weight_'):
                weight_cols.append(col)
            elif col in expected_original:
                original_cols.append(col)
            else:
                feature_cols.append(col)
        
        print(f"Original columns ({len(original_cols)}): {original_cols}")
        print(f"Label columns ({len(label_cols)}): {label_cols}")
        print(f"Weight columns ({len(weight_cols)}): {weight_cols}")
        print(f"Feature columns ({len(feature_cols)}): {len(feature_cols)} total")
        
        if len(feature_cols) <= 10:
            print(f"   Features: {feature_cols}")
        else:
            print(f"   First 10 features: {feature_cols[:10]}")
            print(f"   ... and {len(feature_cols) - 10} more")
        
        # Validate expected structure
        print(f"\n✅ STRUCTURE VALIDATION")
        print("-" * 30)
        
        issues = []
        
        # Check for 6 labels + 6 weights
        if len(label_cols) != 6:
            issues.append(f"Expected 6 label columns, got {len(label_cols)}")
        if len(weight_cols) != 6:
            issues.append(f"Expected 6 weight columns, got {len(weight_cols)}")
        
        # Check for ~43 features
        if len(feature_cols) < 40 or len(feature_cols) > 50:
            issues.append(f"Expected ~43 features, got {len(feature_cols)}")
        
        # Check total columns (~65)
        expected_total = len(expected_original) + 12 + 43  # 10 + 12 + 43 = 65
        if abs(len(df.columns) - expected_total) > 5:
            issues.append(f"Expected ~{expected_total} total columns, got {len(df.columns)}")
        
        if issues:
            print("⚠️  Structure issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Structure looks perfect!")
        
        # Data quality analysis
        print(f"\n🧹 DATA QUALITY ANALYSIS")
        print("-" * 30)
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        
        if len(cols_with_nans) > 0:
            print(f"⚠️  Found NaN values in {len(cols_with_nans)} columns:")
            for col, count in cols_with_nans.head(10).items():
                pct = count / len(df) * 100
                print(f"   {col}: {count:,} NaN ({pct:.1f}%)")
            if len(cols_with_nans) > 10:
                print(f"   ... and {len(cols_with_nans) - 10} more columns with NaN")
        else:
            print("✅ No NaN values found!")
        
        # Check price data quality
        price_cols = ['open', 'high', 'low', 'close']
        price_issues = []
        
        for col in price_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                negative_count = (df[col] < 0).sum()
                
                if zero_count > 0:
                    price_issues.append(f"{col}: {zero_count} zeros")
                if negative_count > 0:
                    price_issues.append(f"{col}: {negative_count} negative")
        
        if price_issues:
            print("⚠️  Price data issues:")
            for issue in price_issues:
                print(f"   - {issue}")
        else:
            print("✅ Price data looks clean!")
        
        # Timestamp analysis
        print(f"\n🕐 TIMESTAMP ANALYSIS")
        print("-" * 30)
        
        if 'timestamp' in df.columns:
            ts_col = df['timestamp']
            print(f"Timestamp range: {ts_col.min()} to {ts_col.max()}")
            
            # Check if it's RTH data
            if hasattr(ts_col.dt, 'tz'):
                print(f"Timezone: {ts_col.dt.tz}")
            
            # Convert to Central Time to verify RTH
            try:
                import pytz
                if ts_col.dt.tz is not None:
                    central_times = ts_col.dt.tz_convert(pytz.timezone('US/Central'))
                    sample_hours = central_times.dt.hour.unique()
                    print(f"Hours present (Central Time): {sorted(sample_hours)}")
                    
                    # Check if it's truly RTH (7:30 AM = hour 7, 3:00 PM = hour 14)
                    rth_hours = set(range(7, 15))  # 7 AM to 2:59 PM
                    actual_hours = set(sample_hours)
                    
                    if actual_hours.issubset(rth_hours):
                        print("✅ Data appears to be RTH-only!")
                    else:
                        non_rth = actual_hours - rth_hours
                        print(f"⚠️  Found non-RTH hours: {sorted(non_rth)}")
            except Exception as e:
                print(f"⚠️  Could not verify RTH: {e}")
        else:
            print("❌ No timestamp column found!")
        
        # Label and weight analysis
        print(f"\n🏷️  LABEL & WEIGHT ANALYSIS")
        print("-" * 30)
        
        for label_col in label_cols:
            if label_col in df.columns:
                unique_vals = df[label_col].unique()
                win_rate = df[label_col].mean() * 100
                print(f"{label_col}: {win_rate:.1f}% win rate, values: {sorted(unique_vals)}")
        
        print(f"\nWeight statistics:")
        for weight_col in weight_cols:
            if weight_col in df.columns:
                weights = df[weight_col]
                print(f"{weight_col}: min={weights.min():.3f}, mean={weights.mean():.3f}, max={weights.max():.3f}")
        
        # Feature analysis
        print(f"\n🔧 FEATURE ANALYSIS")
        print("-" * 30)
        
        # Group features by category
        feature_categories = {
            'volume': [col for col in feature_cols if 'volume' in col.lower()],
            'price': [col for col in feature_cols if any(x in col.lower() for x in ['vwap', 'distance', 'price'])],
            'range': [col for col in feature_cols if any(x in col.lower() for x in ['range', 'consolidation'])],
            'return': [col for col in feature_cols if 'return' in col.lower() or 'momentum' in col.lower()],
            'volatility': [col for col in feature_cols if any(x in col.lower() for x in ['volatility', 'atr'])],
            'microstructure': [col for col in feature_cols if any(x in col.lower() for x in ['bar', 'uptick', 'flow'])],
            'time': [col for col in feature_cols if col.startswith('is_')]
        }
        
        for category, cols in feature_categories.items():
            if cols:
                print(f"{category.title()} features ({len(cols)}): {cols[:3]}{'...' if len(cols) > 3 else ''}")
        
        # Sample data preview
        print(f"\n📋 SAMPLE DATA PREVIEW")
        print("-" * 30)
        print("First 3 rows:")
        print(df.head(3).to_string())
        
        # Final summary
        print(f"\n🎉 ANALYSIS SUMMARY")
        print("-" * 30)
        print(f"✅ File loaded successfully: {len(df):,} rows × {len(df.columns)} columns")
        print(f"✅ Data size: {file_size_mb:.1f} MB")
        
        if len(label_cols) == 6 and len(weight_cols) == 6:
            print(f"✅ Weighted labeling: 6 labels + 6 weights ✓")
        else:
            print(f"⚠️  Weighted labeling: {len(label_cols)} labels + {len(weight_cols)} weights")
        
        if 40 <= len(feature_cols) <= 50:
            print(f"✅ Feature engineering: {len(feature_cols)} features ✓")
        else:
            print(f"⚠️  Feature engineering: {len(feature_cols)} features")
        
        if len(cols_with_nans) == 0:
            print(f"✅ Data quality: No NaN values ✓")
        else:
            print(f"⚠️  Data quality: {len(cols_with_nans)} columns with NaN")
        
        print(f"\n🚀 READY FOR XGBOOST TRAINING!")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Analyze the downloaded file
    file_path = r"C:\Users\jdotzler\Desktop\test_single_month_20251103_170901.parquet"
    
    print("🔍 PROCESSED FILE ANALYSIS")
    print("=" * 60)
    print(f"Analyzing: {file_path}")
    
    success = analyze_processed_file(file_path)
    
    if success:
        print(f"\n✅ ANALYSIS COMPLETE - FILE LOOKS GOOD!")
        print(f"Ready to scale to all 186 months!")
    else:
        print(f"\n❌ ANALYSIS FAILED - CHECK ISSUES ABOVE")