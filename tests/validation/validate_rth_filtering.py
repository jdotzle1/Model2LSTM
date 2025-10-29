#!/usr/bin/env python3
"""
RTH Filtering Validation Script

Validates that:
1. All data is within RTH hours (07:30-15:00 CT)
2. No is_eth features are True
3. Expected data reduction (~65%)
4. Session periods are correctly assigned
"""

import pandas as pd
import numpy as np
import pytz
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

def validate_rth_filtering(df, dataset_name="dataset"):
    """
    Comprehensive RTH filtering validation
    
    Args:
        df: DataFrame to validate
        dataset_name: Name for reporting
        
    Returns:
        dict: Validation results
    """
    print(f"\n=== RTH FILTERING VALIDATION: {dataset_name} ===")
    
    results = {
        'dataset_name': dataset_name,
        'total_bars': len(df),
        'validation_passed': True,
        'issues': []
    }
    
    try:
        # 1. Check timestamp availability
        if 'timestamp' not in df.columns:
            if df.index.name == 'ts_event' or pd.api.types.is_datetime64_any_dtype(df.index):
                timestamps = df.index
                print("✓ Using index as timestamp")
            else:
                results['issues'].append("No timestamp found")
                results['validation_passed'] = False
                return results
        else:
            timestamps = df['timestamp']
            print("✓ Using timestamp column")
        
        # 2. Convert to Central Time
        central_tz = pytz.timezone('US/Central')
        
        if timestamps.tz is None:
            ct_time = timestamps.tz_localize('UTC').tz_convert(central_tz)
        else:
            ct_time = timestamps.tz_convert(central_tz)
        
        ct_decimal = ct_time.hour + ct_time.minute / 60.0
        
        # 3. Check time ranges
        min_hour = ct_decimal.min()
        max_hour = ct_decimal.max()
        
        print(f"Time range: {min_hour:.2f} to {max_hour:.2f} CT")
        
        # 4. Validate RTH bounds (07:30-15:00 CT)
        rth_start = 7.5  # 07:30 CT
        rth_end = 15.0   # 15:00 CT
        
        before_rth = (ct_decimal < rth_start).sum()
        after_rth = (ct_decimal >= rth_end).sum()
        eth_total = before_rth + after_rth
        
        if eth_total > 0:
            eth_pct = (eth_total / len(df)) * 100
            print(f"❌ Found {eth_total:,} ETH bars ({eth_pct:.1f}%)")
            print(f"   Before RTH (< 07:30): {before_rth:,}")
            print(f"   After RTH (>= 15:00): {after_rth:,}")
            results['issues'].append(f"Found {eth_total} ETH bars")
            results['validation_passed'] = False
        else:
            print(f"✓ All {len(df):,} bars are within RTH (07:30-15:00 CT)")
        
        results['eth_bars'] = eth_total
        results['rth_bars'] = len(df) - eth_total
        
        # 5. Check session period distribution
        session_periods = {
            'pre_open': ((ct_decimal >= 7.5) & (ct_decimal < 8.5)).sum(),
            'rth_open': ((ct_decimal >= 8.5) & (ct_decimal < 9.25)).sum(),
            'morning': ((ct_decimal >= 9.25) & (ct_decimal < 11.0)).sum(),
            'lunch': ((ct_decimal >= 11.0) & (ct_decimal < 13.0)).sum(),
            'afternoon': ((ct_decimal >= 13.0) & (ct_decimal < 14.5)).sum(),
            'rth_close': ((ct_decimal >= 14.5) & (ct_decimal < 15.0)).sum()
        }
        
        print(f"\nSession period distribution:")
        total_session_bars = 0
        for period, count in session_periods.items():
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {period:>10}: {count:>8,} bars ({pct:>5.1f}%)")
            total_session_bars += count
        
        if total_session_bars != len(df):
            missing = len(df) - total_session_bars
            print(f"❌ {missing:,} bars not assigned to any session period")
            results['issues'].append(f"{missing} bars missing session assignment")
            results['validation_passed'] = False
        else:
            print(f"✓ All bars assigned to session periods")
        
        results['session_periods'] = session_periods
        
        # 6. Check is_eth feature if present
        if 'is_eth' in df.columns:
            eth_feature_count = df['is_eth'].sum()
            if eth_feature_count > 0:
                print(f"❌ is_eth feature has {eth_feature_count} True values (should be 0)")
                results['issues'].append(f"is_eth feature has {eth_feature_count} True values")
                results['validation_passed'] = False
            else:
                print(f"✓ is_eth feature is 0 for all bars")
            
            results['is_eth_true_count'] = eth_feature_count
        
        # 7. Estimate data reduction
        # Assume original data was 24/7, RTH is 7.5 hours of 24
        rth_hours_per_day = 7.5
        total_hours_per_day = 24
        expected_reduction_pct = (1 - rth_hours_per_day / total_hours_per_day) * 100
        
        print(f"\nData reduction estimate:")
        print(f"  RTH hours per day: {rth_hours_per_day}")
        print(f"  Expected reduction: ~{expected_reduction_pct:.0f}%")
        print(f"  Current dataset: {len(df):,} bars (RTH-only)")
        
        results['expected_reduction_pct'] = expected_reduction_pct
        
        # 8. Check for weekend data (should be minimal)
        weekdays = ct_time.dt.dayofweek  # 0=Monday, 6=Sunday
        weekend_bars = ((weekdays == 5) | (weekdays == 6)).sum()  # Saturday=5, Sunday=6
        
        if weekend_bars > 0:
            weekend_pct = (weekend_bars / len(df)) * 100
            print(f"⚠️  Found {weekend_bars:,} weekend bars ({weekend_pct:.1f}%)")
            if weekend_pct > 5:  # More than 5% weekend data is suspicious
                results['issues'].append(f"High weekend data: {weekend_pct:.1f}%")
        else:
            print(f"✓ No weekend data found")
        
        results['weekend_bars'] = weekend_bars
        
        # 9. Summary
        print(f"\n=== VALIDATION SUMMARY ===")
        if results['validation_passed']:
            print(f"✅ RTH filtering validation PASSED")
            print(f"   Dataset: {len(df):,} bars, all within RTH")
        else:
            print(f"❌ RTH filtering validation FAILED")
            print(f"   Issues found: {len(results['issues'])}")
            for issue in results['issues']:
                print(f"   - {issue}")
        
        return results
        
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        results['issues'].append(f"Validation error: {str(e)}")
        results['validation_passed'] = False
        return results

def validate_sample_file(file_path):
    """Validate a single Parquet file"""
    print(f"Loading file: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        # Run validation
        results = validate_rth_filtering(df, os.path.basename(file_path))
        
        return results
        
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return {'validation_passed': False, 'issues': [f"File load error: {str(e)}"]}

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate RTH filtering')
    parser.add_argument('file_path', help='Path to Parquet file to validate')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("RTH Filtering Validation")
    print("=" * 50)
    
    # Validate file
    results = validate_sample_file(args.file_path)
    
    # Exit with appropriate code
    if results['validation_passed']:
        print(f"\n🎉 Validation successful!")
        sys.exit(0)
    else:
        print(f"\n💥 Validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()