#!/usr/bin/env python3
"""
Enhanced Data Quality Issues Fix for Monthly Processing Pipeline

This module provides enhanced data quality validation and cleaning functions
with better edge case handling, ES futures specific validation, and improved
timezone/RTH filtering with DST support.

Key enhancements:
- Better ES futures price range validation
- Enhanced OHLC relationship validation with tick precision
- Improved volume validation with zero volume handling
- Better outlier detection using IQR method
- Enhanced timezone handling with DST transition support
- Comprehensive RTH filtering validation
"""
import pandas as pd
import numpy as np


def clean_price_data(df):
    """
    Enhanced price data cleaning with better edge case handling
    
    Fixes and enhancements:
    - Better ES futures price range validation (typical range 1000-6000)
    - Enhanced OHLC relationship validation with tick precision
    - Improved volume validation with zero volume handling
    - Better outlier detection using IQR method
    - Enhanced error reporting and statistics
    """
    print(f"   🧹 Enhanced price data cleaning...")
    
    original_rows = len(df)
    price_cols = ['open', 'high', 'low', 'close']
    
    # Step 1: Basic invalid value detection
    invalid_counts = {}
    for col in price_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            negative_count = (df[col] < 0).sum()
            nan_count = df[col].isnull().sum()
            
            invalid_counts[col] = {
                'zeros': zero_count,
                'negatives': negative_count,
                'nans': nan_count
            }
            
            if zero_count > 0 or negative_count > 0 or nan_count > 0:
                print(f"     ⚠️  {col}: {zero_count} zeros, {negative_count} negatives, {nan_count} NaNs")
    
    # Step 2: Remove rows with any invalid prices
    valid_mask = True
    for col in price_cols:
        if col in df.columns:
            valid_mask = valid_mask & (df[col] > 0) & df[col].notna()
    
    df_clean = df[valid_mask].copy()
    basic_removed = original_rows - len(df_clean)
    
    if basic_removed > 0:
        print(f"     🗑️  Removed {basic_removed:,} rows with invalid prices ({basic_removed/original_rows*100:.2f}%)")
    
    # Step 3: Enhanced ES futures price range validation
    if len(df_clean) > 0:
        for col in price_cols:
            if col in df_clean.columns:
                # ES futures typically trade between 1000-6000 points
                # Use more conservative bounds: 500-10000 to avoid false positives
                es_min_price = 500.0
                es_max_price = 10000.0
                
                out_of_range = ((df_clean[col] < es_min_price) | (df_clean[col] > es_max_price)).sum()
                
                if out_of_range > 0:
                    print(f"     ⚠️  {col}: {out_of_range} prices outside ES range ({es_min_price}-{es_max_price})")
                    # Remove extreme outliers
                    df_clean = df_clean[
                        (df_clean[col] >= es_min_price) & (df_clean[col] <= es_max_price)
                    ].copy()
    
    # Step 4: Enhanced OHLC relationship validation with tick precision
    if len(df_clean) > 0 and all(col in df_clean.columns for col in price_cols):
        # Allow for small rounding errors (0.01 points = 4 ticks tolerance)
        tick_tolerance = 0.01
        
        # Enhanced OHLC validation
        invalid_high = (
            (df_clean['high'] < df_clean['open'] - tick_tolerance) | 
            (df_clean['high'] < df_clean['low'] - tick_tolerance) | 
            (df_clean['high'] < df_clean['close'] - tick_tolerance)
        ).sum()
        
        invalid_low = (
            (df_clean['low'] > df_clean['open'] + tick_tolerance) | 
            (df_clean['low'] > df_clean['high'] + tick_tolerance) | 
            (df_clean['low'] > df_clean['close'] + tick_tolerance)
        ).sum()
        
        # Check for impossible bar ranges (high = low but open != close)
        zero_range_bars = (df_clean['high'] == df_clean['low']).sum()
        impossible_zero_range = (
            (df_clean['high'] == df_clean['low']) & 
            (df_clean['open'] != df_clean['close'])
        ).sum()
        
        if invalid_high > 0 or invalid_low > 0 or impossible_zero_range > 0:
            print(f"     ⚠️  OHLC validation: {invalid_high} invalid highs, {invalid_low} invalid lows, {impossible_zero_range} impossible zero-range bars")
            
            # Remove rows with invalid OHLC relationships
            valid_ohlc = (
                (df_clean['high'] >= df_clean['open'] - tick_tolerance) & 
                (df_clean['high'] >= df_clean['low'] - tick_tolerance) & 
                (df_clean['high'] >= df_clean['close'] - tick_tolerance) &
                (df_clean['low'] <= df_clean['open'] + tick_tolerance) & 
                (df_clean['low'] <= df_clean['high'] + tick_tolerance) & 
                (df_clean['low'] <= df_clean['close'] + tick_tolerance) &
                # Allow zero-range bars only if open = close
                ~((df_clean['high'] == df_clean['low']) & (df_clean['open'] != df_clean['close']))
            )
            
            df_clean = df_clean[valid_ohlc].copy()
            ohlc_removed = len(df) - len(df_clean) - basic_removed
            
            if ohlc_removed > 0:
                print(f"     🗑️  Removed {ohlc_removed:,} additional rows with invalid OHLC relationships")
        else:
            print(f"     ✅ OHLC relationships valid ({zero_range_bars:,} zero-range bars allowed)")
    
    # Step 5: Enhanced volume validation
    if 'volume' in df_clean.columns and len(df_clean) > 0:
        negative_volume = (df_clean['volume'] < 0).sum()
        zero_volume = (df_clean['volume'] == 0).sum()
        nan_volume = df_clean['volume'].isnull().sum()
        
        # Remove negative volumes (clear errors)
        if negative_volume > 0:
            print(f"     ⚠️  Found {negative_volume} negative volume values - removing")
            df_clean = df_clean[df_clean['volume'] >= 0].copy()
        
        # Zero volume is acceptable for some bars, just report
        if zero_volume > 0:
            print(f"     ℹ️  Found {zero_volume} zero volume bars (acceptable)")
        
        # Remove NaN volumes
        if nan_volume > 0:
            print(f"     ⚠️  Found {nan_volume} NaN volume values - removing")
            df_clean = df_clean[df_clean['volume'].notna()].copy()
        
        # Check for extremely high volume (potential errors)
        if len(df_clean) > 100:  # Need sufficient data for percentile calculation
            volume_99th = df_clean['volume'].quantile(0.99)
            extreme_volume_threshold = volume_99th * 50  # 50x the 99th percentile
            extreme_volume = (df_clean['volume'] > extreme_volume_threshold).sum()
            
            if extreme_volume > 0:
                print(f"     ⚠️  Found {extreme_volume} extremely high volume values (>{extreme_volume_threshold:,.0f})")
                # Don't remove these automatically as they might be legitimate
    
    # Step 6: Statistical outlier detection using IQR method
    if len(df_clean) > 100:  # Need sufficient data
        outlier_counts = {}
        for col in price_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Use conservative outlier bounds (3 * IQR instead of 1.5)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
                
                if outliers > 0:
                    print(f"     ℹ️  {col}: {outliers} statistical outliers detected (not removed)")
    
    final_rows = len(df_clean)
    total_removed = original_rows - final_rows
    
    # Enhanced completion summary
    if total_removed > 0:
        removal_rate = total_removed / original_rows * 100
        print(f"   ✅ Enhanced data cleaning complete: {final_rows:,} rows")
        print(f"      Removed: {total_removed:,} rows ({removal_rate:.2f}%)")
        print(f"      Data retention: {(1 - removal_rate/100)*100:.1f}%")
    else:
        print(f"   ✅ Enhanced data cleaning complete: {final_rows:,} rows (no issues found)")
    
    return df_clean


def validate_cleaned_data(df):
    """
    Enhanced validation that cleaned data meets all requirements
    
    Enhancements:
    - More comprehensive validation checks
    - ES futures specific validation
    - Better error reporting with statistics
    - Performance validation (data retention rates)
    """
    print(f"   🔍 Enhanced data validation...")
    
    issues = []
    warnings = []
    
    # Check required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    if len(df) == 0:
        issues.append("DataFrame is empty after cleaning")
        print(f"   ❌ Critical validation issues found:")
        for issue in issues:
            print(f"     - {issue}")
        return False
    
    # Enhanced price validation
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            # Check for remaining invalid values
            zero_negative = (df[col] <= 0).sum()
            if zero_negative > 0:
                issues.append(f"{col} still has {zero_negative} zero/negative values")
            
            # Check for NaN values
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                issues.append(f"{col} has {nan_count} NaN values")
            
            # Check for infinite values
            inf_count = (~np.isfinite(df[col])).sum()
            if inf_count > 0:
                issues.append(f"{col} has {inf_count} infinite values")
            
            # ES futures price range validation
            if len(df) > 0:
                min_price = df[col].min()
                max_price = df[col].max()
                
                if min_price < 500 or max_price > 10000:
                    warnings.append(f"{col} has prices outside typical ES range: {min_price:.2f} - {max_price:.2f}")
    
    # Enhanced OHLC relationship validation
    if all(col in df.columns for col in price_cols) and len(df) > 0:
        tick_tolerance = 0.01
        
        invalid_high = (
            (df['high'] < df['open'] - tick_tolerance) | 
            (df['high'] < df['low'] - tick_tolerance) | 
            (df['high'] < df['close'] - tick_tolerance)
        ).sum()
        
        invalid_low = (
            (df['low'] > df['open'] + tick_tolerance) | 
            (df['low'] > df['high'] + tick_tolerance) | 
            (df['low'] > df['close'] + tick_tolerance)
        ).sum()
        
        if invalid_high > 0:
            issues.append(f"Found {invalid_high} bars with invalid high prices")
        if invalid_low > 0:
            issues.append(f"Found {invalid_low} bars with invalid low prices")
        
        # Check bar range statistics
        bar_ranges = df['high'] - df['low']
        zero_range_count = (bar_ranges == 0).sum()
        if zero_range_count > len(df) * 0.1:  # More than 10% zero-range bars
            warnings.append(f"High number of zero-range bars: {zero_range_count} ({zero_range_count/len(df)*100:.1f}%)")
    
    # Enhanced volume validation
    if 'volume' in df.columns:
        # Check for invalid volume values
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            issues.append(f"Volume still has {negative_volume} negative values")
        
        nan_volume = df['volume'].isnull().sum()
        if nan_volume > 0:
            issues.append(f"Volume has {nan_volume} NaN values")
        
        inf_volume = (~np.isfinite(df['volume'])).sum()
        if inf_volume > 0:
            issues.append(f"Volume has {inf_volume} infinite values")
        
        # Volume statistics
        if len(df) > 0:
            zero_volume_count = (df['volume'] == 0).sum()
            if zero_volume_count > len(df) * 0.05:  # More than 5% zero volume
                warnings.append(f"High number of zero volume bars: {zero_volume_count} ({zero_volume_count/len(df)*100:.1f}%)")
    
    # Timestamp validation
    if 'timestamp' in df.columns:
        nan_timestamp = df['timestamp'].isnull().sum()
        if nan_timestamp > 0:
            issues.append(f"Timestamp has {nan_timestamp} NaN values")
        
        # Check for duplicate timestamps
        if len(df) > 1:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                warnings.append(f"Found {duplicate_timestamps} duplicate timestamps")
        
        # Check timestamp ordering
        if len(df) > 1:
            unsorted_count = (df['timestamp'].diff() < pd.Timedelta(0)).sum()
            if unsorted_count > 0:
                warnings.append(f"Found {unsorted_count} out-of-order timestamps")
    
    # Data quality statistics
    if len(df) > 0:
        print(f"     📊 Data quality statistics:")
        print(f"        Rows: {len(df):,}")
        print(f"        Columns: {len(df.columns)}")
        
        if 'timestamp' in df.columns:
            date_range = df['timestamp'].max() - df['timestamp'].min()
            print(f"        Date range: {date_range}")
        
        # Price statistics
        for col in price_cols:
            if col in df.columns:
                print(f"        {col}: {df[col].min():.2f} - {df[col].max():.2f}")
        
        if 'volume' in df.columns:
            print(f"        Volume: {df['volume'].min():,.0f} - {df['volume'].max():,.0f}")
    
    # Report results
    validation_passed = len(issues) == 0
    
    if issues:
        print(f"   ❌ Validation issues found ({len(issues)}):")
        for issue in issues:
            print(f"     - {issue}")
    
    if warnings:
        print(f"   ⚠️  Validation warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"     - {warning}")
    
    if validation_passed:
        if warnings:
            print(f"   ✅ Data validation passed with {len(warnings)} warnings")
        else:
            print(f"   ✅ Data validation passed - all checks successful")
    
    return validation_passed


def validate_rth_filtering(df, expected_reduction_min=0.30, expected_reduction_max=0.40):
    """
    Enhanced RTH filtering validation with DST handling
    
    Validates:
    - Central Time conversion works correctly year-round
    - RTH filtering (07:30-15:00 CT) is accurate
    - Expected 30-40% data reduction from RTH filtering
    - DST transition handling
    
    Args:
        df: DataFrame with timestamp column
        expected_reduction_min: Minimum expected data reduction (30%)
        expected_reduction_max: Maximum expected data reduction (40%)
    
    Returns:
        dict: Validation results with statistics
    """
    print(f"   🕐 Enhanced RTH filtering validation...")
    
    if 'timestamp' not in df.columns:
        print(f"     ❌ No timestamp column found for RTH validation")
        return {'valid': False, 'error': 'No timestamp column'}
    
    if len(df) == 0:
        print(f"     ❌ Empty DataFrame for RTH validation")
        return {'valid': False, 'error': 'Empty DataFrame'}
    
    try:
        import pytz
        from datetime import time as dt_time
        
        # Get timestamps
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Handle timezone conversion properly
        if timestamps.dt.tz is None:
            # Assume UTC if no timezone
            print(f"     ℹ️  Assuming UTC timezone for naive timestamps")
            timestamps = timestamps.dt.tz_localize(pytz.UTC)
        
        # Convert to Central Time with proper DST handling
        central_tz = pytz.timezone('US/Central')
        central_times = timestamps.dt.tz_convert(central_tz)
        
        # Extract time component
        time_only = central_times.dt.time
        
        # Define RTH bounds (07:30-15:00 CT)
        rth_start = dt_time(7, 30)
        rth_end = dt_time(15, 0)
        
        # Check RTH compliance
        rth_mask = (time_only >= rth_start) & (time_only < rth_end)
        rth_bars = rth_mask.sum()
        non_rth_bars = (~rth_mask).sum()
        
        # Calculate statistics
        total_bars = len(df)
        rth_percentage = rth_bars / total_bars if total_bars > 0 else 0
        non_rth_percentage = non_rth_bars / total_bars if total_bars > 0 else 0
        
        print(f"     📊 RTH Analysis:")
        print(f"        Total bars: {total_bars:,}")
        print(f"        RTH bars: {rth_bars:,} ({rth_percentage:.1%})")
        print(f"        Non-RTH bars: {non_rth_bars:,} ({non_rth_percentage:.1%})")
        
        # Validate RTH compliance
        if non_rth_bars == 0:
            print(f"     ✅ All data is RTH-compliant")
            rth_compliant = True
        else:
            print(f"     ⚠️  Found {non_rth_bars:,} non-RTH bars ({non_rth_percentage:.1%})")
            rth_compliant = False
            
            # Show sample non-RTH times for debugging
            non_rth_times = time_only[~rth_mask].unique()[:10]
            print(f"        Sample non-RTH times: {[str(t) for t in non_rth_times]}")
        
        # Validate expected data reduction (only if we have non-RTH data to filter)
        reduction_valid = True
        if non_rth_bars > 0:
            if expected_reduction_min <= non_rth_percentage <= expected_reduction_max:
                print(f"     ✅ Data reduction within expected range ({expected_reduction_min:.0%}-{expected_reduction_max:.0%})")
            else:
                print(f"     ⚠️  Data reduction outside expected range: {non_rth_percentage:.1%} (expected {expected_reduction_min:.0%}-{expected_reduction_max:.0%})")
                reduction_valid = False
        
        # Check for DST transition handling
        dst_issues = validate_dst_transitions(central_times)
        
        # Hour distribution analysis
        hour_dist = central_times.dt.hour.value_counts().sort_index()
        print(f"     📈 Hour distribution (Central Time):")
        for hour in sorted(hour_dist.index):
            count = hour_dist[hour]
            percentage = count / total_bars * 100
            status = "RTH" if 7 <= hour < 15 else "ETH"
            print(f"        {hour:02d}:xx - {count:,} bars ({percentage:.1f}%) [{status}]")
        
        return {
            'valid': rth_compliant and reduction_valid and not dst_issues['has_issues'],
            'rth_compliant': rth_compliant,
            'reduction_valid': reduction_valid,
            'dst_issues': dst_issues,
            'total_bars': total_bars,
            'rth_bars': rth_bars,
            'non_rth_bars': non_rth_bars,
            'rth_percentage': rth_percentage,
            'non_rth_percentage': non_rth_percentage,
            'hour_distribution': hour_dist.to_dict()
        }
        
    except Exception as e:
        print(f"     ❌ RTH validation error: {str(e)}")
        return {'valid': False, 'error': str(e)}


def validate_dst_transitions(central_times):
    """
    Validate DST transition handling in Central Time data
    
    Checks for:
    - Missing hours during spring forward (2 AM -> 3 AM)
    - Duplicate hours during fall back (2 AM appears twice)
    - Proper timezone offset handling
    
    Args:
        central_times: Pandas Series with Central Time timestamps
    
    Returns:
        dict: DST validation results
    """
    print(f"     🔄 Checking DST transition handling...")
    
    try:
        # Get unique dates in the data
        dates = central_times.dt.date.unique()
        
        # Common DST transition dates (approximate - varies by year)
        # Spring forward: 2nd Sunday in March
        # Fall back: 1st Sunday in November
        
        dst_issues = []
        spring_transitions = 0
        fall_transitions = 0
        
        # Check each year in the data
        years = central_times.dt.year.unique()
        
        for year in years:
            year_data = central_times[central_times.dt.year == year]
            
            # Check March for spring forward (2 AM -> 3 AM, hour 2 missing)
            march_data = year_data[year_data.dt.month == 3]
            if len(march_data) > 0:
                march_hours = march_data.dt.hour.unique()
                if 2 not in march_hours and len(march_data) > 100:  # Significant data in March
                    spring_transitions += 1
                    print(f"        ✅ Spring forward detected in March {year} (hour 2 missing)")
            
            # Check November for fall back (2 AM appears twice)
            november_data = year_data[year_data.dt.year == year]
            november_data = november_data[november_data.dt.month == 11]
            if len(november_data) > 0:
                # Check for duplicate 2 AM hours (would show as different UTC offsets)
                nov_2am = november_data[november_data.dt.hour == 2]
                if len(nov_2am) > 0:
                    # Check if we have both standard and daylight time at 2 AM
                    offsets = nov_2am.dt.strftime('%z').unique()
                    if len(offsets) > 1:
                        fall_transitions += 1
                        print(f"        ✅ Fall back detected in November {year} (multiple 2 AM offsets)")
        
        # Summary
        total_years = len(years)
        if spring_transitions > 0 or fall_transitions > 0:
            print(f"        DST transitions detected: {spring_transitions} spring, {fall_transitions} fall")
            has_issues = False
        else:
            if total_years > 1:
                print(f"        ⚠️  No DST transitions detected across {total_years} years")
                has_issues = True
            else:
                print(f"        ℹ️  Single year data - DST transitions may not be present")
                has_issues = False
        
        return {
            'has_issues': has_issues,
            'spring_transitions': spring_transitions,
            'fall_transitions': fall_transitions,
            'years_checked': total_years,
            'issues': dst_issues
        }
        
    except Exception as e:
        print(f"        ❌ DST validation error: {str(e)}")
        return {
            'has_issues': True,
            'error': str(e)
        }


def filter_to_rth_only(df):
    """
    Enhanced RTH filtering with proper DST handling
    
    Filters data to Regular Trading Hours (07:30-15:00 Central Time)
    with proper handling of DST transitions throughout the year.
    
    Args:
        df: DataFrame with timestamp column
    
    Returns:
        DataFrame: Filtered to RTH-only data
    """
    print(f"   🕐 Enhanced RTH filtering...")
    
    if 'timestamp' not in df.columns:
        print(f"     ❌ No timestamp column found")
        return df
    
    original_rows = len(df)
    
    try:
        import pytz
        from datetime import time as dt_time
        
        # Get timestamps and handle timezone conversion
        timestamps = pd.to_datetime(df['timestamp'])
        
        if timestamps.dt.tz is None:
            print(f"     ℹ️  Localizing naive timestamps to UTC")
            timestamps = timestamps.dt.tz_localize(pytz.UTC)
        
        # Convert to Central Time with automatic DST handling
        central_tz = pytz.timezone('US/Central')
        central_times = timestamps.dt.tz_convert(central_tz)
        
        # Extract time component for filtering
        time_only = central_times.dt.time
        
        # Define RTH bounds (07:30-15:00 CT)
        rth_start = dt_time(7, 30)
        rth_end = dt_time(15, 0)
        
        # Create RTH mask
        rth_mask = (time_only >= rth_start) & (time_only < rth_end)
        
        # Apply filter
        df_rth = df[rth_mask].copy()
        
        # Update timestamp column with Central Time
        df_rth['timestamp'] = central_times[rth_mask]
        
        # Statistics
        filtered_rows = len(df_rth)
        removed_rows = original_rows - filtered_rows
        retention_rate = filtered_rows / original_rows if original_rows > 0 else 0
        
        print(f"     📊 RTH filtering results:")
        print(f"        Original: {original_rows:,} rows")
        print(f"        RTH only: {filtered_rows:,} rows")
        print(f"        Removed: {removed_rows:,} rows ({(1-retention_rate)*100:.1f}%)")
        print(f"        Retention: {retention_rate:.1%}")
        
        # Validate expected retention rate
        if 0.60 <= retention_rate <= 0.70:  # 60-70% retention expected
            print(f"     ✅ Retention rate within expected range")
        else:
            print(f"     ⚠️  Retention rate outside expected range (60-70%)")
        
        return df_rth
        
    except Exception as e:
        print(f"     ❌ RTH filtering error: {str(e)}")
        return df


if __name__ == "__main__":
    # Test the enhanced cleaning functions
    print("🧪 Testing enhanced data cleaning functions...")
    
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
    
    print("✅ Enhanced data cleaning functions ready")