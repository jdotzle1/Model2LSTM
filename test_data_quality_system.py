#!/usr/bin/env python3
"""
Test script for the enhanced data quality validation system
"""
import sys
import os
sys.path.insert(0, '.')

from fix_data_quality_issues import clean_price_data, validate_cleaned_data, validate_rth_filtering
import pandas as pd
import numpy as np

def test_data_quality_system():
    """Test the enhanced data quality validation system"""
    print('Testing enhanced data quality validation system...')
    
    # Create test data with various issues
    test_data = {
        'timestamp': pd.date_range('2024-01-01 08:00:00', periods=100, freq='1min', tz='US/Central'),
        'open': np.random.normal(5000, 10, 100),
        'high': np.random.normal(5005, 10, 100),
        'low': np.random.normal(4995, 10, 100),
        'close': np.random.normal(5000, 10, 100),
        'volume': np.random.exponential(1000, 100)
    }
    
    # Add some data quality issues
    test_data['open'][5] = 0  # Zero price
    test_data['volume'][10] = -100  # Negative volume
    test_data['high'][15] = test_data['low'][15] - 1  # Invalid OHLC
    
    df_test = pd.DataFrame(test_data)
    print(f'Created test data with {len(df_test)} rows')
    
    # Test enhanced cleaning
    df_clean = clean_price_data(df_test)
    print(f'After cleaning: {len(df_clean)} rows')
    
    # Test validation
    is_valid = validate_cleaned_data(df_clean)
    print(f'Validation result: {is_valid}')
    
    # Test RTH filtering validation
    rth_results = validate_rth_filtering(df_clean)
    print(f'RTH validation valid: {rth_results.get("valid", False)}')
    
    print('Enhanced data quality validation system test completed!')
    return True

if __name__ == '__main__':
    test_data_quality_system()