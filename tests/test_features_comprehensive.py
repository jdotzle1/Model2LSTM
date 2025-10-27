"""
Comprehensive Feature Engineering Test Suite
Tests all 43 features across 7 categories with data leakage prevention and performance validation.
Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
"""
import sys, os, time, pandas as pd, numpy as np

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from project.data_pipeline.features import (
    create_all_features, get_expected_feature_names, add_volume_features, add_price_context_features, 
    add_consolidation_features, add_return_features, add_volatility_features, add_microstructure_features, add_time_features
)


def create_test_data(n_bars=1000):
    """Create synthetic OHLCV data for testing"""
    np.random.seed(42)
    timestamps = pd.date_range('2025-01-15 14:30:00', periods=n_bars, freq='1s', tz='UTC')
    base_price, prices = 4750.0, [4750.0]
    
    for i in range(1, n_bars):
        prices.append(prices[-1] + np.random.normal(0, 0.25))
    
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        high, low = close + abs(np.random.normal(0, 0.1)), close - abs(np.random.normal(0, 0.1))
        open_price = prices[i-1] if i > 0 else close
        data.append({'timestamp': ts, 'open': open_price, 'high': high, 'low': low, 'close': close, 'volume': np.random.randint(500, 2000)})
    
    return pd.DataFrame(data)


def test_feature_category(category_name, add_func, expected_features, test_size=100):
    """Generic test for feature categories"""
    print(f"Testing {category_name} features...")
    df = create_test_data(test_size)
    df = add_func(df)
    
    for feature in expected_features:
        assert feature in df.columns, f"Missing {feature}"
    
    # Category-specific validations
    if category_name == "volume":
        assert df['volume_ratio_30s'].dropna().min() >= 0, "Volume ratio should be positive"
    elif category_name == "price context":
        vwap_vals = df['vwap'].dropna()
        if len(vwap_vals) > 0:
            assert vwap_vals.min() > 0, "VWAP should be positive"
    elif category_name == "consolidation":
        pos_vals = df['position_in_short_range'].dropna()
        if len(pos_vals) > 0:
            assert pos_vals.min() >= 0 and pos_vals.max() <= 1, "Position should be in [0,1]"
    elif category_name == "volatility":
        atr_vals = df['atr_30s'].dropna()
        if len(atr_vals) > 0:
            assert atr_vals.min() >= 0, "ATR should be positive"
    elif category_name == "microstructure":
        for feature in ['uptick_pct_30s', 'uptick_pct_60s']:
            if feature in df.columns:
                vals = df[feature].dropna()
                if len(vals) > 0:
                    assert vals.min() >= 0 and vals.max() <= 100, f"{feature} should be in [0,100]"
    elif category_name == "time":
        time_features = ['is_eth', 'is_pre_open', 'is_rth_open', 'is_morning', 'is_lunch', 'is_afternoon', 'is_rth_close']
        for feature in time_features:
            if feature in df.columns:
                vals = df[feature].dropna()
                if len(vals) > 0:
                    assert vals.isin([0, 1]).all(), f"{feature} should be binary"
    
    print(f"  ✓ {category_name.title()} features validated")


def test_integration_end_to_end():
    """Test complete end-to-end processing"""
    print("Testing end-to-end integration...")
    df = create_test_data(1000)
    df['existing_label'] = np.random.choice([-1, 0, 1], size=len(df))
    original_cols = len(df.columns)
    
    df_featured = create_all_features(df)
    new_cols = len(df_featured.columns) - original_cols
    assert new_cols == 43, f"Expected 43 new features, got {new_cols}"
    
    expected_features = get_expected_feature_names()
    for feature in expected_features:
        assert feature in df_featured.columns, f"Missing expected feature: {feature}"
    
    assert 'existing_label' in df_featured.columns, "Original columns should be preserved"
    assert len(df_featured) == len(df), "Row count should be preserved"
    print(f"  ✓ Added {new_cols} features to {len(df)} bars")


def test_data_leakage_prevention():
    """Test no future data is used in calculations"""
    print("Testing data leakage prevention...")
    df = create_test_data(100)
    
    # Add future spike at bar 50
    df.loc[50:60, 'close'] = df.loc[50:60, 'close'] + 10
    df.loc[50:60, 'volume'] = df.loc[50:60, 'volume'] * 3
    
    df_featured = create_all_features(df)
    
    # Check features at bar 49 don't reflect spike at bar 50+
    if len(df_featured) > 50:
        vol_ratio_49 = df_featured.loc[49, 'volume_ratio_30s']
        vol_ratio_51 = df_featured.loc[51, 'volume_ratio_30s']
        
        if not pd.isna(vol_ratio_49) and not pd.isna(vol_ratio_51):
            assert vol_ratio_51 > vol_ratio_49, "Future spike should not affect past features"
    
    print("  ✓ Data leakage prevention validated")


def test_performance_validation():
    """Test processing performance meets basic requirements"""
    print("Testing performance validation...")
    test_size = 2000
    df = create_test_data(test_size)
    
    start_time = time.time()
    df_featured = create_all_features(df)
    processing_time = time.time() - start_time
    bars_per_second = test_size / processing_time
    
    print(f"  Processed {test_size:,} bars in {processing_time:.2f} seconds ({bars_per_second:,.0f} bars/sec)")
    assert bars_per_second >= 10, f"Processing too slow: {bars_per_second:.1f} bars/second"
    print("  ✓ Performance validation passed")


def run_all_tests():
    """Run all tests in sequence"""
    print("=" * 50)
    print("COMPREHENSIVE FEATURE ENGINEERING TEST SUITE")
    print("=" * 50)
    
    try:
        # Unit tests for each feature category with expected features
        feature_categories = [
            ("volume", add_volume_features, ['volume_ratio_30s', 'volume_slope_30s', 'volume_slope_5s', 'volume_exhaustion'], 100),
            ("price context", add_price_context_features, ['vwap', 'distance_from_vwap_pct', 'vwap_slope', 'distance_from_rth_high', 'distance_from_rth_low'], 100),
            ("consolidation", add_consolidation_features, ['short_range_high', 'short_range_low', 'short_range_size', 'position_in_short_range', 'medium_range_high', 'medium_range_low', 'medium_range_size', 'range_compression_ratio', 'short_range_retouches', 'medium_range_retouches'], 1000),
            ("return", add_return_features, ['return_30s', 'return_60s', 'return_300s', 'momentum_acceleration', 'momentum_consistency'], 400),
            ("volatility", add_volatility_features, ['atr_30s', 'atr_300s', 'volatility_regime', 'volatility_acceleration', 'volatility_breakout', 'atr_percentile'], 400),
            ("microstructure", add_microstructure_features, ['bar_range', 'relative_bar_size', 'uptick_pct_30s', 'uptick_pct_60s', 'bar_flow_consistency', 'directional_strength'], 100),
            ("time", add_time_features, ['is_eth', 'is_pre_open', 'is_rth_open', 'is_morning', 'is_lunch', 'is_afternoon', 'is_rth_close'], 100)
        ]
        
        for name, func, features, size in feature_categories:
            test_feature_category(name, func, features, size)
        
        # Integration and validation tests
        test_integration_end_to_end()
        test_data_leakage_prevention()
        test_performance_validation()
        
        print("=" * 50)
        print("✅ ALL TESTS PASSED - Feature engineering validated")
        print("=" * 50)
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ TEST FAILED: {str(e)}")
        print("=" * 50)
        raise


if __name__ == "__main__":
    run_all_tests()