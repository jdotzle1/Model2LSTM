#!/usr/bin/env python3
"""
Simple test to validate the enhanced statistics functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_enhanced_get_statistics():
    """Test the enhanced get_statistics method"""
    
    print("🧪 Testing Enhanced get_statistics Method")
    print("=" * 50)
    
    try:
        # Import the weighted labeling module
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, OutputDataFrame
        
        print("✅ Successfully imported weighted labeling modules")
        
        # Create simple test data (RTH hours: 07:30-15:00 CT)
        import pytz
        
        n_bars = 100
        # Create timestamps in Central Time during RTH
        central_tz = pytz.timezone('US/Central')
        start_time_ct = central_tz.localize(datetime(2024, 3, 1, 8, 0, 0))  # 08:00 CT
        
        # Convert to UTC for storage
        timestamps = []
        for i in range(n_bars):
            ct_time = start_time_ct + timedelta(seconds=i)
            utc_time = ct_time.astimezone(pytz.UTC)
            timestamps.append(utc_time)
        
        # Generate simple price data with valid OHLC relationships
        base_price = 4750.0
        data = []
        for i, ts in enumerate(timestamps):
            # Create base price with small random walk
            price = base_price + np.random.normal(0, 0.1)
            
            # Create valid OHLC relationships
            open_price = round(price + np.random.uniform(-0.25, 0.25), 2)
            close_price = round(price + np.random.uniform(-0.25, 0.25), 2)
            
            # Ensure high is at least as high as open and close
            high_price = round(max(open_price, close_price) + np.random.uniform(0, 0.5), 2)
            
            # Ensure low is at most as low as open and close
            low_price = round(min(open_price, close_price) - np.random.uniform(0, 0.5), 2)
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(100, 1000)
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Created test data: {len(df)} rows")
        
        # Process with weighted labeling
        engine = WeightedLabelingEngine()
        df_labeled = engine.process_dataframe(df, validate_performance=False)
        print(f"✅ Weighted labeling complete: {len(df_labeled.columns)} columns")
        
        # Test basic get_statistics (without enhancements)
        output_data = OutputDataFrame(df_labeled, ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        basic_stats = output_data.get_statistics()
        
        print(f"✅ Basic statistics collected for {len(basic_stats)} items")
        
        # Test enhanced get_statistics with additional parameters
        processing_metrics = {
            'processing_time_minutes': 5.2,
            'memory_peak_mb': 1500.0,
            'memory_final_mb': 1200.0,
            'rows_per_minute': 19.2,
            'processing_efficiency_score': 0.75,
            'stage_times': {'labeling': 180.0, 'features': 132.0},
            'slowest_stage': 'labeling',
            'slowest_stage_time': 180.0
        }
        
        rollover_events = [
            {
                'timestamp': datetime(2024, 3, 1, 10, 30, 0),
                'price_gap': 25.5,
                'bars_affected': 6,
                'gap_direction': 'up'
            }
        ]
        
        feature_quality = {
            'features_generated': 43,
            'expected_features': 43,
            'feature_completeness': 100.0,
            'avg_nan_percentage': 15.2,
            'max_nan_percentage': 32.1,
            'high_nan_features': ['feature_1'],
            'features_with_outliers': ['feature_2'],
            'suspicious_ranges': [],
            'quality_score': 0.85
        }
        
        # Test enhanced statistics
        enhanced_stats = output_data.get_statistics(
            processing_metrics=processing_metrics,
            rollover_events=rollover_events,
            feature_quality=feature_quality
        )
        
        print(f"✅ Enhanced statistics collected for {len(enhanced_stats)} items")
        
        # Validate enhanced statistics structure
        dataset_summary = enhanced_stats.get('dataset_summary', {})
        
        # Check processing metrics
        proc_metrics = dataset_summary.get('processing_metrics', {})
        if proc_metrics.get('processing_time_minutes') == 5.2:
            print("✅ Processing metrics correctly included")
        else:
            print(f"❌ Processing metrics issue: {proc_metrics}")
            return False
        
        # Check rollover statistics
        rollover_stats = dataset_summary.get('rollover_statistics', {})
        if rollover_stats.get('total_rollover_events') == 1:
            print("✅ Rollover statistics correctly included")
        else:
            print(f"❌ Rollover statistics issue: {rollover_stats}")
            return False
        
        # Check feature quality metrics
        feature_stats = dataset_summary.get('feature_quality', {})
        if feature_stats.get('features_generated') == 43:
            print("✅ Feature quality metrics correctly included")
        else:
            print(f"❌ Feature quality issue: {feature_stats}")
            return False
        
        # Check that all trading modes have statistics
        expected_modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                         'low_vol_short', 'normal_vol_short', 'high_vol_short']
        
        for mode in expected_modes:
            if mode not in enhanced_stats:
                print(f"❌ Missing mode statistics: {mode}")
                return False
            
            mode_stats = enhanced_stats[mode]
            required_fields = ['win_rate', 'avg_weight', 'validation_passed']
            for field in required_fields:
                if field not in mode_stats:
                    print(f"❌ Missing field {field} in mode {mode}")
                    return False
        
        print("✅ All trading mode statistics validated")
        
        print("\n📊 Sample Enhanced Statistics:")
        print(f"   Processing time: {proc_metrics.get('processing_time_minutes')} minutes")
        print(f"   Peak memory: {proc_metrics.get('memory_peak_mb')} MB")
        print(f"   Rollover events: {rollover_stats.get('total_rollover_events')}")
        print(f"   Features generated: {feature_stats.get('features_generated')}")
        print(f"   Feature quality score: {feature_stats.get('quality_score')}")
        
        # Show sample mode statistics
        sample_mode = enhanced_stats.get('low_vol_long', {})
        print(f"   Sample mode (low_vol_long):")
        print(f"     Win rate: {sample_mode.get('win_rate', 0):.1%}")
        print(f"     Avg weight: {sample_mode.get('avg_weight', 0):.3f}")
        print(f"     Validation passed: {sample_mode.get('validation_passed', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_get_statistics()
    if success:
        print("\n🎯 Enhanced get_statistics Method: WORKING")
        print("The core enhancement to OutputDataFrame.get_statistics() is functional!")
    else:
        print("\n❌ Enhanced get_statistics Method: FAILED")
        print("Please review the errors above.")
    
    sys.exit(0 if success else 1)