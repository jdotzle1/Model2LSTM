#!/usr/bin/env python3
"""
Main script to integrate feature engineering with existing labeled dataset

This script demonstrates the integration functionality for task 9:
- Accepts existing 39-column labeled dataset as input
- Adds all 43 feature columns while preserving labels  
- Ensures feature column names match feature definitions exactly
- Validates feature values are within expected ranges
- Saves enhanced dataset in Parquet format with 82 total columns
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from project.data_pipeline.features import integrate_with_labeled_dataset

def main():
    """Main integration function"""
    
    parser = argparse.ArgumentParser(description='Integrate feature engineering with labeled dataset')
    parser.add_argument('input_path', help='Path to existing labeled dataset (Parquet format)')
    parser.add_argument('--output', '-o', help='Output path for enhanced dataset (optional)')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: add '_featured' suffix to input filename
        base_path = args.input_path.replace('.parquet', '')
        output_path = f"{base_path}_featured.parquet"
    
    print("=" * 70)
    print("ES Futures Feature Engineering Integration")
    print("=" * 70)
    
    try:
        # Run the integration
        df_enhanced = integrate_with_labeled_dataset(args.input_path, output_path)
        
        print("\n" + "=" * 70)
        print("Integration Summary")
        print("=" * 70)
        
        print(f"✓ Successfully enhanced dataset")
        print(f"  Input:  {args.input_path}")
        print(f"  Output: {output_path}")
        print(f"  Shape:  {df_enhanced.shape}")
        
        # Show feature categories added
        feature_categories = {
            'Volume': ['volume_ratio_30s', 'volume_slope_30s', 'volume_slope_5s', 'volume_exhaustion'],
            'Price Context': ['vwap', 'distance_from_vwap_pct', 'vwap_slope', 'distance_from_rth_high', 'distance_from_rth_low'],
            'Consolidation': ['short_range_high', 'short_range_low', 'short_range_size', 'position_in_short_range',
                             'medium_range_high', 'medium_range_low', 'medium_range_size', 'range_compression_ratio',
                             'short_range_retouches', 'medium_range_retouches'],
            'Returns': ['return_30s', 'return_60s', 'return_300s', 'momentum_acceleration', 'momentum_consistency'],
            'Volatility': ['atr_30s', 'atr_300s', 'volatility_regime', 'volatility_acceleration', 'volatility_breakout', 'atr_percentile'],
            'Microstructure': ['bar_range', 'relative_bar_size', 'uptick_pct_30s', 'uptick_pct_60s', 'bar_flow_consistency', 'directional_strength'],
            'Time': ['is_eth', 'is_pre_open', 'is_rth_open', 'is_morning', 'is_lunch', 'is_afternoon', 'is_rth_close']
        }
        
        print(f"\n  Feature Categories Added:")
        for category, features in feature_categories.items():
            present_features = [f for f in features if f in df_enhanced.columns]
            print(f"    {category}: {len(present_features)}/{len(features)} features")
        
        print(f"\n✓ Dataset ready for LSTM model training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)