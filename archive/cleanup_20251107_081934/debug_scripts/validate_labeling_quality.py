#!/usr/bin/env python3
"""
Robust Labeling Quality Validation Script

This script performs comprehensive validation of trading labels to ensure
they are suitable for XGBoost training. It checks for:
- Contract rollover issues
- Abnormal price jumps
- Inverted labeling logic
- Statistical anomalies
- Data quality issues
- High short win rate investigation

Usage:
python3 validate_labeling_quality.py --s3-path s3://bucket/path/to/file.parquet
python3 validate_labeling_quality.py --local-path /path/to/file.parquet
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LabelingValidator:
    """Comprehensive labeling validation system with focus on short trade investigation"""
    
    def __init__(self, data_path: str, is_s3: bool = False):
        """Initialize validator with data path"""
        self.data_path = data_path
        self.is_s3 = is_s3
        self.df = None
        self.tick_size = 0.25
        
    def load_data(self) -> bool:
        """Load data from S3 or local path"""
        try:
            if self.is_s3:
                # Download from S3
                local_file = "/tmp/validation_data.parquet"
                print(f"📥 Downloading {self.data_path}...")
                os.system(f"aws s3 cp {self.data_path} {local_file} --region us-east-1")
                
                if not os.path.exists(local_file):
                    print("❌ Failed to download file from S3")
                    return False
                    
                self.df = pd.read_parquet(local_file)
                os.remove(local_file)  # Clean up
            else:
                self.df = pd.read_parquet(self.data_path)
                
            print(f"✅ Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def validate_basic_structure(self) -> Dict:
        """Validate basic data structure"""
        print("\n🔍 BASIC STRUCTURE VALIDATION")
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        label_cols = [col for col in self.df.columns if col.startswith('label_')]
        weight_cols = [col for col in self.df.columns if col.startswith('weight_')]
        
        results = {
            'required_columns_present': all(col in self.df.columns for col in required_cols),
            'label_columns_count': len(label_cols),
            'weight_columns_count': len(weight_cols),
            'expected_label_columns': 6,
            'expected_weight_columns': 6,
            'label_columns': label_cols,
            'weight_columns': weight_cols
        }
        
        print(f"  Required columns present: {results['required_columns_present']}")
        print(f"  Label columns: {len(label_cols)}/6 expected")
        print(f"  Weight columns: {len(weight_cols)}/6 expected")
        
        return results
    
    def investigate_short_win_rates(self) -> Dict:
        """Deep investigation of short trade win rates"""
        print("\n🚨 SHORT WIN RATE INVESTIGATION")
        
        label_cols = [col for col in self.df.columns if col.startswith('label_')]
        long_cols = [col for col in label_cols if 'long' in col]
        short_cols = [col for col in label_cols if 'short' in col]
        
        results = {
            'long_win_rates': {},
            'short_win_rates': {},
            'suspicious_short_modes': [],
            'sample_analysis': {}
        }
        
        # Calculate win rates
        print("  Win rates by mode:")
        for col in long_cols:
            win_rate = self.df[col].mean()
            results['long_win_rates'][col] = win_rate
            print(f"    {col}: {win_rate:.1%}")
            
        for col in short_cols:
            win_rate = self.df[col].mean()
            results['short_win_rates'][col] = win_rate
            print(f"    {col}: {win_rate:.1%}")
            
            # Flag suspicious short win rates (>50%)
            if win_rate > 0.5:
                results['suspicious_short_modes'].append(col)
                print(f"      🚨 SUSPICIOUS: {col} win rate {win_rate:.1%} too high!")
        
        # Detailed sample analysis for suspicious modes
        if results['suspicious_short_modes']:
            print(f"\n  🔍 DETAILED ANALYSIS OF SUSPICIOUS MODES:")
            for col in results['suspicious_short_modes'][:2]:  # Analyze first 2
                results['sample_analysis'][col] = self._analyze_short_mode_samples(col)
        
        return results
    
    def _analyze_short_mode_samples(self, label_col: str) -> Dict:
        """Analyze specific samples for a short mode to understand high win rates"""
        print(f"\n    📊 Analyzing {label_col}:")
        
        # Get mode parameters
        if 'low_vol' in label_col:
            stop_ticks, target_ticks = 6, 12
        elif 'normal_vol' in label_col:
            stop_ticks, target_ticks = 8, 16
        else:  # high_vol
            stop_ticks, target_ticks = 10, 20
            
        # Sample some winning trades
        winners = self.df[self.df[label_col] == 1].head(10)
        losers = self.df[self.df[label_col] == 0].head(10)
        
        print(f"      Stop: {stop_ticks} ticks, Target: {target_ticks} ticks")
        print(f"      Winners: {len(self.df[self.df[label_col] == 1]):,}")
        print(f"      Losers: {len(self.df[self.df[label_col] == 0]):,}")
        
        # Analyze a few specific winning trades
        analysis = {
            'stop_ticks': stop_ticks,
            'target_ticks': target_ticks,
            'winner_count': len(self.df[self.df[label_col] == 1]),
            'loser_count': len(self.df[self.df[label_col] == 0]),
            'sample_winners': []
        }
        
        print(f"      Sample winning trades:")
        for idx, row in winners.head(5).iterrows():
            # Simulate the trade logic
            entry_price = row['open']  # Simplified - actual uses next bar
            target_price = entry_price - (target_ticks * self.tick_size)
            stop_price = entry_price + (stop_ticks * self.tick_size)
            
            # Check if this makes sense
            target_hit = row['low'] <= target_price
            stop_hit = row['high'] >= stop_price
            
            sample_info = {
                'timestamp': row['timestamp'],
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_price': stop_price,
                'bar_high': row['high'],
                'bar_low': row['low'],
                'target_hit': target_hit,
                'stop_hit': stop_hit,
                'both_hit': target_hit and stop_hit
            }
            
            analysis['sample_winners'].append(sample_info)
            
            print(f"        {row['timestamp']}: Entry={entry_price:.2f}, Target={target_price:.2f}, Stop={stop_price:.2f}")
            print(f"          Bar: H={row['high']:.2f}, L={row['low']:.2f}")
            print(f"          Target hit: {target_hit}, Stop hit: {stop_hit}")
            
            if target_hit and stop_hit:
                print(f"          ⚠️  BOTH HIT - This should be a loss!")
            elif not target_hit:
                print(f"          ⚠️  TARGET NOT HIT - Why is this a winner?")
        
        return analysis
    
    def check_price_continuity(self) -> Dict:
        """Check for price gaps that might indicate rollover issues"""
        print("\n🔍 PRICE CONTINUITY CHECK")
        
        # Calculate price changes
        self.df['price_change'] = self.df['close'].diff()
        self.df['price_change_pct'] = self.df['close'].pct_change()
        
        # Find large gaps (potential rollovers)
        large_gaps = self.df[abs(self.df['price_change']) > 50]  # >50 point gaps
        large_pct_gaps = self.df[abs(self.df['price_change_pct']) > 0.02]  # >2% gaps
        
        results = {
            'total_bars': len(self.df),
            'large_point_gaps': len(large_gaps),
            'large_pct_gaps': len(large_pct_gaps),
            'max_point_gap': abs(self.df['price_change']).max(),
            'max_pct_gap': abs(self.df['price_change_pct']).max()
        }
        
        print(f"  Total bars: {results['total_bars']:,}")
        print(f"  Large point gaps (>50): {results['large_point_gaps']}")
        print(f"  Large % gaps (>2%): {results['large_pct_gaps']}")
        print(f"  Max point gap: {results['max_point_gap']:.2f}")
        print(f"  Max % gap: {results['max_pct_gap']:.2%}")
        
        if len(large_gaps) > 0:
            print(f"  🚨 Found {len(large_gaps)} large price gaps - potential rollover issues!")
            print("  Sample large gaps:")
            for idx, row in large_gaps.head(3).iterrows():
                print(f"    {row['timestamp']}: {row['price_change']:.2f} points")
        
        return results
    
    def validate_label_logic(self) -> Dict:
        """Validate that label logic makes sense"""
        print("\n🔍 LABEL LOGIC VALIDATION")
        
        results = {
            'binary_labels': True,
            'positive_weights': True,
            'reasonable_win_rates': True,
            'issues': []
        }
        
        # Check binary labels
        label_cols = [col for col in self.df.columns if col.startswith('label_')]
        for col in label_cols:
            unique_vals = set(self.df[col].unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                results['binary_labels'] = False
                results['issues'].append(f"{col} has non-binary values: {unique_vals}")
        
        # Check positive weights
        weight_cols = [col for col in self.df.columns if col.startswith('weight_')]
        for col in weight_cols:
            if (self.df[col] <= 0).any():
                results['positive_weights'] = False
                results['issues'].append(f"{col} has non-positive weights")
        
        # Check reasonable win rates
        for col in label_cols:
            win_rate = self.df[col].mean()
            if win_rate > 0.7 or win_rate < 0.1:
                results['reasonable_win_rates'] = False
                results['issues'].append(f"{col} has unreasonable win rate: {win_rate:.1%}")
        
        print(f"  Binary labels: {results['binary_labels']}")
        print(f"  Positive weights: {results['positive_weights']}")
        print(f"  Reasonable win rates: {results['reasonable_win_rates']}")
        
        if results['issues']:
            print("  Issues found:")
            for issue in results['issues']:
                print(f"    - {issue}")
        
        return results
    
    def run_full_validation(self) -> Dict:
        """Run complete validation suite"""
        print("🔍 STARTING COMPREHENSIVE LABELING VALIDATION")
        print("=" * 60)
        
        if not self.load_data():
            return {'error': 'Failed to load data'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_path': self.data_path,
            'total_rows': len(self.df),
            'basic_structure': self.validate_basic_structure(),
            'short_investigation': self.investigate_short_win_rates(),
            'price_continuity': self.check_price_continuity(),
            'label_logic': self.validate_label_logic()
        }
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        
        suspicious_count = len(results['short_investigation']['suspicious_short_modes'])
        if suspicious_count > 0:
            print(f"🚨 Found {suspicious_count} suspicious short modes with high win rates!")
            for mode in results['short_investigation']['suspicious_short_modes']:
                win_rate = results['short_investigation']['short_win_rates'][mode]
                print(f"   - {mode}: {win_rate:.1%}")
        else:
            print("✅ No suspicious short win rates found")
        
        return results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Validate labeling quality')
    parser.add_argument('--s3-path', help='S3 path to parquet file')
    parser.add_argument('--local-path', help='Local path to parquet file')
    
    args = parser.parse_args()
    
    if args.s3_path:
        validator = LabelingValidator(args.s3_path, is_s3=True)
    elif args.local_path:
        validator = LabelingValidator(args.local_path, is_s3=False)
    else:
        # Default to recent S3 file
        s3_path = "s3://es-1-second-data/processed-data/monthly/2011/06/monthly_2011-06_20251106_173444.parquet"
        print(f"No path specified, using default: {s3_path}")
        validator = LabelingValidator(s3_path, is_s3=True)
    
    results = validator.run_full_validation()
    
    # Save results
    import json
    output_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()