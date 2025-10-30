#!/usr/bin/env python3
"""
Labeling Systems Comparison Utility

Compares the weighted labeling system results with the original labeling system.
Analyzes differences in win rates, correlations, and XGBoost compatibility.

Usage:
    python compare_labeling_systems.py [weighted_file.parquet] [--original original_file.parquet]

Requirements: Task 9 - comparison utility vs original labeling system
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.validation_utils import OriginalLabelingComparator


def main():
    parser = argparse.ArgumentParser(description='Compare weighted labeling system with original labeling system')
    parser.add_argument('weighted_file', nargs='?', 
                       help='Input parquet file with weighted labeling results')
    parser.add_argument('--original', type=str, default=None,
                       help='Input parquet file with original labeling results (optional)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for comparison (default: use full dataset)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file to save comparison results (JSON format)')
    parser.add_argument('--generate-original', action='store_true',
                       help='Generate original labeling for comparison if not provided')
    
    args = parser.parse_args()
    
    # Determine weighted labeling file
    if args.weighted_file:
        weighted_file = Path(args.weighted_file)
    else:
        # Look for common weighted labeling output files
        possible_files = [
            'project/data/processed/weighted_labeled_dataset.parquet',
            'project/data/test/weighted_labeled_sample.parquet',
            'weighted_labeling_output.parquet'
        ]
        
        weighted_file = None
        for file_path in possible_files:
            if Path(file_path).exists():
                weighted_file = Path(file_path)
                break
        
        if weighted_file is None:
            print("❌ No weighted labeling file specified and no default files found.")
            print("Available options:")
            for file_path in possible_files:
                print(f"  - {file_path}")
            print("\nUsage: python compare_labeling_systems.py <weighted_file.parquet>")
            sys.exit(1)
    
    if not weighted_file.exists():
        print(f"❌ Weighted labeling file not found: {weighted_file}")
        sys.exit(1)
    
    # Determine original labeling file
    original_file = None
    if args.original:
        original_file = Path(args.original)
        if not original_file.exists():
            print(f"❌ Original labeling file not found: {original_file}")
            sys.exit(1)
    
    print("=" * 80)
    print("LABELING SYSTEMS COMPARISON")
    print("=" * 80)
    print(f"Weighted labeling file: {weighted_file}")
    if original_file:
        print(f"Original labeling file: {original_file}")
    else:
        print("Original labeling file: Will attempt to generate from weighted data")
    
    try:
        # Load weighted labeling data
        print("\nLoading weighted labeling data...")
        df_weighted = pd.read_parquet(weighted_file)
        
        if args.sample_size and len(df_weighted) > args.sample_size:
            print(f"Sampling {args.sample_size:,} rows from {len(df_weighted):,} total rows...")
            df_weighted = df_weighted.sample(n=args.sample_size, random_state=42)
        
        print(f"Weighted dataset size: {len(df_weighted):,} rows, {len(df_weighted.columns)} columns")
        
        # Load original labeling data if provided
        df_original = None
        if original_file:
            print("Loading original labeling data...")
            df_original = pd.read_parquet(original_file)
            
            if args.sample_size and len(df_original) > args.sample_size:
                print(f"Sampling {args.sample_size:,} rows from original data...")
                df_original = df_original.sample(n=args.sample_size, random_state=42)
            
            print(f"Original dataset size: {len(df_original):,} rows, {len(df_original.columns)} columns")
        
        # Run comparison
        print("\nRunning labeling systems comparison...")
        comparator = OriginalLabelingComparator()
        results = comparator.compare_with_original_labeling(df_weighted, df_original)
        
        # Print detailed report
        comparator.print_comparison_report(results)
        
        # Additional analysis
        if results['comparison_available']:
            print("\n" + "=" * 80)
            print("DETAILED COMPARISON ANALYSIS")
            print("=" * 80)
            
            # Mode-by-mode detailed comparison
            if 'differences' in results and results['differences']['mode_comparisons']:
                print("\nMode-by-Mode Detailed Analysis:")
                for mode_name, comparison in results['differences']['mode_comparisons'].items():
                    print(f"\n{mode_name.upper()}:")
                    print(f"  Weighted system win rate: {comparison['weighted_win_rate']:.3%}")
                    print(f"  Original system win rate: {comparison['original_win_rate']:.3%}")
                    
                    diff = comparison['win_rate_difference']
                    diff_direction = "higher" if diff > 0 else "lower" if diff < 0 else "same"
                    print(f"  Difference: {abs(diff):.3%} {diff_direction} in weighted system")
                    
                    if not pd.isna(comparison['correlation']):
                        corr = comparison['correlation']
                        corr_strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                        print(f"  Correlation: {corr:.3f} ({corr_strength})")
            
            # System-level comparison
            weighted_stats = results['weighted_system_stats']
            original_stats = results.get('original_system_stats', {})
            
            if original_stats:
                print(f"\nSystem-Level Comparison:")
                print(f"  Weighted system modes: {len(weighted_stats['mode_stats'])}")
                print(f"  Original system profiles: {len(original_stats['mode_stats'])}")
                
                # Overall win rate comparison
                weighted_overall_win_rate = sum(stats['win_rate'] for stats in weighted_stats['mode_stats'].values()) / len(weighted_stats['mode_stats'])
                original_overall_win_rate = sum(stats['win_rate'] for stats in original_stats['mode_stats'].values()) / len(original_stats['mode_stats'])
                
                print(f"  Average win rate (weighted): {weighted_overall_win_rate:.3%}")
                print(f"  Average win rate (original): {original_overall_win_rate:.3%}")
        
        # Save results if requested
        if args.output_file:
            import json
            import numpy as np
            output_path = Path(args.output_file)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json_results = convert_for_json(results)
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n📄 Results saved to: {output_path}")
        
        # Summary
        comparison_successful = results['comparison_available']
        xgboost_ready = results['compatibility_analysis']['xgboost_ready']
        
        print(f"\n{'✅ SUCCESS' if comparison_successful and xgboost_ready else '❌ ISSUES FOUND'}")
        
        if not comparison_successful:
            print("❌ Could not complete comparison with original labeling system")
            if 'comparison_error' in results:
                print(f"   Error: {results['comparison_error']}")
        
        if not xgboost_ready:
            print("❌ Weighted labeling data not ready for XGBoost training")
            for issue in results['compatibility_analysis']['issues']:
                print(f"   - {issue}")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()