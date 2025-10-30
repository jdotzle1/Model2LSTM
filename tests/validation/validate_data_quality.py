#!/usr/bin/env python3
"""
Data Quality Validation Script

Performs comprehensive data quality checks for NaN/infinite values in the weighted labeling system.
Validates data integrity and identifies quality issues.

Usage:
    python validate_data_quality.py [input_file.parquet]

Requirements: 10.6
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.validation_utils import DataQualityChecker


def main():
    parser = argparse.ArgumentParser(description='Validate data quality for weighted labeling system')
    parser.add_argument('input_file', nargs='?', 
                       help='Input parquet file with weighted labeling results')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for validation (default: use full dataset)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file to save validation results (JSON format)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed column-by-column analysis')
    
    args = parser.parse_args()
    
    # Determine input file
    if args.input_file:
        input_file = Path(args.input_file)
    else:
        # Look for common weighted labeling output files
        possible_files = [
            'project/data/processed/weighted_labeled_dataset.parquet',
            'project/data/test/weighted_labeled_sample.parquet',
            'weighted_labeling_output.parquet'
        ]
        
        input_file = None
        for file_path in possible_files:
            if Path(file_path).exists():
                input_file = Path(file_path)
                break
        
        if input_file is None:
            print("❌ No input file specified and no default files found.")
            print("Available options:")
            for file_path in possible_files:
                print(f"  - {file_path}")
            print("\nUsage: python validate_data_quality.py <input_file.parquet>")
            sys.exit(1)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print("=" * 80)
    print("DATA QUALITY VALIDATION")
    print("=" * 80)
    print(f"Input file: {input_file}")
    
    try:
        # Load data
        print("Loading data...")
        df = pd.read_parquet(input_file)
        
        if args.sample_size and len(df) > args.sample_size:
            print(f"Sampling {args.sample_size:,} rows from {len(df):,} total rows...")
            df = df.sample(n=args.sample_size, random_state=42)
        
        print(f"Dataset size: {len(df):,} rows, {len(df.columns)} columns")
        
        # Run validation
        print("\nRunning data quality checks...")
        checker = DataQualityChecker()
        results = checker.check_data_quality(df)
        
        # Print detailed report
        checker.print_data_quality_report(results)
        
        # Additional detailed analysis if requested
        if args.detailed:
            print("\n" + "=" * 80)
            print("DETAILED COLUMN ANALYSIS")
            print("=" * 80)
            
            for col, analysis in results['column_analysis'].items():
                print(f"\n{col}:")
                print(f"  Type: {analysis['dtype']}")
                print(f"  Non-null: {analysis['non_null_count']:,} ({100 - analysis['null_percentage']:.1f}%)")
                
                if analysis['null_count'] > 0:
                    print(f"  Null: {analysis['null_count']:,} ({analysis['null_percentage']:.2f}%)")
                
                if 'infinite_count' in analysis:
                    if analysis['infinite_count'] > 0:
                        print(f"  Infinite: {analysis['infinite_count']:,} ({analysis['infinite_percentage']:.2f}%)")
                    
                    if not pd.isna(analysis['min_value']):
                        print(f"  Range: {analysis['min_value']:.6f} to {analysis['max_value']:.6f}")
                        print(f"  Mean: {analysis['mean_value']:.6f}")
        
        # Save results if requested
        if args.output_file:
            import json
            output_path = Path(args.output_file)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_for_json(obj):
                if hasattr(obj, 'item'):  # numpy scalar
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
        quality_passed = results['overall_stats']['quality_passed']
        
        print(f"\n{'✅ SUCCESS' if quality_passed else '❌ VALIDATION FAILED'}")
        
        if not quality_passed:
            high_issues = results['overall_stats']['high_severity_issues']
            total_issues = results['overall_stats']['total_issues']
            print(f"Found {high_issues} high-severity issues out of {total_issues} total issues")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()