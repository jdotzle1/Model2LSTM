#!/usr/bin/env python3
"""
Label Distribution Validation Script

Validates label distributions per trading mode for the weighted labeling system.
Checks that win rates are reasonable (5-50% range) and labels contain only 0 or 1 values.

Usage:
    python validate_label_distributions.py [input_file.parquet]

Requirements: 10.1, 10.3, 10.5
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from project.data_pipeline.validation_utils import LabelDistributionValidator


def main():
    parser = argparse.ArgumentParser(description='Validate label distributions for weighted labeling system')
    parser.add_argument('input_file', nargs='?', 
                       help='Input parquet file with weighted labeling results')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for validation (default: use full dataset)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file to save validation results (JSON format)')
    
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
            print("\nUsage: python validate_label_distributions.py <input_file.parquet>")
            sys.exit(1)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print("=" * 80)
    print("LABEL DISTRIBUTION VALIDATION")
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
        print("\nRunning label distribution validation...")
        validator = LabelDistributionValidator()
        results = validator.validate_label_distributions(df)
        
        # Print detailed report
        validator.print_label_distribution_report(results)
        
        # Save results if requested
        if args.output_file:
            import json
            output_path = Path(args.output_file)
            
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for mode_name, stats in results.items():
                json_results[mode_name] = {}
                for key, value in stats.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        json_results[mode_name][key] = value.item()
                    else:
                        json_results[mode_name][key] = value
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n📄 Results saved to: {output_path}")
        
        # Summary
        all_passed = all(stats.get('validation_passed', False) for stats in results.values() 
                        if 'error' not in stats)
        
        print(f"\n{'✅ SUCCESS' if all_passed else '❌ VALIDATION FAILED'}")
        
        if not all_passed:
            failed_modes = [mode for mode, stats in results.items() 
                          if not stats.get('validation_passed', False) or 'error' in stats]
            print(f"Failed modes: {', '.join(failed_modes)}")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()