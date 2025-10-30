#!/usr/bin/env python3
"""
Comprehensive Validation Suite

Runs all validation and quality assurance checks for the weighted labeling system.
This is the main entry point for validating weighted labeling results.

Usage:
    python run_comprehensive_validation.py [input_file.parquet]

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.validation_utils import run_comprehensive_validation


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive validation suite for weighted labeling system')
    parser.add_argument('input_file', nargs='?', 
                       help='Input parquet file with weighted labeling results')
    parser.add_argument('--original', type=str, default=None,
                       help='Input parquet file with original labeling results (optional)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for validation (default: use full dataset)')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory to save all validation results')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip printing detailed reports (only show summary)')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual validation results as separate files')
    
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
            print("\nUsage: python run_comprehensive_validation.py <input_file.parquet>")
            sys.exit(1)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE WEIGHTED LABELING VALIDATION SUITE")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    if args.original:
        print(f"Original labeling file: {args.original}")
    
    try:
        # Load data
        print("\nLoading data...")
        df = pd.read_parquet(input_file)
        
        if args.sample_size and len(df) > args.sample_size:
            print(f"Sampling {args.sample_size:,} rows from {len(df):,} total rows...")
            df = df.sample(n=args.sample_size, random_state=42)
        
        print(f"Dataset size: {len(df):,} rows, {len(df.columns)} columns")
        
        # Load original data if provided
        df_original = None
        if args.original:
            original_file = Path(args.original)
            if original_file.exists():
                print("Loading original labeling data...")
                df_original = pd.read_parquet(original_file)
                
                if args.sample_size and len(df_original) > args.sample_size:
                    df_original = df_original.sample(n=args.sample_size, random_state=42)
                
                print(f"Original dataset size: {len(df_original):,} rows")
            else:
                print(f"⚠ Original file not found: {original_file}")
        
        # Run comprehensive validation
        print("\n" + "=" * 80)
        print("RUNNING COMPREHENSIVE VALIDATION SUITE")
        print("=" * 80)
        
        results = run_comprehensive_validation(
            df=df, 
            df_original=df_original, 
            print_reports=not args.no_reports
        )
        
        # Save comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comprehensive_results_file = output_dir / f'comprehensive_validation_{timestamp}.json'
        
        # Convert results for JSON serialization
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
        
        with open(comprehensive_results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n📄 Comprehensive results saved to: {comprehensive_results_file}")
        
        # Save individual validation results if requested
        if args.save_individual:
            print("\nSaving individual validation results...")
            
            # Label distributions
            label_file = output_dir / f'label_distributions_{timestamp}.json'
            with open(label_file, 'w') as f:
                json.dump(convert_for_json(results['label_distributions']), f, indent=2)
            print(f"  📄 Label distributions: {label_file}")
            
            # Weight distributions
            weight_file = output_dir / f'weight_distributions_{timestamp}.json'
            with open(weight_file, 'w') as f:
                json.dump(convert_for_json(results['weight_distributions']), f, indent=2)
            print(f"  📄 Weight distributions: {weight_file}")
            
            # Data quality
            quality_file = output_dir / f'data_quality_{timestamp}.json'
            with open(quality_file, 'w') as f:
                json.dump(convert_for_json(results['data_quality']), f, indent=2)
            print(f"  📄 Data quality: {quality_file}")
            
            # Original comparison
            comparison_file = output_dir / f'original_comparison_{timestamp}.json'
            with open(comparison_file, 'w') as f:
                json.dump(convert_for_json(results['original_comparison']), f, indent=2)
            print(f"  📄 Original comparison: {comparison_file}")
        
        # Generate summary report
        summary_file = output_dir / f'validation_summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write("WEIGHTED LABELING VALIDATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input File: {input_file}\n")
            f.write(f"Dataset Size: {len(df):,} rows, {len(df.columns)} columns\n\n")
            
            overall = results['overall_validation']
            f.write(f"Overall Validation: {'PASSED' if overall['passed'] else 'FAILED'}\n\n")
            
            summary = overall['summary']
            f.write("Individual Validations:\n")
            f.write(f"  Label validation: {'PASSED' if summary['label_validation_passed'] else 'FAILED'}\n")
            f.write(f"  Weight validation: {'PASSED' if summary['weight_validation_passed'] else 'FAILED'}\n")
            f.write(f"  Data quality: {'PASSED' if summary['data_quality_passed'] else 'FAILED'}\n")
            f.write(f"  XGBoost ready: {'YES' if summary['xgboost_ready'] else 'NO'}\n\n")
            
            # Mode-specific results
            f.write("Mode-Specific Results:\n")
            for mode_name, stats in results['label_distributions'].items():
                if 'error' not in stats:
                    f.write(f"  {mode_name}:\n")
                    f.write(f"    Win rate: {stats['win_rate_percentage']:.1f}%\n")
                    f.write(f"    Samples: {stats['total_samples']:,}\n")
                    f.write(f"    Validation: {'PASSED' if stats['validation_passed'] else 'FAILED'}\n")
            
            # Issues summary
            quality_issues = results['data_quality']['quality_issues']
            high_issues = [i for i in quality_issues if i['severity'] == 'high']
            if high_issues:
                f.write(f"\nHigh-Severity Issues Found: {len(high_issues)}\n")
                for issue in high_issues:
                    f.write(f"  - {issue['column']}: {issue['issue']} ({issue['count']} occurrences)\n")
        
        print(f"📄 Summary report saved to: {summary_file}")
        
        # Final status
        overall_passed = results['overall_validation']['passed']
        
        print("\n" + "=" * 80)
        print("FINAL VALIDATION STATUS")
        print("=" * 80)
        
        if overall_passed:
            print("✅ ALL VALIDATIONS PASSED")
            print("   The weighted labeling data is ready for XGBoost training.")
        else:
            print("❌ VALIDATION FAILED")
            print("   Issues found that need to be addressed before training.")
            
            # Show specific failures
            summary = results['overall_validation']['summary']
            if not summary['label_validation_passed']:
                print("   - Label validation failed")
            if not summary['weight_validation_passed']:
                print("   - Weight validation failed")
            if not summary['data_quality_passed']:
                print("   - Data quality issues found")
            if not summary['xgboost_ready']:
                print("   - Data not ready for XGBoost training")
        
        print("=" * 80)
        
        # Exit with appropriate code
        sys.exit(0 if overall_passed else 1)
        
    except Exception as e:
        print(f"❌ Error during comprehensive validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()