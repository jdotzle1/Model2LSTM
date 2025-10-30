#!/usr/bin/env python3
"""
ES Trading Model - Main Entry Point

This is the main production entry point for the weighted labeling pipeline.
Processes ES futures data through the complete pipeline:
1. Weighted labeling (6 volatility modes)
2. Feature engineering (43 features)
3. Output validation for XGBoost training

Usage:
    python main.py --input data.parquet --output processed_data.parquet
    python main.py --help
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_pipeline.pipeline import process_labeling_and_features, PipelineConfig
from src.data_pipeline.validation_utils import run_comprehensive_validation


def main():
    parser = argparse.ArgumentParser(
        description="ES Trading Model - Weighted Labeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input raw_data.parquet --output processed_data.parquet
  python main.py --input raw_data.parquet --output processed_data.parquet --chunk-size 1000
  python main.py --input raw_data.parquet --output processed_data.parquet --validate
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input parquet file with OHLCV data"
    )
    
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Output parquet file for processed data"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for processing (default: 10000)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run comprehensive validation after processing"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Configure pipeline
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        enable_progress_tracking=True,
        enable_performance_monitoring=args.verbose
    )
    
    print("ES Trading Model - Weighted Labeling Pipeline")
    print("=" * 50)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Chunk size: {args.chunk_size:,}")
    print()
    
    try:
        # Load input data
        import pandas as pd
        print("Loading input data...")
        df = pd.read_parquet(args.input)
        print(f"✓ Loaded {len(df):,} rows")
        
        # Process pipeline
        print("\nProcessing weighted labeling and features...")
        df_processed = process_labeling_and_features(df, config)
        print(f"✓ Generated {len(df_processed.columns)} columns")
        
        # Save output
        print(f"\nSaving to {args.output}...")
        df_processed.to_parquet(args.output, index=False)
        print("✓ Saved successfully")
        
        # Run validation if requested
        if args.validate:
            print("\nRunning comprehensive validation...")
            validation_results = run_comprehensive_validation(df_processed)
            if validation_results['overall_validation']['passed']:
                print("✅ Validation passed")
            else:
                print("❌ Validation failed")
                sys.exit(1)
        
        print("\n🎉 Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()