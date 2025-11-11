#!/usr/bin/env python3 -u
"""
Process 15 years of ES data in monthly batches

Clean CLI script that uses modular components.
This is the PRODUCTION script for monthly batch processing.

Usage:
    python scripts/process_monthly_batches.py
    python scripts/process_monthly_batches.py --start-year 2024 --start-month 1
"""

import sys
import os
from pathlib import Path
import argparse

# Force unbuffered output for nohup/background processes
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.monthly_processor import MonthlyProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Process ES data in monthly batches from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--start-year", type=int, default=2010,
        help="Start year (default: 2010)"
    )
    parser.add_argument(
        "--start-month", type=int, default=7,
        help="Start month (default: 7 for July)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2025,
        help="End year (default: 2025)"
    )
    parser.add_argument(
        "--end-month", type=int, default=10,
        help="End month (default: 10 for October)"
    )
    parser.add_argument(
        "--bucket", default="es-1-second-data",
        help="S3 bucket name (default: es-1-second-data)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip months that are already processed in S3"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ES TRADING MODEL - MONTHLY BATCH PROCESSOR")
    print("=" * 80)
    print(f"Date range: {args.start_year}-{args.start_month:02d} to {args.end_year}-{args.end_month:02d}")
    print(f"S3 bucket: {args.bucket}")
    print(f"Skip existing: {args.skip_existing}")
    print()
    
    # Initialize processor
    processor = MonthlyProcessor(bucket_name=args.bucket)
    
    # Generate file list
    print("📅 Generating monthly file list...")
    monthly_files = processor.generate_monthly_file_list(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month
    )
    print(f"   Generated {len(monthly_files)} months to process")
    
    # Check existing if requested
    if args.skip_existing:
        print("\n🔍 Checking for existing processed files...")
        monthly_files = processor.check_existing_processed(monthly_files)
    
    if not monthly_files:
        print("\n✅ All months already processed!")
        return 0
    
    # Process all months
    print(f"\n🚀 Starting batch processing of {len(monthly_files)} months...")
    results = processor.process_all_months(monthly_files)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total months: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total time: {results['total_time_hours']:.1f} hours")
    
    if results['failed_months']:
        print(f"\n❌ Failed months:")
        for month in results['failed_months']:
            print(f"   - {month}")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
