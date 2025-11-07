#!/usr/bin/env python3
"""
Quick June 2011 Re-run

Simplified version that just re-runs the processing and checks results
"""

import subprocess
import sys

def main():
    """Quick re-run of June 2011"""
    
    print("🚀 QUICK JUNE 2011 RE-RUN")
    print("=" * 40)
    
    # Clear and reprocess
    print("1. Clearing existing results...")
    subprocess.run("aws s3 rm s3://es-1-second-data/processed-data/monthly/2011/06/ --recursive --region us-east-1", shell=True)
    
    print("\n2. Re-processing June 2011...")
    result = subprocess.run("python process_monthly_chunks_fixed.py --test-month 2011-06", shell=True)
    
    if result.returncode == 0:
        print("\n✅ Processing completed")
        print("\n3. Check results with:")
        print("aws s3 ls s3://es-1-second-data/processed-data/monthly/2011/06/statistics/ --region us-east-1")
        print("\n4. Download results with:")
        print("aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/06/statistics/ ./june_2011_rerun/ --recursive --region us-east-1")
    else:
        print(f"\n❌ Processing failed with code: {result.returncode}")

if __name__ == "__main__":
    main()