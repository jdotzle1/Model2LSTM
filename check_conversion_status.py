#!/usr/bin/env python3
"""
Check the status of the DBN conversion process
"""
import os
from pathlib import Path

def check_conversion_status():
    """Check what files exist and what went wrong"""
    work_dir = Path('/tmp/es_30day_processing')
    
    print("🔍 CONVERSION STATUS CHECK")
    print("=" * 50)
    
    # Check if work directory exists
    if not work_dir.exists():
        print(f"❌ Work directory doesn't exist: {work_dir}")
        print("   Creating it now...")
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {work_dir}")
    else:
        print(f"✅ Work directory exists: {work_dir}")
    
    # Check for input file
    dbn_file = work_dir / "es_data_30day.dbn.zst"
    if dbn_file.exists():
        size_gb = dbn_file.stat().st_size / (1024**3)
        print(f"✅ Input file exists: {dbn_file} ({size_gb:.2f} GB)")
    else:
        print(f"❌ Input file missing: {dbn_file}")
        print("   Need to run Step 1 first!")
        return
    
    # Check for output files
    raw_parquet = work_dir / "es_data_raw.parquet"
    if raw_parquet.exists():
        size_mb = raw_parquet.stat().st_size / (1024**2)
        print(f"✅ Raw parquet exists: {raw_parquet} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Raw parquet missing: {raw_parquet}")
    
    rth_parquet = work_dir / "es_data_rth.parquet"
    if rth_parquet.exists():
        size_mb = rth_parquet.stat().st_size / (1024**2)
        print(f"✅ RTH parquet exists: {rth_parquet} ({size_mb:.1f} MB)")
    else:
        print(f"❌ RTH parquet missing: {rth_parquet}")
    
    # Check for log files
    progress_2a = work_dir / "progress_2a.log"
    if progress_2a.exists():
        print(f"✅ Step 2A log exists: {progress_2a}")
        print("   Last 5 lines:")
        with open(progress_2a) as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(f"     {line.strip()}")
    else:
        print(f"❌ Step 2A log missing: {progress_2a}")
    
    progress_2b = work_dir / "progress_2b.log"
    if progress_2b.exists():
        print(f"✅ Step 2B log exists: {progress_2b}")
    else:
        print(f"❌ Step 2B log missing: {progress_2b}")
    
    print("\n📋 NEXT STEPS:")
    if not dbn_file.exists():
        print("1. Run: python3 step1_download_30day.py")
    elif not raw_parquet.exists():
        print("1. Run: python3 step2a_dbn_to_parquet.py")
        print("2. Monitor: tail -f /tmp/es_30day_processing/progress_2a.log")
    elif not rth_parquet.exists():
        print("1. Run: python3 step2b_filter_rth.py")
        print("2. Monitor: tail -f /tmp/es_30day_processing/progress_2b.log")
    else:
        print("✅ All files exist! Ready for processing.")

if __name__ == "__main__":
    check_conversion_status()