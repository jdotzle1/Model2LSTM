#!/usr/bin/env python3
"""
Get Unstuck - Emergency Diagnostic and Recovery Tool

Run this if your DBN conversion is hanging or you're seeing "nohup: ignoring input"
"""

import os
import sys
import psutil
import subprocess
from pathlib import Path

def kill_stuck_python_processes():
    """Find and offer to kill stuck Python processes"""
    print("🔍 FINDING STUCK PYTHON PROCESSES")
    print("=" * 50)
    
    stuck_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'create_time']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                # Check if it's been running for a while with low CPU
                running_time = psutil.time.time() - proc.info['create_time']
                cpu_percent = proc.cpu_percent()
                
                # Consider it stuck if running >10 minutes with <1% CPU
                if running_time > 600 and cpu_percent < 1.0:
                    stuck_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else 'Unknown',
                        'running_time': running_time,
                        'cpu_percent': cpu_percent
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not stuck_processes:
        print("✅ No stuck Python processes found")
        return
    
    print(f"Found {len(stuck_processes)} potentially stuck processes:")
    print()
    
    for i, proc in enumerate(stuck_processes, 1):
        print(f"Process {i}:")
        print(f"  PID: {proc['pid']}")
        print(f"  Command: {proc['cmdline'][:100]}...")
        print(f"  Running: {proc['running_time']/60:.1f} minutes")
        print(f"  CPU: {proc['cpu_percent']:.1f}%")
        print()
    
    # Ask user if they want to kill these processes
    response = input("Kill these stuck processes? (y/N): ").strip().lower()
    
    if response == 'y':
        for proc in stuck_processes:
            try:
                os.kill(proc['pid'], 9)  # SIGKILL
                print(f"✅ Killed process {proc['pid']}")
            except ProcessLookupError:
                print(f"⚠️  Process {proc['pid']} already gone")
            except PermissionError:
                print(f"❌ Permission denied to kill process {proc['pid']}")
        print("Process cleanup complete!")
    else:
        print("Processes left running")

def check_work_directory():
    """Check the work directory and files"""
    print("\n🔍 CHECKING WORK DIRECTORY")
    print("=" * 50)
    
    work_dir = Path('/tmp/es_processing')
    
    if not work_dir.exists():
        print(f"❌ Work directory doesn't exist: {work_dir}")
        print("You may need to run Step 1 (download) first")
        return
    
    print(f"✅ Work directory exists: {work_dir}")
    
    # Check files
    files_to_check = [
        "es_data.dbn.zst",
        "es_data_rth.parquet", 
        "es_data_sample.parquet",
        "es_data_labeled.parquet",
        "es_data_complete.parquet"
    ]
    
    print("\nFile status:")
    for filename in files_to_check:
        filepath = work_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024**2)
            print(f"  ✅ {filename}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {filename}: Not found")

def suggest_next_steps():
    """Suggest what to do next based on current state"""
    print("\n🔧 SUGGESTED NEXT STEPS")
    print("=" * 50)
    
    work_dir = Path('/tmp/es_processing')
    
    # Check what files exist to determine where to resume
    if not work_dir.exists():
        print("1. Create work directory and run Step 1 (download)")
        print("   mkdir -p /tmp/es_processing")
        print("   python3 step1_download.py")
        
    elif not (work_dir / "es_data.dbn.zst").exists():
        print("1. Run Step 1 to download the DBN file")
        print("   python3 step1_download.py")
        
    elif not (work_dir / "es_data_rth.parquet").exists():
        print("1. Run diagnostic tools first:")
        print("   python3 debug_dbn_conversion.py")
        print()
        print("2. Then run Step 2 conversion in a screen session:")
        print("   screen -S conversion")
        print("   python3 step2_convert.py")
        print("   # Press Ctrl+A, then D to detach")
        
    elif not (work_dir / "es_data_sample.parquet").exists():
        print("1. Run Step 3 to create sample data:")
        print("   python3 step3_sample.py")
        
    elif not (work_dir / "es_data_labeled.parquet").exists():
        print("1. Run Step 4 to apply weighted labeling:")
        print("   python3 step4_labeling.py")
        
    elif not (work_dir / "es_data_complete.parquet").exists():
        print("1. Run Step 5 to add features:")
        print("   python3 step5_features.py")
        
    else:
        print("1. All files exist! Run Step 6 to upload to S3:")
        print("   python3 step6_upload.py")
        print()
        print("2. Or check if everything is working:")
        print("   python3 -c \"import pandas as pd; df = pd.read_parquet('/tmp/es_processing/es_data_complete.parquet'); print(f'Rows: {len(df):,}, Columns: {len(df.columns)}')\"")

def main():
    """Main recovery function"""
    print("🚨 GET UNSTUCK - RECOVERY TOOL")
    print("=" * 60)
    print()
    
    # Step 1: Kill stuck processes
    kill_stuck_python_processes()
    
    # Step 2: Check work directory
    check_work_directory()
    
    # Step 3: Suggest next steps
    suggest_next_steps()
    
    print("\n" + "=" * 60)
    print("🎯 QUICK RECOVERY COMMANDS")
    print("=" * 60)
    print()
    print("# Kill all Python processes (nuclear option)")
    print("pkill -f python")
    print()
    print("# Start fresh conversion in screen session")
    print("screen -S conversion")
    print("python3 step2_convert.py")
    print()
    print("# Monitor system resources")
    print("top")
    print()
    print("# Check conversion progress (if using nohup)")
    print("tail -f nohup.out")

if __name__ == "__main__":
    main()