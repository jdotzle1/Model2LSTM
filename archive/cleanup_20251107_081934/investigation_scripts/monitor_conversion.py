#!/usr/bin/env python3
"""
Monitor DBN Conversion Progress
"""
import os
import time
import psutil
from pathlib import Path
from datetime import datetime

def monitor_conversion():
    """Monitor the conversion progress"""
    work_dir = Path('/tmp/es_processing')
    progress_file = work_dir / "progress.log"
    
    print("🔍 CONVERSION PROGRESS MONITOR")
    print("=" * 60)
    
    # Check if progress file exists
    if not progress_file.exists():
        print("❌ No progress file found")
        print("   Either conversion hasn't started or no progress logged yet")
        print(f"   Expected file: {progress_file}")
        return
    
    # Show last few lines of progress
    print("📋 RECENT PROGRESS:")
    print("-" * 40)
    try:
        with open(progress_file, 'r') as f:
            lines = f.readlines()
            # Show last 10 lines
            for line in lines[-10:]:
                print(line.strip())
    except Exception as e:
        print(f"Error reading progress file: {e}")
    
    print()
    print("🔍 PROCESS STATUS:")
    print("-" * 40)
    
    # Check for running Python processes
    conversion_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
        try:
            if (proc.info['name'] and 'python' in proc.info['name'].lower() and 
                proc.info['cmdline'] and any('step2_convert' in str(cmd) for cmd in proc.info['cmdline'])):
                conversion_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if conversion_processes:
        for proc in conversion_processes:
            running_time = time.time() - proc['create_time']
            print(f"✅ Conversion process running:")
            print(f"   PID: {proc['pid']}")
            print(f"   CPU: {proc['cpu_percent']:.1f}%")
            print(f"   Memory: {proc['memory_percent']:.1f}%")
            print(f"   Running: {running_time/60:.1f} minutes")
            
            # Determine status based on CPU usage
            if proc['cpu_percent'] > 10:
                print(f"   Status: 🔄 ACTIVELY WORKING")
            elif running_time > 600:  # 10 minutes
                print(f"   Status: ⚠️  LOW CPU - might be stuck or in slow phase")
            else:
                print(f"   Status: 🔄 STARTING UP")
    else:
        print("❌ No conversion processes found")
    
    print()
    print("📁 OUTPUT FILES:")
    print("-" * 40)
    
    # Check for output files
    files_to_check = [
        "es_data.dbn.zst",
        "es_data_rth.parquet",
        "progress.log"
    ]
    
    for filename in files_to_check:
        filepath = work_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024**2)
            mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
            print(f"✅ {filename}: {size_mb:.1f} MB (modified: {mod_time.strftime('%H:%M:%S')})")
        else:
            print(f"❌ {filename}: Not found")
    
    print()
    print("🔧 MONITORING COMMANDS:")
    print("-" * 40)
    print("# Watch progress in real-time:")
    print("tail -f /tmp/es_processing/progress.log")
    print()
    print("# Check this monitor again:")
    print("python3 monitor_conversion.py")
    print()
    print("# Check all running processes:")
    print("python3 check_running_processes.py")

if __name__ == "__main__":
    monitor_conversion()