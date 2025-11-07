#!/usr/bin/env python3
"""
Restart the DBN conversion process with proper error handling
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def restart_conversion():
    """Restart the conversion process properly"""
    print("🚀 RESTARTING DBN CONVERSION")
    print("=" * 50)
    
    # Ensure work directory exists
    work_dir = Path('/tmp/es_30day_processing')
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Work directory ready: {work_dir}")
    
    # Check if input file exists
    dbn_file = work_dir / "es_data_30day.dbn.zst"
    if not dbn_file.exists():
        print(f"❌ Input file missing: {dbn_file}")
        print("Need to run Step 1 first!")
        return False
    
    size_gb = dbn_file.stat().st_size / (1024**3)
    print(f"✅ Input file ready: {dbn_file} ({size_gb:.2f} GB)")
    
    # Start Step 2A in background
    print("\n🔄 Starting Step 2A conversion...")
    print("This will take 45-90 minutes for the full conversion")
    
    # Run in background with nohup
    cmd = ["nohup", "python3", "step2a_dbn_to_parquet.py"]
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=open('conversion_2a.log', 'w'),
            stderr=subprocess.STDOUT,
            cwd=os.getcwd()
        )
        
        print(f"✅ Process started with PID: {process.pid}")
        print(f"📋 Monitor progress with:")
        print(f"   tail -f /tmp/es_30day_processing/progress_2a.log")
        print(f"📋 Check process status with:")
        print(f"   ps aux | grep {process.pid}")
        print(f"📋 View conversion log with:")
        print(f"   tail -f conversion_2a.log")
        
        # Wait a few seconds to see if it starts properly
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Process is running successfully")
            return True
        else:
            print(f"❌ Process exited immediately with code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start process: {e}")
        return False

if __name__ == "__main__":
    success = restart_conversion()
    if not success:
        print("\n💥 Failed to start conversion")
        sys.exit(1)
    else:
        print("\n🎉 Conversion started successfully!")
        print("Check progress in a few minutes with:")
        print("  python3 check_conversion_status.py")