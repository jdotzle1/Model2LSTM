#!/usr/bin/env python3
"""
Debug DBN Conversion Issues

This script helps diagnose why the DBN conversion might be hanging.
"""

import sys
import os
import time
import psutil
from pathlib import Path

def check_system_resources():
    """Check available system resources"""
    print("🔍 SYSTEM RESOURCE CHECK")
    print("=" * 50)
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"Memory Total: {memory.total / (1024**3):.1f} GB")
    print(f"Memory Available: {memory.available / (1024**3):.1f} GB")
    print(f"Memory Used: {memory.percent:.1f}%")
    
    # Disk space
    disk = psutil.disk_usage('/')
    print(f"Disk Total: {disk.total / (1024**3):.1f} GB")
    print(f"Disk Free: {disk.free / (1024**3):.1f} GB")
    print(f"Disk Used: {disk.percent:.1f}%")
    
    # CPU
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    
    print()

def check_file_exists(file_path):
    """Check if file exists and get info"""
    print(f"🔍 FILE CHECK: {file_path}")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    
    print(f"✅ File exists: {file_path}")
    print(f"File size: {file_size:,} bytes")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"File size: {file_size_gb:.2f} GB")
    
    # Check if we have enough memory to process this file
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if file_size_gb > available_gb * 0.5:
        print(f"⚠️  WARNING: File is large ({file_size_gb:.2f} GB) compared to available memory ({available_gb:.1f} GB)")
        print("   This may cause memory issues during processing")
    else:
        print(f"✅ File size looks manageable for available memory")
    
    print()
    return True

def test_databento_import():
    """Test if databento can be imported and works"""
    print("🔍 DATABENTO IMPORT TEST")
    print("=" * 50)
    
    try:
        import databento as db
        print("✅ databento imported successfully")
        print(f"databento version: {db.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import databento: {e}")
        print("Try: pip install databento")
        return False
    except Exception as e:
        print(f"❌ Error with databento: {e}")
        return False

def test_small_dbn_read(dbn_path):
    """Test reading just the first few records from DBN file"""
    print("🔍 SMALL DBN READ TEST")
    print("=" * 50)
    
    try:
        import databento as db
        
        print(f"Attempting to read DBN file: {dbn_path}")
        print("This is just a test read to see if the file is valid...")
        
        # Try to open the store
        print("Opening DBN store...")
        store = db.DBNStore.from_file(dbn_path)
        
        print("✅ DBN store opened successfully")
        
        # Try to get metadata without reading all data
        print("Getting metadata...")
        metadata = store.metadata
        print(f"Dataset: {metadata.dataset}")
        print(f"Schema: {metadata.schema}")
        print(f"Start: {metadata.start}")
        print(f"End: {metadata.end}")
        
        # Try to read just first few records
        print("Reading first 10 records...")
        df_sample = store.to_df(limit=10)
        
        print(f"✅ Successfully read {len(df_sample)} sample records")
        print(f"Columns: {df_sample.columns.tolist()}")
        print("Sample data:")
        print(df_sample.head(3))
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to read DBN file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic checks"""
    print("🚀 DBN CONVERSION DIAGNOSTIC TOOL")
    print("=" * 60)
    print()
    
    # Check system resources
    check_system_resources()
    
    # Test databento import
    if not test_databento_import():
        print("❌ Cannot proceed without databento library")
        return
    
    print()
    
    # Check for DBN file in common locations
    possible_paths = [
        "/tmp/es_processing/es_data.dbn.zst",
        "data/raw/es_data.dbn.zst", 
        "es_data.dbn.zst"
    ]
    
    dbn_file = None
    for path in possible_paths:
        if check_file_exists(path):
            dbn_file = path
            break
    
    if not dbn_file:
        print("❌ No DBN file found in common locations")
        print("Expected locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print()
        print("Please ensure you've downloaded the DBN file first (Step 1)")
        return
    
    # Test reading the DBN file
    if test_small_dbn_read(dbn_file):
        print()
        print("✅ DBN file appears to be valid and readable")
        print()
        print("🔧 RECOMMENDATIONS:")
        print("1. The DBN file is working - the hang might be due to file size")
        print("2. Try processing in smaller chunks or with more memory")
        print("3. Monitor system resources during conversion")
        print("4. Consider using 'screen' or 'tmux' for long-running processes")
    else:
        print()
        print("❌ DBN file has issues - this is likely the cause of hanging")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("1. Re-download the DBN file (it may be corrupted)")
        print("2. Check disk space and permissions")
        print("3. Verify the file is complete (not partially downloaded)")

if __name__ == "__main__":
    main()