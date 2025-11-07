#!/usr/bin/env python3
"""
Step 2A: Pure DBN to Parquet Conversion (NO FILTERING)

This script ONLY converts DBN to Parquet with no modifications.
No timezone conversion, no filtering, no complex operations.
"""
import sys
import os
import time
import psutil
from pathlib import Path
from datetime import datetime

def log_progress(message, progress_file="/tmp/es_30day_processing/progress_2a.log"):
    """Write progress message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_msg = f"[{timestamp}] {message}"
    print(progress_msg)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # Also write to progress file
    with open(progress_file, "a") as f:
        f.write(progress_msg + "\n")
        f.flush()

def convert_dbn_to_raw_parquet():
    """Convert DBN file to raw Parquet with NO modifications"""
    work_dir = Path('/tmp/es_30day_processing')
    dbn_file = work_dir / "es_data_30day.dbn.zst"
    parquet_file = work_dir / "es_data_raw.parquet"  # Raw, unfiltered
    progress_file = work_dir / "progress_2a.log"
    
    # Clear previous progress log
    if progress_file.exists():
        progress_file.unlink()
    
    log_progress("🚀 STEP 2A: PURE DBN TO PARQUET CONVERSION")
    log_progress(f"Input: {dbn_file}")
    log_progress(f"Output: {parquet_file}")
    log_progress("📋 NO FILTERING - Raw conversion only")
    log_progress("=" * 60)
    
    # Check input file exists
    if not dbn_file.exists():
        log_progress(f"❌ Input file not found: {dbn_file}")
        log_progress("Run Step 1 first!")
        return None
    
    # Check system resources
    memory = psutil.virtual_memory()
    file_size_gb = dbn_file.stat().st_size / (1024**3)
    
    log_progress(f"File size: {file_size_gb:.2f} GB")
    log_progress(f"Available memory: {memory.available / (1024**3):.1f} GB")
    
    log_progress("")
    start_time = time.time()
    
    try:
        # Import libraries
        log_progress("📦 Importing databento...")
        import databento as db
        import pandas as pd
        log_progress("✅ Libraries imported successfully")
        
        log_progress("📖 Opening DBN store...")
        store = db.DBNStore.from_file(str(dbn_file))
        log_progress("✅ DBN store opened successfully")
        
        log_progress("📊 Getting metadata...")
        metadata = store.metadata
        log_progress(f"   Dataset: {metadata.dataset}")
        log_progress(f"   Schema: {metadata.schema}")
        log_progress(f"   Period: {metadata.start} to {metadata.end}")
        log_progress("✅ Metadata retrieved successfully")
        
        log_progress("")
        log_progress("🔄 Converting to DataFrame...")
        log_progress("   ⏳ THIS IS THE SLOW STEP - WILL TAKE 45-90 MINUTES")
        log_progress("   💡 Pure conversion - no filtering or timezone changes")
        log_progress("   🔍 Monitor with: tail -f /tmp/es_30day_processing/progress_2a.log")
        log_progress("")
        log_progress(f"   Started conversion at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Monitor memory during conversion
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        
        # PURE CONVERSION - With timestamp preservation
        df_start = time.time()
        df = store.to_df()
        
        # Convert nanosecond timestamps to proper datetime index
        log_progress("   🕐 Converting timestamps...")
        log_progress(f"   Original index type: {type(df.index)}")
        log_progress(f"   Original index values: {df.index[:3]}")
        
        # The timestamps are in the metadata, need to reconstruct
        start_ns = metadata.start
        end_ns = metadata.end
        total_rows = len(df)
        
        # Create timestamp range from start to end
        timestamps = pd.date_range(
            start=pd.to_datetime(start_ns, unit='ns', utc=True),
            end=pd.to_datetime(end_ns, unit='ns', utc=True),
            periods=total_rows
        )
        
        df.index = timestamps
        df.index.name = 'timestamp'
        
        log_progress(f"   New index type: {type(df.index)}")
        log_progress(f"   New index values: {df.index[:3]}")
        
        df_time = time.time() - df_start
        
        current_memory = process.memory_info().rss / (1024**2)
        
        log_progress("")
        log_progress("✅ DataFrame conversion COMPLETE!")
        log_progress(f"   Rows: {len(df):,}")
        log_progress(f"   Columns: {df.columns.tolist()}")
        log_progress(f"   Conversion time: {df_time/60:.1f} minutes")
        log_progress(f"   Memory used: {current_memory - initial_memory:.1f} MB")
        
        # Show sample of data
        log_progress("")
        log_progress("📋 Sample data (first 3 rows):")
        for i, row in df.head(3).iterrows():
            log_progress(f"   Row {i}: {row.to_dict()}")
        
        # Save to Parquet - WITH TIMESTAMPS
        log_progress("")
        log_progress("💾 Saving raw data to Parquet...")
        log_progress("   Including timestamp index in Parquet file...")
        df.to_parquet(parquet_file, index=True)  # Save WITH index (timestamps)
        
        # Final stats
        output_size_mb = parquet_file.stat().st_size / (1024**2)
        total_time = time.time() - start_time
        
        log_progress("")
        log_progress("🎉 RAW CONVERSION COMPLETE!")
        log_progress(f"   Output file: {parquet_file}")
        log_progress(f"   Output size: {output_size_mb:.1f} MB")
        log_progress(f"   Total time: {total_time/60:.1f} minutes")
        log_progress(f"   Compression ratio: {file_size_gb*1024/output_size_mb:.1f}x")
        log_progress("")
        log_progress("🚀 Ready for Step 2B: python3 step2b_filter_rth.py")
        
        return str(parquet_file)
        
    except KeyboardInterrupt:
        log_progress("\n⚠️  Conversion interrupted by user")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        log_progress(f"\n❌ Conversion failed after {elapsed/60:.1f} minutes")
        log_progress(f"Error: {e}")
        import traceback
        log_progress("Full error traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                log_progress(f"   {line}")
        return None

if __name__ == "__main__":
    result = convert_dbn_to_raw_parquet()
    if result:
        log_progress(f"\n🎉 STEP 2A COMPLETE: {result}")
    else:
        log_progress(f"\n💥 STEP 2A FAILED")
        sys.exit(1)