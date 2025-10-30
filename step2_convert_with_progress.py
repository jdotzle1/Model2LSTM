#!/usr/bin/env python3
"""
Step 2: Convert DBN to Parquet with DETAILED progress tracking
"""
import sys
import os
import time
import psutil
from pathlib import Path
from datetime import datetime

def log_progress(message, progress_file="/tmp/es_processing/progress.log"):
    """Write progress message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_msg = f"[{timestamp}] {message}"
    print(progress_msg)
    
    # Also write to progress file
    with open(progress_file, "a") as f:
        f.write(progress_msg + "\n")
        f.flush()

def convert_dbn_to_parquet():
    """Convert DBN file to Parquet with detailed progress tracking"""
    work_dir = Path('/tmp/es_processing')
    dbn_file = work_dir / "es_data.dbn.zst"
    parquet_file = work_dir / "es_data_rth.parquet"
    progress_file = work_dir / "progress.log"
    
    # Clear previous progress log
    if progress_file.exists():
        progress_file.unlink()
    
    log_progress("🚀 STEP 2: CONVERTING DBN TO PARQUET (RTH ONLY)")
    log_progress(f"Input: {dbn_file}")
    log_progress(f"Output: {parquet_file}")
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
    
    if file_size_gb > memory.available / (1024**3) * 0.5:
        log_progress("⚠️  WARNING: Large file relative to available memory")
        log_progress("   This conversion may take 45-90 minutes")
    
    log_progress("")
    start_time = time.time()
    
    try:
        # Import libraries
        log_progress("📦 Importing libraries...")
        import databento as db
        import pandas as pd
        import numpy as np
        import pytz
        from datetime import time as dt_time
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
        log_progress("   💡 Process is working even if no new messages appear")
        log_progress("   🔍 Monitor with: tail -f /tmp/es_processing/progress.log")
        log_progress("   📊 Check CPU usage with: python3 check_running_processes.py")
        log_progress("")
        log_progress(f"   Started DataFrame conversion at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Monitor memory during conversion
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        
        # This is the slow step - convert to DataFrame
        df_start = time.time()
        df = store.to_df()
        df_time = time.time() - df_start
        
        current_memory = process.memory_info().rss / (1024**2)
        
        log_progress("")
        log_progress("✅ DataFrame conversion COMPLETE!")
        log_progress(f"   Rows: {len(df):,}")
        log_progress(f"   Columns: {df.columns.tolist()}")
        log_progress(f"   Conversion time: {df_time/60:.1f} minutes")
        log_progress(f"   Memory used: {current_memory - initial_memory:.1f} MB")
        
        # Apply RTH filtering
        log_progress("")
        log_progress("🕐 Applying RTH filter (07:30-15:00 Central Time)...")
        
        # Convert to Central Time
        central_tz = pytz.timezone('US/Central')
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(central_tz)
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert(central_tz)
        
        log_progress("   Timezone conversion complete")
        
        # Filter to RTH
        rth_start = dt_time(7, 30)
        rth_end = dt_time(15, 0)
        df_time = df['timestamp'].dt.time
        rth_mask = (df_time >= rth_start) & (df_time < rth_end)
        
        original_count = len(df)
        df_filtered = df[rth_mask].copy()
        
        log_progress(f"   Original rows: {original_count:,}")
        log_progress(f"   RTH rows: {len(df_filtered):,}")
        log_progress(f"   Filtered out: {original_count - len(df_filtered):,} ({(original_count - len(df_filtered))/original_count:.1%})")
        
        # Convert back to UTC
        utc_tz = pytz.UTC
        df_filtered['timestamp'] = df_filtered['timestamp'].dt.tz_convert(utc_tz)
        log_progress("✅ RTH filtering complete")
        
        # Save to Parquet
        log_progress("")
        log_progress("💾 Saving to Parquet...")
        df_filtered.to_parquet(parquet_file, index=False)
        
        # Final stats
        output_size_mb = parquet_file.stat().st_size / (1024**2)
        total_time = time.time() - start_time
        
        log_progress("")
        log_progress("🎉 CONVERSION COMPLETE!")
        log_progress(f"   Output file: {parquet_file}")
        log_progress(f"   Output size: {output_size_mb:.1f} MB")
        log_progress(f"   Total time: {total_time/60:.1f} minutes")
        log_progress(f"   Compression ratio: {file_size_gb*1024/output_size_mb:.1f}x")
        log_progress("")
        log_progress("🚀 Ready for Step 3: python3 step3_sample.py")
        
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
    result = convert_dbn_to_parquet()
    if result:
        log_progress(f"\n🎉 STEP 2 COMPLETE: {result}")
    else:
        log_progress(f"\n💥 STEP 2 FAILED")
        sys.exit(1)