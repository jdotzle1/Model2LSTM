#!/usr/bin/env python3
"""
Step 2B: Filter Raw Parquet to RTH Only

This script reads the raw Parquet file and applies RTH filtering.
Much faster and more reliable than doing it during DBN conversion.
"""
import sys
import os
import time
import psutil
from pathlib import Path
from datetime import datetime

def log_progress(message, progress_file="/tmp/es_processing/progress_2b.log"):
    """Write progress message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_msg = f"[{timestamp}] {message}"
    print(progress_msg)
    
    # Also write to progress file
    with open(progress_file, "a") as f:
        f.write(progress_msg + "\n")
        f.flush()

def filter_parquet_to_rth():
    """Filter raw Parquet file to RTH only"""
    work_dir = Path('/tmp/es_processing')
    raw_parquet = work_dir / "es_data_raw.parquet"
    rth_parquet = work_dir / "es_data_rth.parquet"
    progress_file = work_dir / "progress_2b.log"
    
    # Clear previous progress log
    if progress_file.exists():
        progress_file.unlink()
    
    log_progress("🚀 STEP 2B: FILTER TO RTH (07:30-15:00 Central Time)")
    log_progress(f"Input: {raw_parquet}")
    log_progress(f"Output: {rth_parquet}")
    log_progress("=" * 60)
    
    # Check input file exists
    if not raw_parquet.exists():
        log_progress(f"❌ Input file not found: {raw_parquet}")
        log_progress("Run Step 2A first!")
        return None
    
    # Check system resources
    memory = psutil.virtual_memory()
    file_size_mb = raw_parquet.stat().st_size / (1024**2)
    
    log_progress(f"Input file size: {file_size_mb:.1f} MB")
    log_progress(f"Available memory: {memory.available / (1024**3):.1f} GB")
    
    log_progress("")
    start_time = time.time()
    
    try:
        # Import libraries
        log_progress("📦 Importing libraries...")
        import pandas as pd
        import numpy as np
        import pytz
        from datetime import time as dt_time
        log_progress("✅ Libraries imported successfully")
        
        log_progress("📖 Loading raw Parquet file...")
        df = pd.read_parquet(raw_parquet)
        
        log_progress(f"✅ Loaded {len(df):,} rows")
        log_progress(f"   Columns: {df.columns.tolist()}")
        log_progress(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Show timezone info
        if df['timestamp'].dt.tz is None:
            log_progress("   Timezone: None (assuming UTC)")
        else:
            log_progress(f"   Timezone: {df['timestamp'].dt.tz}")
        
        log_progress("")
        log_progress("🕐 Applying RTH filter...")
        
        # Convert to Central Time
        log_progress("   Converting to Central Time...")
        central_tz = pytz.timezone('US/Central')
        
        if df['timestamp'].dt.tz is None:
            # Assume UTC if no timezone
            utc_tz = pytz.UTC
            df['timestamp'] = df['timestamp'].dt.tz_localize(utc_tz)
        
        df['timestamp'] = df['timestamp'].dt.tz_convert(central_tz)
        log_progress("   ✅ Timezone conversion complete")
        
        # Filter to RTH (07:30-15:00 Central)
        log_progress("   Applying time filter (07:30-15:00 CT)...")
        rth_start = dt_time(7, 30)
        rth_end = dt_time(15, 0)
        
        df_time = df['timestamp'].dt.time
        rth_mask = (df_time >= rth_start) & (df_time < rth_end)
        
        original_count = len(df)
        df_filtered = df[rth_mask].copy()
        filtered_count = len(df_filtered)
        
        log_progress(f"   Original rows: {original_count:,}")
        log_progress(f"   RTH rows: {filtered_count:,}")
        log_progress(f"   Filtered out: {original_count - filtered_count:,} ({(original_count - filtered_count)/original_count:.1%})")
        
        # Convert back to UTC for consistency
        log_progress("   Converting back to UTC...")
        utc_tz = pytz.UTC
        df_filtered['timestamp'] = df_filtered['timestamp'].dt.tz_convert(utc_tz)
        log_progress("   ✅ RTH filtering complete")
        
        # Save filtered data
        log_progress("")
        log_progress("💾 Saving RTH-filtered data...")
        df_filtered.to_parquet(rth_parquet, index=False)
        
        # Final stats
        output_size_mb = rth_parquet.stat().st_size / (1024**2)
        total_time = time.time() - start_time
        
        log_progress("")
        log_progress("🎉 RTH FILTERING COMPLETE!")
        log_progress(f"   Output file: {rth_parquet}")
        log_progress(f"   Output size: {output_size_mb:.1f} MB")
        log_progress(f"   Total time: {total_time:.1f} seconds")
        log_progress(f"   Size reduction: {file_size_mb/output_size_mb:.1f}x")
        log_progress("")
        log_progress("🚀 Ready for Step 3: python3 step3_sample.py")
        
        return str(rth_parquet)
        
    except KeyboardInterrupt:
        log_progress("\n⚠️  Filtering interrupted by user")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        log_progress(f"\n❌ Filtering failed after {elapsed:.1f} seconds")
        log_progress(f"Error: {e}")
        import traceback
        log_progress("Full error traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                log_progress(f"   {line}")
        return None

if __name__ == "__main__":
    result = filter_parquet_to_rth()
    if result:
        log_progress(f"\n🎉 STEP 2B COMPLETE: {result}")
    else:
        log_progress(f"\n💥 STEP 2B FAILED")
        sys.exit(1)