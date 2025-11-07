#!/usr/bin/env python3
"""
Simple DBN Converter with Progress Monitoring

A more robust version of the DBN converter that provides better progress feedback
and handles large files more gracefully.
"""

import sys
import os
import time
import psutil
from pathlib import Path

def convert_dbn_with_progress(dbn_path, output_path, rth_only=True):
    """
    Convert DBN to Parquet with progress monitoring
    """
    print(f"🚀 STARTING DBN CONVERSION")
    print(f"Input: {dbn_path}")
    print(f"Output: {output_path}")
    print(f"RTH Only: {rth_only}")
    print("=" * 60)
    
    # Check system resources before starting
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / (1024**3):.1f} GB")
    
    file_size_gb = os.path.getsize(dbn_path) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")
    
    if file_size_gb > memory.available / (1024**3) * 0.5:
        print("⚠️  WARNING: Large file relative to available memory")
        print("   This conversion may take a long time or use swap space")
    
    print()
    
    start_time = time.time()
    
    try:
        # Import here to catch import errors early
        import databento as db
        import pandas as pd
        import numpy as np
        import pytz
        from datetime import time as dt_time
        
        print("📖 Opening DBN store...")
        store = db.DBNStore.from_file(dbn_path)
        
        print("📊 Getting metadata...")
        metadata = store.metadata
        print(f"   Dataset: {metadata.dataset}")
        print(f"   Schema: {metadata.schema}")
        print(f"   Period: {metadata.start} to {metadata.end}")
        
        print("🔄 Converting to DataFrame...")
        print("   This is the slow step - please be patient...")
        
        # Monitor memory usage during conversion
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Convert to DataFrame
        df = store.to_df()
        
        conversion_time = time.time() - start_time
        current_memory = process.memory_info().rss / (1024**2)  # MB
        
        print(f"✅ Conversion complete!")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Time: {conversion_time/60:.1f} minutes")
        print(f"   Memory used: {current_memory - initial_memory:.1f} MB")
        
        if rth_only:
            print()
            print("🕐 Applying RTH filter (07:30-15:00 Central Time)...")
            df = filter_rth_only(df)
            print(f"   After RTH filtering: {len(df):,} rows")
        
        print()
        print("💾 Saving to Parquet...")
        df.to_parquet(output_path, index=False)
        
        # Check output file
        output_size_mb = os.path.getsize(output_path) / (1024**2)
        total_time = time.time() - start_time
        
        print(f"✅ CONVERSION COMPLETE!")
        print(f"   Output file: {output_path}")
        print(f"   Output size: {output_size_mb:.1f} MB")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Compression ratio: {file_size_gb*1024/output_size_mb:.1f}x")
        
        return output_path
        
    except KeyboardInterrupt:
        print("\n⚠️  Conversion interrupted by user")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Conversion failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def filter_rth_only(df):
    """Filter to Regular Trading Hours with progress feedback"""
    import pandas as pd
    import numpy as np
    import pytz
    from datetime import time as dt_time
    
    print("   Converting timestamps to Central Time...")
    
    # Ensure timestamp is datetime
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    
    # Convert to Central Time
    central_tz = pytz.timezone('US/Central')
    
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(central_tz)
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert(central_tz)
    
    print("   Applying time filter...")
    
    # Define RTH hours (07:30-15:00 Central)
    rth_start = dt_time(7, 30)
    rth_end = dt_time(15, 0)
    
    # Filter to RTH only
    df_time = df['timestamp'].dt.time
    rth_mask = (df_time >= rth_start) & (df_time < rth_end)
    
    original_count = len(df)
    df_filtered = df[rth_mask].copy()
    filtered_count = len(df_filtered)
    
    print(f"   Removed {original_count - filtered_count:,} bars outside RTH")
    print(f"   Kept {filtered_count:,} bars ({filtered_count/original_count:.1%})")
    
    # Convert back to UTC for consistency
    print("   Converting back to UTC...")
    utc_tz = pytz.UTC
    df_filtered['timestamp'] = df_filtered['timestamp'].dt.tz_convert(utc_tz)
    
    return df_filtered

def main():
    """Main conversion function"""
    # Default paths
    work_dir = Path('/tmp/es_processing')
    dbn_file = work_dir / "es_data.dbn.zst"
    parquet_file = work_dir / "es_data_rth.parquet"
    
    # Check if input file exists
    if not dbn_file.exists():
        print(f"❌ DBN file not found: {dbn_file}")
        print("Please run Step 1 (download) first!")
        return
    
    # Create output directory
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run conversion
    result = convert_dbn_with_progress(str(dbn_file), str(parquet_file), rth_only=True)
    
    if result:
        print(f"\n🎉 SUCCESS: {result}")
    else:
        print(f"\n💥 FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()