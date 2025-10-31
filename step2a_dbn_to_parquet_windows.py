#!/usr/bin/env python3
"""
Step 2A: Pure DBN to Parquet Conversion - WINDOWS VERSION

This script ONLY converts DBN to Parquet with no modifications.
No timezone conversion, no filtering, no complex operations.
"""
import sys
import os
import time
import zipfile
from pathlib import Path
from datetime import datetime

def log_progress(message, progress_file=None):
    """Write progress message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_msg = f"[{timestamp}] {message}"
    print(progress_msg)
    
    # Also write to progress file if specified
    if progress_file:
        with open(progress_file, "a") as f:
            f.write(progress_msg + "\n")
            f.flush()

def extract_dbn_from_zip(zip_path, extract_dir):
    """Extract DBN file from ZIP archive"""
    log_progress(f"📦 Extracting ZIP file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List contents
        file_list = zip_ref.namelist()
        log_progress(f"   ZIP contents: {file_list}")
        
        # Find DBN file
        dbn_files = [f for f in file_list if f.endswith('.dbn.zst')]
        if not dbn_files:
            log_progress("❌ No DBN.ZST file found in ZIP")
            return None
        
        dbn_file = dbn_files[0]
        log_progress(f"   Found DBN file: {dbn_file}")
        
        # Extract to directory
        zip_ref.extractall(extract_dir)
        extracted_path = extract_dir / dbn_file
        
        log_progress(f"✅ Extracted to: {extracted_path}")
        return extracted_path

def convert_dbn_to_raw_parquet():
    """Convert DBN file to raw Parquet with NO modifications"""
    
    # Windows paths
    zip_file = Path(r"C:\Users\jdotzler\Desktop\GLBX-20251023-G5FBXRA73N.zip")
    work_dir = Path(r"C:\Users\jdotzler\Desktop\es_processing")
    progress_file = work_dir / "progress_2a.log"
    
    # Create work directory
    work_dir.mkdir(exist_ok=True)
    
    # Clear previous progress log
    if progress_file.exists():
        progress_file.unlink()
    
    log_progress("🚀 STEP 2A: PURE DBN TO PARQUET CONVERSION (WINDOWS)", progress_file)
    log_progress(f"ZIP file: {zip_file}", progress_file)
    log_progress(f"Work directory: {work_dir}", progress_file)
    log_progress("📋 NO FILTERING - Raw conversion only", progress_file)
    log_progress("=" * 60, progress_file)
    
    # Check ZIP file exists
    if not zip_file.exists():
        log_progress(f"❌ ZIP file not found: {zip_file}", progress_file)
        return None
    
    # Check file size
    file_size_mb = zip_file.stat().st_size / (1024**2)
    log_progress(f"ZIP file size: {file_size_mb:.1f} MB", progress_file)
    
    log_progress("", progress_file)
    start_time = time.time()
    
    try:
        # Extract DBN file from ZIP
        dbn_file = extract_dbn_from_zip(zip_file, work_dir)
        if not dbn_file:
            return None
        
        # Check extracted DBN file size
        dbn_size_mb = dbn_file.stat().st_size / (1024**2)
        log_progress(f"DBN file size: {dbn_size_mb:.1f} MB", progress_file)
        
        # Import libraries
        log_progress("📦 Importing databento...", progress_file)
        import databento as db
        import pandas as pd
        log_progress("✅ Libraries imported successfully", progress_file)
        
        log_progress("📖 Opening DBN store...", progress_file)
        store = db.DBNStore.from_file(str(dbn_file))
        log_progress("✅ DBN store opened successfully", progress_file)
        
        log_progress("📊 Getting metadata...", progress_file)
        metadata = store.metadata
        log_progress(f"   Dataset: {metadata.dataset}", progress_file)
        log_progress(f"   Schema: {metadata.schema}", progress_file)
        log_progress(f"   Period: {metadata.start} to {metadata.end}", progress_file)
        log_progress("✅ Metadata retrieved successfully", progress_file)
        
        log_progress("", progress_file)
        log_progress("🔄 Converting to DataFrame...", progress_file)
        log_progress("   ⏳ This should take 5-15 minutes for 30 days of data", progress_file)
        log_progress("   💡 Pure conversion - no filtering or timezone changes", progress_file)
        log_progress("", progress_file)
        log_progress(f"   Started conversion at: {datetime.now().strftime('%H:%M:%S')}", progress_file)
        
        # PURE CONVERSION - No modifications
        df_start = time.time()
        df = store.to_df()
        df_time = time.time() - df_start
        
        log_progress("", progress_file)
        log_progress("✅ DataFrame conversion COMPLETE!", progress_file)
        log_progress(f"   Rows: {len(df):,}", progress_file)
        log_progress(f"   Columns: {df.columns.tolist()}", progress_file)
        log_progress(f"   Conversion time: {df_time/60:.1f} minutes", progress_file)
        
        # Show sample of data
        log_progress("", progress_file)
        log_progress("📋 Sample data (first 3 rows):", progress_file)
        for i, row in df.head(3).iterrows():
            log_progress(f"   Row {i}: {row.to_dict()}", progress_file)
        
        # Save to Parquet - NO MODIFICATIONS
        parquet_file = work_dir / "es_data_raw.parquet"
        log_progress("", progress_file)
        log_progress("💾 Saving raw data to Parquet...", progress_file)
        df.to_parquet(parquet_file, index=False)
        
        # Final stats
        output_size_mb = parquet_file.stat().st_size / (1024**2)
        total_time = time.time() - start_time
        
        log_progress("", progress_file)
        log_progress("🎉 RAW CONVERSION COMPLETE!", progress_file)
        log_progress(f"   Output file: {parquet_file}", progress_file)
        log_progress(f"   Output size: {output_size_mb:.1f} MB", progress_file)
        log_progress(f"   Total time: {total_time/60:.1f} minutes", progress_file)
        log_progress(f"   Compression ratio: {dbn_size_mb/output_size_mb:.1f}x", progress_file)
        log_progress("", progress_file)
        log_progress("🚀 Ready for Step 2B: python step2b_filter_rth_windows.py", progress_file)
        
        return str(parquet_file)
        
    except KeyboardInterrupt:
        log_progress("\n⚠️  Conversion interrupted by user", progress_file)
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        log_progress(f"\n❌ Conversion failed after {elapsed/60:.1f} minutes", progress_file)
        log_progress(f"Error: {e}", progress_file)
        import traceback
        log_progress("Full error traceback:", progress_file)
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                log_progress(f"   {line}", progress_file)
        return None

if __name__ == "__main__":
    result = convert_dbn_to_raw_parquet()
    if result:
        print(f"\n🎉 STEP 2A COMPLETE: {result}")
    else:
        print(f"\n💥 STEP 2A FAILED")
        sys.exit(1)