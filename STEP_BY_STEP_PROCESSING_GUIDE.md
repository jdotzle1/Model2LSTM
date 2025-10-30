# Step-by-Step ES Data Processing Guide

## Overview - Why Break It Into Steps?

Instead of running one massive script for 6+ hours, we break the processing into **6 logical steps**. This allows you to:

✅ **Monitor progress** at each stage  
✅ **Debug issues** without losing all work  
✅ **Resume from failures** without starting over  
✅ **Validate results** at each step  
✅ **Save intermediate files** for inspection  

**Total time remains the same, but now you have control and visibility!**

---

## Prerequisites

✅ **EC2 Instance**: Running with Session Manager access  
✅ **S3 Bucket**: `es-1-second-data` with DBN files  
✅ **Code Repository**: Cloned to EC2 instance  
✅ **Environment**: Python packages installed

---

## 🚨 TROUBLESHOOTING: "nohup: ignoring input" Issue

If you're seeing "nohup: ignoring input" and the process seems stuck:

### Quick Diagnosis
```bash
# 1. Check what processes are running
python3 check_running_processes.py

# 2. Test your DBN file and system resources  
python3 debug_dbn_conversion.py

# 3. Check if there's a stuck process
ps aux | grep python
```

### Common Causes & Solutions

**Problem**: DBN conversion hangs during `store.to_df()`
- **Cause**: Large file (1.3GB) takes 45-90 minutes to process
- **Solution**: Use `screen` session instead of `nohup`, monitor with diagnostic tools

**Problem**: "nohup: ignoring input" message
- **Cause**: Process is running in background but may be waiting for input or stuck
- **Solution**: Kill the process and restart with better monitoring

**Problem**: Out of memory during conversion
- **Cause**: 1.3GB compressed file expands to ~4-6GB in memory
- **Solution**: Ensure EC2 instance has at least 8GB RAM

### Recommended Approach
1. **Kill any stuck processes**: `kill -9 <PID>`
2. **Use screen instead of nohup**: `screen -S conversion`
3. **Run diagnostic tools first**: Check system resources and file validity
4. **Monitor progress**: Use the improved Step 2 script with progress feedback  

---

## Step 1: Download DBN File from S3

### 1.1 Create Download Script
```bash
cat > step1_download.py << 'EOF'
#!/usr/bin/env python3
"""
Step 1: Download DBN file from S3
"""
import boto3
import os
from pathlib import Path
import time

def download_dbn_file():
    """Download the DBN file from S3"""
    bucket_name = "es-1-second-data"
    s3_key = "raw-data/databento/glbx-mdp3-20100606-20251021.ohlcv-1s.dbn.zst"
    
    # Create work directory
    work_dir = Path('/tmp/es_processing')
    work_dir.mkdir(exist_ok=True)
    
    local_file = work_dir / "es_data.dbn.zst"
    
    print(f"🚀 STEP 1: DOWNLOADING DBN FILE FROM S3")
    print(f"Bucket: s3://{bucket_name}")
    print(f"Key: {s3_key}")
    print(f"Local: {local_file}")
    
    start_time = time.time()
    
    try:
        s3_client = boto3.client('s3')
        
        # Get file size for progress
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = response['ContentLength']
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"File size: {file_size_mb:.1f} MB")
        print("Downloading...")
        
        # Download file
        s3_client.download_file(bucket_name, s3_key, str(local_file))
        
        # Verify download
        if local_file.exists():
            actual_size = local_file.stat().st_size
            if actual_size == file_size:
                elapsed = time.time() - start_time
                print(f"✅ Download complete!")
                print(f"Time: {elapsed:.1f} seconds")
                print(f"File: {local_file}")
                print(f"Size: {actual_size:,} bytes")
                return str(local_file)
            else:
                print(f"❌ Size mismatch: expected {file_size}, got {actual_size}")
                return None
        else:
            print(f"❌ File not found after download")
            return None
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

if __name__ == "__main__":
    result = download_dbn_file()
    if result:
        print(f"\n🎉 STEP 1 COMPLETE: {result}")
    else:
        print(f"\n💥 STEP 1 FAILED")
        exit(1)
EOF

chmod +x step1_download.py
```

### 1.2 Run Download
```bash
# Run Step 1 (should take 2-5 minutes)
python3 step1_download.py
```

**Expected output:**
```
🚀 STEP 1: DOWNLOADING DBN FILE FROM S3
File size: 1300.5 MB
Downloading...
✅ Download complete!
🎉 STEP 1 COMPLETE: /tmp/es_processing/es_data.dbn.zst
```

---

## Step 2: Convert DBN to Parquet with RTH Filtering

### 2.1 First - Diagnose Any Issues
```bash
# Check if there are any hanging processes from previous attempts
python3 check_running_processes.py

# Test the DBN file and system resources
python3 debug_dbn_conversion.py
```

### 2.2 Create Robust Conversion Script
```bash
cat > step2_convert.py << 'EOF'
#!/usr/bin/env python3
"""
Step 2: Convert DBN to Parquet with RTH filtering - ROBUST VERSION
"""
import sys
import os
import time
import psutil
from pathlib import Path

def convert_dbn_to_parquet():
    """Convert DBN file to Parquet with RTH filtering and progress monitoring"""
    work_dir = Path('/tmp/es_processing')
    dbn_file = work_dir / "es_data.dbn.zst"
    parquet_file = work_dir / "es_data_rth.parquet"
    
    print(f"🚀 STEP 2: CONVERTING DBN TO PARQUET (RTH ONLY)")
    print(f"Input: {dbn_file}")
    print(f"Output: {parquet_file}")
    print("=" * 60)
    
    # Check input file exists
    if not dbn_file.exists():
        print(f"❌ Input file not found: {dbn_file}")
        print("Run Step 1 first!")
        return None
    
    # Check system resources
    memory = psutil.virtual_memory()
    file_size_gb = dbn_file.stat().st_size / (1024**3)
    
    print(f"File size: {file_size_gb:.2f} GB")
    print(f"Available memory: {memory.available / (1024**3):.1f} GB")
    
    if file_size_gb > memory.available / (1024**3) * 0.5:
        print("⚠️  WARNING: Large file relative to available memory")
        print("   This conversion may take 45-90 minutes and use significant resources")
        print("   Consider running in 'screen' or 'tmux' session")
    
    print()
    start_time = time.time()
    
    try:
        # Import libraries
        import databento as db
        import pandas as pd
        import numpy as np
        import pytz
        from datetime import time as dt_time
        
        print("📖 Opening DBN store...")
        store = db.DBNStore.from_file(str(dbn_file))
        
        print("📊 Getting metadata...")
        metadata = store.metadata
        print(f"   Dataset: {metadata.dataset}")
        print(f"   Period: {metadata.start} to {metadata.end}")
        
        print("🔄 Converting to DataFrame...")
        print("   ⏳ This is the slow step - may take 45-90 minutes for large files")
        print("   💡 The process is working even if it seems stuck")
        
        # Monitor memory during conversion
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        
        # Convert to DataFrame
        df = store.to_df()
        
        conversion_time = time.time() - start_time
        current_memory = process.memory_info().rss / (1024**2)
        
        print(f"✅ DataFrame conversion complete!")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Time: {conversion_time/60:.1f} minutes")
        print(f"   Memory used: {current_memory - initial_memory:.1f} MB")
        
        # Apply RTH filtering
        print()
        print("🕐 Applying RTH filter (07:30-15:00 Central Time)...")
        
        # Convert to Central Time
        central_tz = pytz.timezone('US/Central')
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(central_tz)
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert(central_tz)
        
        # Filter to RTH
        rth_start = dt_time(7, 30)
        rth_end = dt_time(15, 0)
        df_time = df['timestamp'].dt.time
        rth_mask = (df_time >= rth_start) & (df_time < rth_end)
        
        original_count = len(df)
        df_filtered = df[rth_mask].copy()
        
        print(f"   Original rows: {original_count:,}")
        print(f"   RTH rows: {len(df_filtered):,}")
        print(f"   Filtered out: {original_count - len(df_filtered):,} ({(original_count - len(df_filtered))/original_count:.1%})")
        
        # Convert back to UTC
        utc_tz = pytz.UTC
        df_filtered['timestamp'] = df_filtered['timestamp'].dt.tz_convert(utc_tz)
        
        # Save to Parquet
        print()
        print("💾 Saving to Parquet...")
        df_filtered.to_parquet(parquet_file, index=False)
        
        # Final stats
        output_size_mb = parquet_file.stat().st_size / (1024**2)
        total_time = time.time() - start_time
        
        print(f"✅ CONVERSION COMPLETE!")
        print(f"   Output file: {parquet_file}")
        print(f"   Output size: {output_size_mb:.1f} MB")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Compression ratio: {file_size_gb*1024/output_size_mb:.1f}x")
        
        return str(parquet_file)
        
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

if __name__ == "__main__":
    result = convert_dbn_to_parquet()
    if result:
        print(f"\n🎉 STEP 2 COMPLETE: {result}")
    else:
        print(f"\n💥 STEP 2 FAILED")
        exit(1)
EOF

chmod +x step2_convert.py
```

### 2.3 Run Conversion (Choose One Method)

**Method A: Direct Run (Recommended for testing)**
```bash
# Run directly - you can monitor progress
python3 step2_convert.py
```

**Method B: Screen Session (Recommended for production)**
```bash
# Start a screen session (survives disconnection)
screen -S dbn_conversion
python3 step2_convert.py
# Press Ctrl+A, then D to detach
# Reconnect with: screen -r dbn_conversion
```

**Method C: Background with Logging**
```bash
# Run in background with logging
nohup python3 step2_convert.py > conversion.log 2>&1 &
# Monitor with: tail -f conversion.log
```

### 2.4 If Conversion Gets Stuck

**Check what's happening:**
```bash
# Check if process is still running
python3 check_running_processes.py

# Check the log file (if using nohup)
tail -f conversion.log

# Check system resources
top
```

**If you need to kill a stuck process:**
```bash
# Find the process ID
ps aux | grep python

# Kill it (replace XXXX with actual PID)
kill -9 XXXX
```

**Expected output:**
```
🚀 STEP 2: CONVERTING DBN TO PARQUET (RTH ONLY)
Converting DBN to Parquet...
✅ Conversion complete!
Rows: 2,847,392
Output size: 245.3 MB
🎉 STEP 2 COMPLETE: /tmp/es_processing/es_data_rth.parquet
```

---

## Step 3: Sample Data for Testing

### 3.1 Create Sampling Script
```bash
cat > step3_sample.py << 'EOF'
#!/usr/bin/env python3
"""
Step 3: Sample data for testing (100k rows)
"""
import pandas as pd
import time
from pathlib import Path

def sample_data_for_testing():
    """Sample first 100k rows for testing"""
    work_dir = Path('/tmp/es_processing')
    input_file = work_dir / "es_data_rth.parquet"
    output_file = work_dir / "es_data_sample.parquet"
    
    print(f"🚀 STEP 3: SAMPLING DATA FOR TESTING")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Check input file exists
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Run Step 2 first!")
        return None
    
    start_time = time.time()
    
    try:
        print("Loading Parquet file...")
        df = pd.read_parquet(input_file)
        
        original_rows = len(df)
        print(f"Original rows: {original_rows:,}")
        
        # Sample first 100k rows
        sample_size = min(100000, original_rows)
        df_sample = df.head(sample_size).copy()
        
        print(f"Sampling {sample_size:,} rows...")
        
        # Save sample
        df_sample.to_parquet(output_file, index=False)
        
        elapsed = time.time() - start_time
        output_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"✅ Sampling complete!")
        print(f"Sample rows: {len(df_sample):,}")
        print(f"Sample size: {output_size_mb:.1f} MB")
        print(f"Date range: {df_sample['timestamp'].min()} to {df_sample['timestamp'].max()}")
        print(f"Time: {elapsed:.1f} seconds")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = sample_data_for_testing()
    if result:
        print(f"\n🎉 STEP 3 COMPLETE: {result}")
    else:
        print(f"\n💥 STEP 3 FAILED")
        exit(1)
EOF

chmod +x step3_sample.py
```

### 3.2 Run Sampling
```bash
# Run Step 3 (should take 1-2 minutes)
python3 step3_sample.py
```

---

## Step 4: Apply Weighted Labeling

### 4.1 Create Labeling Script
```bash
cat > step4_labeling.py << 'EOF'
#!/usr/bin/env python3
"""
Step 4: Apply weighted labeling (6 trading modes)
"""
import sys
import pandas as pd
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.data_pipeline.weighted_labeling import process_weighted_labeling

def apply_weighted_labeling():
    """Apply weighted labeling to sample data"""
    work_dir = Path('/tmp/es_processing')
    input_file = work_dir / "es_data_sample.parquet"
    output_file = work_dir / "es_data_labeled.parquet"
    
    print(f"🚀 STEP 4: APPLYING WEIGHTED LABELING")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Check input file exists
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Run Step 3 first!")
        return None
    
    start_time = time.time()
    
    try:
        print("Loading sample data...")
        df = pd.read_parquet(input_file)
        
        print(f"Input rows: {len(df):,}")
        print(f"Input columns: {len(df.columns)}")
        
        print("Applying weighted labeling...")
        print("- 6 trading modes (Low/Normal/High vol × Long/Short)")
        print("- Binary labels (0=loss, 1=win)")
        print("- Quality weights (MAE-based)")
        print("- Velocity weights (speed-based)")
        print("- Time decay weights (recency-based)")
        
        # Apply weighted labeling
        df_labeled = process_weighted_labeling(df)
        
        # Check results
        label_cols = [col for col in df_labeled.columns if col.startswith('label_')]
        weight_cols = [col for col in df_labeled.columns if col.startswith('weight_')]
        
        print(f"✅ Labeling complete!")
        print(f"Output rows: {len(df_labeled):,}")
        print(f"Output columns: {len(df_labeled.columns)}")
        print(f"Added {len(label_cols)} label columns")
        print(f"Added {len(weight_cols)} weight columns")
        
        # Show win rates
        print("\nWin rates by trading mode:")
        for label_col in label_cols:
            win_rate = df_labeled[label_col].mean()
            mode_name = label_col.replace('label_', '')
            print(f"  {mode_name}: {win_rate:.1%}")
        
        # Save labeled data
        print("Saving labeled data...")
        df_labeled.to_parquet(output_file, index=False)
        
        elapsed = time.time() - start_time
        output_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"✅ Labeled data saved!")
        print(f"Output size: {output_size_mb:.1f} MB")
        print(f"Time: {elapsed/60:.1f} minutes")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Labeling failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = apply_weighted_labeling()
    if result:
        print(f"\n🎉 STEP 4 COMPLETE: {result}")
    else:
        print(f"\n💥 STEP 4 FAILED")
        exit(1)
EOF

chmod +x step4_labeling.py
```

### 4.2 Run Labeling
```bash
# Run Step 4 (may take 10-20 minutes)
python3 step4_labeling.py
```

---

## Step 5: Add Technical Features

### 5.1 Create Feature Engineering Script
```bash
cat > step5_features.py << 'EOF'
#!/usr/bin/env python3
"""
Step 5: Add technical features (43 features)
"""
import sys
import pandas as pd
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.data_pipeline.features import create_all_features

def add_technical_features():
    """Add 43 technical features to labeled data"""
    work_dir = Path('/tmp/es_processing')
    input_file = work_dir / "es_data_labeled.parquet"
    output_file = work_dir / "es_data_complete.parquet"
    
    print(f"🚀 STEP 5: ADDING TECHNICAL FEATURES")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Check input file exists
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Run Step 4 first!")
        return None
    
    start_time = time.time()
    
    try:
        print("Loading labeled data...")
        df = pd.read_parquet(input_file)
        
        print(f"Input rows: {len(df):,}")
        print(f"Input columns: {len(df.columns)}")
        
        print("Adding technical features...")
        print("- Volume features (4)")
        print("- Price context features (5)")
        print("- Consolidation features (10)")
        print("- Return features (5)")
        print("- Volatility features (6)")
        print("- Microstructure features (6)")
        print("- Time features (7)")
        
        # Add features
        df_features = create_all_features(df)
        
        # Check results
        original_cols = 6  # timestamp, open, high, low, close, volume
        label_weight_cols = len([col for col in df_features.columns 
                               if col.startswith(('label_', 'weight_'))])
        feature_cols = len(df_features.columns) - original_cols - label_weight_cols
        
        print(f"✅ Feature engineering complete!")
        print(f"Output rows: {len(df_features):,}")
        print(f"Output columns: {len(df_features.columns)}")
        print(f"Added {feature_cols} technical features")
        
        # Check for NaN values
        nan_counts = df_features.isnull().sum()
        high_nan_cols = nan_counts[nan_counts > len(df_features) * 0.5]
        
        if len(high_nan_cols) > 0:
            print(f"⚠️  Warning: {len(high_nan_cols)} columns have >50% NaN values")
            for col in high_nan_cols.head(5).index:
                pct = nan_counts[col] / len(df_features) * 100
                print(f"  {col}: {pct:.1f}% NaN")
        else:
            print("✅ NaN levels acceptable")
        
        # Save complete data
        print("Saving complete dataset...")
        df_features.to_parquet(output_file, index=False)
        
        elapsed = time.time() - start_time
        output_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"✅ Complete dataset saved!")
        print(f"Output size: {output_size_mb:.1f} MB")
        print(f"Time: {elapsed/60:.1f} minutes")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = add_technical_features()
    if result:
        print(f"\n🎉 STEP 5 COMPLETE: {result}")
    else:
        print(f"\n💥 STEP 5 FAILED")
        exit(1)
EOF

chmod +x step5_features.py
```

### 5.2 Run Feature Engineering
```bash
# Run Step 5 (may take 5-10 minutes)
python3 step5_features.py
```

---

## Step 6: Upload Results to S3

### 6.1 Create Upload Script
```bash
cat > step6_upload.py << 'EOF'
#!/usr/bin/env python3
"""
Step 6: Upload final results to S3
"""
import boto3
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

def upload_results_to_s3():
    """Upload final dataset and metadata to S3"""
    bucket_name = "es-1-second-data"
    work_dir = Path('/tmp/es_processing')
    input_file = work_dir / "es_data_complete.parquet"
    
    print(f"🚀 STEP 6: UPLOADING RESULTS TO S3")
    print(f"Bucket: s3://{bucket_name}")
    print(f"Input: {input_file}")
    
    # Check input file exists
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Run Step 5 first!")
        return None
    
    start_time = time.time()
    
    try:
        s3_client = boto3.client('s3')
        
        # Load data for metadata
        print("Loading dataset for metadata...")
        df = pd.read_parquet(input_file)
        
        file_size_mb = input_file.stat().st_size / (1024 * 1024)
        
        print(f"Dataset info:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Size: {file_size_mb:.1f} MB")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Upload main dataset
        s3_key = 'processed/weighted_labeling/weighted_labeled_es_dataset.parquet'
        print(f"Uploading dataset to s3://{bucket_name}/{s3_key}...")
        
        s3_client.upload_file(str(input_file), bucket_name, s3_key)
        print(f"✅ Dataset uploaded!")
        
        # Create and upload metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'file_size_mb': file_size_mb,
            'test_mode': True,
            'processing_steps': [
                'Download DBN from S3',
                'Convert DBN to Parquet with RTH filtering',
                'Sample 100k rows for testing',
                'Apply weighted labeling (6 modes)',
                'Add technical features (43 features)',
                'Upload results to S3'
            ],
            'column_breakdown': {
                'original': 6,
                'labels': len([col for col in df.columns if col.startswith('label_')]),
                'weights': len([col for col in df.columns if col.startswith('weight_')]),
                'features': len(df.columns) - 6 - len([col for col in df.columns if col.startswith(('label_', 'weight_'))])
            }
        }
        
        metadata_file = work_dir / 'processing_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        metadata_s3_key = 'processed/weighted_labeling/processing_metadata.json'
        s3_client.upload_file(str(metadata_file), bucket_name, metadata_s3_key)
        print(f"✅ Metadata uploaded!")
        
        elapsed = time.time() - start_time
        
        print(f"✅ Upload complete!")
        print(f"Time: {elapsed:.1f} seconds")
        print(f"Dataset: s3://{bucket_name}/{s3_key}")
        print(f"Metadata: s3://{bucket_name}/{metadata_s3_key}")
        
        return f"s3://{bucket_name}/{s3_key}"
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = upload_results_to_s3()
    if result:
        print(f"\n🎉 STEP 6 COMPLETE: {result}")
        print(f"\n🏆 ALL STEPS COMPLETE! Your data is ready for XGBoost training!")
    else:
        print(f"\n💥 STEP 6 FAILED")
        exit(1)
EOF

chmod +x step6_upload.py
```

### 6.2 Run Upload
```bash
# Run Step 6 (should take 1-2 minutes)
python3 step6_upload.py
```

---

## Complete Pipeline Runner

### Create All-in-One Script (Optional)
```bash
cat > run_all_steps.py << 'EOF'
#!/usr/bin/env python3
"""
Run all processing steps in sequence
"""
import subprocess
import sys
import time

def run_step(step_name, script_name):
    """Run a processing step"""
    print(f"\n{'='*60}")
    print(f"STARTING {step_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True, check=True)
        
        elapsed = time.time() - start_time
        print(f"\n✅ {step_name} completed in {elapsed/60:.1f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {step_name} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False

def main():
    """Run all processing steps"""
    steps = [
        ("STEP 1: DOWNLOAD", "step1_download.py"),
        ("STEP 2: CONVERT", "step2_convert.py"),
        ("STEP 3: SAMPLE", "step3_sample.py"),
        ("STEP 4: LABELING", "step4_labeling.py"),
        ("STEP 5: FEATURES", "step5_features.py"),
        ("STEP 6: UPLOAD", "step6_upload.py")
    ]
    
    overall_start = time.time()
    
    print("🚀 STARTING COMPLETE ES DATA PROCESSING PIPELINE")
    print("This will run all 6 steps in sequence...")
    
    for step_name, script_name in steps:
        success = run_step(step_name, script_name)
        if not success:
            print(f"\n💥 PIPELINE FAILED AT {step_name}")
            print("You can resume by running the individual step scripts.")
            sys.exit(1)
    
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'='*60}")
    print("🎉 COMPLETE PIPELINE SUCCESS!")
    print(f"Total time: {overall_elapsed/3600:.1f} hours")
    print("Your data is ready for XGBoost training!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
EOF

chmod +x run_all_steps.py
```

---

## Usage Guide

### **Option 1: Run Individual Steps (Test Mode - 100k rows)**
```bash
# Run each step individually for maximum control
python3 step1_download.py      # 2-5 minutes
python3 step2_convert.py       # 45-90 minutes (BOTTLENECK - processes full 1.3GB file)
python3 step3_sample.py        # 1-2 minutes
python3 step4_labeling.py      # 10-20 minutes
python3 step5_features.py      # 5-10 minutes
python3 step6_upload.py        # 1-2 minutes
```

### **Option 2: Run All Steps Together (Test Mode)**
```bash
# Run complete pipeline (1.5-2.5 hours total)
python3 run_all_steps.py
```

### **Option 3: Full Dataset Processing**
```bash
# For full dataset (all ~2.8M rows) - 6-8 hours total
# Step 2: 45-90 minutes (same DBN conversion)
# Step 4: 3-4 hours (labeling 2.8M rows)
# Step 5: 1-2 hours (features for 2.8M rows)
```

### **Option 3: Resume from Any Step**
```bash
# If Step 2 failed, you can resume from there:
python3 step3_sample.py
python3 step4_labeling.py
python3 step5_features.py
python3 step6_upload.py
```

---

## Benefits of Step-by-Step Approach

✅ **Visibility**: See exactly what each step is doing  
✅ **Control**: Stop, inspect, and resume at any point  
✅ **Debugging**: Isolate issues to specific steps  
✅ **Flexibility**: Skip or repeat individual steps  
✅ **Progress Tracking**: Know exactly how much is left  
✅ **Intermediate Files**: Inspect results at each stage  
✅ **Resume Capability**: Don't lose work if something fails  

---

## File Structure After Completion

```
/tmp/es_processing/
├── es_data.dbn.zst           # Step 1: Downloaded DBN file (1.3 GB)
├── es_data_rth.parquet       # Step 2: RTH-filtered data (~245 MB)
├── es_data_sample.parquet    # Step 3: 100k sample (~8 MB)
├── es_data_labeled.parquet   # Step 4: With weighted labels (~12 MB)
├── es_data_complete.parquet  # Step 5: With all features (~15 MB)
├── processing_metadata.json  # Step 6: Processing metadata
└── processing.log            # Overall processing log
```

---

## Next Steps - XGBoost Training

After completion, your processed data will be in S3:
- **Dataset**: `s3://es-1-second-data/processed/weighted_labeling/weighted_labeled_es_dataset.parquet`
- **Metadata**: `s3://es-1-second-data/processed/weighted_labeling/processing_metadata.json`

**Ready for:**
1. Download processed dataset
2. Train 6 XGBoost models (one per trading mode)
3. Evaluate model performance
4. Deploy for live trading

🎉 **Your ES trading data is now ready for machine learning!**