# Complete EC2 Deployment Guide - Novice Level

## Overview - What We're Going to Do

You have compressed DBN files in S3, and we need to:
1. **Connect** to your EC2 instance via Session Manager
2. **Download** your compressed data from S3
3. **Convert** DBN files to Parquet format (RTH only: 07:30-15:00 Central Time)
4. **Apply** weighted labeling (6 trading strategies)
5. **Add** 43 technical features
6. **Upload** the processed data back to S3 for model training

**Total Time:** 6-8 hours for full dataset, 30-60 minutes for test mode

---

## Prerequisites - What You Need

✅ **EC2 Instance**: See instance selection guide below  
✅ **S3 Bucket**: With your DBN files in `s3://your-bucket/raw/dbn/`  
✅ **IAM Permissions**: EC2 instance can read/write to your S3 bucket  
✅ **Session Manager**: Access to connect to your EC2 instance  

### 💰 **EC2 Instance Selection Guide**

Based on processing requirements (DBN conversion, weighted labeling, 43 features), here are your options:

#### **🏆 Recommended: c5.4xlarge (Best Balance)**
- **Specs**: 16 vCPU, 32 GB RAM
- **Cost**: ~$0.68/hour ($5.44 for 8 hours)
- **Expected Time**: 6-8 hours for full dataset
- **Why**: Optimal CPU/memory ratio for pandas operations
- **Best For**: Most users - good performance without overspending

#### **💸 Budget Option: c5.2xlarge**
- **Specs**: 8 vCPU, 16 GB RAM  
- **Cost**: ~$0.34/hour ($4.08 for 12 hours)
- **Expected Time**: 10-12 hours for full dataset
- **Risk**: May run out of memory on very large datasets
- **Best For**: Cost-conscious users willing to wait longer

#### **⚡ Performance Option: c5.9xlarge**
- **Specs**: 36 vCPU, 72 GB RAM
- **Cost**: ~$1.53/hour ($7.65 for 5 hours)  
- **Expected Time**: 4-5 hours for full dataset
- **Why**: Faster processing, more memory headroom
- **Best For**: Users who want results quickly

#### **🚀 Maximum Performance: c5.18xlarge**
- **Specs**: 72 vCPU, 144 GB RAM
- **Cost**: ~$3.06/hour ($12.24 for 4 hours)
- **Expected Time**: 3-4 hours for full dataset
- **Overkill**: More CPU than needed, but maximum memory
- **Best For**: Users with large budgets who want fastest results

#### **📊 Cost vs Time Comparison**

| Instance | Cost/Hour | Total Cost* | Processing Time | Memory Safety |
|----------|-----------|-------------|-----------------|---------------|
| c5.2xlarge | $0.34 | $4.08 | 10-12 hours | ⚠️ Tight |
| **c5.4xlarge** | **$0.68** | **$5.44** | **6-8 hours** | **✅ Good** |
| c5.9xlarge | $1.53 | $7.65 | 4-5 hours | ✅ Excellent |
| c5.18xlarge | $3.06 | $12.24 | 3-4 hours | ✅ Overkill |

*Total cost estimate for full 15-year dataset processing

#### **🎯 Recommendation Logic**

**Choose c5.4xlarge if:**
- You want good performance at reasonable cost
- You're processing a typical 15-year ES dataset
- You can wait 6-8 hours for results

**Choose c5.2xlarge if:**
- Budget is tight
- You can wait 10-12 hours
- Your dataset isn't extremely large

**Choose c5.9xlarge if:**
- You want results in half the time
- Budget isn't a major concern
- You're running multiple experiments

#### **⚙️ Instance Setup Notes**
- **Storage**: 100 GB EBS (gp3) minimum for temporary files
- **AMI**: Amazon Linux 2 (free tier eligible)
- **Security Group**: Allow Session Manager access only
- **IAM Role**: EC2 role with S3 read/write permissions

---

## Step 1: Connect to Your EC2 Instance

### 1.1 Open AWS Console
1. Go to **AWS Console** → **EC2** → **Instances**
2. Find your EC2 instance
3. Select it (click the checkbox)

### 1.2 Connect via Session Manager
1. Click **"Connect"** button (top right)
2. Choose **"Session Manager"** tab
3. Click **"Connect"**
4. A new browser tab opens with a terminal

**You should see something like:**
```
sh-4.2$ 
```

---

## Step 2: Set Up the Environment

### 2.1 Update System and Install Tools
```bash
# Update the system (takes 2-3 minutes)
sudo yum update -y

# Install essential tools
sudo yum install -y git python3 python3-pip htop unzip

# Check Python version (should be 3.7+)
python3 --version
```

### 2.2 Install Python Packages
```bash
# Install required packages (takes 5-10 minutes)
pip3 install --user pandas numpy xgboost scikit-learn pyarrow boto3 pytz databento psutil

# Verify installation
python3 -c "import pandas; print('✓ Pandas installed')"
python3 -c "import xgboost; print('✓ XGBoost installed')"
```

### 2.3 Set Up Permissions and Directories
```bash
# Create and set permissions for working directories
sudo mkdir -p /tmp/es_processing
sudo chmod 777 /tmp/es_processing

# Ensure we can write files in our home directory
chmod 755 ~
cd ~
```

---

## Step 3: Get the Code

### 3.1 Make Repository Public (Temporarily)
**Before connecting to EC2:**
1. Go to **GitHub.com** → Your repository → **Settings**
2. Scroll down to **"Danger Zone"**
3. Click **"Change repository visibility"**
4. Select **"Make public"** → Type repository name → **Confirm**

### 3.2 Clone the Repository
```bash
# Go to home directory
cd /home/ssm-user

# Clone your now-public repository (no authentication needed)
git clone https://github.com/jdotzle1/Model2LSTM.git

# Go into the project
cd Model2LSTM

# Check that files are there
ls -la
```

**You should see:**
```
main.py
requirements.txt
src/
tests/
scripts/
```

### 3.3 Make Repository Private Again (After Cloning)
**Once you've successfully cloned the repo:**
1. Go back to **GitHub.com** → Your repository → **Settings**
2. Scroll down to **"Danger Zone"**
3. Click **"Change repository visibility"**
4. Select **"Make private"** → Type repository name → **Confirm**

**Your repo is now private again!** The cloned code on EC2 will continue to work.

### 3.4 Set Your S3 Bucket Name
```bash
# Replace 'your-actual-bucket-name' with your real bucket name
export S3_BUCKET=your-actual-bucket-name

# Make this permanent for this session
echo "export S3_BUCKET=your-actual-bucket-name" >> ~/.bashrc

# Verify it's set
echo $S3_BUCKET
```

---

## Step 4: Understanding Data Filtering

### 📊 **When RTH Filtering Occurs**

**RTH (Regular Trading Hours) filtering happens during DBN to Parquet conversion:**

- **Time Range**: 07:30-15:00 Central Time (Chicago time)
- **What's Removed**: All data outside these hours (overnight, pre-market, after-hours)
- **Why**: ES futures have different characteristics during RTH vs extended hours
- **When**: During Step 1 of processing (DBN → Parquet conversion)

**Example:**
```
Original DBN file: 1,000,000 bars (24 hours/day)
After RTH filtering: ~300,000 bars (7.5 hours/day RTH only)
```

**Time Zones:**
- **Input**: Your DBN file timestamps (likely UTC or exchange time)
- **Filter**: Convert to Central Time, keep only 07:30-15:00 CT
- **Output**: Convert back to UTC for consistency

**🔄 Day Transition Handling:**

When we remove overnight data, we create gaps between trading days. The system handles this properly:

**Rolling Calculations:**
- **Problem**: 30-bar rolling average shouldn't include bars from previous day
- **Solution**: Session boundaries prevent rolling calculations from spanning days
- **Example**: First 30 bars of each day will have NaN for 30-bar features (expected)

**Lookforward Labeling:**
- **Problem**: 15-minute lookforward shouldn't cross into next trading day
- **Solution**: Lookforward stops at session end (15:00 CT)
- **Example**: Trade at 14:50 only looks forward 10 minutes, not 15

**Session Identification:**
- Each trading day (07:30-15:00 CT) is treated as separate session
- Features reset at session boundaries
- No contamination between trading days

---

## Step 5: Create the Processing Script

Since we reorganized the directory, let me create the complete processing script for you:

### 5.1 Create the S3 Processing Script
```bash
# Create the processing script with proper permissions
cat > process_s3_data.py << 'EOF'
#!/usr/bin/env python3
"""
Complete S3 Data Processing Pipeline
"""

import argparse
import os
import sys
import boto3
import pandas as pd
import time
from pathlib import Path
import shutil
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline.weighted_labeling import process_weighted_labeling
from src.data_pipeline.features import create_all_features
from src.data_pipeline.validation_utils import run_comprehensive_validation
import src.convert_dbn as convert_dbn


class S3DataProcessor:
    def __init__(self, bucket_name, test_mode=False):
        self.bucket_name = bucket_name
        self.test_mode = test_mode
        self.s3_client = boto3.client('s3')
        self.work_dir = Path('/tmp/es_processing')
        self.work_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.log_file = self.work_dir / 'processing.log'
        
    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def list_dbn_files(self):
        """List all DBN files in S3"""
        self.log("Listing DBN files in S3...")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='raw/dbn/',
                MaxKeys=1000
            )
            
            if 'Contents' not in response:
                raise Exception(f"No files found in s3://{self.bucket_name}/raw/dbn/")
            
            dbn_files = [obj['Key'] for obj in response['Contents'] 
                        if obj['Key'].endswith('.dbn.zst')]
            
            self.log(f"Found {len(dbn_files)} DBN files to process")
            return dbn_files
            
        except Exception as e:
            self.log(f"Error listing S3 files: {e}")
            raise
    
    def download_file(self, s3_key, local_path):
        """Download a file from S3"""
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            return True
        except Exception as e:
            self.log(f"Error downloading {s3_key}: {e}")
            return False
    
    def upload_file(self, local_path, s3_key):
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            self.log(f"✓ Uploaded to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            self.log(f"Error uploading {s3_key}: {e}")
            return False
    
    def convert_dbn_to_parquet(self, dbn_files):
        """Convert DBN files to Parquet format"""
        self.log("=== STEP 1: CONVERTING DBN TO PARQUET ===")
        
        parquet_files = []
        
        for i, s3_key in enumerate(dbn_files, 1):
            self.log(f"Processing file {i}/{len(dbn_files)}: {s3_key}")
            
            # Download DBN file
            local_dbn = self.work_dir / Path(s3_key).name
            if not self.download_file(s3_key, local_dbn):
                continue
            
            try:
                # Convert to Parquet
                parquet_name = Path(s3_key).stem + '.parquet'  # Remove .dbn.zst, add .parquet
                local_parquet = self.work_dir / parquet_name
                
                # Use the convert_dbn module with session boundary handling
                df = convert_dbn.convert_dbn_file(str(local_dbn), rth_only=True)
                df.to_parquet(local_parquet, index=False)
                
                parquet_files.append(local_parquet)
                self.log(f"✓ Converted to Parquet: {len(df):,} rows")
                
                # Clean up DBN file to save space
                local_dbn.unlink()
                
            except Exception as e:
                self.log(f"Error converting {s3_key}: {e}")
                continue
        
        self.log(f"✓ Converted {len(parquet_files)} files to Parquet")
        return parquet_files
    
    def combine_parquet_files(self, parquet_files):
        """Combine all Parquet files into one dataset"""
        self.log("=== STEP 2: COMBINING PARQUET FILES ===")
        
        if not parquet_files:
            raise Exception("No Parquet files to combine")
        
        # If only one file, just load it directly
        if len(parquet_files) == 1:
            df = pd.read_parquet(parquet_files[0])
            self.log(f"✓ Loaded single file: {len(df):,} rows")
            
            # Apply test mode sampling AFTER conversion
            if self.test_mode:
                # Take first 100,000 rows for testing
                original_rows = len(df)
                df = df.head(100000).copy()
                self.log(f"TEST MODE: Sampled {len(df):,} rows from {original_rows:,} total rows")
            
            # Clean up file to save space
            parquet_files[0].unlink()
            return df
        
        # Multiple files - combine them
        dfs = []
        
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                dfs.append(df)
                self.log(f"✓ Loaded {parquet_file.name}: {len(df):,} rows")
                
                # Clean up individual file to save space
                parquet_file.unlink()
                
            except Exception as e:
                self.log(f"Error loading {parquet_file}: {e}")
                continue
        
        if not dfs:
            raise Exception("No valid Parquet files loaded")
        
        # Combine all dataframes
        self.log("Combining all data...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply test mode sampling AFTER combining
        if self.test_mode:
            original_rows = len(combined_df)
            combined_df = combined_df.head(100000).copy()
            self.log(f"TEST MODE: Sampled {len(combined_df):,} rows from {original_rows:,} total rows")
        
        self.log(f"✓ Final dataset: {len(combined_df):,} rows")
        return combined_df
    
    def apply_weighted_labeling(self, df):
        """Apply weighted labeling system"""
        self.log("=== STEP 3: APPLYING WEIGHTED LABELING ===")
        
        try:
            df_labeled = process_weighted_labeling(df)
            
            # Check results
            label_cols = [col for col in df_labeled.columns if col.startswith('label_')]
            weight_cols = [col for col in df_labeled.columns if col.startswith('weight_')]
            
            self.log(f"✓ Added {len(label_cols)} label columns")
            self.log(f"✓ Added {len(weight_cols)} weight columns")
            
            # Show win rates
            for label_col in label_cols:
                win_rate = df_labeled[label_col].mean()
                self.log(f"  {label_col}: {win_rate:.1%} win rate")
            
            return df_labeled
            
        except Exception as e:
            self.log(f"Error in weighted labeling: {e}")
            raise
    
    def add_features(self, df):
        """Add technical features"""
        self.log("=== STEP 4: ADDING TECHNICAL FEATURES ===")
        
        try:
            df_features = create_all_features(df)
            
            # Check results
            original_cols = 6  # timestamp, open, high, low, close, volume
            label_weight_cols = len([col for col in df_features.columns 
                                   if col.startswith(('label_', 'weight_'))])
            feature_cols = len(df_features.columns) - original_cols - label_weight_cols
            
            self.log(f"✓ Added {feature_cols} technical features")
            self.log(f"✓ Total columns: {len(df_features.columns)}")
            
            return df_features
            
        except Exception as e:
            self.log(f"Error adding features: {e}")
            raise
    
    def save_and_upload_results(self, df):
        """Save results and upload to S3"""
        self.log("=== STEP 5: SAVING AND UPLOADING RESULTS ===")
        
        try:
            # Save locally first
            output_file = self.work_dir / 'weighted_labeled_es_dataset.parquet'
            df.to_parquet(output_file, index=False)
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            self.log(f"✓ Saved dataset: {file_size_mb:.1f} MB")
            
            # Upload to S3
            s3_key = 'processed/weighted_labeling/weighted_labeled_es_dataset.parquet'
            if self.upload_file(output_file, s3_key):
                self.log(f"✅ Dataset uploaded to s3://{self.bucket_name}/{s3_key}")
            
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
                'test_mode': self.test_mode
            }
            
            metadata_file = self.work_dir / 'processing_metadata.json'
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            metadata_s3_key = 'processed/weighted_labeling/processing_metadata.json'
            self.upload_file(metadata_file, metadata_s3_key)
            
            return True
            
        except Exception as e:
            self.log(f"Error saving results: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete processing pipeline"""
        start_time = time.time()
        
        try:
            self.log("🚀 STARTING COMPLETE S3 DATA PROCESSING PIPELINE")
            self.log(f"Bucket: s3://{self.bucket_name}")
            self.log(f"Test mode: {self.test_mode}")
            
            # Step 1: List and download DBN files
            dbn_files = self.list_dbn_files()
            
            # Step 2: Convert to Parquet
            parquet_files = self.convert_dbn_to_parquet(dbn_files)
            
            # Step 3: Combine all Parquet files
            df = self.combine_parquet_files(parquet_files)
            
            # Step 4: Apply weighted labeling
            df = self.apply_weighted_labeling(df)
            
            # Step 5: Add technical features
            df = self.add_features(df)
            
            # Step 6: Save and upload
            upload_success = self.save_and_upload_results(df)
            
            # Summary
            elapsed_time = time.time() - start_time
            self.log("=" * 60)
            self.log("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            self.log(f"Total time: {elapsed_time/3600:.1f} hours")
            self.log(f"Final dataset: {len(df):,} rows × {len(df.columns)} columns")
            self.log(f"Upload: {'✅ Success' if upload_success else '❌ Failed'}")
            self.log("=" * 60)
            
            return True
            
        except Exception as e:
            self.log(f"❌ PIPELINE FAILED: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False


def main():
    parser = argparse.ArgumentParser(description="Complete S3 Data Processing Pipeline")
    
    parser.add_argument("--bucket", "-b", required=True, help="S3 bucket name")
    parser.add_argument("--test-mode", action="store_true", help="Process only 2 files for testing")
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = S3DataProcessor(args.bucket, args.test_mode)
    success = processor.run_complete_pipeline()
    
    if success:
        print(f"\n✅ Processing completed successfully!")
        print(f"Check your results at: s3://{args.bucket}/processed/weighted_labeling/")
    else:
        print(f"\n❌ Processing failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
EOF

# Make it executable and writable
chmod 755 process_s3_data.py

# Also ensure we can write to temp directories
sudo mkdir -p /tmp/es_processing
sudo chmod 777 /tmp/es_processing
```

---

## Step 6: Test the Setup

### 6.1 Quick Test
```bash
# Test that imports work
python3 -c "from src.data_pipeline.weighted_labeling import TRADING_MODES; print('✓ Imports working')"

# Test S3 connection
python3 -c "import boto3; s3=boto3.client('s3'); print('✓ S3 connection working')"
```

### 6.2 List Your S3 Files
```bash
# Check what DBN files you have
aws s3 ls s3://$S3_BUCKET/raw/dbn/ --human-readable

# You should see files like:
# 2024-01-15.dbn.zst
# 2024-01-16.dbn.zst
# etc.
```

---

## Step 7: Run the Processing Pipeline

### 🔄 **Background Processing Explained**

Since processing takes hours, we use `nohup` (no hang up) to run processes in the background:

- **`nohup`**: Keeps the process running even if you close Session Manager
- **`&`**: Runs the command in the background
- **`> logfile.log 2>&1`**: Saves all output to a log file
- **Process ID**: We save the process ID so you can check if it's still running

**This means you can:**
- Start the process
- Close your laptop
- Go to sleep
- Come back hours later and check results!

### 7.1 Test Mode First (RECOMMENDED)

Since you have one large DBN file, test mode will process the first 100,000 rows:

```bash
# Process first 100k rows to test everything works
nohup python3 process_s3_data.py --bucket $S3_BUCKET --test-mode > test_processing.log 2>&1 &

# Get the process ID for monitoring
echo $! > test_process.pid
echo "Test process started with PID: $(cat test_process.pid)"
```

**This will:**
- Download your large DBN file from S3
- Convert it to Parquet with RTH filtering (07:30-15:00 Central Time)
- Sample first 100,000 rows for testing
- Apply weighted labeling
- Add 43 features
- Upload test results back to S3

**Expected time:** 30-60 minutes

**Why nohup?** This runs in the background and continues even if you close Session Manager.

### 7.2 Monitor Progress
You can monitor progress even if you close Session Manager:

```bash
# Watch the main log file
tail -f test_processing.log

# Watch the detailed processing log
tail -f /tmp/es_processing/processing.log

# Check if process is still running
ps aux | grep process_s3_data.py

# Check system resources
htop

# Check process status by PID
if [ -f test_process.pid ]; then
    pid=$(cat test_process.pid)
    if ps -p $pid > /dev/null; then
        echo "Process $pid is still running"
    else
        echo "Process $pid has finished"
    fi
fi
```

### 7.3 Full Processing (After Test Works)

**IMPORTANT:** This runs in the background and continues even if you disconnect:

```bash
# Process your complete dataset in the background
nohup python3 process_s3_data.py --bucket $S3_BUCKET > full_processing.log 2>&1 &

# Get the process ID for monitoring
echo $! > full_process.pid
echo "Full processing started with PID: $(cat full_process.pid)"

# You can now safely close Session Manager - the process will continue!
```

**Expected time:** 6-8 hours for 15 years of data

### 7.4 Reconnect and Monitor Later
You can disconnect from Session Manager and reconnect later:

```bash
# Reconnect to Session Manager anytime and check progress:

# Check if process is still running
if [ -f full_process.pid ]; then
    pid=$(cat full_process.pid)
    if ps -p $pid > /dev/null; then
        echo "✅ Process $pid is still running"
        # Show recent progress
        tail -20 full_processing.log
    else
        echo "🎉 Process $pid has finished!"
        # Show final results
        tail -50 full_processing.log
    fi
fi

# Monitor live progress
tail -f full_processing.log
```

---

## Step 8: Check Your Results

### 8.1 Verify Upload
```bash
# Check that results were uploaded
aws s3 ls s3://$S3_BUCKET/processed/weighted_labeling/ --human-readable

# You should see:
# weighted_labeled_es_dataset.parquet
# processing_metadata.json
```

### 8.2 Download and Inspect (Optional)
```bash
# Download the metadata to check
aws s3 cp s3://$S3_BUCKET/processed/weighted_labeling/processing_metadata.json .

# Look at the metadata
cat processing_metadata.json
```

---

## What You Get - Final Output

After successful processing, you'll have in S3:

### Main Dataset
**File:** `s3://your-bucket/processed/weighted_labeling/weighted_labeled_es_dataset.parquet`

**Contains:**
- **6 original columns:** timestamp, open, high, low, close, volume
- **12 labeling columns:** 6 labels + 6 weights for different trading strategies
- **43 feature columns:** Technical indicators and market features
- **Total:** 61 columns ready for XGBoost model training

### Metadata File
**File:** `s3://your-bucket/processed/weighted_labeling/processing_metadata.json`

**Contains:**
- Processing date and time
- Total rows and columns
- Date range of data
- File size information

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "No files found in S3"
```bash
# Check your bucket name and file structure
aws s3 ls s3://$S3_BUCKET/
aws s3 ls s3://$S3_BUCKET/raw/
aws s3 ls s3://$S3_BUCKET/raw/dbn/

# Your file should end with .dbn.zst
# Example: 15_years_es_data.dbn.zst
```

#### 2. "Permission denied" errors
- Make sure your EC2 instance has IAM role with S3 permissions
- Check that the role can read from and write to your bucket

#### 3. "Out of memory" errors
- Your EC2 instance might be too small
- **Solution**: Upgrade to larger instance:
  - c5.2xlarge → c5.4xlarge (double the RAM)
  - c5.4xlarge → c5.9xlarge (more memory headroom)
- **Quick fix**: Stop instance, change instance type, restart

#### 4. Python import errors
```bash
# Reinstall packages
pip3 install --user --upgrade pandas numpy xgboost scikit-learn pyarrow boto3 pytz databento psutil
```

#### 5. Processing stops or fails
```bash
# Check the main log file
tail -50 full_processing.log  # or test_processing.log

# Check the detailed processing log
cat /tmp/es_processing/processing.log

# Check if process is still running
ps aux | grep process_s3_data.py

# Check system resources
htop
df -h  # Check disk space

# Check process by PID
if [ -f full_process.pid ]; then
    pid=$(cat full_process.pid)
    ps -p $pid
fi
```

#### 6. "Permission denied" when writing files
```bash
# Fix permissions for temp directory
sudo chmod 777 /tmp/es_processing

# Fix permissions for current directory
chmod 755 .
chmod 644 *.py *.log 2>/dev/null || true

# If still having issues, run with sudo (not recommended but works)
sudo python3 process_s3_data.py --bucket $S3_BUCKET --test-mode
```

#### 7. Questions about day transitions and gaps
**Q: Will removing overnight data affect rolling calculations?**
**A:** Yes, but this is handled properly:
- Rolling calculations respect session boundaries
- First 30 bars of each day will have NaN for 30-bar features (expected)
- No contamination between trading days

**Q: Will 15-minute lookforward cross into next day?**
**A:** No, lookforward stops at session end (15:00 CT):
- Trade at 14:50 only looks forward 10 minutes
- Trade at 14:59 only looks forward 1 minute
- This prevents overnight gaps from affecting labels

---

## Quick Reference Commands

```bash
# Essential commands for copy/paste:

# FIRST: Launch EC2 instance (recommended: c5.4xlarge)
# THEN: Make GitHub repo public temporarily (in browser)
# GitHub.com → Settings → Danger Zone → Make public

# Set bucket name
export S3_BUCKET=your-actual-bucket-name

# Set up permissions
sudo mkdir -p /tmp/es_processing && sudo chmod 777 /tmp/es_processing

# Test mode (30-60 minutes, runs in background)
nohup python3 process_s3_data.py --bucket $S3_BUCKET --test-mode > test_processing.log 2>&1 &
echo $! > test_process.pid

# Full processing (6-8 hours, runs in background)
nohup python3 process_s3_data.py --bucket $S3_BUCKET > full_processing.log 2>&1 &
echo $! > full_process.pid

# Monitor progress (can reconnect anytime)
tail -f full_processing.log
tail -f /tmp/es_processing/processing.log

# Check if process is running
ps aux | grep process_s3_data.py
ps -p $(cat full_process.pid) 2>/dev/null && echo "Running" || echo "Finished"

# Check results
aws s3 ls s3://$S3_BUCKET/processed/weighted_labeling/ --human-readable

# Check system resources
htop

# AFTER: Make GitHub repo private again (in browser)
# GitHub.com → Settings → Danger Zone → Make private
```

---

## Next Steps - Model Training

After your data is processed, you can use it to train XGBoost models:

1. **Download** the processed dataset from S3
2. **Split** into training/validation sets
3. **Train** 6 separate XGBoost models (one for each trading strategy)
4. **Evaluate** model performance
5. **Deploy** for live trading

The processed dataset is now ready for machine learning with proper labels, weights, and features! 🎉