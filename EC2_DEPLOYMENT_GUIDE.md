# Complete EC2 Deployment Guide - Novice Level

## Overview - What We're Going to Do

You have compressed DBN files in S3, and we need to:
1. **Connect** to your EC2 instance via Session Manager
2. **Download** your compressed data from S3
3. **Convert** DBN files to Parquet format (RTH only)
4. **Apply** weighted labeling (6 trading strategies)
5. **Add** 43 technical features
6. **Upload** the processed data back to S3 for model training

**Total Time:** 6-8 hours for full dataset, 30-60 minutes for test mode

---

## Prerequisites - What You Need

✅ **EC2 Instance**: c5.4xlarge or larger (16 CPU, 32 GB RAM recommended)  
✅ **S3 Bucket**: With your DBN files in `s3://your-bucket/raw/dbn/`  
✅ **IAM Permissions**: EC2 instance can read/write to your S3 bucket  
✅ **Session Manager**: Access to connect to your EC2 instance  

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

---

## Step 3: Get the Code

### 3.1 Clone the Repository
```bash
# Go to home directory
cd /home/ssm-user

# Clone your repository
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

### 3.2 Set Your S3 Bucket Name
```bash
# Replace 'your-actual-bucket-name' with your real bucket name
export S3_BUCKET=your-actual-bucket-name

# Make this permanent for this session
echo "export S3_BUCKET=your-actual-bucket-name" >> ~/.bashrc

# Verify it's set
echo $S3_BUCKET
```

---

## Step 4: Create the Processing Script

Since we reorganized the directory, let me create the complete processing script for you:

### 4.1 Create the S3 Processing Script
```bash
# Create the processing script
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
            
            if self.test_mode:
                dbn_files = dbn_files[:2]  # Only process 2 files in test mode
                self.log(f"TEST MODE: Processing only {len(dbn_files)} files")
            
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
                
                # Use the convert_dbn module
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
        
        self.log(f"✓ Combined dataset: {len(combined_df):,} rows")
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

# Make it executable
chmod +x process_s3_data.py
```

---

## Step 5: Test the Setup

### 5.1 Quick Test
```bash
# Test that imports work
python3 -c "from src.data_pipeline.weighted_labeling import TRADING_MODES; print('✓ Imports working')"

# Test S3 connection
python3 -c "import boto3; s3=boto3.client('s3'); print('✓ S3 connection working')"
```

### 5.2 List Your S3 Files
```bash
# Check what DBN files you have
aws s3 ls s3://$S3_BUCKET/raw/dbn/ --human-readable

# You should see files like:
# 2024-01-15.dbn.zst
# 2024-01-16.dbn.zst
# etc.
```

---

## Step 6: Run the Processing Pipeline

### 6.1 Test Mode First (RECOMMENDED)
```bash
# Process only 2 files to test everything works
python3 process_s3_data.py --bucket $S3_BUCKET --test-mode
```

**This will:**
- Download 2 DBN files from S3
- Convert them to Parquet
- Apply weighted labeling
- Add 43 features
- Upload results back to S3

**Expected time:** 30-60 minutes

### 6.2 Monitor Progress (Open Second Terminal)
While the test is running, open a second Session Manager connection and run:
```bash
# Watch the log file
tail -f /tmp/es_processing/processing.log

# Check system resources
htop
```

### 6.3 Full Processing (After Test Works)
```bash
# Process all your DBN files
python3 process_s3_data.py --bucket $S3_BUCKET
```

**Expected time:** 6-8 hours for 15 years of data

---

## Step 7: Check Your Results

### 7.1 Verify Upload
```bash
# Check that results were uploaded
aws s3 ls s3://$S3_BUCKET/processed/weighted_labeling/ --human-readable

# You should see:
# weighted_labeled_es_dataset.parquet
# processing_metadata.json
```

### 7.2 Download and Inspect (Optional)
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
```

#### 2. "Permission denied" errors
- Make sure your EC2 instance has IAM role with S3 permissions
- Check that the role can read from and write to your bucket

#### 3. "Out of memory" errors
- Your EC2 instance might be too small
- Recommended: c5.4xlarge (16 CPU, 32 GB RAM) or larger

#### 4. Python import errors
```bash
# Reinstall packages
pip3 install --user --upgrade pandas numpy xgboost scikit-learn pyarrow boto3 pytz databento psutil
```

#### 5. Processing stops or fails
```bash
# Check the log file
cat /tmp/es_processing/processing.log

# Check system resources
htop
df -h  # Check disk space
```

---

## Quick Reference Commands

```bash
# Essential commands for copy/paste:

# Set bucket name
export S3_BUCKET=your-actual-bucket-name

# Test mode (30-60 minutes)
python3 process_s3_data.py --bucket $S3_BUCKET --test-mode

# Full processing (6-8 hours)
python3 process_s3_data.py --bucket $S3_BUCKET

# Monitor progress
tail -f /tmp/es_processing/processing.log

# Check results
aws s3 ls s3://$S3_BUCKET/processed/weighted_labeling/ --human-readable

# Check system resources
htop
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