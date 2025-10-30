# Simple EC2 Deployment Instructions

## What You Need
- EC2 instance (c5.4xlarge recommended - 16 CPU, 32 GB RAM)
- S3 bucket with your DBN files
- Session Manager access to your EC2 instance

## Step-by-Step Instructions

### Step 1: Connect to Your EC2 Instance
1. Go to AWS Console → EC2 → Instances
2. Select your instance
3. Click "Connect" → "Session Manager" → "Connect"
4. You should now have a terminal in your browser

### Step 2: Install Basic Tools
```bash
# Update the system
sudo yum update -y

# Install git and python tools
sudo yum install -y git python3 python3-pip htop

# Install required Python packages
pip3 install --user pandas numpy xgboost scikit-learn pyarrow boto3 pytz databento psutil joblib
```

### Step 3: Get the Code from GitHub
```bash
# Go to home directory
cd /home/ssm-user

# Clone your repository
git clone https://github.com/jdotzle1/Model2LSTM.git

# Go into the project
cd Model2LSTM
```

### Step 4: Set Up Your S3 Bucket Name
```bash
# Replace 'your-bucket-name' with your actual S3 bucket
export S3_BUCKET=your-bucket-name

# Make this permanent (optional)
echo "export S3_BUCKET=your-bucket-name" >> ~/.bashrc
```

### Step 5: Test the Setup
```bash
# Test that everything works
python3 aws_setup/validate_ec2_integration.py --quick
```

### Step 6: Run the Pipeline

#### Option A: Test Mode (Recommended First)
```bash
# Process just 2 DBN files to test everything works
python3 aws_setup/ec2_weighted_labeling_pipeline.py --test-mode --bucket $S3_BUCKET
```
**This takes about 30-60 minutes**

#### Option B: Full Pipeline (All Your Data)
```bash
# Process all DBN files (only run this after test mode works)
python3 aws_setup/ec2_weighted_labeling_pipeline.py --bucket $S3_BUCKET
```
**This takes about 6-8 hours for 15 years of data**

### Step 7: Monitor Progress
Open a second Session Manager window and run:
```bash
# Watch the progress
tail -f /tmp/es_weighted_pipeline/pipeline.log

# Check system resources
htop
```

### Step 8: Check Results
When it's done, your results will be in S3:
```bash
# List the output files
aws s3 ls s3://$S3_BUCKET/processed/weighted_labeling/

# The main file you want is:
# weighted_labeled_es_dataset.parquet
```

## What the Pipeline Does
1. **Downloads** your DBN files from S3
2. **Converts** them to Parquet format (RTH only)
3. **Labels** the data with 6 different trading strategies
4. **Adds** 43 technical features
5. **Trains** 6 XGBoost models
6. **Uploads** everything back to S3

## Output Files You'll Get
- `weighted_labeled_es_dataset.parquet` - Your complete dataset
- `models/` folder - 6 trained XGBoost models
- `weighted_pipeline_metadata.json` - Summary of results

## If Something Goes Wrong
1. **Out of memory**: Your instance might be too small
2. **S3 permissions**: Make sure your EC2 has S3 access
3. **Python errors**: Try reinstalling packages:
   ```bash
   pip3 install --user --upgrade pandas numpy xgboost scikit-learn pyarrow boto3 pytz databento psutil joblib
   ```

## Quick Commands Reference
```bash
# Connect to EC2 via Session Manager (in AWS Console)
# Update system: sudo yum update -y
# Get code: git clone https://github.com/jdotzle1/Model2LSTM.git
# Set bucket: export S3_BUCKET=your-bucket-name
# Test run: python3 aws_setup/ec2_weighted_labeling_pipeline.py --test-mode --bucket $S3_BUCKET
# Full run: python3 aws_setup/ec2_weighted_labeling_pipeline.py --bucket $S3_BUCKET
# Monitor: tail -f /tmp/es_weighted_pipeline/pipeline.log
```
