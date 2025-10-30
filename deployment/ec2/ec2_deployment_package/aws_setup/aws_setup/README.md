# AWS Setup Guide for ES Data Processing

## Overview
This guide walks you through processing 15 years of ES futures data on AWS using:
1. **EC2** for complete pipeline: DBN → Parquet → Labeling → Features → XGBoost Training
2. **Single instance approach** for simplicity and cost control

## Prerequisites

### 1. AWS Account Setup
- AWS account with appropriate permissions
- S3 bucket created for data storage
- IAM role for SageMaker with S3 access

### 2. Data Upload
Upload your compressed DBN files to S3:
```bash
aws s3 cp your-data/ s3://your-bucket/raw/dbn/ --recursive
```

## Complete Pipeline on Single EC2 Instance

### Step 1: Launch EC2 Instance
```bash
# Launch on-demand instance for complete pipeline
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \  # Amazon Linux 2
    --instance-type c5.4xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --iam-instance-profile Name=EC2-S3-Access-Role
```

### Step 2: Setup EC2 Environment
SSH into your instance and run:
```bash
# Update system
sudo yum update -y

# Install Python 3.9+
sudo yum install python3 python3-pip git -y

# Install required packages for complete pipeline
pip3 install --user databento pandas pyarrow boto3 xgboost scikit-learn numpy

# Configure AWS credentials (if not using IAM role)
aws configure

# Clone your project
git clone https://github.com/your-username/es-trading-model.git
cd es-trading-model
```

### Step 3: Run Complete Pipeline
```bash
# Edit configuration
nano aws_setup/ec2_complete_pipeline.py

# Update these variables:
S3_BUCKET = "your-actual-bucket-name"
S3_DBN_PREFIX = "raw/dbn/"
S3_OUTPUT_PREFIX = "processed/"

# Run complete pipeline (conversion + labeling + features + training)
python3 aws_setup/ec2_complete_pipeline.py
```

**Expected output:**
```
ES Data Conversion: DBN → Parquet
==================================================
Found 15 DBN files:
  ES-2010.dbn.zst (2.1 GB)
  ES-2011.dbn.zst (2.3 GB)
  ...
Total size: 45.2 GB

[1/15] Processing ES-2010.dbn.zst
  Downloading from S3...
  Loading DBN data...
  Loaded 8,234,567 bars
  Saving as Parquet...
  ✓ Conversion successful: 8,234,567 bars
  Uploading Parquet to S3...
  ✓ Uploaded to s3://your-bucket/raw/parquet/ES-2010.parquet
...

🎉 Conversion complete!
  Successful: 15
  Failed: 0
  Parquet files available at: s3://your-bucket/raw/parquet/
```

### Step 4: Verify Conversion
```bash
# List converted files
aws s3 ls s3://your-bucket/raw/parquet/

# Check file sizes (should be smaller due to compression)
aws s3 ls s3://your-bucket/raw/parquet/ --human-readable --summarize
```

### Step 4: Monitor Progress
The complete pipeline will run automatically and provide progress updates:

```bash
# Expected pipeline stages:
# 1. DBN → Parquet conversion (1-2 hours)
# 2. Weighted labeling system (2-3 hours)  
# 3. Feature engineering (1 hour)
# 4. XGBoost model training (1-2 hours)
# Total: ~6-8 hours
```

**Expected output:**
```
SageMaker Processing Job Launcher
==================================================
Configuration:
  Region: us-east-1
  S3 Bucket: your-bucket
  Instance: ml.c5.4xlarge
  Spot instances: True

💰 COST ESTIMATES
Instance: ml.c5.4xlarge (spot)
Hourly cost: $0.245

Estimated costs by processing time:
   2 hours: $0.49
   6 hours: $1.47
  12 hours: $2.94
  24 hours: $5.88

Spot savings (12h estimate): $7.86 (70%)

Proceed with job creation? (y/n): y

✓ Processing job created successfully!
Job Name: es-data-processing-20241028-143022

Monitoring job: es-data-processing-20241028-143022
[0m] Status: InProgress
[1m] Status: InProgress
         Processing time: 1 minutes
...
[180m] Status: Completed

==================================================
Job completed with status: Completed
✅ SUCCESS!
Output location: s3://your-bucket/processed/
Processing duration: 178.5 minutes
```

### Step 4: Verify Results
```bash
# Check output files
aws s3 ls s3://your-bucket/processed/

# Download metadata to check results
aws s3 cp s3://your-bucket/processed/dataset_metadata.json .
cat dataset_metadata.json
```

## Cost Estimates

### EC2 Conversion Costs
- **Instance:** m5.xlarge spot (~$0.05/hour)
- **Storage:** 100 GB EBS (~$10/month, prorated)
- **Data transfer:** S3 download/upload (minimal cost)
- **Total estimated:** $5-15 for conversion

### Complete Pipeline Costs (EC2 On-Demand)
- **Instance:** c5.4xlarge (~$0.816/hour)
- **Estimated time:** 6-8 hours for complete pipeline
- **Compute cost:** $4.90-6.53
- **Storage (EBS):** 200 GB × $0.08/GB/month × 0.25 months = $4.00
- **Storage (S3):** 50 GB × $0.023/GB/month = $1.15
- **Data transfer:** ~$1.00

### RTH Filtering Benefits
- **Time range:** 07:30-15:00 CT (7.5 hours of 24-hour day)
- **Data reduction:** ~65% smaller datasets
- **Processing speedup:** ~3x faster due to smaller size
- **Cost reduction:** ~65% lower compute costs
- **Quality improvement:** Focus on liquid, predictable market hours

### Total Project Cost: ~$12 (simple, predictable pricing)

## Monitoring and Troubleshooting

### CloudWatch Logs
Monitor SageMaker job progress:
```bash
# View logs in AWS Console
# Navigate to: CloudWatch → Log groups → /aws/sagemaker/ProcessingJobs
# Find your job: es-data-processing-YYYYMMDD-HHMMSS
```

### Common Issues

**1. Spot Instance Interruption**
- SageMaker automatically handles spot interruptions
- Job will resume when capacity is available
- Use checkpointing (built into our script)

**2. Memory Issues**
- Reduce chunk_size parameter
- Use larger instance type (ml.c5.9xlarge)

**3. Timeout Issues**
- Increase max_runtime_seconds
- Check data size vs processing capacity

**4. S3 Permission Errors**
- Verify IAM roles have S3 access
- Check bucket policies

## Next Steps

After processing completes:

1. **Download sample for validation:**
```bash
aws s3 cp s3://your-bucket/processed/final_es_dataset.parquet sample_dataset.parquet
```

2. **Validate results locally:**
```bash
python tests/validation/validate_final_dataset.py sample_dataset.parquet
```

3. **Proceed to model training:**
- Use processed dataset for 6 XGBoost model training
- Consider SageMaker Training Jobs for model development
- Deploy trained models as ensemble for real-time inference

## Support

If you encounter issues:
1. Check CloudWatch logs for detailed error messages
2. Verify IAM permissions and S3 bucket access
3. Test with smaller dataset first
4. Consider using larger instance types for memory-intensive operations

The processed dataset will be ready for XGBoost model training with:
- ✅ 6 trading profile labels (optimal/suboptimal/loss) 
- ✅ 43 engineered features across 7 categories
- ✅ Proper data leakage prevention
- ✅ RTH-only data for consistent market conditions
- ✅ Optimized for 6 specialized XGBoost models