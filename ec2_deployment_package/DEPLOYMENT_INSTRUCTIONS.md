# EC2 Deployment Instructions for Weighted Labeling Pipeline

## Prerequisites

1. **EC2 Instance**: c5.4xlarge (16 vCPU, 32 GB RAM) recommended
2. **AWS CLI**: Configured with appropriate permissions
3. **S3 Bucket**: With DBN files in `raw/dbn/` prefix
4. **IAM Permissions**: S3 read/write, EC2 describe

## Deployment Steps

### 1. Upload and Extract Pipeline Code

```bash
# Upload deployment package to EC2
scp -i your-key.pem ec2_deployment_package.tar.gz ec2-user@your-instance:/home/ec2-user/

# Extract on EC2
ssh -i your-key.pem ec2-user@your-instance
cd /home/ec2-user
tar -xzf ec2_deployment_package.tar.gz
cd weighted_labeling_pipeline
```

### 2. Set Up Environment

```bash
# Run environment setup
./setup_ec2_environment.sh

# Set S3 bucket (replace with your bucket)
export S3_BUCKET=your-es-data-bucket

# Validate setup
./validate_ec2_setup.sh
```

### 3. Run Pipeline

```bash
# Test mode (first 2 DBN files only)
python3 aws_setup/ec2_weighted_labeling_pipeline.py --test-mode

# Full pipeline (all DBN files)
python3 aws_setup/ec2_weighted_labeling_pipeline.py --bucket $S3_BUCKET
```

### 4. Monitor Progress

```bash
# Monitor in real-time
tail -f /tmp/es_weighted_pipeline/pipeline.log

# Check system resources
htop

# Check S3 uploads
aws s3 ls s3://$S3_BUCKET/processed/weighted_labeling/
```

## Expected Runtime

- **Test mode**: 30-60 minutes (2 DBN files)
- **Full pipeline**: 6-8 hours (15 years of data)

## Output Files

- `weighted_labeled_es_dataset.parquet`: Complete dataset with labels, weights, and features
- `models/`: 6 trained XGBoost models (one per volatility mode)
- `weighted_pipeline_metadata.json`: Comprehensive pipeline summary

## Troubleshooting

1. **Memory issues**: Reduce chunk_size in pipeline_config.json
2. **S3 permissions**: Check IAM role has S3 access
3. **Package issues**: Re-run setup_ec2_environment.sh
4. **Performance issues**: Check instance type and available resources

## Validation

After completion, validate results:

```bash
python3 run_comprehensive_validation.py s3://$S3_BUCKET/processed/weighted_labeling/weighted_labeled_es_dataset.parquet
```
