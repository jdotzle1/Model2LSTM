# SageMaker Deployment Checklist

## Pre-Deployment
- [ ] Verify pandas/numpy/pytz versions in SageMaker environment
- [ ] Test feature engineering on sample data (10K rows)
- [ ] Validate chunked processing with target chunk size
- [ ] Confirm S3 bucket access and permissions

## Deployment Configuration
- [ ] Instance type: ml.m5.xlarge or larger for 88M rows
- [ ] Chunk size: 100K-500K rows based on instance memory
- [ ] Enable CloudWatch logging for progress monitoring
- [ ] Set appropriate timeout (6-12 hours for full dataset)

## Data Pipeline
- [ ] Input: Parquet files with 39 columns (OHLCV + labels)
- [ ] Processing: Add 43 feature columns using chunked processing
- [ ] Output: Enhanced Parquet files with 82 columns total
- [ ] Validation: Verify feature ranges and column count

## Monitoring
- [ ] Track processing progress (log every 100K rows)
- [ ] Monitor memory usage and instance health
- [ ] Validate sample results against laptop processing
- [ ] Check final output file integrity

## Post-Processing
- [ ] Verify all 43 features present in output
- [ ] Validate feature value distributions
- [ ] Compare processing time against expectations
- [ ] Archive processing logs for future reference
