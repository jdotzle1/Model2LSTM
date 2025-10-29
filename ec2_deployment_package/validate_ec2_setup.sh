#!/bin/bash
# Validate EC2 Setup for Weighted Labeling Pipeline

set -e

echo "=== VALIDATING EC2 SETUP ==="

# Check Python packages
echo "Checking Python packages..."
python3 -c "
import pandas, numpy, xgboost, sklearn, pyarrow, boto3, pytz, databento, psutil, joblib
print('✓ All required packages installed')
"

# Check AWS access
echo "Checking AWS access..."
if aws sts get-caller-identity &>/dev/null; then
    echo "✓ AWS CLI configured and working"
else
    echo "❌ AWS CLI not configured"
    exit 1
fi

# Test pipeline components
echo "Testing pipeline components..."
cd /home/ec2-user/weighted_labeling_pipeline
python3 -c "
from project.data_pipeline.weighted_labeling import process_weighted_labeling
from project.data_pipeline.pipeline import process_labeling_and_features
print('✓ Pipeline imports successful')
"

# Run integration validation
echo "Running integration validation..."
python3 aws_setup/validate_ec2_integration.py --quick

echo "✅ EC2 setup validation complete"
echo "Ready to run: python3 aws_setup/ec2_weighted_labeling_pipeline.py"
