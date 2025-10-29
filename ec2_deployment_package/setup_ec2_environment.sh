#!/bin/bash
# EC2 Environment Setup for Weighted Labeling Pipeline

set -e

echo "=== SETTING UP EC2 ENVIRONMENT ==="

# Update system
echo "Updating system packages..."
sudo yum update -y
sudo yum install -y python3 python3-pip git htop tmux

# Install Python packages
echo "Installing Python packages..."
pip3 install --user --upgrade pip
pip3 install --user pandas numpy xgboost scikit-learn pyarrow boto3 pytz databento psutil joblib

# Set up AWS CLI (if not already configured)
if ! aws sts get-caller-identity &>/dev/null; then
    echo "⚠️  AWS CLI not configured. Please run 'aws configure' manually."
fi

# Create working directories
mkdir -p /tmp/es_weighted_pipeline/{data,models,logs}

# Set environment variables
export PYTHONPATH="/home/ec2-user/weighted_labeling_pipeline:$PYTHONPATH"

echo "✓ Environment setup complete"
echo "Next: Run ./validate_ec2_setup.sh to test the installation"
