#!/usr/bin/env python3
"""
EC2 Deployment Preparation Script

This script prepares all necessary files and configurations for EC2 deployment
of the weighted labeling pipeline. It validates the complete setup and creates
deployment packages.

Tasks performed:
1. Validate all pipeline components
2. Create deployment package
3. Generate EC2 configuration files
4. Test deployment scripts
5. Create monitoring and validation tools
"""

import os
import sys
import json
import shutil
import tarfile
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)


def validate_pipeline_components():
    """Validate all pipeline components are present and working"""
    print("=== VALIDATING PIPELINE COMPONENTS ===")
    
    required_files = [
        'project/data_pipeline/weighted_labeling.py',
        'project/data_pipeline/features.py',
        'project/data_pipeline/pipeline.py',
        'project/data_pipeline/validation_utils.py',
        'project/data_pipeline/performance_monitor.py',
        'aws_setup/ec2_weighted_labeling_pipeline.py',
        'aws_setup/validate_ec2_integration.py',
        'examples/basic_usage_example.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print(f"\n✓ All {len(required_files)} required files present")
    
    # Test imports
    print("\nTesting imports...")
    try:
        from src.data_pipeline.weighted_labeling import process_weighted_labeling
        from src.data_pipeline.pipeline import process_labeling_and_features
        from src.data_pipeline.validation_utils import run_comprehensive_validation
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def create_deployment_package():
    """Create deployment package for EC2"""
    print("\n=== CREATING DEPLOYMENT PACKAGE ===")
    
    # Create deployment directory
    deploy_dir = Path('ec2_deployment_package')
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    print(f"Created deployment directory: {deploy_dir}")
    
    # Copy essential files
    files_to_copy = [
        # Core pipeline
        ('project/', 'project/'),
        ('aws_setup/', 'aws_setup/'),
        ('examples/', 'examples/'),
        ('tests/validation/', 'tests/validation/'),
        
        # Documentation
        ('docs/weighted_labeling_usage_guide.md', 'docs/'),
        ('docs/weighted_labeling_api_reference.md', 'docs/'),
        ('docs/weighted_labeling_troubleshooting.md', 'docs/'),
        
        # Test and validation scripts
        ('test_final_integration_1000_bars.py', './'),
        ('validate_performance_target.py', './'),
        ('run_comprehensive_validation.py', './'),
        
        # Configuration
        ('README.md', './'),
    ]
    
    for src, dst_dir in files_to_copy:
        src_path = Path(src)
        dst_path = deploy_dir / dst_dir
        
        if src_path.is_file():
            dst_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path / src_path.name)
            print(f"  ✓ Copied file: {src} -> {dst_path / src_path.name}")
        elif src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_path, dst_path / src_path.name, dirs_exist_ok=True)
            print(f"  ✓ Copied directory: {src} -> {dst_path / src_path.name}")
        else:
            print(f"  ⚠️  Skipped missing: {src}")
    
    return deploy_dir


def create_ec2_configuration_files(deploy_dir):
    """Create EC2-specific configuration files"""
    print("\n=== CREATING EC2 CONFIGURATION FILES ===")
    
    # 1. Environment setup script
    env_setup = '''#!/bin/bash
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
'''
    
    env_setup_path = deploy_dir / 'setup_ec2_environment.sh'
    with open(env_setup_path, 'w', encoding='utf-8') as f:
        f.write(env_setup)
    os.chmod(env_setup_path, 0o755)
    print(f"  ✓ Created: {env_setup_path}")
    
    # 2. Validation script
    validation_script = '''#!/bin/bash
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
from src.data_pipeline.weighted_labeling import process_weighted_labeling
from src.data_pipeline.pipeline import process_labeling_and_features
print('✓ Pipeline imports successful')
"

# Run integration validation
echo "Running integration validation..."
python3 aws_setup/validate_ec2_integration.py --quick

echo "✅ EC2 setup validation complete"
echo "Ready to run: python3 aws_setup/ec2_weighted_labeling_pipeline.py"
'''
    
    validation_path = deploy_dir / 'validate_ec2_setup.sh'
    with open(validation_path, 'w', encoding='utf-8') as f:
        f.write(validation_script)
    os.chmod(validation_path, 0o755)
    print(f"  ✓ Created: {validation_path}")
    
    # 3. Pipeline configuration
    pipeline_config = {
        "pipeline_version": "2.0_weighted_labeling",
        "ec2_instance_type": "c5.4xlarge",
        "recommended_specs": {
            "cpu_cores": 16,
            "memory_gb": 32,
            "storage_gb": 100
        },
        "s3_configuration": {
            "bucket": "${S3_BUCKET}",
            "dbn_prefix": "raw/dbn/",
            "output_prefix": "processed/weighted_labeling/"
        },
        "processing_configuration": {
            "chunk_size": 500000,
            "timeout_seconds": 900,
            "performance_target_rows_per_minute": 167000,
            "memory_limit_gb": 8.0
        },
        "trading_modes": [
            {"name": "low_vol_long", "stop_ticks": 6, "target_ticks": 12, "direction": "long"},
            {"name": "normal_vol_long", "stop_ticks": 8, "target_ticks": 16, "direction": "long"},
            {"name": "high_vol_long", "stop_ticks": 10, "target_ticks": 20, "direction": "long"},
            {"name": "low_vol_short", "stop_ticks": 6, "target_ticks": 12, "direction": "short"},
            {"name": "normal_vol_short", "stop_ticks": 8, "target_ticks": 16, "direction": "short"},
            {"name": "high_vol_short", "stop_ticks": 10, "target_ticks": 20, "direction": "short"}
        ]
    }
    
    config_path = deploy_dir / 'pipeline_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline_config, f, indent=2)
    print(f"  ✓ Created: {config_path}")
    
    # 4. Deployment instructions
    instructions = '''# EC2 Deployment Instructions for Weighted Labeling Pipeline

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
'''
    
    instructions_path = deploy_dir / 'DEPLOYMENT_INSTRUCTIONS.md'
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    print(f"  ✓ Created: {instructions_path}")


def create_monitoring_tools(deploy_dir):
    """Create monitoring and validation tools for EC2"""
    print("\n=== CREATING MONITORING TOOLS ===")
    
    # 1. Resource monitoring script
    monitor_script = '''#!/bin/bash
# Resource Monitoring for Weighted Labeling Pipeline

echo "=== PIPELINE RESOURCE MONITORING ==="
echo "Timestamp: $(date)"
echo

# System resources
echo "System Resources:"
echo "  CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "  Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
echo

# Pipeline processes
echo "Pipeline Processes:"
pgrep -f "ec2_weighted_labeling_pipeline" | while read pid; do
    if [ -n "$pid" ]; then
        echo "  PID $pid: $(ps -p $pid -o %cpu,%mem,etime,cmd --no-headers)"
    fi
done

# Working directory size
if [ -d "/tmp/es_weighted_pipeline" ]; then
    echo "Working Directory:"
    echo "  Size: $(du -sh /tmp/es_weighted_pipeline | cut -f1)"
    echo "  Files: $(find /tmp/es_weighted_pipeline -type f | wc -l)"
fi

# Recent log entries
if [ -f "/tmp/es_weighted_pipeline/pipeline.log" ]; then
    echo
    echo "Recent Log Entries:"
    tail -5 /tmp/es_weighted_pipeline/pipeline.log | sed 's/^/  /'
fi

echo
echo "=== END MONITORING ==="
'''
    
    monitor_path = deploy_dir / 'monitor_pipeline.sh'
    with open(monitor_path, 'w', encoding='utf-8') as f:
        f.write(monitor_script)
    os.chmod(monitor_path, 0o755)
    print(f"  ✓ Created: {monitor_path}")
    
    # 2. Progress checker
    progress_checker = '''#!/usr/bin/env python3
"""
Pipeline Progress Checker

Monitors the weighted labeling pipeline progress and provides estimates.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta

def check_pipeline_progress():
    """Check current pipeline progress"""
    
    log_file = "/tmp/es_weighted_pipeline/pipeline.log"
    
    if not os.path.exists(log_file):
        print("❌ Pipeline log file not found")
        print(f"Expected: {log_file}")
        return
    
    print("=== PIPELINE PROGRESS CHECK ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Read log file
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find progress indicators
    current_step = "Unknown"
    rows_processed = 0
    total_rows = 0
    start_time = None
    
    for line in lines:
        if "STEP" in line and "===" in line:
            current_step = line.strip()
        elif "rows processed" in line.lower():
            # Extract numbers from progress lines
            words = line.split()
            for i, word in enumerate(words):
                if word.replace(',', '').isdigit():
                    rows_processed = int(word.replace(',', ''))
                    break
        elif "total rows" in line.lower() or "dataset:" in line.lower():
            words = line.split()
            for word in words:
                if word.replace(',', '').isdigit():
                    total_rows = max(total_rows, int(word.replace(',', '')))
        elif "Pipeline" in line and "complete" in line.lower():
            current_step = "✅ COMPLETE"
    
    print(f"Current Step: {current_step}")
    
    if rows_processed > 0 and total_rows > 0:
        progress_pct = (rows_processed / total_rows) * 100
        print(f"Progress: {rows_processed:,} / {total_rows:,} rows ({progress_pct:.1f}%)")
        
        # Estimate remaining time
        if progress_pct > 5:  # Only estimate after 5% progress
            # Assume linear processing rate
            elapsed_lines = [l for l in lines if "elapsed" in l.lower() or "time:" in l.lower()]
            if elapsed_lines:
                # Simple time estimation based on progress
                remaining_pct = 100 - progress_pct
                estimated_remaining = (remaining_pct / progress_pct) * 60  # Rough estimate in minutes
                print(f"Estimated remaining: {estimated_remaining:.0f} minutes")
    
    # Show recent activity
    print(f"\\nRecent Activity (last 5 lines):")
    for line in lines[-5:]:
        print(f"  {line.strip()}")
    
    # Check for errors
    error_lines = [l for l in lines if "error" in l.lower() or "failed" in l.lower()]
    if error_lines:
        print(f"\\n⚠️  Potential Issues Found:")
        for line in error_lines[-3:]:  # Show last 3 errors
            print(f"  {line.strip()}")

if __name__ == "__main__":
    check_pipeline_progress()
'''
    
    progress_path = deploy_dir / 'check_progress.py'
    with open(progress_path, 'w', encoding='utf-8') as f:
        f.write(progress_checker)
    os.chmod(progress_path, 0o755)
    print(f"  ✓ Created: {progress_path}")


def create_deployment_archive(deploy_dir):
    """Create compressed archive for deployment"""
    print("\n=== CREATING DEPLOYMENT ARCHIVE ===")
    
    archive_name = f'ec2_deployment_package_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tar.gz'
    
    with tarfile.open(archive_name, 'w:gz') as tar:
        tar.add(deploy_dir, arcname='weighted_labeling_pipeline')
    
    # Get archive size
    archive_size = os.path.getsize(archive_name) / (1024 * 1024)  # MB
    
    print(f"✓ Created deployment archive: {archive_name}")
    print(f"  Size: {archive_size:.1f} MB")
    print(f"  Contents: {deploy_dir}")
    
    # Create checksum
    import hashlib
    with open(archive_name, 'rb') as f:
        checksum = hashlib.sha256(f.read()).hexdigest()
    
    checksum_file = f'{archive_name}.sha256'
    with open(checksum_file, 'w', encoding='utf-8') as f:
        f.write(f'{checksum}  {archive_name}\n')
    
    print(f"  Checksum: {checksum_file}")
    
    return archive_name, checksum_file


def test_deployment_scripts(deploy_dir):
    """Test deployment scripts for syntax and basic functionality"""
    print("\n=== TESTING DEPLOYMENT SCRIPTS ===")
    
    scripts_to_test = [
        'setup_ec2_environment.sh',
        'validate_ec2_setup.sh',
        'monitor_pipeline.sh',
        'check_progress.py'
    ]
    
    for script in scripts_to_test:
        script_path = deploy_dir / script
        
        if script_path.exists():
            if script.endswith('.sh'):
                # Test bash syntax
                result = subprocess.run(['bash', '-n', str(script_path)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✓ {script}: Bash syntax OK")
                else:
                    print(f"  ❌ {script}: Bash syntax error")
                    print(f"    {result.stderr}")
            
            elif script.endswith('.py'):
                # Test Python syntax
                result = subprocess.run([sys.executable, '-m', 'py_compile', str(script_path)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✓ {script}: Python syntax OK")
                else:
                    print(f"  ❌ {script}: Python syntax error")
                    print(f"    {result.stderr}")
        else:
            print(f"  ⚠️  {script}: Not found")


def generate_deployment_summary():
    """Generate deployment summary and next steps"""
    print("\n=== DEPLOYMENT SUMMARY ===")
    
    summary = {
        'deployment_prepared': True,
        'timestamp': datetime.now().isoformat(),
        'pipeline_version': '2.0_weighted_labeling',
        'components_validated': True,
        'deployment_package_created': True,
        'ec2_scripts_created': True,
        'monitoring_tools_created': True,
        'next_steps': [
            '1. Upload deployment archive to EC2 instance',
            '2. Extract archive and run setup_ec2_environment.sh',
            '3. Set S3_BUCKET environment variable',
            '4. Run validate_ec2_setup.sh to test installation',
            '5. Execute ec2_weighted_labeling_pipeline.py',
            '6. Monitor progress with monitor_pipeline.sh and check_progress.py'
        ]
    }
    
    summary_file = 'deployment_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Deployment preparation complete!")
    print(f"  Summary saved to: {summary_file}")
    print(f"\\nNext Steps:")
    for step in summary['next_steps']:
        print(f"  {step}")
    
    return summary_file


def main():
    """Main deployment preparation function"""
    print("EC2 DEPLOYMENT PREPARATION")
    print("=" * 50)
    print("Preparing weighted labeling pipeline for EC2 deployment...")
    
    try:
        # Step 1: Validate pipeline components
        if not validate_pipeline_components():
            print("❌ Pipeline validation failed")
            return 1
        
        # Step 2: Create deployment package
        deploy_dir = create_deployment_package()
        
        # Step 3: Create EC2 configuration files
        create_ec2_configuration_files(deploy_dir)
        
        # Step 4: Create monitoring tools
        create_monitoring_tools(deploy_dir)
        
        # Step 5: Test deployment scripts
        test_deployment_scripts(deploy_dir)
        
        # Step 6: Create deployment archive
        archive_name, checksum_file = create_deployment_archive(deploy_dir)
        
        # Step 7: Generate summary
        summary_file = generate_deployment_summary()
        
        print(f"\\n🎉 EC2 DEPLOYMENT PREPARATION COMPLETE!")
        print(f"\\nFiles created:")
        print(f"  📦 Deployment archive: {archive_name}")
        print(f"  🔍 Checksum: {checksum_file}")
        print(f"  📋 Summary: {summary_file}")
        print(f"  📁 Package directory: {deploy_dir}")
        
        print(f"\\nReady for EC2 deployment!")
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ DEPLOYMENT PREPARATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())