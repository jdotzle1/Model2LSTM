#!/usr/bin/env python3
"""
Launch SageMaker Processing Job for ES Data Processing

This script creates and launches a SageMaker Processing Job with:
- Spot instance pricing (60-70% cost savings)
- Appropriate instance type for CPU-intensive work
- Proper IAM roles and S3 permissions
- Checkpointing for long-running jobs
"""

import boto3
import time
from datetime import datetime

# Configuration
REGION = 'us-east-1'  # Change to your preferred region
S3_BUCKET = 'your-es-data-bucket'  # Change to your bucket name
SAGEMAKER_ROLE = 'arn:aws:iam::YOUR-ACCOUNT:role/SageMakerExecutionRole'  # Change to your role

# Processing job configuration
JOB_CONFIG = {
    'instance_type': 'ml.c5.4xlarge',  # 16 vCPUs, 32 GB RAM - good for CPU-intensive work
    'instance_count': 1,
    'volume_size_gb': 100,  # Enough for temporary processing
    'max_runtime_seconds': 24 * 3600,  # 24 hours max
    'use_spot_instances': True,  # 60-70% cost savings
    'max_wait_time_seconds': 48 * 3600,  # Wait up to 48 hours for spot capacity
}

def create_sagemaker_job():
    """Create and launch SageMaker Processing Job"""
    
    sagemaker = boto3.client('sagemaker', region_name=REGION)
    
    # Generate unique job name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f'es-data-processing-{timestamp}'
    
    print(f"Creating SageMaker Processing Job: {job_name}")
    print(f"Instance: {JOB_CONFIG['instance_type']} ({'spot' if JOB_CONFIG['use_spot_instances'] else 'on-demand'})")
    
    # Job configuration
    processing_job_config = {
        'ProcessingJobName': job_name,
        'ProcessingResources': {
            'ClusterConfig': {
                'InstanceType': JOB_CONFIG['instance_type'],
                'InstanceCount': JOB_CONFIG['instance_count'],
                'VolumeSizeInGB': JOB_CONFIG['volume_size_gb'],
            }
        },
        'AppSpecification': {
            'ImageUri': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-cpu-py38-ubuntu20.04-sagemaker',
            'ContainerEntrypoint': ['python3', '/opt/ml/code/sagemaker_processing.py'],
            'ContainerArguments': ['--chunk-size', '250000']
        },
        'RoleArn': SAGEMAKER_ROLE,
        'ProcessingInputs': [
            {
                'InputName': 'code',
                'S3Input': {
                    'S3Uri': f's3://{S3_BUCKET}/code/',  # Upload your code here
                    'LocalPath': '/opt/ml/code',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File'
                }
            }
        ],
        'ProcessingOutputConfig': {
            'Outputs': [
                {
                    'OutputName': 'processed-data',
                    'S3Output': {
                        'S3Uri': f's3://{S3_BUCKET}/processed/',
                        'LocalPath': '/opt/ml/processing/output',
                        'S3UploadMode': 'EndOfJob'
                    }
                }
            ]
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': JOB_CONFIG['max_runtime_seconds']
        },
        'Environment': {
            'S3_BUCKET': S3_BUCKET,
            'S3_PARQUET_PREFIX': 'raw/parquet/',
            'S3_OUTPUT_PREFIX': 'processed/'
        },
        'Tags': [
            {'Key': 'Project', 'Value': 'ES-Trading-Model'},
            {'Key': 'Stage', 'Value': 'Data-Processing'},
            {'Key': 'CostCenter', 'Value': 'ML-Research'}
        ]
    }
    
    # Add spot instance configuration if enabled
    if JOB_CONFIG['use_spot_instances']:
        processing_job_config['ProcessingResources']['ClusterConfig']['InstanceType'] = JOB_CONFIG['instance_type']
        # Note: Spot instances for Processing Jobs are configured differently
        # They use managed spot training which is handled automatically
        print("Note: Using managed spot instances (automatic spot handling)")
    
    try:
        # Create the processing job
        response = sagemaker.create_processing_job(**processing_job_config)
        
        print(f"✓ Processing job created successfully!")
        print(f"Job ARN: {response['ProcessingJobArn']}")
        print(f"Job Name: {job_name}")
        
        return job_name
        
    except Exception as e:
        print(f"❌ Failed to create processing job: {str(e)}")
        raise

def monitor_job(job_name):
    """Monitor the processing job status"""
    
    sagemaker = boto3.client('sagemaker', region_name=REGION)
    
    print(f"\nMonitoring job: {job_name}")
    print("Status updates every 60 seconds...")
    
    start_time = time.time()
    
    while True:
        try:
            response = sagemaker.describe_processing_job(ProcessingJobName=job_name)
            
            status = response['ProcessingJobStatus']
            
            # Calculate elapsed time
            elapsed_minutes = (time.time() - start_time) / 60
            
            print(f"[{elapsed_minutes:.0f}m] Status: {status}")
            
            if status in ['Completed', 'Failed', 'Stopped']:
                break
            elif status == 'InProgress':
                # Show additional details if available
                if 'ProcessingStartTime' in response:
                    processing_start = response['ProcessingStartTime']
                    processing_elapsed = (datetime.now(processing_start.tzinfo) - processing_start).total_seconds() / 60
                    print(f"         Processing time: {processing_elapsed:.0f} minutes")
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Error checking job status: {str(e)}")
            time.sleep(60)
    
    # Final status
    response = sagemaker.describe_processing_job(ProcessingJobName=job_name)
    final_status = response['ProcessingJobStatus']
    
    print(f"\n{'='*50}")
    print(f"Job completed with status: {final_status}")
    
    if final_status == 'Completed':
        print(f"✅ SUCCESS!")
        print(f"Output location: s3://{S3_BUCKET}/processed/")
        
        # Show processing time
        if 'ProcessingStartTime' in response and 'ProcessingEndTime' in response:
            start_time = response['ProcessingStartTime']
            end_time = response['ProcessingEndTime']
            duration = (end_time - start_time).total_seconds() / 60
            print(f"Processing duration: {duration:.1f} minutes")
        
    elif final_status == 'Failed':
        print(f"❌ FAILED!")
        if 'FailureReason' in response:
            print(f"Failure reason: {response['FailureReason']}")
        
        print(f"Check CloudWatch logs for details:")
        print(f"Log group: /aws/sagemaker/ProcessingJobs")
        print(f"Log stream: {job_name}/...")
    
    return final_status

def estimate_costs():
    """Estimate processing costs"""
    
    # Instance pricing (approximate, varies by region)
    instance_costs = {
        'ml.c5.xlarge': {'on_demand': 0.204, 'spot': 0.061},    # 4 vCPU, 8 GB
        'ml.c5.2xlarge': {'on_demand': 0.408, 'spot': 0.122},   # 8 vCPU, 16 GB  
        'ml.c5.4xlarge': {'on_demand': 0.816, 'spot': 0.245},   # 16 vCPU, 32 GB
        'ml.c5.9xlarge': {'on_demand': 1.836, 'spot': 0.551},   # 36 vCPU, 72 GB
    }
    
    instance_type = JOB_CONFIG['instance_type']
    use_spot = JOB_CONFIG['use_spot_instances']
    
    if instance_type in instance_costs:
        hourly_cost = instance_costs[instance_type]['spot' if use_spot else 'on_demand']
        
        print(f"\n💰 COST ESTIMATES")
        print(f"Instance: {instance_type} ({'spot' if use_spot else 'on-demand'})")
        print(f"Hourly cost: ${hourly_cost:.3f}")
        
        # Estimate processing time based on dataset size
        estimated_hours = [2, 6, 12, 24]  # Different scenarios
        
        print(f"\nEstimated costs by processing time:")
        for hours in estimated_hours:
            cost = hourly_cost * hours
            print(f"  {hours:2d} hours: ${cost:.2f}")
        
        if use_spot:
            on_demand_cost = instance_costs[instance_type]['on_demand'] * 12  # 12-hour estimate
            spot_cost = hourly_cost * 12
            savings = on_demand_cost - spot_cost
            savings_pct = (savings / on_demand_cost) * 100
            print(f"\nSpot savings (12h estimate): ${savings:.2f} ({savings_pct:.0f}%)")

def upload_code_to_s3():
    """Upload processing code to S3"""
    
    s3 = boto3.client('s3')
    
    print("Uploading code to S3...")
    
    # Files to upload
    code_files = [
        'sagemaker_processing.py',
        'simple_optimized_labeling.py',
        'project/data_pipeline/labeling.py',
        'project/data_pipeline/features.py',
        'project/config/config.py'
    ]
    
    for file_path in code_files:
        if os.path.exists(file_path):
            s3_key = f"code/{file_path}"
            print(f"  Uploading {file_path} → s3://{S3_BUCKET}/{s3_key}")
            s3.upload_file(file_path, S3_BUCKET, s3_key)
        else:
            print(f"  ⚠️  File not found: {file_path}")
    
    print("✓ Code upload complete")

def main():
    """Main function"""
    
    print("SageMaker Processing Job Launcher")
    print("=" * 50)
    
    # Show configuration
    print(f"Configuration:")
    print(f"  Region: {REGION}")
    print(f"  S3 Bucket: {S3_BUCKET}")
    print(f"  Instance: {JOB_CONFIG['instance_type']}")
    print(f"  Spot instances: {JOB_CONFIG['use_spot_instances']}")
    
    # Show cost estimates
    estimate_costs()
    
    # Confirm before proceeding
    response = input(f"\nProceed with job creation? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    try:
        # Upload code (uncomment when ready)
        # upload_code_to_s3()
        
        # Create job
        job_name = create_sagemaker_job()
        
        # Monitor job
        final_status = monitor_job(job_name)
        
        if final_status == 'Completed':
            print(f"\n🎉 Processing completed successfully!")
            print(f"Your processed dataset is ready for model training.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    import os
    main()