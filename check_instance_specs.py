#!/usr/bin/env python3
"""
Check current EC2 instance specifications and recommend setup for full dataset processing
"""
import psutil
import boto3
import subprocess
import json

def check_current_instance():
    """Check current instance specifications"""
    print("🔍 CURRENT INSTANCE SPECIFICATIONS")
    print("=" * 50)
    
    # Memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    print(f"💾 Memory: {memory_gb:.1f} GB total, {available_gb:.1f} GB available")
    
    # CPU
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        print(f"🖥️  CPU: {cpu_count} cores @ {cpu_freq.current:.0f} MHz")
    else:
        print(f"🖥️  CPU: {cpu_count} cores")
    
    # Disk
    disk = psutil.disk_usage('/')
    disk_gb = disk.total / (1024**3)
    free_gb = disk.free / (1024**3)
    print(f"💽 Disk: {disk_gb:.1f} GB total, {free_gb:.1f} GB free")
    
    # Try to get EC2 instance type
    try:
        result = subprocess.run(['curl', '-s', 'http://169.254.169.254/latest/meta-data/instance-type'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            instance_type = result.stdout.strip()
            print(f"☁️  Instance Type: {instance_type}")
        else:
            print("☁️  Instance Type: Unknown (not on EC2 or metadata unavailable)")
    except:
        print("☁️  Instance Type: Unknown")
    
    return {
        'memory_gb': memory_gb,
        'available_gb': available_gb,
        'cpu_count': cpu_count,
        'disk_free_gb': free_gb
    }

def estimate_full_dataset_requirements():
    """Estimate requirements for full 15-year dataset"""
    print("\n📊 FULL DATASET REQUIREMENTS ESTIMATE")
    print("=" * 50)
    
    # Based on 30-day sample: 295K rows, 35MB processed
    sample_rows = 295_939
    sample_size_mb = 35.2
    sample_days = 30
    
    # Estimate for 15 years
    full_years = 15
    full_days = full_years * 365
    
    estimated_rows = (sample_rows / sample_days) * full_days
    estimated_size_gb = (sample_size_mb / sample_days) * full_days / 1024
    
    print(f"📅 Time period: {full_years} years ({full_days} days)")
    print(f"📊 Estimated rows: {estimated_rows:,.0f}")
    print(f"📦 Estimated processed size: {estimated_size_gb:.1f} GB")
    
    # Memory requirements (need 3-4x data size for processing)
    memory_needed = estimated_size_gb * 4
    print(f"💾 Memory needed: {memory_needed:.0f} GB (4x data size)")
    
    # Disk requirements (raw + processed + temp)
    disk_needed = estimated_size_gb * 6  # Raw DBN + processed + temp files
    print(f"💽 Disk needed: {disk_needed:.0f} GB")
    
    # Processing time estimate
    sample_processing_time = 399.5  # seconds from our test
    estimated_processing_hours = (sample_processing_time / sample_rows) * estimated_rows / 3600
    print(f"⏱️  Estimated processing time: {estimated_processing_hours:.1f} hours")
    
    return {
        'estimated_rows': estimated_rows,
        'estimated_size_gb': estimated_size_gb,
        'memory_needed_gb': memory_needed,
        'disk_needed_gb': disk_needed,
        'processing_hours': estimated_processing_hours
    }

def recommend_instance_upgrade(current_specs, requirements):
    """Recommend EC2 instance upgrade if needed"""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    memory_ok = current_specs['memory_gb'] >= requirements['memory_needed_gb']
    disk_ok = current_specs['disk_free_gb'] >= requirements['disk_needed_gb']
    cpu_ok = current_specs['cpu_count'] >= 8  # Minimum for good performance
    
    print(f"Memory: {'✅' if memory_ok else '❌'} {current_specs['memory_gb']:.0f} GB available vs {requirements['memory_needed_gb']:.0f} GB needed")
    print(f"Disk:   {'✅' if disk_ok else '❌'} {current_specs['disk_free_gb']:.0f} GB free vs {requirements['disk_needed_gb']:.0f} GB needed")
    print(f"CPU:    {'✅' if cpu_ok else '❌'} {current_specs['cpu_count']} cores (recommend 8+)")
    
    if memory_ok and disk_ok and cpu_ok:
        print("\n✅ Current instance is suitable for full dataset processing!")
        print("🚀 You can proceed with the full dataset processing.")
    else:
        print("\n⚠️  Current instance may not be suitable for full dataset processing.")
        print("\n📋 Recommended EC2 instances for full dataset:")
        print("   • r5.4xlarge  (16 vCPU, 128 GB RAM) - Good balance")
        print("   • r5.8xlarge  (32 vCPU, 256 GB RAM) - Faster processing")
        print("   • r5.12xlarge (48 vCPU, 384 GB RAM) - Maximum performance")
        print("\n💡 Alternative approaches:")
        print("   1. Process in smaller chunks with current instance")
        print("   2. Use AWS Batch for distributed processing")
        print("   3. Upgrade to larger instance temporarily")

def check_s3_access():
    """Check S3 access for full dataset"""
    print("\n🔗 CHECKING S3 ACCESS")
    print("=" * 30)
    
    try:
        s3_client = boto3.client('s3')
        
        # List buckets to test access
        response = s3_client.list_buckets()
        print(f"✅ S3 access confirmed ({len(response['Buckets'])} buckets accessible)")
        
        # Check the test bucket we used
        test_bucket = "es-1-second-30-days"
        try:
            response = s3_client.list_objects_v2(Bucket=test_bucket, MaxKeys=1)
            print(f"✅ Test bucket accessible: {test_bucket}")
        except Exception as e:
            print(f"⚠️  Test bucket issue: {e}")
        
    except Exception as e:
        print(f"❌ S3 access failed: {e}")
        print("   Check AWS credentials and permissions")

def main():
    """Main function to check specifications and provide recommendations"""
    current_specs = check_current_instance()
    requirements = estimate_full_dataset_requirements()
    recommend_instance_upgrade(current_specs, requirements)
    check_s3_access()
    
    print(f"\n📋 NEXT STEPS")
    print("=" * 30)
    print("1. If current instance is suitable: Run python3 process_full_dataset.py")
    print("2. If upgrade needed: Launch larger instance and transfer code")
    print("3. Alternative: Process in smaller chunks with current setup")

if __name__ == "__main__":
    main()