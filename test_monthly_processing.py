#!/usr/bin/env python3
"""
Test monthly processing on a single month to verify everything works
"""
import sys
import os
import boto3
from pathlib import Path
import calendar

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_s3_structure():
    """Check the actual S3 structure and find monthly files"""
    print("🔍 CHECKING S3 STRUCTURE")
    print("=" * 50)
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        
        # List objects in the bucket
        print("📋 Listing files in bucket...")
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            MaxKeys=20  # Just get first 20 to see structure
        )
        
        if 'Contents' not in response:
            print("❌ No files found in bucket")
            return []
        
        print(f"📊 Found {len(response['Contents'])} files (showing first 20):")
        
        monthly_files = []
        for obj in response['Contents']:
            key = obj['Key']
            size_mb = obj['Size'] / (1024**2)
            print(f"   {key} ({size_mb:.1f} MB)")
            
            # Check if it's a monthly DBN file
            if key.endswith('.dbn.zst') and 'glbx-mdp3-' in key:
                monthly_files.append(key)
        
        print(f"\n📦 Found {len(monthly_files)} monthly DBN files")
        
        return monthly_files
        
    except Exception as e:
        print(f"❌ Failed to check S3: {e}")
        return []

def generate_test_file_info():
    """Generate file info for July 2010 (first month)"""
    year = 2010
    month = 7
    
    # Calculate first and last day of month
    first_day = 1
    last_day = calendar.monthrange(year, month)[1]
    
    # Generate date strings
    start_date = f"{year:04d}{month:02d}{first_day:02d}"
    end_date = f"{year:04d}{month:02d}{last_day:02d}"
    
    # Generate file info
    month_str = f"{year:04d}-{month:02d}"
    filename = f"glbx-mdp3-{start_date}-{end_date}.ohlcv-1s.dbn.zst"
    
    return {
        'year': year,
        'month': month,
        'month_str': month_str,
        'filename': filename,
        's3_key': f"raw-data/databento/{filename}",  # Try this path first
        'local_file': f"/tmp/monthly_test/input.dbn.zst",
        'output_file': f"/tmp/monthly_test/processed.parquet"
    }

def test_single_month_download():
    """Test downloading and processing a single month"""
    print("\n🧪 TESTING SINGLE MONTH PROCESSING")
    print("=" * 50)
    
    # Generate test file info
    file_info = generate_test_file_info()
    
    print(f"📅 Testing month: {file_info['month_str']}")
    print(f"📁 Expected filename: {file_info['filename']}")
    print(f"🔗 S3 key: {file_info['s3_key']}")
    
    # Create test directory
    test_dir = Path("/tmp/monthly_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        
        # Try to find the file in S3
        print("\n🔍 Searching for file in S3...")
        
        # Try different possible paths
        possible_paths = [
            f"raw-data/databento/{file_info['filename']}",
            f"databento/{file_info['filename']}",
            f"{file_info['filename']}",
            f"raw-data/{file_info['filename']}"
        ]
        
        found_path = None
        for path in possible_paths:
            try:
                response = s3_client.head_object(Bucket=bucket_name, Key=path)
                found_path = path
                file_size_mb = response['ContentLength'] / (1024**2)
                print(f"✅ Found file at: {path}")
                print(f"   Size: {file_size_mb:.1f} MB")
                break
            except:
                print(f"   ❌ Not found at: {path}")
        
        if not found_path:
            print("❌ Could not find the monthly file in any expected location")
            print("\n💡 Please check the S3 bucket structure and update the paths")
            return False
        
        # Update file info with correct path
        file_info['s3_key'] = found_path
        
        # Test download
        print(f"\n📥 Testing download...")
        local_file = Path(file_info['local_file'])
        
        s3_client.download_file(bucket_name, found_path, str(local_file))
        
        downloaded_size_mb = local_file.stat().st_size / (1024**2)
        print(f"✅ Download successful!")
        print(f"   Local file: {local_file}")
        print(f"   Size: {downloaded_size_mb:.1f} MB")
        
        # Test basic DBN reading
        print(f"\n📖 Testing DBN reading...")
        
        try:
            import databento as db
            
            store = db.DBNStore.from_file(str(local_file))
            metadata = store.metadata
            
            print(f"✅ DBN file is valid!")
            print(f"   Dataset: {metadata.dataset}")
            print(f"   Schema: {metadata.schema}")
            print(f"   Period: {metadata.start} to {metadata.end}")
            
            # Quick conversion test (just first 1000 rows)
            df = store.to_df()
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {df.columns.tolist()}")
            
            # Cleanup
            local_file.unlink()
            
            print(f"\n✅ SINGLE MONTH TEST SUCCESSFUL!")
            print(f"Ready to process all months with correct S3 path: {found_path.replace(file_info['filename'], '')}")
            
            return True
            
        except Exception as e:
            print(f"❌ DBN reading failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 MONTHLY PROCESSING TEST")
    print("=" * 60)
    
    # Check S3 structure
    monthly_files = check_s3_structure()
    
    # Test single month
    if test_single_month_download():
        print(f"\n🚀 READY FOR FULL MONTHLY PROCESSING!")
        print(f"Run: python3 process_monthly_chunks.py")
    else:
        print(f"\n💥 TEST FAILED - Fix issues before running full processing")

if __name__ == "__main__":
    main()