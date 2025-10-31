#!/usr/bin/env python3
"""
Process the full 15-year ES dataset - optimized for your current instance
68GB RAM, 36 cores, 195GB disk - perfect for this job!
"""
import sys
import os
import time
import boto3
import pandas as pd
from pathlib import Path
from datetime import datetime
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_processing_environment():
    """Set up directories and check readiness"""
    print("🚀 SETTING UP FULL DATASET PROCESSING")
    print("=" * 60)
    
    # Create processing directory
    work_dir = Path("/tmp/es_full_processing")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Work directory: {work_dir}")
    
    # Check current resources
    memory = psutil.virtual_memory()
    print(f"💾 Available memory: {memory.available / (1024**3):.1f} GB")
    print(f"🖥️  CPU cores: {psutil.cpu_count()}")
    print(f"💽 Free disk: {psutil.disk_usage('/').free / (1024**3):.1f} GB")
    
    return work_dir

def download_full_dataset_from_s3():
    """Download the full 15-year dataset"""
    print("\n📥 DOWNLOADING FULL 15-YEAR DATASET")
    print("=" * 50)
    
    # Actual S3 details for the full dataset
    bucket_name = "es-1-second-data"
    s3_key = "raw-data/databento/glbx-mdp3-20100606-20251021.ohlcv-1s.dbn.zst"
    
    local_file = Path("/tmp/es_full_processing/es_15year_data.dbn.zst")
    
    print(f"📤 S3: s3://{bucket_name}/{s3_key}")
    print(f"📥 Local: {local_file}")
    
    # Check if already downloaded
    if local_file.exists():
        size_gb = local_file.stat().st_size / (1024**3)
        print(f"✅ File already exists ({size_gb:.1f} GB)")
        return str(local_file)
    
    try:
        s3_client = boto3.client('s3')
        
        # Get file info
        print("📊 Getting file information...")
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = response['ContentLength']
        file_size_gb = file_size / (1024**3)
        
        print(f"📦 File size: {file_size_gb:.2f} GB")
        print(f"📅 Date range: 2010-06-06 to 2025-10-21 (15+ years)")
        print("⏳ Download will take 5-15 minutes...")
        
        start_time = time.time()
        
        # Download the file
        s3_client.download_file(bucket_name, s3_key, str(local_file))
        
        download_time = time.time() - start_time
        download_speed = file_size_gb / (download_time / 60)  # GB/min
        
        print(f"✅ Download complete!")
        print(f"   Time: {download_time/60:.1f} minutes")
        print(f"   Speed: {download_speed:.1f} GB/min")
        print(f"   Local file: {local_file}")
        
        return str(local_file)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("📋 Manual download command:")
        print(f"   aws s3 cp s3://{bucket_name}/{s3_key} {local_file}")
        return None

def process_full_dataset_optimized(input_file=None):
    """Process the full dataset optimized for your 68GB RAM instance"""
    print("\n🔄 PROCESSING FULL 15-YEAR DATASET")
    print("=" * 50)
    
    if input_file:
        # Process actual full dataset
        dbn_file = Path(input_file)
        print(f"🗂️  Processing full dataset: {dbn_file}")
        
        if not dbn_file.exists():
            print(f"❌ Input file not found: {dbn_file}")
            return None
        
        # Convert DBN to DataFrame with timestamps
        print("📖 Loading and converting DBN file...")
        print("   ⏳ This will take 10-30 minutes for the full dataset...")
        
        try:
            import databento as db
            
            conversion_start = time.time()
            
            # Open DBN store
            store = db.DBNStore.from_file(str(dbn_file))
            metadata = store.metadata
            
            print(f"📊 Dataset: {metadata.dataset}")
            print(f"📅 Period: {metadata.start} to {metadata.end}")
            
            # Convert to DataFrame
            df = store.to_df()
            
            # Add proper timestamps
            if hasattr(df.index, 'astype'):
                start_ns = metadata.start
                end_ns = metadata.end
                total_rows = len(df)
                
                timestamps = pd.date_range(
                    start=pd.to_datetime(start_ns, unit='ns', utc=True),
                    end=pd.to_datetime(end_ns, unit='ns', utc=True),
                    periods=total_rows
                )
                
                df.index = timestamps
                df.index.name = 'timestamp'
            
            # Convert to column format
            df = df.reset_index()
            
            conversion_time = time.time() - conversion_start
            
            print(f"✅ DBN conversion complete!")
            print(f"   Rows: {len(df):,}")
            print(f"   Conversion time: {conversion_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"❌ DBN conversion failed: {e}")
            return None
            
    else:
        # Use test file for validation
        test_file = Path("project/data/processed/es_30day_rth.parquet")
        
        if not test_file.exists():
            print("❌ Test file not found. Run the 30-day pipeline first.")
            return None
        
        print("🧪 TESTING WITH 30-DAY SAMPLE (scaled processing)")
        
        # Load test data
        print("📖 Loading test data...")
        df = pd.read_parquet(test_file)
        
        # Convert timestamp index to column if needed
        if 'timestamp' not in df.columns:
            df = df.reset_index()
        
        print(f"📊 Loaded {len(df):,} rows for testing")
    
    try:
        # Import processing modules
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
        from src.data_pipeline.features import create_all_features
        
        print("✅ Processing modules imported")
        
        # Load test data
        print("📖 Loading test data...")
        df = pd.read_parquet(test_file)
        
        # Convert timestamp index to column if needed
        if 'timestamp' not in df.columns:
            df = df.reset_index()
        
        print(f"📊 Loaded {len(df):,} rows for testing")
        
        # Initialize processing engine
        engine = WeightedLabelingEngine()
        
        # Process with full power of your instance
        print("\n🏷️  STEP 1: WEIGHTED LABELING")
        print("-" * 40)
        
        start_time = time.time()
        
        # Process without performance limits (you have plenty of resources)
        df_labeled = engine.process_dataframe(df, validate_performance=False)
        
        labeling_time = time.time() - start_time
        
        print(f"✅ Weighted labeling complete in {labeling_time:.1f} seconds")
        print(f"   Rows processed: {len(df_labeled):,}")
        print(f"   Processing rate: {len(df_labeled)/labeling_time:.0f} rows/second")
        
        # Feature engineering
        print("\n🔧 STEP 2: FEATURE ENGINEERING")
        print("-" * 40)
        
        features_start = time.time()
        df_final = create_all_features(df_labeled)
        features_time = time.time() - features_start
        
        print(f"✅ Feature engineering complete in {features_time:.1f} seconds")
        print(f"   Final columns: {len(df_final.columns)}")
        
        # Save results
        output_file = Path("/tmp/es_full_processing/test_full_pipeline.parquet")
        df_final.to_parquet(output_file, index=False)
        
        total_time = time.time() - start_time
        file_size_mb = output_file.stat().st_size / (1024**2)
        
        print(f"\n🎉 TEST PROCESSING COMPLETE!")
        print(f"   Output: {output_file}")
        print(f"   Size: {file_size_mb:.1f} MB")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Columns: {len(df_final.columns)}")
        
        # Extrapolate to full dataset
        sample_rows = len(df_final)
        full_dataset_rows = 54_000_000  # Estimated from earlier
        
        scaling_factor = full_dataset_rows / sample_rows
        estimated_full_time = total_time * scaling_factor
        estimated_full_size = file_size_mb * scaling_factor / 1024  # GB
        
        print(f"\n📊 FULL DATASET PROJECTIONS:")
        print(f"   Estimated rows: {full_dataset_rows:,}")
        print(f"   Estimated time: {estimated_full_time/3600:.1f} hours")
        print(f"   Estimated size: {estimated_full_size:.1f} GB")
        print(f"   Memory usage: Well within your {psutil.virtual_memory().total/(1024**3):.0f} GB")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def upload_full_results(processed_file):
    """Upload the processed full dataset results"""
    print("\n📤 UPLOADING FULL DATASET RESULTS")
    print("=" * 50)
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"  # Same bucket as source
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"processed-data/es_15year_labeled_features_{timestamp}.parquet"
        
        file_size_gb = Path(processed_file).stat().st_size / (1024**3)
        
        print(f"📥 Local: {processed_file}")
        print(f"📤 S3: s3://{bucket_name}/{s3_key}")
        print(f"📊 Size: {file_size_gb:.1f} GB")
        print("⏳ Upload will take 10-30 minutes...")
        
        start_time = time.time()
        
        s3_client.upload_file(
            processed_file,
            bucket_name,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'source': 'full_15year_processing',
                    'date_range': '2010-06-06_to_2025-10-21',
                    'processing_date': timestamp,
                    'features': '43',
                    'labels': '6',
                    'weights': '6'
                }
            }
        )
        
        upload_time = time.time() - start_time
        
        print(f"✅ Upload complete!")
        print(f"   Time: {upload_time/60:.1f} minutes")
        print(f"   S3 URL: s3://{bucket_name}/{s3_key}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def create_full_dataset_plan():
    """Create a detailed plan for processing the actual full dataset"""
    print("\n📋 FULL DATASET PROCESSING PLAN")
    print("=" * 50)
    
    print("🎯 PHASE 1: PREPARATION")
    print("   1. Get S3 bucket/key details for 15-year dataset")
    print("   2. Download full dataset (estimated 20-50 GB compressed)")
    print("   3. Verify data integrity and format")
    
    print("\n🎯 PHASE 2: PROCESSING")
    print("   1. Convert DBN to Parquet with timestamps")
    print("   2. Filter to RTH hours (reduce ~70% of data)")
    print("   3. Apply weighted labeling (6 volatility modes)")
    print("   4. Generate 43 engineered features")
    print("   5. Save final dataset")
    
    print("\n🎯 PHASE 3: DEPLOYMENT")
    print("   1. Upload processed dataset to S3")
    print("   2. Create training/validation splits")
    print("   3. Begin XGBoost model training")
    
    print("\n⏱️  ESTIMATED TIMELINE:")
    print("   • Download: 1-2 hours")
    print("   • Processing: 18-24 hours")
    print("   • Upload: 2-4 hours")
    print("   • Total: ~30 hours")
    
    print("\n💾 RESOURCE USAGE:")
    print(f"   • Peak memory: ~40 GB (you have {psutil.virtual_memory().total/(1024**3):.0f} GB)")
    print(f"   • Disk space: ~100 GB (you have {psutil.disk_usage('/').free/(1024**3):.0f} GB)")
    print("   • CPU: All 36 cores utilized")

def main():
    """Main function for full dataset processing"""
    work_dir = setup_processing_environment()
    
    print("\n🤔 PROCESSING OPTIONS:")
    print("1. Test pipeline with scaled processing (recommended first)")
    print("2. Process actual full dataset (need S3 details)")
    print("3. Show detailed processing plan")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        print("\n🧪 RUNNING SCALED TEST...")
        result = process_full_dataset_optimized()
        if result:
            print(f"\n✅ Test successful! Ready for full dataset.")
            print(f"📁 Test output: {result}")
        
    elif choice == "2":
        print("\n⚠️  FULL DATASET PROCESSING:")
        print("   📦 File: s3://es-1-second-data/raw-data/databento/glbx-mdp3-20100606-20251021.ohlcv-1s.dbn.zst")
        print("   📊 Size: 1.3 GB compressed → ~6 GB processed")
        print("   ⏱️  Time: ~20 hours total (download + processing)")
        print("   💾 Memory: ~40 GB peak usage")
        print()
        
        confirm = input("Proceed with full 15-year dataset processing? (yes/no): ")
        if confirm.lower() == "yes":
            print("\n🚀 STARTING FULL DATASET PROCESSING...")
            
            # Download the full dataset
            input_file = download_full_dataset_from_s3()
            if not input_file:
                print("❌ Download failed. Cannot proceed.")
                return
            
            # Process the full dataset
            result = process_full_dataset_optimized(input_file)
            if result:
                print(f"\n🎉 FULL DATASET PROCESSING COMPLETE!")
                print(f"📁 Output: {result}")
                
                # Upload to S3
                upload_choice = input("\nUpload results to S3? (yes/no): ")
                if upload_choice.lower() == "yes":
                    upload_full_results(result)
            else:
                print("\n❌ Full dataset processing failed")
        else:
            print("❌ Full dataset processing cancelled")
    
    elif choice == "3":
        create_full_dataset_plan()
    
    else:
        print("❌ Invalid option")

if __name__ == "__main__":
    main()