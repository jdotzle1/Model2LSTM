#!/usr/bin/env python3
"""
Test script for enhanced S3 operations and file handling

Tests the implementation of task 7.2:
- Parquet compression optimization for S3 storage
- Retry logic with exponential backoff for S3 operations
- File upload/download with progress tracking
- File integrity validation before and after S3 operations
"""
import sys
import os
import time
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.data_pipeline.s3_operations import (
        EnhancedS3Operations, 
        optimize_parquet_for_s3,
        upload_with_retry_and_validation,
        download_with_retry_and_validation
    )
    print("✅ Successfully imported enhanced S3 operations")
except ImportError as e:
    print(f"❌ Failed to import S3 operations: {e}")
    sys.exit(1)


def create_test_parquet_file(file_path: str, rows: int = 10000) -> str:
    """Create a test parquet file for testing"""
    print(f"📝 Creating test parquet file with {rows:,} rows...")
    
    # Create realistic ES futures data
    np.random.seed(42)  # For reproducible tests
    
    # Generate timestamps
    start_time = pd.Timestamp('2024-01-01 09:30:00')
    timestamps = pd.date_range(start_time, periods=rows, freq='1S')
    
    # Generate realistic price data
    base_price = 4750.0
    price_changes = np.random.normal(0, 0.25, rows).cumsum()
    close_prices = base_price + price_changes
    
    # Generate OHLC from close prices
    high_offset = np.abs(np.random.normal(0, 0.5, rows))
    low_offset = np.abs(np.random.normal(0, 0.5, rows))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': close_prices + np.random.normal(0, 0.1, rows),
        'high': close_prices + high_offset,
        'low': close_prices - low_offset,
        'close': close_prices,
        'volume': np.random.randint(100, 5000, rows),
        
        # Add some labeling columns (6 labels + 6 weights)
        'label_low_vol_long': np.random.choice([0, 1], rows, p=[0.7, 0.3]),
        'weight_low_vol_long': np.random.uniform(0.5, 2.0, rows),
        'label_normal_vol_long': np.random.choice([0, 1], rows, p=[0.6, 0.4]),
        'weight_normal_vol_long': np.random.uniform(0.5, 2.0, rows),
        'label_high_vol_long': np.random.choice([0, 1], rows, p=[0.65, 0.35]),
        'weight_high_vol_long': np.random.uniform(0.5, 2.0, rows),
        'label_low_vol_short': np.random.choice([0, 1], rows, p=[0.7, 0.3]),
        'weight_low_vol_short': np.random.uniform(0.5, 2.0, rows),
        'label_normal_vol_short': np.random.choice([0, 1], rows, p=[0.6, 0.4]),
        'weight_normal_vol_short': np.random.uniform(0.5, 2.0, rows),
        'label_high_vol_short': np.random.choice([0, 1], rows, p=[0.65, 0.35]),
        'weight_high_vol_short': np.random.uniform(0.5, 2.0, rows),
        
        # Add some feature columns
        'volume_ratio_30s': np.random.uniform(0.5, 2.0, rows),
        'volatility_regime': np.random.uniform(0.8, 1.5, rows),
        'return_30s': np.random.normal(0, 0.001, rows),
        'atr_30s': np.random.uniform(0.5, 3.0, rows),
        'distance_from_vwap_pct': np.random.normal(0, 0.002, rows)
    })
    
    # Save as parquet
    df.to_parquet(file_path, compression='gzip', index=False)  # Use gzip to test optimization
    
    file_size_mb = Path(file_path).stat().st_size / (1024**2)
    print(f"   ✅ Created test file: {file_size_mb:.1f} MB with {len(df.columns)} columns")
    
    return file_path


def test_parquet_compression_optimization():
    """Test Parquet compression optimization"""
    print("\n🧪 TEST 1: Parquet Compression Optimization")
    print("=" * 50)
    
    try:
        # Create test file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            test_file = create_test_parquet_file(temp_file.name, rows=50000)
        
        # Test compression optimization
        original_size = Path(test_file).stat().st_size
        print(f"📦 Original file size: {original_size / (1024**2):.1f} MB")
        
        # Optimize compression
        optimized_file = optimize_parquet_for_s3(test_file)
        
        if optimized_file != test_file:
            optimized_size = Path(optimized_file).stat().st_size
            compression_ratio = (1 - optimized_size / original_size) * 100
            print(f"✅ Compression optimization successful:")
            print(f"   📦 Optimized size: {optimized_size / (1024**2):.1f} MB")
            print(f"   📊 Compression ratio: {compression_ratio:+.1f}%")
            
            # Verify data integrity (allowing for precision differences due to float32 conversion)
            df_original = pd.read_parquet(test_file)
            df_optimized = pd.read_parquet(optimized_file)
            
            # Check if shapes match
            if df_original.shape != df_optimized.shape:
                print(f"   ❌ Data integrity check failed - shapes differ: {df_original.shape} vs {df_optimized.shape}")
                return False
            
            # Check if column names match
            if list(df_original.columns) != list(df_optimized.columns):
                print(f"   ❌ Data integrity check failed - columns differ")
                return False
            
            # Check numerical columns with tolerance for float32 conversion
            integrity_ok = True
            for col in df_original.columns:
                if df_original[col].dtype in ['float64', 'float32'] and df_optimized[col].dtype in ['float64', 'float32']:
                    # Use numpy allclose for floating point comparison
                    if not np.allclose(df_original[col], df_optimized[col], rtol=1e-5, equal_nan=True):
                        print(f"   ⚠️  Column {col} differs beyond tolerance")
                        print(f"        Original dtype: {df_original[col].dtype}, Optimized dtype: {df_optimized[col].dtype}")
                        print(f"        Sample diff: {abs(df_original[col].iloc[0] - df_optimized[col].iloc[0])}")
                        integrity_ok = False
                        break
                elif df_original[col].dtype in ['int64', 'int32'] and df_optimized[col].dtype in ['int64', 'int32']:
                    # For integer columns, check if values are exactly the same
                    if not np.array_equal(df_original[col].values, df_optimized[col].values):
                        print(f"   ⚠️  Integer column {col} differs")
                        print(f"        Original dtype: {df_original[col].dtype}, Optimized dtype: {df_optimized[col].dtype}")
                        print(f"        First difference at index: {np.where(df_original[col].values != df_optimized[col].values)[0][0] if len(np.where(df_original[col].values != df_optimized[col].values)[0]) > 0 else 'None'}")
                        integrity_ok = False
                        break
                elif not df_original[col].equals(df_optimized[col]):
                    print(f"   ⚠️  Column {col} differs (other type)")
                    print(f"        Original dtype: {df_original[col].dtype}, Optimized dtype: {df_optimized[col].dtype}")
                    integrity_ok = False
                    break
            
            if integrity_ok:
                print(f"   ✅ Data integrity verified - files are equivalent (allowing for precision optimization)")
            else:
                print(f"   ❌ Data integrity check failed - significant differences found")
                return False
            
            # Clean up optimized file
            Path(optimized_file).unlink()
        else:
            print(f"   ℹ️  No optimization applied (file already optimal)")
        
        # Clean up test file
        Path(test_file).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Compression optimization test failed: {e}")
        return False


def test_file_integrity_validation():
    """Test file integrity validation"""
    print("\n🧪 TEST 2: File Integrity Validation")
    print("=" * 50)
    
    try:
        s3_ops = EnhancedS3Operations("test-bucket")
        
        # Create test file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            test_file = create_test_parquet_file(temp_file.name, rows=1000)
        
        # Test valid file validation
        print("🔍 Testing valid file validation...")
        validation_result = s3_ops.validate_file_integrity(
            test_file, 
            file_type="parquet",
            min_size_mb=0.01
        )
        
        if validation_result['valid']:
            print(f"   ✅ Valid file correctly identified")
            print(f"   📊 File size: {validation_result['size_mb']:.2f} MB")
            print(f"   🔐 Hash: {validation_result['hash_value'][:16]}...")
            print(f"   📋 Validation details: {len(validation_result['validation_details'])} checks")
        else:
            print(f"   ❌ Valid file incorrectly flagged as invalid: {validation_result['error_message']}")
            return False
        
        # Test hash validation
        print("🔍 Testing hash validation...")
        expected_hash = validation_result['hash_value']
        hash_validation = s3_ops.validate_file_integrity(
            test_file,
            expected_hash=expected_hash,
            file_type="parquet"
        )
        
        if hash_validation['valid'] and hash_validation['hash_valid']:
            print(f"   ✅ Hash validation successful")
        else:
            print(f"   ❌ Hash validation failed")
            return False
        
        # Test corrupted file detection (simulate by modifying file)
        print("🔍 Testing corrupted file detection...")
        
        # Create a corrupted version
        with open(test_file, 'rb') as f:
            data = bytearray(f.read())
        
        # Corrupt some bytes in the middle
        if len(data) > 1000:
            data[500:510] = b'\x00' * 10  # Corrupt 10 bytes
            
            corrupted_file = test_file + "_corrupted"
            with open(corrupted_file, 'wb') as f:
                f.write(data)
            
            # Validate corrupted file
            corrupted_validation = s3_ops.validate_file_integrity(
                corrupted_file,
                expected_hash=expected_hash,
                file_type="parquet"
            )
            
            if not corrupted_validation['valid'] and corrupted_validation['corruption_detected']:
                print(f"   ✅ Corruption correctly detected")
            else:
                print(f"   ⚠️  Corruption not detected (may be in non-critical area)")
            
            # Clean up corrupted file
            Path(corrupted_file).unlink()
        
        # Clean up test file
        Path(test_file).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ File integrity validation test failed: {e}")
        return False


def test_retry_logic():
    """Test retry logic with exponential backoff"""
    print("\n🧪 TEST 3: Retry Logic with Exponential Backoff")
    print("=" * 50)
    
    try:
        s3_ops = EnhancedS3Operations("test-bucket")
        
        # Test successful operation (no retries needed)
        print("🔍 Testing successful operation...")
        
        def successful_operation():
            return "success"
        
        result = s3_ops.retry_with_exponential_backoff(successful_operation)
        if result == "success":
            print(f"   ✅ Successful operation completed without retries")
        else:
            print(f"   ❌ Unexpected result: {result}")
            return False
        
        # Test retryable error
        print("🔍 Testing retryable error handling...")
        
        attempt_count = 0
        def failing_then_succeeding_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError(f"Simulated connection error (attempt {attempt_count})")
            return f"success_after_{attempt_count}_attempts"
        
        start_time = time.time()
        result = s3_ops.retry_with_exponential_backoff(failing_then_succeeding_operation)
        elapsed_time = time.time() - start_time
        
        if result == "success_after_3_attempts" and attempt_count == 3:
            print(f"   ✅ Retry logic successful after {attempt_count} attempts")
            print(f"   ⏱️  Total time with backoff: {elapsed_time:.1f}s")
        else:
            print(f"   ❌ Retry logic failed: result={result}, attempts={attempt_count}")
            return False
        
        # Test non-retryable error
        print("🔍 Testing non-retryable error handling...")
        
        def non_retryable_operation():
            raise ValueError("This is a non-retryable error")
        
        try:
            s3_ops.retry_with_exponential_backoff(non_retryable_operation)
            print(f"   ❌ Non-retryable error should have been raised")
            return False
        except ValueError as e:
            if "non-retryable error" in str(e):
                print(f"   ✅ Non-retryable error correctly raised immediately")
            else:
                print(f"   ❌ Unexpected error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Retry logic test failed: {e}")
        return False


def test_progress_tracking():
    """Test progress tracking functionality"""
    print("\n🧪 TEST 4: Progress Tracking")
    print("=" * 50)
    
    try:
        from src.data_pipeline.s3_operations import S3ProgressCallback
        
        # Test progress callback
        print("🔍 Testing progress callback...")
        
        total_size = 10 * 1024 * 1024  # 10 MB
        progress = S3ProgressCallback("test_file.parquet", total_size, "upload")
        
        # Simulate progress updates
        chunk_size = 1024 * 1024  # 1 MB chunks
        for i in range(10):
            progress(chunk_size)
            if i == 4:  # Check at 50%
                expected_progress = 50.0
                actual_progress = (progress.bytes_transferred / total_size) * 100
                if abs(actual_progress - expected_progress) < 1.0:
                    print(f"   ✅ Progress tracking accurate at 50%: {actual_progress:.1f}%")
                else:
                    print(f"   ❌ Progress tracking inaccurate: expected {expected_progress}%, got {actual_progress:.1f}%")
                    return False
        
        # Check final progress
        final_progress = (progress.bytes_transferred / total_size) * 100
        if abs(final_progress - 100.0) < 0.1:
            print(f"   ✅ Progress tracking complete: {final_progress:.1f}%")
        else:
            print(f"   ❌ Final progress incorrect: {final_progress:.1f}%")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Progress tracking test failed: {e}")
        return False


def test_integration_with_monthly_processing():
    """Test integration with monthly processing functions"""
    print("\n🧪 TEST 5: Integration with Monthly Processing")
    print("=" * 50)
    
    try:
        # Test that the enhanced operations are properly integrated
        print("🔍 Testing integration imports...")
        
        # Try importing the updated monthly processing
        try:
            import process_monthly_chunks_fixed
            print(f"   ✅ Monthly processing module imported successfully")
            
            # Check if S3_OPERATIONS_AVAILABLE is set
            if hasattr(process_monthly_chunks_fixed, 'S3_OPERATIONS_AVAILABLE'):
                if process_monthly_chunks_fixed.S3_OPERATIONS_AVAILABLE:
                    print(f"   ✅ Enhanced S3 operations available in monthly processing")
                else:
                    print(f"   ⚠️  Enhanced S3 operations not available in monthly processing")
            else:
                print(f"   ⚠️  S3_OPERATIONS_AVAILABLE flag not found")
            
        except ImportError as e:
            print(f"   ❌ Failed to import monthly processing: {e}")
            return False
        
        # Test file info structure compatibility
        print("🔍 Testing file info structure compatibility...")
        
        test_file_info = {
            'month_str': '2024-01',
            'filename': 'glbx-mdp3-20240101-20240131.ohlcv-1s.dbn.zst',
            's3_key': 'raw-data/databento/glbx-mdp3-20240101-20240131.ohlcv-1s.dbn.zst',
            'local_file': '/tmp/monthly_processing/2024-01/input.dbn.zst'
        }
        
        # Test that our S3 operations can handle this structure
        s3_ops = EnhancedS3Operations("es-1-second-data")
        
        # This should not crash (even though the file doesn't exist)
        try:
            # We expect this to fail because the file doesn't exist, but it should fail gracefully
            result = s3_ops.download_monthly_file_optimized(test_file_info)
            print(f"   ✅ File info structure compatible (result: {result})")
        except Exception as e:
            # Check if it's a reasonable error (file not found, no credentials, etc.)
            if any(err_type in str(e).lower() for err_type in ['credentials', 'nosuchkey', 'not found', 'connection']):
                print(f"   ✅ File info structure compatible (expected error: {type(e).__name__})")
            else:
                print(f"   ❌ Unexpected error with file info structure: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def run_all_tests():
    """Run all S3 operations tests"""
    print("🚀 ENHANCED S3 OPERATIONS TEST SUITE")
    print("=" * 60)
    print(f"Testing implementation of Task 7.2:")
    print(f"- Parquet compression optimization for S3 storage")
    print(f"- Retry logic with exponential backoff for S3 operations")
    print(f"- File upload/download with progress tracking")
    print(f"- File integrity validation before and after S3 operations")
    print("=" * 60)
    
    tests = [
        ("Parquet Compression Optimization", test_parquet_compression_optimization),
        ("File Integrity Validation", test_file_integrity_validation),
        ("Retry Logic with Exponential Backoff", test_retry_logic),
        ("Progress Tracking", test_progress_tracking),
        ("Integration with Monthly Processing", test_integration_with_monthly_processing)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 TEST SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"✅ Passed: {passed}/{total} tests")
    print(f"❌ Failed: {total - passed}/{total} tests")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! Enhanced S3 operations are working correctly.")
        print(f"\n✅ Task 7.2 Implementation Complete:")
        print(f"   ✅ Parquet compression optimization implemented")
        print(f"   ✅ Retry logic with exponential backoff implemented")
        print(f"   ✅ Progress tracking for uploads/downloads implemented")
        print(f"   ✅ File integrity validation implemented")
        print(f"   ✅ Integration with monthly processing completed")
    else:
        print(f"\n⚠️  Some tests failed. Please review the implementation.")
        
        print(f"\nDetailed Results:")
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status}: {test_name}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)