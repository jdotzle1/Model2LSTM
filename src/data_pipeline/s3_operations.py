#!/usr/bin/env python3
"""
Enhanced S3 Operations and File Handling

Implements optimized S3 operations with:
- Parquet compression optimization for S3 storage
- Retry logic with exponential backoff for S3 operations
- File upload/download with progress tracking
- File integrity validation before and after S3 operations

Requirements: 8.4, 7.3, 8.7
"""
import os
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
import numpy as np
import psutil
from datetime import datetime
import json


class S3ProgressCallback:
    """Progress callback for S3 operations with detailed tracking"""
    
    def __init__(self, filename: str, total_size: int, operation: str = "upload"):
        self.filename = filename
        self.total_size = total_size
        self.operation = operation
        self.bytes_transferred = 0
        self.start_time = time.time()
        self.last_update = 0
        
    def __call__(self, bytes_amount: int):
        """Called by boto3 during transfer"""
        self.bytes_transferred += bytes_amount
        current_time = time.time()
        
        # Update progress every 2 seconds or at completion
        if current_time - self.last_update >= 2.0 or self.bytes_transferred >= self.total_size:
            self.last_update = current_time
            
            progress_pct = (self.bytes_transferred / self.total_size) * 100
            elapsed_time = current_time - self.start_time
            
            if elapsed_time > 0:
                speed_mbps = (self.bytes_transferred / (1024**2)) / elapsed_time
                
                if self.bytes_transferred < self.total_size:
                    eta_seconds = (self.total_size - self.bytes_transferred) / (self.bytes_transferred / elapsed_time)
                    eta_str = f", ETA: {eta_seconds:.0f}s"
                else:
                    eta_str = ""
                
                print(f"   📊 {self.operation.title()}: {progress_pct:.1f}% "
                      f"({self.bytes_transferred/(1024**2):.1f}/{self.total_size/(1024**2):.1f} MB) "
                      f"@ {speed_mbps:.1f} MB/s{eta_str}")


class EnhancedS3Operations:
    """Enhanced S3 operations with optimization and retry logic"""
    
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        
        # Configuration - set before initializing client
        self.max_retries = 3
        self.base_delay = 1
        self.max_delay = 60
        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks for multipart uploads
        
        # Initialize S3 client
        self.s3_client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize S3 client with retry configuration"""
        try:
            from botocore.config import Config
            
            # Configure retries and timeouts
            config = Config(
                region_name=self.region,
                retries={
                    'max_attempts': self.max_retries,
                    'mode': 'adaptive'
                },
                max_pool_connections=50,
                read_timeout=300,  # 5 minutes
                connect_timeout=60  # 1 minute
            )
            
            self.s3_client = boto3.client('s3', config=config)
            
        except Exception as e:
            print(f"❌ Failed to initialize S3 client: {e}")
            raise
    
    def retry_with_exponential_backoff(self, operation_func, *args, **kwargs):
        """
        Execute operation with exponential backoff retry logic
        
        Args:
            operation_func: Function to retry
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of operation_func
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                    # Add jitter to prevent thundering herd
                    jitter = delay * 0.1 * (0.5 - np.random.random())
                    actual_delay = delay + jitter
                    
                    print(f"   🔄 Retry attempt {attempt}/{self.max_retries} after {actual_delay:.1f}s delay")
                    time.sleep(actual_delay)
                
                return operation_func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Check if error is retryable
                if self._is_retryable_error(e):
                    if attempt < self.max_retries:
                        print(f"   ⚠️  Retryable error on attempt {attempt + 1}: {e}")
                        continue
                    else:
                        print(f"   ❌ Max retries ({self.max_retries}) reached")
                else:
                    print(f"   ❌ Non-retryable error: {e}")
                    break
        
        raise last_exception
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            ClientError
        )
        
        if isinstance(error, retryable_errors):
            return True
        
        # Check for specific S3 error codes that are retryable
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            retryable_codes = [
                'ServiceUnavailable',
                'Throttling',
                'ThrottlingException',
                'RequestTimeout',
                'InternalError',
                'SlowDown'
            ]
            return error_code in retryable_codes
        
        # Check for network-related errors in error message
        error_str = str(error).lower()
        network_errors = ['timeout', 'connection', 'network', 'dns']
        return any(net_err in error_str for net_err in network_errors)
    
    def optimize_parquet_compression(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Optimize Parquet file compression for S3 storage
        
        Args:
            input_file: Path to input parquet file
            output_file: Path to output optimized file (optional)
            
        Returns:
            Path to optimized file
        """
        try:
            if output_file is None:
                # Create optimized version in same directory
                input_path = Path(input_file)
                output_file = str(input_path.parent / f"{input_path.stem}_optimized{input_path.suffix}")
            
            print(f"   🗜️  Optimizing Parquet compression...")
            
            # Read the parquet file
            df = pd.read_parquet(input_file)
            original_memory = df.memory_usage(deep=True).sum()
            
            # Optimize data types for better compression
            optimizations_applied = []
            
            for col in df.columns:
                original_dtype = df[col].dtype
                
                if df[col].dtype == 'float64':
                    # Check if we can downcast to float32 without losing precision
                    if (df[col].min() >= np.finfo(np.float32).min and 
                        df[col].max() <= np.finfo(np.float32).max):
                        # Additional check for precision loss - use more lenient tolerance for test data
                        test_conversion = df[col].astype('float32').astype('float64')
                        if np.allclose(df[col], test_conversion, rtol=1e-5, equal_nan=True):
                            df[col] = df[col].astype('float32')
                            optimizations_applied.append(f"{col}: float64→float32")
                
                elif df[col].dtype == 'int64':
                    # Check if we can downcast to int32
                    if (df[col].min() >= np.iinfo(np.int32).min and 
                        df[col].max() <= np.iinfo(np.int32).max):
                        # Only optimize if all values are integers (no precision loss)
                        if df[col].equals(df[col].astype('int32').astype('int64')):
                            df[col] = df[col].astype('int32')
                            optimizations_applied.append(f"{col}: int64→int32")
                
                elif df[col].dtype == 'object':
                    # Try to optimize string columns
                    if df[col].dtype == 'object' and isinstance(df[col].iloc[0], str):
                        try:
                            df[col] = df[col].astype('category')
                            optimizations_applied.append(f"{col}: object→category")
                        except:
                            pass
            
            # Save with optimal compression settings
            df.to_parquet(
                output_file,
                compression='snappy',  # Good balance of speed and compression
                index=False,
                engine='pyarrow',
                row_group_size=50000,  # Optimize for S3 access patterns
                use_dictionary=True,   # Enable dictionary encoding
                compression_level=None  # Use default level for snappy
            )
            
            # Calculate compression results
            original_size = Path(input_file).stat().st_size
            optimized_size = Path(output_file).stat().st_size
            optimized_memory = df.memory_usage(deep=True).sum()
            
            size_reduction = (1 - optimized_size / original_size) * 100
            memory_reduction = (1 - optimized_memory / original_memory) * 100
            
            print(f"   ✅ Compression optimization complete:")
            print(f"      📦 File size: {original_size/(1024**2):.1f}MB → {optimized_size/(1024**2):.1f}MB ({size_reduction:+.1f}%)")
            print(f"      💾 Memory usage: {original_memory/(1024**2):.1f}MB → {optimized_memory/(1024**2):.1f}MB ({memory_reduction:+.1f}%)")
            
            if optimizations_applied:
                print(f"      🔧 Data type optimizations: {len(optimizations_applied)} columns")
                for opt in optimizations_applied[:3]:  # Show first 3
                    print(f"         {opt}")
                if len(optimizations_applied) > 3:
                    print(f"         ... and {len(optimizations_applied) - 3} more")
            
            return output_file
            
        except Exception as e:
            print(f"   ⚠️  Compression optimization failed: {e}")
            return input_file
    
    def calculate_file_hash(self, file_path: str, algorithm: str = 'md5') -> str:
        """Calculate file hash for integrity validation"""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def validate_file_integrity(self, file_path: str, expected_hash: Optional[str] = None, 
                              file_type: str = "unknown", min_size_mb: float = 0.1) -> Dict[str, Any]:
        """
        Comprehensive file integrity validation
        
        Args:
            file_path: Path to file to validate
            expected_hash: Expected file hash (optional)
            file_type: Type of file for format-specific validation
            min_size_mb: Minimum expected file size in MB
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            'valid': False,
            'file_exists': False,
            'size_mb': 0,
            'size_valid': False,
            'format_valid': False,
            'hash_valid': False,
            'hash_value': None,
            'corruption_detected': False,
            'error_message': None,
            'validation_details': []
        }
        
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                validation_result['error_message'] = f"File does not exist: {file_path}"
                return validation_result
            
            validation_result['file_exists'] = True
            validation_result['validation_details'].append("File exists")
            
            # Check file size
            file_size = file_path.stat().st_size
            validation_result['size_mb'] = file_size / (1024**2)
            validation_result['size_valid'] = validation_result['size_mb'] >= min_size_mb
            
            if not validation_result['size_valid']:
                validation_result['error_message'] = f"File too small: {validation_result['size_mb']:.2f} MB < {min_size_mb} MB"
                return validation_result
            
            validation_result['validation_details'].append(f"Size valid: {validation_result['size_mb']:.2f} MB")
            
            # Calculate file hash
            try:
                validation_result['hash_value'] = self.calculate_file_hash(str(file_path))
                validation_result['validation_details'].append(f"Hash calculated: {validation_result['hash_value'][:8]}...")
                
                if expected_hash:
                    validation_result['hash_valid'] = validation_result['hash_value'] == expected_hash
                    if not validation_result['hash_valid']:
                        validation_result['error_message'] = "Hash mismatch - file may be corrupted"
                        validation_result['corruption_detected'] = True
                        return validation_result
                    validation_result['validation_details'].append("Hash matches expected value")
                else:
                    validation_result['hash_valid'] = True  # No expected hash to compare
                    
            except Exception as e:
                validation_result['error_message'] = f"Hash calculation failed: {e}"
                return validation_result
            
            # Format-specific validation
            if file_type == "parquet":
                try:
                    # Try to read parquet metadata and a small sample
                    # Use head() method instead of nrows parameter which is not supported in read_parquet
                    df_sample = pd.read_parquet(file_path).head(10)
                    
                    if len(df_sample.columns) == 0:
                        validation_result['error_message'] = "Parquet file has no columns"
                        validation_result['corruption_detected'] = True
                        return validation_result
                    
                    # Check for required columns in processed files
                    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    missing_columns = [col for col in required_columns if col not in df_sample.columns]
                    
                    if missing_columns:
                        validation_result['error_message'] = f"Missing required columns: {missing_columns}"
                        validation_result['corruption_detected'] = True
                        return validation_result
                    
                    validation_result['format_valid'] = True
                    validation_result['validation_details'].append(f"Parquet format valid: {len(df_sample.columns)} columns")
                    
                except Exception as e:
                    validation_result['error_message'] = f"Parquet format validation failed: {e}"
                    validation_result['corruption_detected'] = True
                    return validation_result
            
            elif file_type == "dbn":
                try:
                    import databento as db
                    store = db.DBNStore.from_file(str(file_path))
                    metadata = store.metadata
                    
                    # Basic metadata validation
                    if not hasattr(metadata, 'start') or not hasattr(metadata, 'end'):
                        validation_result['error_message'] = "Invalid DBN metadata - missing start/end times"
                        validation_result['corruption_detected'] = True
                        return validation_result
                    
                    validation_result['format_valid'] = True
                    validation_result['validation_details'].append("DBN format valid")
                    
                except Exception as e:
                    validation_result['error_message'] = f"DBN format validation failed: {e}"
                    validation_result['corruption_detected'] = True
                    return validation_result
            else:
                validation_result['format_valid'] = True  # No specific format validation
                validation_result['validation_details'].append("No format-specific validation required")
            
            # If we get here, file is valid
            validation_result['valid'] = True
            validation_result['validation_details'].append("All validations passed")
            
        except Exception as e:
            validation_result['error_message'] = f"File validation error: {e}"
        
        return validation_result
    
    def upload_file_with_progress(self, local_file: str, s3_key: str, 
                                metadata: Optional[Dict[str, str]] = None,
                                validate_before: bool = True,
                                validate_after: bool = True) -> Dict[str, Any]:
        """
        Upload file to S3 with progress tracking and integrity validation
        
        Args:
            local_file: Path to local file
            s3_key: S3 key for the uploaded file
            metadata: Optional metadata to attach to S3 object
            validate_before: Whether to validate file before upload
            validate_after: Whether to validate upload success
            
        Returns:
            Dict with upload results and validation info
        """
        result = {
            'success': False,
            'local_file': local_file,
            's3_key': s3_key,
            'file_size_mb': 0,
            'upload_time_seconds': 0,
            'upload_speed_mbps': 0,
            'pre_validation': None,
            'post_validation': None,
            'error_message': None
        }
        
        try:
            local_path = Path(local_file)
            
            # Pre-upload validation
            if validate_before:
                print(f"   🔍 Validating file before upload...")
                file_type = "parquet" if local_path.suffix.lower() == '.parquet' else "unknown"
                pre_validation = self.validate_file_integrity(local_file, file_type=file_type)
                result['pre_validation'] = pre_validation
                
                if not pre_validation['valid']:
                    result['error_message'] = f"Pre-upload validation failed: {pre_validation['error_message']}"
                    return result
                
                print(f"   ✅ Pre-upload validation passed")
            
            # Get file info
            file_size = local_path.stat().st_size
            result['file_size_mb'] = file_size / (1024**2)
            
            # Calculate file hash for integrity checking
            local_hash = self.calculate_file_hash(local_file)
            
            # Prepare metadata
            upload_metadata = {
                'upload_timestamp': datetime.now().isoformat(),
                'file_size_bytes': str(file_size),
                'file_hash_md5': local_hash,
                'upload_tool': 'enhanced_s3_operations'
            }
            
            if metadata:
                upload_metadata.update(metadata)
            
            # Setup progress callback
            progress_callback = S3ProgressCallback(
                filename=local_path.name,
                total_size=file_size,
                operation="upload"
            )
            
            print(f"   📤 Uploading {local_path.name} ({result['file_size_mb']:.1f} MB) to s3://{self.bucket_name}/{s3_key}")
            
            # Perform upload with retry logic
            start_time = time.time()
            
            def upload_operation():
                return self.s3_client.upload_file(
                    local_file,
                    self.bucket_name,
                    s3_key,
                    Callback=progress_callback,
                    ExtraArgs={'Metadata': upload_metadata}
                )
            
            self.retry_with_exponential_backoff(upload_operation)
            
            upload_time = time.time() - start_time
            result['upload_time_seconds'] = upload_time
            result['upload_speed_mbps'] = result['file_size_mb'] / upload_time if upload_time > 0 else 0
            
            print(f"   ✅ Upload completed in {upload_time:.1f}s @ {result['upload_speed_mbps']:.1f} MB/s")
            
            # Post-upload validation
            if validate_after:
                print(f"   🔍 Validating upload integrity...")
                
                def validate_upload():
                    # Check object exists and get metadata
                    response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                    
                    # Verify file size
                    s3_size = response['ContentLength']
                    if s3_size != file_size:
                        raise ValueError(f"Size mismatch: local={file_size}, s3={s3_size}")
                    
                    # Verify hash if available in metadata
                    s3_metadata = response.get('Metadata', {})
                    s3_hash = s3_metadata.get('file_hash_md5')
                    if s3_hash and s3_hash != local_hash:
                        raise ValueError(f"Hash mismatch: local={local_hash}, s3={s3_hash}")
                    
                    return {
                        'size_match': True,
                        'hash_match': s3_hash == local_hash if s3_hash else None,
                        's3_size': s3_size,
                        's3_hash': s3_hash,
                        'last_modified': response.get('LastModified')
                    }
                
                post_validation = self.retry_with_exponential_backoff(validate_upload)
                result['post_validation'] = post_validation
                
                print(f"   ✅ Post-upload validation passed")
            
            result['success'] = True
            return result
            
        except Exception as e:
            result['error_message'] = f"Upload failed: {e}"
            print(f"   ❌ Upload failed: {e}")
            return result
    
    def download_file_with_progress(self, s3_key: str, local_file: str,
                                  validate_after: bool = True) -> Dict[str, Any]:
        """
        Download file from S3 with progress tracking and integrity validation
        
        Args:
            s3_key: S3 key of the file to download
            local_file: Local path to save the file
            validate_after: Whether to validate download integrity
            
        Returns:
            Dict with download results and validation info
        """
        result = {
            'success': False,
            's3_key': s3_key,
            'local_file': local_file,
            'file_size_mb': 0,
            'download_time_seconds': 0,
            'download_speed_mbps': 0,
            'post_validation': None,
            'error_message': None
        }
        
        try:
            local_path = Path(local_file)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get S3 object info
            def get_object_info():
                return self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            
            print(f"   🔍 Checking S3 object info...")
            s3_info = self.retry_with_exponential_backoff(get_object_info)
            
            file_size = s3_info['ContentLength']
            result['file_size_mb'] = file_size / (1024**2)
            s3_metadata = s3_info.get('Metadata', {})
            expected_hash = s3_metadata.get('file_hash_md5')
            
            print(f"   📥 Downloading s3://{self.bucket_name}/{s3_key} ({result['file_size_mb']:.1f} MB)")
            
            # Setup progress callback
            progress_callback = S3ProgressCallback(
                filename=local_path.name,
                total_size=file_size,
                operation="download"
            )
            
            # Perform download with retry logic
            start_time = time.time()
            
            def download_operation():
                return self.s3_client.download_file(
                    self.bucket_name,
                    s3_key,
                    local_file,
                    Callback=progress_callback
                )
            
            self.retry_with_exponential_backoff(download_operation)
            
            download_time = time.time() - start_time
            result['download_time_seconds'] = download_time
            result['download_speed_mbps'] = result['file_size_mb'] / download_time if download_time > 0 else 0
            
            print(f"   ✅ Download completed in {download_time:.1f}s @ {result['download_speed_mbps']:.1f} MB/s")
            
            # Post-download validation
            if validate_after:
                print(f"   🔍 Validating download integrity...")
                
                file_type = "parquet" if local_path.suffix.lower() == '.parquet' else "dbn" if local_path.suffix.lower() in ['.dbn', '.zst'] else "unknown"
                post_validation = self.validate_file_integrity(
                    local_file, 
                    expected_hash=expected_hash,
                    file_type=file_type
                )
                result['post_validation'] = post_validation
                
                if not post_validation['valid']:
                    result['error_message'] = f"Post-download validation failed: {post_validation['error_message']}"
                    # Remove corrupted file
                    try:
                        local_path.unlink()
                    except:
                        pass
                    return result
                
                print(f"   ✅ Post-download validation passed")
            
            result['success'] = True
            return result
            
        except Exception as e:
            result['error_message'] = f"Download failed: {e}"
            print(f"   ❌ Download failed: {e}")
            return result
    
    def upload_monthly_results_optimized(self, file_info: Dict[str, Any], 
                                       processed_file: str,
                                       monthly_statistics: Optional[Any] = None) -> bool:
        """
        Optimized upload of monthly results with compression and comprehensive validation
        
        Args:
            file_info: Monthly file information
            processed_file: Path to processed parquet file
            monthly_statistics: Optional statistics object
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            print(f"   🚀 Starting optimized upload for {file_info['month_str']}")
            
            # Step 1: Optimize compression
            optimized_file = self.optimize_parquet_compression(processed_file)
            
            # Step 2: Prepare S3 key with organized structure
            year, month = file_info['month_str'].split('-')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"processed-data/monthly/{year}/{month}/monthly_{file_info['month_str']}_{timestamp}.parquet"
            
            # Step 3: Prepare comprehensive metadata
            metadata = {
                'source': 'enhanced_monthly_processing_pipeline',
                'month': file_info['month_str'],
                'processing_date': timestamp,
                'pipeline_version': '4.0_optimized_s3',
                'compression': 'snappy_optimized',
                'data_format': 'parquet'
            }
            
            # Add statistics to metadata if available
            if monthly_statistics:
                metadata.update({
                    'overall_quality_score': str(getattr(monthly_statistics, 'overall_quality_score', 0)),
                    'requires_reprocessing': str(getattr(monthly_statistics, 'requires_reprocessing', False)),
                    'processing_successful': str(getattr(monthly_statistics, 'processing_successful', True)),
                    'total_rollover_events': str(getattr(monthly_statistics, 'total_rollover_events', 0))
                })
            
            # Step 4: Upload main file with validation
            upload_result = self.upload_file_with_progress(
                local_file=optimized_file,
                s3_key=s3_key,
                metadata=metadata,
                validate_before=True,
                validate_after=True
            )
            
            if not upload_result['success']:
                print(f"   ❌ Main file upload failed: {upload_result['error_message']}")
                return False
            
            print(f"   ✅ Main file uploaded successfully")
            print(f"      📊 Size: {upload_result['file_size_mb']:.1f} MB")
            print(f"      ⚡ Speed: {upload_result['upload_speed_mbps']:.1f} MB/s")
            
            # Step 5: Upload statistics JSON if available
            print(f"   🔍 Checking statistics: type={type(monthly_statistics)}, is_dict={isinstance(monthly_statistics, dict)}", flush=True)
            if monthly_statistics:
                stats_s3_key = f"processed-data/monthly/{year}/{month}/statistics/monthly_{file_info['month_str']}_{timestamp}_statistics.json"
                print(f"   📊 Preparing statistics upload...", flush=True)
                print(f"      Statistics keys: {list(monthly_statistics.keys()) if isinstance(monthly_statistics, dict) else 'N/A'}", flush=True)
                
                try:
                    # Create temporary JSON file with real statistics content
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                        if hasattr(monthly_statistics, 'to_json'):
                            temp_file.write(monthly_statistics.to_json())
                        elif isinstance(monthly_statistics, dict):
                            # Use the actual statistics dict from process_complete_pipeline
                            stats_output = {
                                'month': file_info['month_str'],
                                'processing_date': timestamp,
                                'statistics_available': True,
                                'pipeline_statistics': monthly_statistics
                            }
                            # Use default handler for non-serializable types
                            json.dump(stats_output, temp_file, indent=2, default=str)
                        else:
                            # Only use fallback if no real statistics available
                            json.dump({
                                'month': file_info['month_str'],
                                'processing_date': timestamp,
                                'statistics_available': False,
                                'note': 'Statistics object not provided'
                            }, temp_file, indent=2)
                        temp_json_path = temp_file.name
                    
                    print(f"      📝 Statistics JSON created: {temp_json_path}", flush=True)
                    print(f"      📏 File size: {Path(temp_json_path).stat().st_size} bytes", flush=True)
                    
                    # Upload statistics with validation
                    stats_metadata = {
                        'content_type': 'application/json',
                        'month': file_info['month_str'],
                        'statistics_version': '1.0',
                        'related_data_file': s3_key
                    }
                    
                    stats_upload_result = self.upload_file_with_progress(
                        local_file=temp_json_path,
                        s3_key=stats_s3_key,
                        metadata=stats_metadata,
                        validate_before=False,  # JSON files are small, skip pre-validation
                        validate_after=True
                    )
                    
                    # Clean up temporary file
                    Path(temp_json_path).unlink()
                    
                    if stats_upload_result['success']:
                        print(f"   📈 Statistics uploaded successfully", flush=True)
                        print(f"      📍 Location: s3://{self.bucket_name}/{stats_s3_key}", flush=True)
                    else:
                        print(f"   ⚠️  Statistics upload failed: {stats_upload_result['error_message']}", flush=True)
                        # Don't fail the main upload for statistics failure
                    
                except Exception as stats_error:
                    print(f"   ⚠️  Statistics upload error: {stats_error}", flush=True)
                    import traceback
                    traceback.print_exc()
                    # Continue - statistics upload failure is not critical
            else:
                print(f"   ⚠️  No statistics provided for upload", flush=True)
            
            # Skip problematic MD report upload - focus on JSON statistics
            print(f"   📊 Skipping MD report upload - JSON statistics contain all needed metrics")
            
            # Step 6: Clean up optimized file if it's different from original
            if optimized_file != processed_file:
                try:
                    Path(optimized_file).unlink()
                    print(f"   🧹 Cleaned up temporary optimized file")
                except Exception as cleanup_error:
                    print(f"   ⚠️  Cleanup warning: {cleanup_error}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Optimized upload failed: {e}")
            return False
    
    def download_monthly_file_optimized(self, file_info: Dict[str, Any]) -> bool:
        """
        Optimized download of monthly file with multiple path attempts and validation
        
        Args:
            file_info: Monthly file information with s3_key and local_file paths
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            print(f"   🚀 Starting optimized download for {file_info['month_str']}")
            
            # Check if file already exists and is valid
            local_file = Path(file_info['local_file'])
            if local_file.exists():
                print(f"   📁 Existing file found, validating...")
                validation_result = self.validate_file_integrity(
                    str(local_file), 
                    file_type="dbn",
                    min_size_mb=1.0
                )
                
                if validation_result['valid']:
                    print(f"   ✅ Existing file is valid ({validation_result['size_mb']:.1f} MB)")
                    return True
                else:
                    print(f"   🔧 Existing file invalid: {validation_result['error_message']}")
                    try:
                        local_file.unlink()
                        print(f"   🗑️  Removed invalid file")
                    except Exception as e:
                        print(f"   ⚠️  Could not remove invalid file: {e}")
            
            # Try multiple S3 paths
            s3_paths_to_try = [
                file_info['s3_key'],  # Original path
                f"databento/{file_info['filename']}",
                f"raw/{file_info['filename']}",
                f"es-data/{file_info['filename']}",
                f"raw-data/{file_info['filename']}",
                f"monthly/{file_info['filename']}",
                f"dbn/{file_info['filename']}"
            ]
            
            for path_index, s3_path in enumerate(s3_paths_to_try, 1):
                print(f"   🔍 Trying S3 path {path_index}/{len(s3_paths_to_try)}: {s3_path}")
                
                try:
                    # Check if object exists
                    def check_object():
                        return self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_path)
                    
                    s3_info = self.retry_with_exponential_backoff(check_object)
                    file_size_mb = s3_info['ContentLength'] / (1024**2)
                    
                    print(f"   📦 Found file: {file_size_mb:.1f} MB")
                    
                    # Validate expected file size
                    if file_size_mb < 1.0:
                        print(f"   ⚠️  File suspiciously small: {file_size_mb:.1f} MB")
                        continue
                    
                    # Download with validation
                    download_result = self.download_file_with_progress(
                        s3_key=s3_path,
                        local_file=str(local_file),
                        validate_after=True
                    )
                    
                    if download_result['success']:
                        print(f"   ✅ Download successful from path {path_index}")
                        return True
                    else:
                        print(f"   ❌ Download failed: {download_result['error_message']}")
                        continue
                
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        print(f"   ❌ File not found at: {s3_path}")
                        continue
                    else:
                        print(f"   ❌ S3 error for path {s3_path}: {e}")
                        continue
                except Exception as e:
                    print(f"   ❌ Error checking path {s3_path}: {e}")
                    continue
            
            print(f"   ❌ File not found in any S3 path for {file_info['month_str']}")
            return False
            
        except Exception as e:
            print(f"   ❌ Optimized download failed: {e}")
            return False


# Convenience functions for backward compatibility
def optimize_parquet_for_s3(input_file: str, output_file: Optional[str] = None) -> str:
    """Convenience function for Parquet optimization"""
    s3_ops = EnhancedS3Operations("dummy-bucket")  # Bucket not needed for compression
    return s3_ops.optimize_parquet_compression(input_file, output_file)


def upload_with_retry_and_validation(bucket_name: str, local_file: str, s3_key: str,
                                   metadata: Optional[Dict[str, str]] = None) -> bool:
    """Convenience function for optimized upload"""
    s3_ops = EnhancedS3Operations(bucket_name)
    result = s3_ops.upload_file_with_progress(local_file, s3_key, metadata)
    return result['success']


def download_with_retry_and_validation(bucket_name: str, s3_key: str, local_file: str) -> bool:
    """Convenience function for optimized download"""
    s3_ops = EnhancedS3Operations(bucket_name)
    result = s3_ops.download_file_with_progress(s3_key, local_file)
    return result['success']