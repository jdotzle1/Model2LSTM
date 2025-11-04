#!/usr/bin/env python3
"""
Test enhanced error handling and recovery in monthly processing pipeline
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
# import pytest  # Not needed for basic testing

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the enhanced functions
from process_monthly_chunks_fixed import (
    handle_processing_error,
    retry_with_backoff,
    validate_file_integrity,
    log_progress
)

def test_handle_processing_error():
    """Test enhanced error handling function"""
    print("Testing enhanced error handling...")
    
    # Test different error types
    test_cases = [
        (FileNotFoundError("File not found"), "download", True),
        (MemoryError("Out of memory"), "processing", True),
        (ConnectionError("Network error"), "upload", True),
        (ValueError("Invalid data"), "processing", False),
    ]
    
    for error, stage, expected_critical in test_cases:
        error_info = handle_processing_error(error, stage, "2023-01", critical=expected_critical)
        
        assert error_info['error_type'] == type(error).__name__
        assert error_info['stage'] == stage
        assert error_info['month'] == "2023-01"
        assert error_info['critical'] == expected_critical
        assert 'recovery_strategy' in error_info
        assert 'retry_recommended' in error_info
        
        print(f"   ✅ {type(error).__name__} handled correctly")
    
    print("   ✅ Error handling test passed")

def test_retry_with_backoff():
    """Test retry logic with exponential backoff"""
    print("Testing retry with backoff...")
    
    # Test successful operation
    def successful_operation():
        return "success"
    
    result = retry_with_backoff(successful_operation, max_retries=2)
    assert result == "success"
    print("   ✅ Successful operation test passed")
    
    # Test operation that fails then succeeds
    call_count = 0
    def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary failure")
        return "success"
    
    call_count = 0
    result = retry_with_backoff(flaky_operation, max_retries=3, base_delay=0.1)
    assert result == "success"
    assert call_count == 3
    print("   ✅ Flaky operation retry test passed")
    
    # Test operation that always fails
    def failing_operation():
        raise ValueError("Permanent failure")
    
    try:
        retry_with_backoff(failing_operation, max_retries=2, base_delay=0.1)
        assert False, "Should have raised exception"
    except ValueError as e:
        assert str(e) == "Permanent failure"
    
    print("   ✅ Permanent failure test passed")

def test_validate_file_integrity():
    """Test file integrity validation"""
    print("Testing file integrity validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test non-existent file
        non_existent = temp_path / "nonexistent.txt"
        result = validate_file_integrity(non_existent)
        assert not result['valid']
        assert not result['file_exists']
        print("   ✅ Non-existent file test passed")
        
        # Test file too small
        small_file = temp_path / "small.txt"
        small_file.write_text("small")
        result = validate_file_integrity(small_file, expected_min_size_mb=1.0)
        assert not result['valid']
        assert result['file_exists']
        assert not result['size_valid']
        print("   ✅ Small file test passed")
        
        # Test valid file
        large_file = temp_path / "large.txt"
        large_file.write_text("x" * (1024 * 1024 + 100))  # > 1MB
        result = validate_file_integrity(large_file, expected_min_size_mb=1.0)
        assert result['valid']
        assert result['file_exists']
        assert result['size_valid']
        print("   ✅ Valid file test passed")

def test_enhanced_logging():
    """Test enhanced logging functionality"""
    print("Testing enhanced logging...")
    
    # Test basic logging
    log_progress("Test message")
    print("   ✅ Basic logging test passed")
    
    # Test error logging with details
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_progress("Error occurred", level="ERROR", error_details=e, context={"test": "value"})
    
    print("   ✅ Error logging test passed")

def test_corruption_detection():
    """Test corruption detection in file validation"""
    print("Testing corruption detection...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a mock parquet file that will fail validation
        corrupt_file = temp_path / "corrupt.parquet"
        corrupt_file.write_text("This is not a valid parquet file")
        
        result = validate_file_integrity(corrupt_file, "parquet", expected_min_size_mb=0.001)
        
        # Should detect corruption or at least fail validation
        assert not result['valid']
        # Note: corruption_detected might not be set if pandas/pyarrow not available
        # The important thing is that validation fails
        print(f"   📊 Validation result: {result}")
        print("   ✅ Corruption detection test passed")

def run_all_tests():
    """Run all error handling tests"""
    print("🧪 TESTING ENHANCED ERROR HANDLING AND RECOVERY")
    print("=" * 60)
    
    try:
        test_handle_processing_error()
        test_retry_with_backoff()
        test_validate_file_integrity()
        test_enhanced_logging()
        test_corruption_detection()
        
        print("=" * 60)
        print("✅ ALL ERROR HANDLING TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)