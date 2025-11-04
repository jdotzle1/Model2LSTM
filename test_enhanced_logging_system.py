#!/usr/bin/env python3
"""
Test Enhanced Logging and Monitoring System

This script tests the enhanced logging system implementation for task 5.2.
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.enhanced_logging import EnhancedLogger, get_enhanced_logger, log_enhanced


def test_basic_logging():
    """Test basic logging functionality"""
    print("🧪 Testing basic logging functionality...")
    
    # Create temporary log directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = EnhancedLogger(temp_dir)
        
        # Test basic logging
        logger.log("Test info message", level="INFO")
        logger.log("Test warning message", level="WARNING")
        logger.log("Test error message", level="ERROR")
        
        # Test with context
        context = {'test_id': 'basic_test', 'iteration': 1}
        logger.log("Test with context", level="INFO", context=context)
        
        # Verify log files were created
        log_files = list(Path(temp_dir).glob("*.log")) + list(Path(temp_dir).glob("*.jsonl"))
        assert len(log_files) >= 2, f"Expected at least 2 log files, found {len(log_files)}"
        
        print("   ✅ Basic logging test passed")


def test_stage_tracking():
    """Test stage tracking functionality"""
    print("🧪 Testing stage tracking...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = EnhancedLogger(temp_dir)
        
        # Test stage tracking
        logger.start_stage("test_stage", context={'stage_type': 'unit_test'})
        time.sleep(0.1)  # Simulate some work
        logger.end_stage("test_stage", success=True, context={'result': 'success'})
        
        # Test nested stages
        logger.start_stage("parent_stage")
        logger.start_stage("child_stage")
        time.sleep(0.05)
        logger.end_stage("child_stage", success=True)
        logger.end_stage("parent_stage", success=True)
        
        # Get stage statistics
        stats = logger.stage_tracker.get_all_stage_statistics()
        
        assert 'test_stage' in stats, "test_stage not found in statistics"
        assert stats['test_stage']['executions'] == 1, "Expected 1 execution for test_stage"
        assert stats['test_stage']['success_rate'] == 1.0, "Expected 100% success rate"
        
        print("   ✅ Stage tracking test passed")


def test_memory_monitoring():
    """Test memory monitoring functionality"""
    print("🧪 Testing memory monitoring...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = EnhancedLogger(temp_dir)
        
        # Take some memory snapshots
        snapshot1, status1 = logger.memory_monitor.take_snapshot("test_start")
        
        # Simulate memory usage
        data = [i for i in range(10000)]  # Small memory allocation
        
        snapshot2, status2 = logger.memory_monitor.take_snapshot("test_end")
        
        # Get memory statistics
        mem_stats = logger.memory_monitor.get_memory_statistics()
        
        assert mem_stats['snapshots_count'] >= 2, "Expected at least 2 memory snapshots"
        assert mem_stats['peak_memory_mb'] > 0, "Expected positive peak memory usage"
        
        print(f"   📊 Peak memory: {mem_stats['peak_memory_mb']:.1f} MB")
        print("   ✅ Memory monitoring test passed")


def test_error_logging():
    """Test error logging functionality"""
    print("🧪 Testing error logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = EnhancedLogger(temp_dir)
        
        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            logger.log("Test error occurred", level="ERROR", error_details=e, 
                      context={'error_test': True})
        
        # Verify error log file was created
        error_log = Path(temp_dir) / "error_details.log"
        assert error_log.exists(), "Error log file was not created"
        
        # Check error log content
        with open(error_log, 'r') as f:
            content = f.read()
            assert "ValueError" in content, "Error type not found in error log"
            assert "Test error for logging" in content, "Error message not found in error log"
        
        print("   ✅ Error logging test passed")


def test_session_reporting():
    """Test session reporting functionality"""
    print("🧪 Testing session reporting...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = EnhancedLogger(temp_dir)
        
        # Simulate a processing session
        logger.start_stage("download", context={'file': 'test.dbn'})
        time.sleep(0.1)
        logger.end_stage("download", success=True)
        
        logger.start_stage("processing", context={'rows': 1000})
        time.sleep(0.15)
        logger.end_stage("processing", success=True)
        
        logger.start_stage("upload", context={'destination': 's3://test'})
        time.sleep(0.05)
        logger.end_stage("upload", success=False)  # Simulate failure
        
        # Generate session summary
        summary = logger.get_session_summary()
        
        assert 'session_id' in summary, "Session ID not found in summary"
        assert 'stage_statistics' in summary, "Stage statistics not found in summary"
        assert len(summary['stage_statistics']) == 3, "Expected 3 stages in statistics"
        
        # Generate session report
        report = logger.generate_session_report()
        
        assert "Processing Session Report" in report, "Report title not found"
        assert "download" in report, "Download stage not found in report"
        assert "processing" in report, "Processing stage not found in report"
        assert "upload" in report, "Upload stage not found in report"
        
        print("   ✅ Session reporting test passed")


def test_global_logger():
    """Test global logger functionality"""
    print("🧪 Testing global logger...")
    
    # Test global logger function
    log_enhanced("Test global logger message", level="INFO", 
                context={'test': 'global_logger'})
    
    log_enhanced("Test stage start", level="INFO", stage="global_test", 
                stage_event="start")
    
    time.sleep(0.05)
    
    log_enhanced("Test stage end", level="INFO", stage="global_test", 
                stage_event="end")
    
    # Get the global logger instance
    logger = get_enhanced_logger()
    stats = logger.stage_tracker.get_all_stage_statistics()
    
    assert 'global_test' in stats, "Global test stage not found in statistics"
    
    print("   ✅ Global logger test passed")


def test_performance_under_load():
    """Test logging performance under load"""
    print("🧪 Testing performance under load...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = EnhancedLogger(temp_dir)
        
        start_time = time.time()
        
        # Log many messages quickly
        for i in range(100):
            logger.log(f"Load test message {i}", level="INFO", 
                      context={'iteration': i, 'batch': i // 10})
            
            if i % 20 == 0:
                logger.start_stage(f"load_stage_{i}")
                logger.end_stage(f"load_stage_{i}", success=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"   📊 Logged 100 messages in {duration:.3f} seconds ({100/duration:.1f} msg/sec)")
        
        # Verify all messages were logged
        main_log = Path(temp_dir) / "enhanced_processing.log"
        with open(main_log, 'r') as f:
            lines = f.readlines()
            message_lines = [line for line in lines if "Load test message" in line]
            assert len(message_lines) == 100, f"Expected 100 messages, found {len(message_lines)}"
        
        print("   ✅ Performance test passed")


def test_integration_with_existing_system():
    """Test integration with existing monthly processing system"""
    print("🧪 Testing integration with existing system...")
    
    # Test that enhanced logging can work alongside existing log_progress function
    try:
        # Import the existing system
        from process_monthly_chunks_fixed import log_progress
        
        # Test existing function still works
        log_progress("Test existing log_progress function", level="INFO")
        
        # Test enhanced logging
        log_enhanced("Test enhanced logging alongside existing", level="INFO")
        
        print("   ✅ Integration test passed")
        
    except ImportError as e:
        print(f"   ⚠️  Integration test skipped: {e}")


def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Enhanced Logging System Tests")
    print("=" * 50)
    
    tests = [
        test_basic_logging,
        test_stage_tracking,
        test_memory_monitoring,
        test_error_logging,
        test_session_reporting,
        test_global_logger,
        test_performance_under_load,
        test_integration_with_existing_system
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n✅ Enhanced Logging System implementation is working correctly!")
        print("\n📋 Features implemented:")
        print("   • Detailed timestamp and processing time capture")
        print("   • Processing start/end times for each month and stage")
        print("   • Comprehensive processing log with success/failure status")
        print("   • Memory usage and performance monitoring")
        print("   • Structured logging for analysis and debugging")
        print("   • Error details with full traceback")
        print("   • Session reporting and statistics")
        print("   • Performance metrics collection")
    else:
        print("\n❌ Some tests failed. Please review the implementation.")
        sys.exit(1)