#!/usr/bin/env python3
"""
Integration test for monthly processor

Tests that all components work together:
- S3 operations
- Monthly processor
- Pipeline integration
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.data_pipeline.monthly_processor import MonthlyProcessor
        print("✅ MonthlyProcessor imported")
    except ImportError as e:
        print(f"❌ Failed to import MonthlyProcessor: {e}")
        return False
    
    try:
        from src.data_pipeline.s3_operations import EnhancedS3Operations
        print("✅ EnhancedS3Operations imported")
    except ImportError as e:
        print(f"❌ Failed to import EnhancedS3Operations: {e}")
        return False
    
    try:
        from src.data_pipeline.corrected_contract_filtering import process_complete_pipeline
        print("✅ process_complete_pipeline imported")
    except ImportError as e:
        print(f"❌ Failed to import process_complete_pipeline: {e}")
        return False
    
    try:
        from src.data_pipeline.pipeline import process_labeling_and_features, PipelineConfig
        print("✅ Pipeline components imported")
    except ImportError as e:
        print(f"❌ Failed to import pipeline components: {e}")
        return False
    
    return True


def test_monthly_processor_initialization():
    """Test that MonthlyProcessor can be initialized"""
    print("\nTesting MonthlyProcessor initialization...")
    
    try:
        from src.data_pipeline.monthly_processor import MonthlyProcessor
        
        processor = MonthlyProcessor(bucket_name="es-1-second-data")
        print("✅ MonthlyProcessor initialized successfully")
        
        # Test file list generation
        monthly_files = processor.generate_monthly_file_list(
            start_year=2025,
            start_month=10,
            end_year=2025,
            end_month=10
        )
        
        if len(monthly_files) == 1:
            print(f"✅ Generated file list: {len(monthly_files)} month")
            print(f"   Month: {monthly_files[0]['month_str']}")
            print(f"   Filename: {monthly_files[0]['filename']}")
        else:
            print(f"❌ Expected 1 month, got {len(monthly_files)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ MonthlyProcessor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_s3_operations_initialization():
    """Test that S3 operations can be initialized"""
    print("\nTesting S3 operations initialization...")
    
    try:
        from src.data_pipeline.s3_operations import EnhancedS3Operations
        
        s3_ops = EnhancedS3Operations(bucket_name="es-1-second-data")
        print("✅ EnhancedS3Operations initialized successfully")
        
        # Check that required methods exist
        required_methods = [
            'download_monthly_file_optimized',
            'upload_monthly_results_optimized'
        ]
        
        for method_name in required_methods:
            if hasattr(s3_ops, method_name):
                print(f"✅ Method '{method_name}' exists")
            else:
                print(f"❌ Method '{method_name}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ S3 operations initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("=" * 80)
    print("MONTHLY PROCESSOR INTEGRATION TEST")
    print("=" * 80)
    
    tests = [
        ("Import Test", test_imports),
        ("MonthlyProcessor Initialization", test_monthly_processor_initialization),
        ("S3 Operations Initialization", test_s3_operations_initialization)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print(f"{'='*80}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
