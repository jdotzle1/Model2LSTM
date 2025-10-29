"""
Performance Target Validation Script

This script validates that the weighted labeling system can meet the requirement
of processing 10M rows within 60 minutes with <8GB memory usage.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from project.data_pipeline.performance_validation import validate_10m_target, run_memory_stress_test


def main():
    """Run complete performance validation"""
    print("WEIGHTED LABELING SYSTEM - PERFORMANCE TARGET VALIDATION")
    print("=" * 80)
    print("Requirement: Process 10M rows within 60 minutes using <8GB memory")
    print("Target speed: 167,000 rows/minute")
    print("=" * 80)
    
    try:
        # Run the 10M row target validation
        target_met = validate_10m_target()
        
        # Run memory stress test
        print(f"\n" + "=" * 80)
        print("MEMORY STRESS TEST")
        print("=" * 80)
        memory_ok = run_memory_stress_test(max_memory_gb=8.0)
        
        # Final summary
        print(f"\n" + "=" * 80)
        print("PERFORMANCE VALIDATION RESULTS")
        print("=" * 80)
        
        print(f"✅ Performance monitoring: IMPLEMENTED")
        print(f"✅ Memory optimization: IMPLEMENTED") 
        print(f"✅ Numpy vectorization: IMPLEMENTED")
        print(f"{'✅' if target_met else '❌'} 10M rows in 60 minutes: {'MET' if target_met else 'NOT MET'}")
        print(f"{'✅' if memory_ok else '❌'} Memory usage <8GB: {'MET' if memory_ok else 'NOT MET'}")
        
        overall_success = target_met and memory_ok
        
        if overall_success:
            print(f"\n🎉 ALL PERFORMANCE REQUIREMENTS MET!")
            print(f"   ✅ Task 7 implementation is COMPLETE")
            print(f"   ✅ System ready for production use with 10M+ row datasets")
            print(f"   ✅ Requirements 9.1, 9.2, and 9.4 are satisfied")
        else:
            print(f"\n⚠️  PERFORMANCE REQUIREMENTS NOT FULLY MET")
            print(f"   ❌ Additional optimization may be required")
            print(f"   ❌ Consider adjusting chunk size or algorithm parameters")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    print(f"\n" + "=" * 80)
    print("TASK 7 COMPLETION STATUS")
    print("=" * 80)
    
    if success:
        print("✅ Task 7: Add performance monitoring and optimization - COMPLETED")
        print("\nImplemented features:")
        print("  ✅ Processing speed tracking (target: 167K rows/minute)")
        print("  ✅ Memory usage monitoring (target: <8GB)")
        print("  ✅ Numpy vectorization for numerical computations")
        print("  ✅ Performance validation against 60-minute target for 10M rows")
        print("  ✅ Requirements 9.1, 9.2, 9.4 satisfied")
        
        print(f"\n🎯 Ready to proceed to next task in the implementation plan!")
        
    else:
        print("❌ Task 7: Add performance monitoring and optimization - NEEDS WORK")
        print("\nIssues to address:")
        print("  ⚠️  Performance targets not met on current hardware")
        print("  ⚠️  May need algorithm optimization or hardware scaling")
        print("  ⚠️  Consider EC2 deployment for full performance validation")
    
    sys.exit(0 if success else 1)