#!/usr/bin/env python3
"""
Enhanced Monthly Processing Integration

This script demonstrates how to integrate the enhanced logging and monitoring system
with the existing monthly processing pipeline for task 5.2.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.enhanced_logging import get_enhanced_logger, log_enhanced


def enhanced_process_single_month(file_info, use_enhanced_logging=True):
    """
    Enhanced single month processing with integrated logging and monitoring
    
    This function demonstrates how to integrate the enhanced logging system
    with the existing monthly processing workflow.
    """
    month_str = file_info['month_str']
    
    if use_enhanced_logging:
        logger = get_enhanced_logger()
        
        # Start month processing with enhanced logging
        logger.start_stage("month_processing", context={
            'month': month_str,
            'file_key': file_info.get('s3_key', 'unknown'),
            'processing_start': datetime.now().isoformat()
        })
    else:
        # Fallback to existing logging
        from process_monthly_chunks_fixed import log_progress
        log_progress(f"🔄 PROCESSING {month_str}", stage="month_processing", stage_event="start")
    
    try:
        # Stage 1: Download with enhanced monitoring
        if use_enhanced_logging:
            logger.start_stage("download", context={
                'month': month_str,
                'file_key': file_info.get('s3_key', 'unknown')
            })
        
        # Simulate download process
        download_success = simulate_download_process(file_info, use_enhanced_logging)
        
        if use_enhanced_logging:
            logger.end_stage("download", success=download_success, context={
                'result': 'success' if download_success else 'failed',
                'file_size_mb': 150.5 if download_success else 0
            })
        
        if not download_success:
            raise Exception("Download failed")
        
        # Stage 2: Data Processing with enhanced monitoring
        if use_enhanced_logging:
            logger.start_stage("data_processing", context={
                'month': month_str,
                'input_file': file_info.get('local_file', 'unknown')
            })
        
        processing_result = simulate_data_processing(file_info, use_enhanced_logging)
        
        if use_enhanced_logging:
            logger.end_stage("data_processing", success=processing_result['success'], context={
                'raw_rows': processing_result.get('raw_rows', 0),
                'final_rows': processing_result.get('final_rows', 0),
                'retention_rate': processing_result.get('retention_rate', 0),
                'processing_time_seconds': processing_result.get('processing_time', 0)
            })
        
        if not processing_result['success']:
            raise Exception("Data processing failed")
        
        # Stage 3: Feature Engineering with enhanced monitoring
        if use_enhanced_logging:
            logger.start_stage("feature_engineering", context={
                'month': month_str,
                'input_rows': processing_result.get('final_rows', 0)
            })
        
        feature_result = simulate_feature_engineering(processing_result, use_enhanced_logging)
        
        if use_enhanced_logging:
            logger.end_stage("feature_engineering", success=feature_result['success'], context={
                'features_generated': feature_result.get('features_generated', 0),
                'expected_features': 43,
                'feature_quality_score': feature_result.get('quality_score', 0),
                'nan_percentage': feature_result.get('nan_percentage', 0)
            })
        
        if not feature_result['success']:
            raise Exception("Feature engineering failed")
        
        # Stage 4: Statistics Collection with enhanced monitoring
        if use_enhanced_logging:
            logger.start_stage("statistics_collection", context={
                'month': month_str,
                'final_rows': feature_result.get('final_rows', 0)
            })
        
        stats_result = simulate_statistics_collection(feature_result, use_enhanced_logging)
        
        if use_enhanced_logging:
            logger.end_stage("statistics_collection", success=stats_result['success'], context={
                'quality_score': stats_result.get('quality_score', 0),
                'win_rates': stats_result.get('win_rates', {}),
                'rollover_events': stats_result.get('rollover_events', 0),
                'requires_reprocessing': stats_result.get('requires_reprocessing', False)
            })
        
        # Stage 5: Upload with enhanced monitoring
        if use_enhanced_logging:
            logger.start_stage("upload", context={
                'month': month_str,
                'output_file_size_mb': 75.2,
                'destination': 's3://es-1-second-data/processed/'
            })
        
        upload_success = simulate_upload_process(stats_result, use_enhanced_logging)
        
        if use_enhanced_logging:
            logger.end_stage("upload", success=upload_success, context={
                'upload_time_seconds': 45.3 if upload_success else 0,
                'compression_ratio': 0.5 if upload_success else 0
            })
        
        # Stage 6: Cleanup
        if use_enhanced_logging:
            logger.start_stage("cleanup", context={'month': month_str})
        
        cleanup_success = simulate_cleanup_process(file_info, use_enhanced_logging)
        
        if use_enhanced_logging:
            logger.end_stage("cleanup", success=cleanup_success, context={
                'files_cleaned': 3 if cleanup_success else 0,
                'disk_space_freed_mb': 225.7 if cleanup_success else 0
            })
        
        # Complete month processing
        if use_enhanced_logging:
            logger.end_stage("month_processing", success=True, context={
                'month': month_str,
                'processing_end': datetime.now().isoformat(),
                'overall_success': True,
                'final_quality_score': stats_result.get('quality_score', 0)
            })
            
            # Log final success message
            logger.log(f"✅ {month_str} completed successfully", level="INFO", 
                      context={'final_status': 'success'})
        
        return {
            'success': True,
            'month': month_str,
            'quality_score': stats_result.get('quality_score', 0),
            'processing_time': time.time() - (time.time() - 300),  # Simulated
            'final_rows': feature_result.get('final_rows', 0)
        }
        
    except Exception as e:
        # Enhanced error handling with detailed logging
        if use_enhanced_logging:
            logger.log(f"❌ {month_str} processing failed", level="ERROR", 
                      error_details=e, context={
                          'month': month_str,
                          'failure_stage': 'unknown',
                          'processing_end': datetime.now().isoformat()
                      })
            
            logger.end_stage("month_processing", success=False, context={
                'month': month_str,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
        
        return {
            'success': False,
            'month': month_str,
            'error': str(e),
            'processing_time': time.time() - (time.time() - 300)  # Simulated
        }


def simulate_download_process(file_info, use_enhanced_logging):
    """Simulate download process with logging"""
    month_str = file_info['month_str']
    
    if use_enhanced_logging:
        log_enhanced(f"📥 Downloading {month_str} from S3", level="INFO", 
                    stage="download", stage_event="progress",
                    context={'file_key': file_info.get('s3_key', 'unknown')})
    
    # Simulate download time
    time.sleep(0.1)
    
    # Simulate occasional download failures
    import random
    success = random.random() > 0.1  # 90% success rate
    
    if use_enhanced_logging:
        if success:
            log_enhanced(f"✅ Download completed: 150.5 MB", level="INFO",
                        stage="download", stage_event="progress")
        else:
            log_enhanced(f"❌ Download failed: Network timeout", level="ERROR",
                        stage="download", stage_event="progress")
    
    return success


def simulate_data_processing(file_info, use_enhanced_logging):
    """Simulate data processing with logging"""
    month_str = file_info['month_str']
    
    if use_enhanced_logging:
        log_enhanced(f"🔧 Processing data for {month_str}", level="INFO",
                    stage="data_processing", stage_event="progress")
    
    # Simulate processing time
    time.sleep(0.15)
    
    # Simulate processing results
    raw_rows = 2500000  # 2.5M rows
    final_rows = int(raw_rows * 0.35)  # 35% retention after RTH filtering
    retention_rate = final_rows / raw_rows
    
    if use_enhanced_logging:
        log_enhanced(f"📊 Data flow: {raw_rows:,} → {final_rows:,} rows ({retention_rate:.1%} retention)",
                    level="INFO", stage="data_processing", stage_event="progress",
                    context={'raw_rows': raw_rows, 'final_rows': final_rows})
    
    return {
        'success': True,
        'raw_rows': raw_rows,
        'final_rows': final_rows,
        'retention_rate': retention_rate,
        'processing_time': 0.15
    }


def simulate_feature_engineering(processing_result, use_enhanced_logging):
    """Simulate feature engineering with logging"""
    
    if use_enhanced_logging:
        log_enhanced(f"🔬 Generating features", level="INFO",
                    stage="feature_engineering", stage_event="progress")
    
    # Simulate feature engineering time
    time.sleep(0.1)
    
    # Simulate feature engineering results
    features_generated = 43
    quality_score = 0.85
    nan_percentage = 12.5
    
    if use_enhanced_logging:
        log_enhanced(f"📈 Generated {features_generated} features, quality: {quality_score:.2f}",
                    level="INFO", stage="feature_engineering", stage_event="progress",
                    context={'features_count': features_generated, 'quality_score': quality_score})
    
    return {
        'success': True,
        'features_generated': features_generated,
        'quality_score': quality_score,
        'nan_percentage': nan_percentage,
        'final_rows': processing_result['final_rows']
    }


def simulate_statistics_collection(feature_result, use_enhanced_logging):
    """Simulate statistics collection with logging"""
    
    if use_enhanced_logging:
        log_enhanced(f"📊 Collecting comprehensive statistics", level="INFO",
                    stage="statistics_collection", stage_event="progress")
    
    # Simulate statistics collection time
    time.sleep(0.08)
    
    # Simulate statistics results
    win_rates = {
        'low_vol_long': 0.23,
        'normal_vol_long': 0.31,
        'high_vol_long': 0.18,
        'low_vol_short': 0.25,
        'normal_vol_short': 0.29,
        'high_vol_short': 0.21
    }
    
    quality_score = 0.78
    rollover_events = 3
    requires_reprocessing = quality_score < 0.7
    
    if use_enhanced_logging:
        avg_win_rate = sum(win_rates.values()) / len(win_rates)
        log_enhanced(f"📈 Win rates: {avg_win_rate:.1%} avg, Quality: {quality_score:.2f}",
                    level="INFO", stage="statistics_collection", stage_event="progress",
                    context={'avg_win_rate': avg_win_rate, 'quality_score': quality_score})
    
    return {
        'success': True,
        'quality_score': quality_score,
        'win_rates': win_rates,
        'rollover_events': rollover_events,
        'requires_reprocessing': requires_reprocessing
    }


def simulate_upload_process(stats_result, use_enhanced_logging):
    """Simulate upload process with logging"""
    
    if use_enhanced_logging:
        log_enhanced(f"📤 Uploading results to S3", level="INFO",
                    stage="upload", stage_event="progress")
    
    # Simulate upload time
    time.sleep(0.05)
    
    # Simulate upload success
    import random
    success = random.random() > 0.05  # 95% success rate
    
    if use_enhanced_logging:
        if success:
            log_enhanced(f"✅ Upload completed: 75.2 MB compressed", level="INFO",
                        stage="upload", stage_event="progress")
        else:
            log_enhanced(f"❌ Upload failed: S3 timeout", level="ERROR",
                        stage="upload", stage_event="progress")
    
    return success


def simulate_cleanup_process(file_info, use_enhanced_logging):
    """Simulate cleanup process with logging"""
    
    if use_enhanced_logging:
        log_enhanced(f"🧹 Cleaning up temporary files", level="INFO",
                    stage="cleanup", stage_event="progress")
    
    # Simulate cleanup time
    time.sleep(0.02)
    
    if use_enhanced_logging:
        log_enhanced(f"✅ Cleanup completed: 225.7 MB freed", level="INFO",
                    stage="cleanup", stage_event="progress")
    
    return True


def demonstrate_enhanced_logging():
    """Demonstrate the enhanced logging system with monthly processing"""
    print("🚀 Enhanced Monthly Processing Integration Demo")
    print("=" * 60)
    
    # Sample file info for demonstration
    sample_files = [
        {
            'month_str': '2024-01',
            's3_key': 'raw-data/databento/glbx-mdp3-20240101-20240131.ohlcv-1s.dbn.zst',
            'local_file': '/tmp/monthly_processing/2024-01/input.dbn.zst'
        },
        {
            'month_str': '2024-02',
            's3_key': 'raw-data/databento/glbx-mdp3-20240201-20240229.ohlcv-1s.dbn.zst',
            'local_file': '/tmp/monthly_processing/2024-02/input.dbn.zst'
        },
        {
            'month_str': '2024-03',
            's3_key': 'raw-data/databento/glbx-mdp3-20240301-20240331.ohlcv-1s.dbn.zst',
            'local_file': '/tmp/monthly_processing/2024-03/input.dbn.zst'
        }
    ]
    
    results = []
    
    print("\n📊 Processing months with enhanced logging...")
    
    for file_info in sample_files:
        print(f"\n🔄 Processing {file_info['month_str']}...")
        
        result = enhanced_process_single_month(file_info, use_enhanced_logging=True)
        results.append(result)
        
        if result['success']:
            print(f"   ✅ {file_info['month_str']}: Quality {result['quality_score']:.2f}, {result['final_rows']:,} rows")
        else:
            print(f"   ❌ {file_info['month_str']}: {result['error']}")
    
    # Generate session report
    print("\n📋 Generating session report...")
    logger = get_enhanced_logger()
    report = logger.generate_session_report()
    
    print("\n" + "=" * 60)
    print("SESSION REPORT")
    print("=" * 60)
    print(report)
    
    # Summary statistics
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    if successful > 0:
        avg_quality = sum(r['quality_score'] for r in results if r['success']) / successful
        total_rows = sum(r['final_rows'] for r in results if r['success'])
    else:
        avg_quality = 0
        total_rows = 0
    
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"📊 Months processed: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Average quality score: {avg_quality:.3f}")
    print(f"📋 Total rows processed: {total_rows:,}")
    
    # Cleanup
    logger.cleanup()
    
    return results


if __name__ == "__main__":
    results = demonstrate_enhanced_logging()
    
    print("\n🎉 Enhanced logging integration demonstration completed!")
    print("\n📋 Key features demonstrated:")
    print("   • Stage-by-stage timing and monitoring")
    print("   • Memory usage tracking throughout processing")
    print("   • Comprehensive error logging with full context")
    print("   • Structured logging for analysis and debugging")
    print("   • Performance metrics collection")
    print("   • Session reporting and statistics")
    print("   • Integration with existing monthly processing workflow")
    
    successful_count = sum(1 for r in results if r['success'])
    if successful_count == len(results):
        print(f"\n✅ All {len(results)} months processed successfully!")
    else:
        print(f"\n⚠️  {successful_count}/{len(results)} months processed successfully")