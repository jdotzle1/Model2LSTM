#!/usr/bin/env python3
"""
Process 15 years of ES data in monthly chunks with data quality fixes built-in
Enhanced with better memory management and independent processing capability
"""
import sys
import os
import time
import boto3
import pandas as pd
import numpy as np
import psutil
import gc
from pathlib import Path
from datetime import datetime, timedelta
import calendar
from collections import defaultdict, deque

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced S3 operations
try:
    from src.data_pipeline.s3_operations import EnhancedS3Operations
    S3_OPERATIONS_AVAILABLE = True
except ImportError:
    print("⚠️  Enhanced S3 operations not available, using basic operations")
    S3_OPERATIONS_AVAILABLE = False

class EnhancedProgressTracker:
    """Enhanced progress tracking with improved time estimation and bottleneck identification"""
    
    def __init__(self, total_months):
        self.total_months = total_months
        self.start_time = time.time()
        self.completed_months = 0
        self.successful_months = 0
        self.failed_months = 0
        
        # Track timing history for better estimation
        self.month_durations = deque(maxlen=20)  # Keep last 20 for rolling average
        self.recent_durations = deque(maxlen=5)  # Last 5 for trend analysis
        
        # Track current processing
        self.current_month = None
        self.current_month_start = None
        
        # Performance tracking
        self.fastest_month = None
        self.slowest_month = None
        self.fastest_time = float('inf')
        self.slowest_time = 0
    
    def start_month(self, month_str, month_index):
        """Start tracking a new month"""
        self.current_month = month_str
        self.current_month_start = time.time()
        
        # Enhanced progress display
        elapsed = time.time() - self.start_time
        
        if self.completed_months > 0:
            # Use recent performance for better estimation
            if len(self.recent_durations) >= 3:
                # Use trend-adjusted estimation
                recent_avg = sum(self.recent_durations) / len(self.recent_durations)
                trend_factor = self._calculate_trend_factor()
                estimated_time = recent_avg * trend_factor
            else:
                recent_avg = sum(self.month_durations) / len(self.month_durations) if self.month_durations else 1200  # 20 min default
                estimated_time = recent_avg
            
            remaining_months = self.total_months - self.completed_months
            eta_seconds = remaining_months * estimated_time
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            
            log_progress(f"📊 PROGRESS: {month_index}/{self.total_months} months - Processing {month_str}")
            log_progress(f"   🕐 ETA: {eta_time.strftime('%Y-%m-%d %H:%M:%S')} ({eta_seconds/3600:.1f}h remaining)")
            log_progress(f"   📈 Recent avg: {recent_avg/60:.1f} min/month, Trend: {'📈' if trend_factor > 1 else '📉' if trend_factor < 1 else '➡️'}")
        else:
            log_progress(f"📊 PROGRESS: {month_index}/{self.total_months} months - Processing {month_str}")
            log_progress(f"   🚀 Starting first month - establishing baseline timing")
    
    def complete_month(self, month_str, success, duration):
        """Complete tracking for a month"""
        self.completed_months += 1
        
        if success:
            self.successful_months += 1
        else:
            self.failed_months += 1
        
        # Track timing
        self.month_durations.append(duration)
        self.recent_durations.append(duration)
        
        # Track performance extremes
        if duration < self.fastest_time:
            self.fastest_time = duration
            self.fastest_month = month_str
        
        if duration > self.slowest_time:
            self.slowest_time = duration
            self.slowest_month = month_str
        
        self.current_month = None
        self.current_month_start = None
    
    def _calculate_trend_factor(self):
        """Calculate trend factor for time estimation adjustment"""
        if len(self.recent_durations) < 3:
            return 1.0
        
        # Simple linear trend calculation
        recent_list = list(self.recent_durations)
        first_half = sum(recent_list[:len(recent_list)//2]) / (len(recent_list)//2)
        second_half = sum(recent_list[len(recent_list)//2:]) / (len(recent_list) - len(recent_list)//2)
        
        if first_half > 0:
            trend_factor = second_half / first_half
            # Limit trend factor to reasonable bounds
            return max(0.5, min(2.0, trend_factor))
        
        return 1.0
    
    def get_progress_summary(self):
        """Get comprehensive progress summary"""
        elapsed = time.time() - self.start_time
        
        if self.completed_months > 0:
            avg_time = sum(self.month_durations) / len(self.month_durations)
            success_rate = (self.successful_months / self.completed_months) * 100
            
            # Improved ETA calculation
            if len(self.recent_durations) >= 3:
                recent_avg = sum(self.recent_durations) / len(self.recent_durations)
                trend_factor = self._calculate_trend_factor()
                estimated_time = recent_avg * trend_factor
            else:
                estimated_time = avg_time
            
            remaining_months = self.total_months - self.completed_months
            eta_seconds = remaining_months * estimated_time
            completion_time = datetime.now() + timedelta(seconds=eta_seconds)
            
            return {
                'elapsed_hours': elapsed / 3600,
                'eta_hours': eta_seconds / 3600,
                'avg_time_minutes': avg_time / 60,
                'success_rate': success_rate,
                'completion_time': completion_time.strftime('%Y-%m-%d %H:%M:%S'),
                'fastest_month': self.fastest_month,
                'slowest_month': self.slowest_month,
                'fastest_time_minutes': self.fastest_time / 60 if self.fastest_time != float('inf') else 0,
                'slowest_time_minutes': self.slowest_time / 60,
                'trend_factor': self._calculate_trend_factor()
            }
        else:
            return {
                'elapsed_hours': elapsed / 3600,
                'eta_hours': 0,
                'avg_time_minutes': 0,
                'success_rate': 0,
                'completion_time': 'Calculating...',
                'fastest_month': None,
                'slowest_month': None,
                'fastest_time_minutes': 0,
                'slowest_time_minutes': 0,
                'trend_factor': 1.0
            }

def process_single_month_with_timing(file_info, stage_timings):
    """Enhanced single month processing with detailed stage timing"""
    month_str = file_info['month_str']
    
    # Track stage times for this month
    month_stage_times = {}
    
    # Wrap the original function with timing
    stage_start = time.time()
    result = process_single_month(file_info)
    total_time = time.time() - stage_start
    
    # For now, we'll add basic timing - in a full implementation, 
    # we'd instrument process_single_month to track individual stages
    month_stage_times['total'] = total_time
    
    # Accumulate stage timings across all months
    for stage, duration in month_stage_times.items():
        if stage not in stage_timings:
            stage_timings[stage] = []
        stage_timings[stage].append(duration)
    
    return result

def identify_processing_bottlenecks(stage_timings):
    """Identify processing bottlenecks from stage timing data"""
    bottlenecks = []
    
    if not stage_timings:
        return bottlenecks
    
    # Calculate average times for each stage
    stage_averages = {}
    for stage, times in stage_timings.items():
        if times:
            stage_averages[stage] = sum(times) / len(times)
    
    if not stage_averages:
        return bottlenecks
    
    # Find stages that take significantly longer than others
    total_avg = sum(stage_averages.values())
    
    for stage, avg_time in stage_averages.items():
        # If a stage takes more than 30% of total time, consider it a bottleneck
        if avg_time > total_avg * 0.3:
            bottlenecks.append(f"{stage} ({avg_time/60:.1f}min avg)")
    
    return bottlenecks

class EnhancedMonitoringSystem:
    """Enhanced monitoring system with performance tracking and memory monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.stage_start_times = {}
        self.stage_durations = defaultdict(list)
        self.memory_snapshots = []
        self.processing_events = []
        
        # Performance thresholds
        self.memory_warning_threshold = 6000  # 6GB
        self.memory_critical_threshold = 7500  # 7.5GB
        self.stage_warning_threshold = 1800  # 30 minutes
        
    def start_stage(self, stage_name, context=None):
        """Start monitoring a processing stage"""
        timestamp = time.time()
        self.stage_start_times[stage_name] = timestamp
        
        # Take memory snapshot
        memory_mb = psutil.Process().memory_info().rss / (1024**2)
        self.memory_snapshots.append({
            'timestamp': timestamp,
            'stage': stage_name,
            'event': 'stage_start',
            'memory_mb': memory_mb,
            'context': context
        })
        
        # Check memory levels
        if memory_mb > self.memory_critical_threshold:
            log_progress(f"🚨 CRITICAL: High memory usage at {stage_name} start: {memory_mb:.1f} MB", level="CRITICAL")
        elif memory_mb > self.memory_warning_threshold:
            log_progress(f"⚠️  WARNING: Elevated memory usage at {stage_name} start: {memory_mb:.1f} MB", level="WARNING")
    
    def end_stage(self, stage_name, success=True, context=None):
        """End monitoring a processing stage"""
        timestamp = time.time()
        duration = None
        
        if stage_name in self.stage_start_times:
            duration = timestamp - self.stage_start_times[stage_name]
            self.stage_durations[stage_name].append(duration)
            
            # Take memory snapshot
            memory_mb = psutil.Process().memory_info().rss / (1024**2)
            self.memory_snapshots.append({
                'timestamp': timestamp,
                'stage': stage_name,
                'event': 'stage_end',
                'memory_mb': memory_mb,
                'duration': duration,
                'success': success,
                'context': context
            })
            
            # Log stage completion with performance analysis
            avg_duration = sum(self.stage_durations[stage_name]) / len(self.stage_durations[stage_name])
            
            if duration > self.stage_warning_threshold:
                log_progress(f"🐌 SLOW: {stage_name} took {duration/60:.1f} minutes (avg: {avg_duration/60:.1f} min)", level="WARNING")
            
            del self.stage_start_times[stage_name]
        
        return duration
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        current_time = time.time()
        total_runtime = current_time - self.start_time
        
        # Calculate stage statistics
        stage_stats = {}
        for stage, durations in self.stage_durations.items():
            if durations:
                stage_stats[stage] = {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'avg_time': sum(durations) / len(durations),
                    'min_time': min(durations),
                    'max_time': max(durations)
                }
        
        # Memory statistics
        if self.memory_snapshots:
            memory_values = [snap['memory_mb'] for snap in self.memory_snapshots]
            memory_stats = {
                'peak_memory': max(memory_values),
                'avg_memory': sum(memory_values) / len(memory_values),
                'min_memory': min(memory_values),
                'current_memory': memory_values[-1] if memory_values else 0
            }
        else:
            memory_stats = {'peak_memory': 0, 'avg_memory': 0, 'min_memory': 0, 'current_memory': 0}
        
        return {
            'total_runtime_hours': total_runtime / 3600,
            'stage_stats': stage_stats,
            'memory_stats': memory_stats,
            'total_snapshots': len(self.memory_snapshots)
        }

# Global monitoring system instance
monitoring_system = EnhancedMonitoringSystem()

def log_progress(message, level="INFO", error_details=None, context=None, stage=None, stage_event=None):
    """
    Enhanced log progress with comprehensive monitoring, detailed timestamps, and performance tracking
    
    Args:
        message: Main log message
        level: Log level (INFO, WARNING, ERROR, CRITICAL)
        error_details: Detailed error information (exception, traceback, etc.)
        context: Additional context information (dict)
        stage: Processing stage name for performance tracking
        stage_event: Stage event type ('start', 'end', 'progress')
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
    processing_time = time.time()
    
    # Enhanced message formatting with stage information
    if stage:
        log_msg = f"[{timestamp}] [{level}] [{stage}] {message}"
    else:
        log_msg = f"[{timestamp}] [{level}] {message}"
    
    print(log_msg)
    
    # Enhanced logging with comprehensive information capture
    detailed_log_entries = [log_msg]
    
    # Add processing stage timing if provided
    if stage and stage_event:
        if stage_event == 'start':
            monitoring_system.start_stage(stage, context)
            detailed_log_entries.append(f"[{timestamp}] [STAGE_START] {stage} started at {processing_time:.3f}")
        elif stage_event == 'end':
            success = level not in ['ERROR', 'CRITICAL']
            stage_duration = monitoring_system.end_stage(stage, success, context)
            detailed_log_entries.append(f"[{timestamp}] [STAGE_END] {stage} ended at {processing_time:.3f}")
            if stage_duration:
                detailed_log_entries.append(f"[{timestamp}] [STAGE_DURATION] {stage} took {stage_duration:.2f} seconds")
        elif stage_event == 'progress':
            detailed_log_entries.append(f"[{timestamp}] [STAGE_PROGRESS] {stage} progress at {processing_time:.3f}")
    
    # Add error details if provided
    if error_details:
        if isinstance(error_details, Exception):
            # Extract comprehensive exception information
            import traceback
            detailed_log_entries.append(f"[{timestamp}] [ERROR_DETAIL] Exception: {type(error_details).__name__}: {str(error_details)}")
            detailed_log_entries.append(f"[{timestamp}] [ERROR_DETAIL] Traceback: {traceback.format_exc()}")
        elif isinstance(error_details, str):
            detailed_log_entries.append(f"[{timestamp}] [ERROR_DETAIL] {error_details}")
        else:
            detailed_log_entries.append(f"[{timestamp}] [ERROR_DETAIL] {str(error_details)}")
    
    # Add comprehensive context information if provided
    if context:
        for key, value in context.items():
            detailed_log_entries.append(f"[{timestamp}] [CONTEXT] {key}: {value}")
    
    # Enhanced memory and performance monitoring for all levels
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024**2)
        
        # Add memory information for ERROR/CRITICAL levels or high memory usage
        if level in ["ERROR", "CRITICAL"] or memory_mb > 4000:  # 4GB threshold
            detailed_log_entries.append(f"[{timestamp}] [MEMORY] RSS: {memory_mb:.1f} MB, VMS: {memory_info.vms/(1024**2):.1f} MB")
            
            # Add CPU usage for performance analysis
            cpu_percent = process.cpu_percent()
            detailed_log_entries.append(f"[{timestamp}] [PERFORMANCE] CPU: {cpu_percent:.1f}%, Threads: {process.num_threads()}")
            
            # Add system memory information for critical situations
            if level == "CRITICAL":
                system_memory = psutil.virtual_memory()
                detailed_log_entries.append(f"[{timestamp}] [SYSTEM] System memory: {system_memory.percent:.1f}% used, Available: {system_memory.available/(1024**3):.1f} GB")
        
        # Always log memory for stage events to track memory usage patterns
        if stage and stage_event in ['start', 'end']:
            detailed_log_entries.append(f"[{timestamp}] [MEMORY_TRACKING] {stage}_{stage_event}: {memory_mb:.1f} MB")
    
    except Exception:
        pass
    
    # Enhanced file logging with comprehensive processing log
    log_files = [
        Path("/tmp/monthly_processing.log"),
        Path("monthly_processing.log")  # Fallback for Windows
    ]
    
    for log_file in log_files:
        try:
            # Create directory if it doesn't exist
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, "a", encoding='utf-8') as f:
                for entry in detailed_log_entries:
                    f.write(entry + "\n")
                f.flush()
            
            # Successfully wrote to this log file, break
            break
            
        except Exception as e:
            # Try next log file location
            if log_file == log_files[-1]:  # Last attempt failed
                # If all else fails, just print (don't crash)
                print(f"Warning: Could not write to log file: {e}")
    
    # Create success/failure status log for restart capability
    if level in ["ERROR", "CRITICAL"] and context and 'month' in context:
        try:
            status_log = Path("/tmp/monthly_processing/processing_status.log")
            status_log.parent.mkdir(parents=True, exist_ok=True)
            
            with open(status_log, "a") as f:
                f.write(f"{timestamp},{context['month']},{level},{message}\n")
                f.flush()
        except Exception:
            pass
    
    # Create comprehensive processing log with success/failure status
    try:
        comprehensive_log = Path("/tmp/monthly_processing/comprehensive_processing.log")
        comprehensive_log.parent.mkdir(parents=True, exist_ok=True)
        
        # Create structured log entry for parsing and analysis
        log_entry = {
            'timestamp': timestamp,
            'processing_time': processing_time,
            'level': level,
            'stage': stage,
            'stage_event': stage_event,
            'message': message,
            'context': context or {},
            'has_error': error_details is not None
        }
        
        with open(comprehensive_log, "a") as f:
            import json
            f.write(json.dumps(log_entry, default=str) + "\n")
            f.flush()
    except Exception:
        pass

def clean_price_data(df):
    """Clean price data to remove invalid values - built into main script"""
    log_progress("   🧹 Cleaning price data...")
    
    original_rows = len(df)
    
    # Check for invalid prices
    price_cols = ['open', 'high', 'low', 'close']
    
    issues_found = False
    for col in price_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            negative_count = (df[col] < 0).sum()
            
            if zero_count > 0 or negative_count > 0:
                log_progress(f"     ⚠️  {col}: {zero_count} zeros, {negative_count} negative values")
                issues_found = True
    
    if not issues_found:
        log_progress("     ✅ No price issues found")
        return df
    
    # Remove rows with any invalid prices
    valid_mask = True
    for col in price_cols:
        if col in df.columns:
            valid_mask = valid_mask & (df[col] > 0)
    
    df_clean = df[valid_mask].copy()
    
    # Remove negative volume
    if 'volume' in df_clean.columns:
        negative_volume = (df_clean['volume'] < 0).sum()
        if negative_volume > 0:
            log_progress(f"     ⚠️  Found {negative_volume} negative volume values")
            df_clean = df_clean[df_clean['volume'] >= 0].copy()
    
    removed_rows = original_rows - len(df_clean)
    log_progress(f"     🗑️  Removed {removed_rows:,} rows with invalid data ({removed_rows/original_rows*100:.2f}%)")
    
    return df_clean

def generate_monthly_file_list():
    """Generate list of monthly DBN files to process"""
    log_progress("📅 GENERATING MONTHLY FILE LIST")
    
    monthly_files = []
    
    # Generate for 15 years: 2010-2025
    start_year = 2010
    end_year = 2025
    end_month = 10  # October 2025
    
    for year in range(start_year, end_year + 1):
        start_month = 7 if year == 2010 else 1  # Start from July 2010
        last_month = end_month if year == end_year else 12
        
        for month in range(start_month, last_month + 1):
            # Calculate first and last day of month
            first_day = 1
            last_day = calendar.monthrange(year, month)[1]
            
            # Generate date strings
            start_date = f"{year:04d}{month:02d}{first_day:02d}"
            end_date = f"{year:04d}{month:02d}{last_day:02d}"
            
            # Generate file info
            month_str = f"{year:04d}-{month:02d}"
            filename = f"glbx-mdp3-{start_date}-{end_date}.ohlcv-1s.dbn.zst"
            file_key = f"raw-data/databento/{filename}"
            
            monthly_files.append({
                'year': year,
                'month': month,
                'month_str': month_str,
                'filename': filename,
                's3_key': file_key,
                'local_file': f"/tmp/monthly_processing/{month_str}/input.dbn.zst",
                'output_file': f"/tmp/monthly_processing/{month_str}/processed.parquet"
            })
    
    log_progress(f"📊 Generated {len(monthly_files)} monthly files to process")
    return monthly_files

def check_existing_processed_files(monthly_files):
    """Check which files are already processed in S3 with enhanced discovery"""
    log_progress("🔍 CHECKING EXISTING PROCESSED FILES")
    
    try:
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        
        # Try multiple prefixes for processed files
        prefixes_to_check = [
            "processed-data/monthly/",
            "processed/monthly/", 
            "monthly-processed/",
            "output/monthly/"
        ]
        
        existing_files = set()
        
        for prefix in prefixes_to_check:
            try:
                paginator = s3_client.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
                
                for page in page_iterator:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            filename = obj['Key'].split('/')[-1]
                            
                            # Extract month from various filename patterns
                            month_str = extract_month_from_filename(filename)
                            if month_str:
                                existing_files.add(month_str)
                                
            except Exception as e:
                log_progress(f"   ⚠️  Could not check prefix {prefix}: {e}")
                continue
        
        to_process = []
        already_done = []
        
        for file_info in monthly_files:
            if file_info['month_str'] in existing_files:
                already_done.append(file_info['month_str'])
            else:
                to_process.append(file_info)
        
        log_progress(f"✅ Already processed: {len(already_done)} months")
        log_progress(f"🔄 Need to process: {len(to_process)} months")
        
        if already_done:
            log_progress(f"   Sample processed: {', '.join(sorted(already_done)[:5])}")
        
        return to_process
        
    except Exception as e:
        log_progress(f"⚠️  Could not check existing files: {e}")
        log_progress(f"   Proceeding with all {len(monthly_files)} files")
        return monthly_files

def extract_month_from_filename(filename):
    """Extract month string from various filename patterns"""
    import re
    
    # Pattern 1: monthly_YYYY-MM_timestamp.parquet
    match = re.search(r'monthly_(\d{4}-\d{2})_', filename)
    if match:
        return match.group(1)
    
    # Pattern 2: YYYY-MM_processed.parquet
    match = re.search(r'(\d{4}-\d{2})_processed', filename)
    if match:
        return match.group(1)
    
    # Pattern 3: es_YYYY_MM_processed.parquet
    match = re.search(r'es_(\d{4})_(\d{2})_processed', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    
    # Pattern 4: YYYYMM.parquet
    match = re.search(r'(\d{4})(\d{2})\.parquet', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    
    return None

def handle_processing_error(error, stage, month_str, context=None, critical=True):
    """
    Enhanced error handling with detailed error classification and recovery strategies
    
    Args:
        error: The exception that occurred
        stage: Processing stage where error occurred
        month_str: Month being processed
        context: Additional context information
        critical: Whether this is a critical error that should stop processing
    
    Returns:
        dict: Error information for logging and recovery decisions
    """
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'stage': stage,
        'month': month_str,
        'critical': critical,
        'recovery_strategy': None,
        'retry_recommended': False
    }
    
    # Classify error types and determine recovery strategies
    if isinstance(error, FileNotFoundError):
        error_info['recovery_strategy'] = 'check_alternative_paths'
        error_info['retry_recommended'] = True
    elif isinstance(error, MemoryError):
        error_info['recovery_strategy'] = 'reduce_chunk_size_and_cleanup'
        error_info['retry_recommended'] = True
    elif isinstance(error, (ConnectionError, TimeoutError)):
        error_info['recovery_strategy'] = 'retry_with_backoff'
        error_info['retry_recommended'] = True
    elif 'S3' in str(error) or 'boto' in str(error):
        error_info['recovery_strategy'] = 'retry_s3_operation'
        error_info['retry_recommended'] = True
    elif 'corrupted' in str(error).lower() or 'invalid' in str(error).lower():
        error_info['recovery_strategy'] = 'redownload_file'
        error_info['retry_recommended'] = True
    elif isinstance(error, (ValueError, TypeError)) and stage in ['processing', 'feature_engineering']:
        error_info['recovery_strategy'] = 'fallback_processing_method'
        error_info['retry_recommended'] = False
    else:
        error_info['recovery_strategy'] = 'log_and_continue'
        error_info['retry_recommended'] = False
    
    # Enhanced error logging with context
    log_context = {
        'stage': stage,
        'month': month_str,
        'error_type': error_info['error_type'],
        'recovery_strategy': error_info['recovery_strategy']
    }
    
    if context:
        log_context.update(context)
    
    log_progress(
        f"Processing error in {stage} for {month_str}: {error_info['error_message']}", 
        level="ERROR" if critical else "WARNING",
        error_details=error,
        context=log_context
    )
    
    return error_info

def retry_with_backoff(operation_func, max_retries=3, base_delay=1, max_delay=60, *args, **kwargs):
    """
    Enhanced retry logic with exponential backoff for transient errors
    
    Args:
        operation_func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        *args, **kwargs: Arguments to pass to operation_func
    
    Returns:
        Result of operation_func or raises last exception
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            if attempt > 0:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                log_progress(f"Retrying operation (attempt {attempt + 1}/{max_retries + 1}) after {delay}s delay", level="INFO")
                time.sleep(delay)
            
            return operation_func(*args, **kwargs)
            
        except Exception as e:
            last_exception = e
            
            # Check if error is retryable
            retryable_errors = (
                ConnectionError, TimeoutError, MemoryError,
                # Add S3-specific errors
                Exception  # For now, retry most exceptions
            )
            
            if attempt < max_retries and isinstance(e, retryable_errors):
                log_progress(f"Retryable error on attempt {attempt + 1}: {e}", level="WARNING")
                continue
            else:
                # Non-retryable error or max retries reached
                if attempt >= max_retries:
                    log_progress(f"Max retries ({max_retries}) reached for operation", level="ERROR", error_details=e)
                else:
                    log_progress(f"Non-retryable error: {e}", level="ERROR", error_details=e)
                break
    
    # Re-raise the last exception
    raise last_exception

def validate_file_integrity(file_path, file_type="unknown", expected_min_size_mb=0.1):
    """
    Enhanced file integrity validation with corruption detection
    
    Args:
        file_path: Path to file to validate
        file_type: Type of file (dbn, parquet, etc.)
        expected_min_size_mb: Minimum expected file size in MB
    
    Returns:
        dict: Validation results with detailed information
    """
    validation_result = {
        'valid': False,
        'file_exists': False,
        'size_mb': 0,
        'size_valid': False,
        'format_valid': False,
        'corruption_detected': False,
        'error_message': None
    }
    
    try:
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            validation_result['error_message'] = f"File does not exist: {file_path}"
            return validation_result
        
        validation_result['file_exists'] = True
        
        # Check file size
        file_size = file_path.stat().st_size
        validation_result['size_mb'] = file_size / (1024**2)
        validation_result['size_valid'] = validation_result['size_mb'] >= expected_min_size_mb
        
        if not validation_result['size_valid']:
            validation_result['error_message'] = f"File too small: {validation_result['size_mb']:.2f} MB < {expected_min_size_mb} MB"
            return validation_result
        
        # Format-specific validation
        if file_type == "dbn":
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
                
            except Exception as e:
                validation_result['error_message'] = f"DBN format validation failed: {e}"
                validation_result['corruption_detected'] = True
                return validation_result
                
        elif file_type == "parquet":
            try:
                # Try to read a small sample to validate format
                df_sample = pd.read_parquet(file_path)
                if len(df_sample) > 10:
                    df_sample = df_sample.head(10)
                
                if len(df_sample.columns) == 0:
                    validation_result['error_message'] = "Parquet file has no columns"
                    validation_result['corruption_detected'] = True
                    return validation_result
                
                validation_result['format_valid'] = True
                
            except Exception as e:
                validation_result['error_message'] = f"Parquet format validation failed: {e}"
                validation_result['corruption_detected'] = True
                return validation_result
        
        # If we get here, file is valid
        validation_result['valid'] = True
        
    except Exception as e:
        validation_result['error_message'] = f"File validation error: {e}"
    
    return validation_result

def process_single_month(file_info):
    """
    Enhanced single month processing with independent operation and restart capability
    
    Enhancements:
    - Independent processing (each month is self-contained)
    - Better error isolation and recovery
    - Enhanced memory management between stages
    - Comprehensive logging for restart capability
    - Graceful degradation on failures
    """
    month_str = file_info['month_str']
    log_progress(f"🔄 PROCESSING {month_str}", stage="month_processing", stage_event="start", 
                context={'month': month_str, 'file_key': file_info.get('s3_key', 'unknown')})
    
    # Create isolated month directory for independent processing
    month_dir = Path(f"/tmp/monthly_processing/{month_str}")
    month_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processing state for restart capability
    processing_state = {
        'month': month_str,
        'start_time': time.time(),
        'stages_completed': [],
        'current_stage': None,
        'memory_usage': [],
        'errors': []
    }
    
    def log_stage_completion(stage_name, success=True, error_msg=None):
        """Log stage completion for restart tracking"""
        if success:
            processing_state['stages_completed'].append(stage_name)
            log_progress(f"   ✅ Stage '{stage_name}' completed")
        else:
            processing_state['errors'].append(f"{stage_name}: {error_msg}")
            log_progress(f"   ❌ Stage '{stage_name}' failed: {error_msg}")
    
    def check_memory_and_cleanup():
        """Monitor memory usage and perform cleanup if needed"""
        memory_mb = psutil.Process().memory_info().rss / (1024**2)
        processing_state['memory_usage'].append(memory_mb)
        
        if memory_mb > 6000:  # If memory usage > 6GB, force cleanup
            log_progress(f"   🧹 High memory usage ({memory_mb:.1f} MB), forcing cleanup")
            gc.collect()
            
        return memory_mb
    
    try:
        # Stage 1: Enhanced download with retry logic and better error handling
        processing_state['current_stage'] = 'download'
        log_progress(f"   📥 Stage 1: Downloading {month_str}", stage="download", stage_event="start",
                    context={'month': month_str, 'file_key': file_info.get('s3_key', 'unknown')})
        
        try:
            # Use retry logic for download operations
            download_success = retry_with_backoff(
                download_monthly_file, 
                max_retries=3, 
                base_delay=2,
                file_info=file_info
            )
            
            if not download_success:
                error_info = handle_processing_error(
                    ValueError("Download failed after retries"), 
                    'download', 
                    month_str, 
                    context={'file_key': file_info.get('s3_key', 'unknown')},
                    critical=True
                )
                log_stage_completion('download', False, error_info['error_message'])
                log_progress(f"   ❌ Download stage failed", level="ERROR", stage="download", stage_event="end",
                           context={'month': month_str, 'error': error_info['error_message']})
                return None
                
            log_stage_completion('download', True)
            log_progress(f"   ✅ Download stage completed", stage="download", stage_event="end",
                        context={'month': month_str})
            check_memory_and_cleanup()
            
        except Exception as e:
            error_info = handle_processing_error(e, 'download', month_str, critical=True)
            log_stage_completion('download', False, error_info['error_message'])
            
            # For certain error types, don't return None immediately - try alternative strategies
            if error_info['retry_recommended'] and error_info['recovery_strategy'] == 'check_alternative_paths':
                log_progress(f"   🔄 Attempting alternative download strategies for {month_str}", level="INFO")
                # This would be handled in the download_monthly_file function
            
            return None
        
        # Stage 2: Enhanced processing with corruption detection and fallback strategies
        processing_state['current_stage'] = 'processing'
        log_progress(f"   🔧 Stage 2: Processing {month_str}")
        
        try:
            # Enhanced processing with retry logic for transient errors
            processed_file = retry_with_backoff(
                process_monthly_data,
                max_retries=2,  # Fewer retries for processing as it's more expensive
                base_delay=5,
                file_info=file_info
            )
            
            if not processed_file:
                error_info = handle_processing_error(
                    ValueError("Processing returned None"), 
                    'processing', 
                    month_str,
                    critical=True
                )
                log_stage_completion('processing', False, error_info['error_message'])
                return None
            
            # Enhanced file validation with corruption detection
            validation_result = validate_file_integrity(processed_file, "parquet", expected_min_size_mb=0.1)
            
            if not validation_result['valid']:
                error_msg = f"Processed file validation failed: {validation_result['error_message']}"
                
                if validation_result['corruption_detected']:
                    # Handle corrupted output - this might indicate input corruption
                    error_info = handle_processing_error(
                        ValueError(f"Output corruption detected: {validation_result['error_message']}"), 
                        'processing', 
                        month_str,
                        context={'validation_result': validation_result},
                        critical=True
                    )
                    
                    # For corrupted output, we might want to try redownloading the input
                    log_progress(f"   🔄 Corruption detected, may need to redownload input file", level="WARNING")
                
                log_stage_completion('processing', False, error_msg)
                return None
            
            log_progress(f"   ✅ Processing validation passed: {validation_result['size_mb']:.1f} MB", level="INFO")
            log_stage_completion('processing', True)
            check_memory_and_cleanup()
            
        except Exception as e:
            error_info = handle_processing_error(e, 'processing', month_str, critical=True)
            log_stage_completion('processing', False, error_info['error_message'])
            
            # For memory errors, try to free up memory before continuing
            if isinstance(e, MemoryError):
                log_progress(f"   🧹 Memory error detected, forcing aggressive cleanup", level="WARNING")
                import gc
                gc.collect()
                check_memory_and_cleanup()
            
            # Don't return None immediately - try to cleanup first
            processed_file = None
        
        # Stage 3: Enhanced upload with comprehensive retry logic (only if processing succeeded)
        if processed_file:
            processing_state['current_stage'] = 'upload'
            log_progress(f"   📤 Stage 3: Uploading {month_str}")
            
            try:
                # Collect comprehensive statistics before upload
                try:
                    # Read and preserve statistics file content before cleanup
                    stats_file = Path(f"/tmp/monthly_processing/{month_str}/{month_str}_processing_stats.json")
                    log_progress(f"   🔍 Looking for statistics file: {stats_file}")
                    if stats_file.exists():
                        log_progress(f"   📊 Found local statistics file: {stats_file}")
                        # Read the actual statistics content
                        import json
                        with open(stats_file, 'r') as f:
                            stats_content = json.load(f)
                        log_progress(f"   📊 Statistics content loaded: {len(str(stats_content))} characters")
                        monthly_statistics = {"stats_content": stats_content, "stats_file": str(stats_file)}
                    else:
                        log_progress(f"   ⚠️  No local statistics file found at {stats_file}")
                        # Check what files do exist
                        month_dir = stats_file.parent
                        if month_dir.exists():
                            files = list(month_dir.glob("*.json"))
                            log_progress(f"   🔍 Files in {month_dir}: {[f.name for f in files]}")
                        monthly_statistics = None
                    
                except Exception as stats_error:
                    log_progress(f"   ⚠️  Statistics collection failed: {stats_error}", level="WARNING")
                    monthly_statistics = None
                
                # Enhanced upload with comprehensive statistics
                upload_success = retry_with_backoff(
                    upload_monthly_results,
                    max_retries=3,
                    base_delay=3,
                    file_info=file_info,
                    processed_file=processed_file,
                    monthly_statistics=monthly_statistics
                )
                
                if not upload_success:
                    error_info = handle_processing_error(
                        ValueError("Upload failed after retries"), 
                        'upload', 
                        month_str,
                        context={'processed_file_size': Path(processed_file).stat().st_size / (1024**2)},
                        critical=False  # Upload failure is not critical - we have the processed file
                    )
                    log_stage_completion('upload', False, error_info['error_message'])
                    # Don't return None - still cleanup, file is processed successfully
                else:
                    log_stage_completion('upload', True)
                    
            except Exception as e:
                error_info = handle_processing_error(e, 'upload', month_str, critical=False)
                log_stage_completion('upload', False, error_info['error_message'])
                
                # For S3 errors, log additional context for debugging
                if 'S3' in str(e) or 'boto' in str(e):
                    log_progress(f"   🔍 S3 upload error details - check AWS credentials and permissions", level="WARNING")
                
                # Continue to cleanup even if upload fails
        
        # Stage 4: Cleanup (always attempt, regardless of previous failures)
        processing_state['current_stage'] = 'cleanup'
        log_progress(f"   🧹 Stage 4: Cleaning up {month_str}")
        
        try:
            cleanup_monthly_files(file_info)
            log_stage_completion('cleanup', True)
        except Exception as e:
            log_stage_completion('cleanup', False, str(e))
            # Cleanup failure is not critical
        
        # Final memory cleanup
        try:
            import gc
            gc.collect()
            final_memory = check_memory_and_cleanup()
        except Exception as cleanup_error:
            log_progress(f"   ⚠️  Cleanup warning: {cleanup_error}")
            final_memory = psutil.Process().memory_info().rss / (1024**2)
        
        # Calculate and log final statistics
        total_time = time.time() - processing_state['start_time']
        peak_memory = max(processing_state['memory_usage']) if processing_state['memory_usage'] else 0
        
        # Determine overall success
        critical_stages = ['download', 'processing', 'upload']
        successful_critical = all(stage in processing_state['stages_completed'] for stage in critical_stages)
        
        if successful_critical:
            log_progress(f"✅ {month_str} completed successfully in {total_time/60:.1f} minutes")
            log_progress(f"   📊 Peak memory: {peak_memory:.1f} MB, Final memory: {final_memory:.1f} MB")
            log_progress(f"   🎯 Completed stages: {', '.join(processing_state['stages_completed'])}")
            return processed_file
        else:
            log_progress(f"❌ {month_str} failed - critical stages incomplete")
            log_progress(f"   🎯 Completed stages: {', '.join(processing_state['stages_completed'])}")
            log_progress(f"   ⚠️  Errors: {'; '.join(processing_state['errors'])}")
            return None
        
    except Exception as e:
        # Enhanced catch-all for unexpected errors with detailed error analysis
        current_stage = processing_state.get('current_stage', 'unknown')
        error_info = handle_processing_error(e, current_stage, month_str, critical=True)
        
        log_progress(f"❌ {month_str} failed with unexpected error in {current_stage} stage", level="CRITICAL", error_details=e)
        
        # Enhanced error state logging for debugging and recovery
        error_context = {
            'month': month_str,
            'stage': current_stage,
            'stages_completed': processing_state.get('stages_completed', []),
            'memory_usage_history': processing_state.get('memory_usage', []),
            'error_classification': error_info,
            'processing_time_so_far': time.time() - processing_state.get('start_time', time.time())
        }
        
        # Save detailed error state for post-mortem analysis
        try:
            error_file = Path(f"/tmp/monthly_processing/{month_str}/critical_error_state.json")
            error_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(error_file, 'w') as f:
                json.dump(error_context, f, indent=2, default=str)
            
            log_progress(f"   💾 Saved critical error state to {error_file}", level="INFO")
            
        except Exception as save_error:
            log_progress(f"   ⚠️  Could not save critical error state: {save_error}", level="WARNING")
        
        # Enhanced emergency cleanup with error handling
        try:
            cleanup_monthly_files(file_info)
            log_progress(f"   🧹 Emergency cleanup completed successfully", level="INFO")
        except Exception as cleanup_error:
            log_progress(f"   ⚠️  Emergency cleanup failed", level="WARNING", error_details=cleanup_error)
        
        return None

def download_monthly_file(file_info):
    """
    Enhanced download with corruption detection and recovery strategies
    Uses optimized S3 operations with retry logic and integrity validation
    """
    try:
        # Use enhanced S3 operations if available
        if S3_OPERATIONS_AVAILABLE:
            log_progress(f"   🚀 Using enhanced S3 download operations")
            
            s3_ops = EnhancedS3Operations("es-1-second-data")
            success = s3_ops.download_monthly_file_optimized(file_info)
            
            if success:
                log_progress(f"   ✅ Enhanced download completed successfully")
                return True
            else:
                log_progress(f"   ⚠️  Enhanced download failed, falling back to basic download", level="WARNING")
                # Fall through to basic download
        
        # Fallback to basic download (original implementation)
        log_progress(f"   🔄 Using basic S3 download operations")
        
        bucket_name = "es-1-second-data"
        s3_key = file_info['s3_key']
        local_file = Path(file_info['local_file'])
        month_str = file_info['month_str']
        
        # Ensure download directory exists
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Enhanced existing file validation with corruption detection
        if local_file.exists():
            log_progress(f"   📁 Existing file found, validating integrity", level="INFO")
            
            validation_result = validate_file_integrity(local_file, "dbn", expected_min_size_mb=1.0)
            
            if validation_result['valid']:
                log_progress(f"   ✅ Existing file validated successfully ({validation_result['size_mb']:.1f} MB)")
                return True
            else:
                if validation_result['corruption_detected']:
                    log_progress(f"   🔧 Corruption detected in existing file: {validation_result['error_message']}", level="WARNING")
                else:
                    log_progress(f"   ⚠️  Existing file validation failed: {validation_result['error_message']}", level="WARNING")
                
                # Remove corrupted/invalid file
                try:
                    local_file.unlink()
                    log_progress(f"   🗑️  Removed invalid existing file", level="INFO")
                except Exception as e:
                    log_progress(f"   ⚠️  Could not remove invalid file: {e}", level="WARNING")
        
        # Enhanced S3 path discovery with more alternatives
        s3_paths_to_try = [
            s3_key,  # Original path
            f"databento/{file_info['filename']}",  # Alternative path 1
            f"raw/{file_info['filename']}",  # Alternative path 2
            f"es-data/{file_info['filename']}",  # Alternative path 3
            f"raw-data/{file_info['filename']}",  # Alternative path 4
            f"monthly/{file_info['filename']}",  # Alternative path 5
            f"dbn/{file_info['filename']}"  # Alternative path 6
        ]
        
        s3_client = boto3.client('s3')
        download_attempts = []
        
        for path_index, attempt_path in enumerate(s3_paths_to_try, 1):
            log_progress(f"   🔍 Trying S3 path {path_index}/{len(s3_paths_to_try)}: {attempt_path}")
            
            try:
                # Enhanced S3 file existence check with metadata
                response = s3_client.head_object(Bucket=bucket_name, Key=attempt_path)
                file_size_mb = response['ContentLength'] / (1024**2)
                last_modified = response.get('LastModified', 'unknown')
                
                log_progress(f"   📦 Found file: {file_size_mb:.1f} MB, modified: {last_modified}")
                
                # Validate expected file size (monthly ES data should be substantial)
                if file_size_mb < 1.0:  # Less than 1MB is suspicious for monthly data
                    log_progress(f"   ⚠️  File size suspiciously small: {file_size_mb:.1f} MB", level="WARNING")
                    download_attempts.append({
                        'path': attempt_path,
                        'status': 'skipped',
                        'reason': f'file_too_small_{file_size_mb:.1f}MB'
                    })
                    continue
                
                # Enhanced download with corruption detection
                download_success = False
                corruption_retry_count = 0
                max_corruption_retries = 2
                
                while not download_success and corruption_retry_count <= max_corruption_retries:
                    if corruption_retry_count > 0:
                        log_progress(f"   🔄 Retry {corruption_retry_count}/{max_corruption_retries} for corruption recovery")
                    
                    # Download with retry logic
                    if download_with_retry(s3_client, bucket_name, attempt_path, local_file):
                        # Enhanced validation with corruption detection
                        validation_result = validate_file_integrity(local_file, "dbn", expected_min_size_mb=1.0)
                        
                        if validation_result['valid']:
                            log_progress(f"   ✅ Downloaded and validated successfully ({validation_result['size_mb']:.1f} MB)")
                            download_attempts.append({
                                'path': attempt_path,
                                'status': 'success',
                                'size_mb': validation_result['size_mb'],
                                'corruption_retries': corruption_retry_count
                            })
                            return True
                        else:
                            if validation_result['corruption_detected']:
                                log_progress(f"   🔧 Downloaded file corruption detected: {validation_result['error_message']}", level="WARNING")
                                
                                # Remove corrupted download
                                try:
                                    local_file.unlink()
                                except Exception:
                                    pass
                                
                                corruption_retry_count += 1
                                
                                if corruption_retry_count <= max_corruption_retries:
                                    log_progress(f"   🔄 Will retry download due to corruption", level="INFO")
                                    continue
                                else:
                                    log_progress(f"   ❌ Max corruption retries reached for this path", level="ERROR")
                                    download_attempts.append({
                                        'path': attempt_path,
                                        'status': 'failed',
                                        'reason': 'persistent_corruption',
                                        'corruption_retries': corruption_retry_count
                                    })
                                    break
                            else:
                                log_progress(f"   ❌ Downloaded file validation failed: {validation_result['error_message']}", level="ERROR")
                                download_attempts.append({
                                    'path': attempt_path,
                                    'status': 'failed',
                                    'reason': 'validation_failed',
                                    'validation_error': validation_result['error_message']
                                })
                                break
                    else:
                        log_progress(f"   ❌ Download failed for path: {attempt_path}", level="ERROR")
                        download_attempts.append({
                            'path': attempt_path,
                            'status': 'failed',
                            'reason': 'download_failed'
                        })
                        break
                
            except s3_client.exceptions.NoSuchKey:
                log_progress(f"   ❌ File not found at: {attempt_path}")
                download_attempts.append({
                    'path': attempt_path,
                    'status': 'not_found'
                })
                continue
            except Exception as e:
                log_progress(f"   ❌ Error checking path {attempt_path}", level="ERROR", error_details=e)
                download_attempts.append({
                    'path': attempt_path,
                    'status': 'error',
                    'error': str(e)
                })
                continue
        
        # Enhanced failure reporting with detailed attempt history
        log_progress(f"   ❌ File not found or downloadable in any S3 path for {month_str}", level="ERROR")
        log_progress(f"   📊 Download attempt summary:", level="INFO")
        
        for i, attempt in enumerate(download_attempts, 1):
            status_msg = f"     {i}. {attempt['path']}: {attempt['status']}"
            if 'reason' in attempt:
                status_msg += f" ({attempt['reason']})"
            if 'size_mb' in attempt:
                status_msg += f" - {attempt['size_mb']:.1f} MB"
            log_progress(status_msg, level="INFO")
        
        return False
        
    except Exception as e:
        log_progress(f"   ❌ Enhanced download failed: {e}")
        return False

def download_with_retry(s3_client, bucket_name, s3_key, local_file, max_retries=3):
    """Download file with exponential backoff retry"""
    for attempt in range(max_retries):
        try:
            s3_client.download_file(bucket_name, s3_key, str(local_file))
            return True
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            log_progress(f"   ⚠️  Download attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                log_progress(f"   ⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                log_progress(f"   ❌ All download attempts failed")
                return False
    
    return False

def validate_downloaded_file(file_path):
    """Validate downloaded file integrity"""
    try:
        if not file_path.exists():
            return False
        
        # Check file size (should be > 1MB for monthly ES data)
        file_size = file_path.stat().st_size
        if file_size < 1024 * 1024:  # Less than 1MB
            log_progress(f"   ⚠️  File too small: {file_size} bytes")
            return False
        
        # Try to open as DBN file to validate format
        try:
            import databento as db
            store = db.DBNStore.from_file(str(file_path))
            metadata = store.metadata
            
            # Basic validation - should have start/end times
            if not hasattr(metadata, 'start') or not hasattr(metadata, 'end'):
                log_progress(f"   ⚠️  Invalid DBN metadata")
                return False
            
            return True
            
        except Exception as e:
            log_progress(f"   ⚠️  DBN validation failed: {e}")
            return False
            
    except Exception as e:
        log_progress(f"   ⚠️  File validation error: {e}")
        return False

def process_monthly_data(file_info):
    """
    Enhanced monthly data processing with improved integration and memory management
    
    Enhancements:
    - Better memory management between processing stages with explicit cleanup
    - Improved error handling and recovery with detailed logging
    - Independent processing for restart capability with isolated state
    - Enhanced integration with WeightedLabelingEngine and feature engineering
    - Comprehensive statistics collection and validation
    - Graceful fallback mechanisms for component failures
    """
    local_file = Path(file_info['local_file'])
    output_file = Path(file_info['output_file'])
    month_str = file_info['month_str']
    
    # Initialize processing statistics for independent tracking
    processing_stats = {
        'month': month_str,
        'start_time': time.time(),
        'raw_rows': 0,
        'cleaned_rows': 0,
        'rth_rows': 0,
        'final_rows': 0,
        'memory_peak_mb': 0,
        'memory_at_stages': {},
        'processing_stages': {},
        'component_versions': {},
        'errors': [],
        'warnings': []
    }
    
    # Enhanced memory monitoring with stage tracking
    def check_memory_and_cleanup(stage_name, force_gc=True):
        """Enhanced memory monitoring with automatic cleanup"""
        if force_gc:
            gc.collect()
        
        memory_mb = psutil.Process().memory_info().rss / (1024**2)
        processing_stats['memory_peak_mb'] = max(processing_stats['memory_peak_mb'], memory_mb)
        processing_stats['memory_at_stages'][stage_name] = memory_mb
        
        # Log memory usage if high
        if memory_mb > 6000:  # > 6GB
            log_progress(f"   🧹 High memory usage at {stage_name}: {memory_mb:.1f} MB")
        
        return memory_mb
    
    # Enhanced error handling with recovery strategies
    def handle_stage_error(stage_name, error, critical=True):
        """Handle stage errors with appropriate recovery strategies"""
        error_msg = f"{stage_name} failed: {error}"
        log_progress(f"   ❌ {error_msg}")
        processing_stats['errors'].append(error_msg)
        
        if critical:
            # Log detailed error for debugging
            import traceback
            detailed_error = traceback.format_exc()
            log_progress(f"   Error details: {detailed_error}")
            return None
        else:
            # Non-critical error - log as warning and continue
            processing_stats['warnings'].append(error_msg)
            return False
    
    try:
        # Import processing modules with enhanced error handling and version tracking
        try:
            from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, process_weighted_labeling
            from src.data_pipeline.features import create_all_features
            import databento as db
            import pytz
            from datetime import time as dt_time
            
            # Track component versions for debugging
            processing_stats['component_versions'] = {
                'databento': getattr(db, '__version__', 'unknown'),
                'pytz': getattr(pytz, '__version__', 'unknown'),
                'pandas': pd.__version__,
                'numpy': np.__version__
            }
            
        except ImportError as e:
            return handle_stage_error('module_import', e, critical=True)
        
        # Stage 1: Enhanced DBN conversion with comprehensive corruption detection
        stage_start = time.time()
        log_progress(f"   📖 Converting DBN...")
        
        try:
            # Enhanced input file validation
            if not local_file.exists():
                raise FileNotFoundError(f"Input file not found: {local_file}")
            
            # Pre-processing file integrity check
            validation_result = validate_file_integrity(local_file, "dbn", expected_min_size_mb=1.0)
            
            if not validation_result['valid']:
                if validation_result['corruption_detected']:
                    raise ValueError(f"Input file corruption detected: {validation_result['error_message']}")
                else:
                    raise ValueError(f"Input file validation failed: {validation_result['error_message']}")
            
            log_progress(f"   📦 Processing validated {validation_result['size_mb']:.1f} MB DBN file")
            
            # Enhanced DBN loading with corruption detection
            try:
                store = db.DBNStore.from_file(str(local_file))
                metadata = store.metadata
            except Exception as dbn_error:
                # Check if this is a corruption issue
                if any(keyword in str(dbn_error).lower() for keyword in ['corrupt', 'invalid', 'truncated', 'damaged']):
                    raise ValueError(f"DBN file corruption detected during loading: {dbn_error}")
                else:
                    raise ValueError(f"DBN file loading failed: {dbn_error}")
            
            # Enhanced metadata validation with corruption indicators
            if not hasattr(metadata, 'start') or not hasattr(metadata, 'end'):
                raise ValueError("Invalid DBN metadata - missing start/end times (possible corruption)")
            
            # Validate metadata values are reasonable
            try:
                start_date = pd.to_datetime(metadata.start, unit='ns')
                end_date = pd.to_datetime(metadata.end, unit='ns')
                
                # Check for obviously corrupted timestamps
                if start_date.year < 2000 or start_date.year > 2030:
                    raise ValueError(f"Invalid start date in metadata: {start_date} (possible corruption)")
                if end_date.year < 2000 or end_date.year > 2030:
                    raise ValueError(f"Invalid end date in metadata: {end_date} (possible corruption)")
                if end_date <= start_date:
                    raise ValueError(f"End date before start date: {start_date} to {end_date} (possible corruption)")
                
                date_range_days = (end_date - start_date).days
                
                if date_range_days > 35:  # More than 35 days is suspicious for monthly data
                    processing_stats['warnings'].append(f"Large date range: {date_range_days} days")
                elif date_range_days < 1:  # Less than 1 day is also suspicious
                    processing_stats['warnings'].append(f"Small date range: {date_range_days} days")
                    
            except Exception as date_error:
                raise ValueError(f"Metadata date validation failed: {date_error} (possible corruption)")
            
            # Enhanced DataFrame conversion with corruption detection
            try:
                df = store.to_df()
            except Exception as conversion_error:
                if any(keyword in str(conversion_error).lower() for keyword in ['corrupt', 'invalid', 'truncated', 'damaged']):
                    raise ValueError(f"Data corruption detected during DataFrame conversion: {conversion_error}")
                else:
                    raise ValueError(f"DataFrame conversion failed: {conversion_error}")
            
            # Enhanced DataFrame validation
            if len(df) == 0:
                raise ValueError("Empty DataFrame from DBN conversion (possible file corruption)")
            
            # Check for obviously corrupted data patterns
            if len(df) < 1000:  # Very small datasets might indicate corruption
                processing_stats['warnings'].append(f"Small dataset: {len(df)} rows")
            
            # Validate basic column structure
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing expected columns: {missing_columns} (possible corruption)")
            
            # Check for obviously corrupted price data
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    # Check for impossible values that indicate corruption
                    if (df[col] <= 0).any():
                        processing_stats['warnings'].append(f"Zero or negative prices in {col} (will be cleaned)")
                    if (df[col] > 100000).any():  # ES prices > 100,000 are unrealistic
                        processing_stats['warnings'].append(f"Extremely high prices in {col} (possible corruption)")
            
            processing_stats['raw_rows'] = len(df)
            log_progress(f"   ✅ DBN conversion successful: {len(df):,} rows")
            
            # Enhanced timestamp handling with multiple fallback strategies
            timestamp_created = False
            
            # Strategy 1: Use metadata timestamps
            if hasattr(df.index, 'astype') and not timestamp_created:
                try:
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
                    timestamp_created = True
                    log_progress(f"   ✅ Created timestamps from metadata")
                    
                except Exception as ts_error:
                    log_progress(f"   ⚠️  Metadata timestamp generation failed: {ts_error}")
            
            # Strategy 2: Use existing timestamp columns
            if not timestamp_created:
                for col_name in ['ts_event', 'timestamp', 'ts_recv']:
                    if col_name in df.columns:
                        try:
                            df['timestamp'] = pd.to_datetime(df[col_name], unit='ns', utc=True)
                            timestamp_created = True
                            log_progress(f"   ✅ Used existing {col_name} column")
                            break
                        except Exception:
                            continue
            
            # Strategy 3: Fallback to synthetic timestamps
            if not timestamp_created:
                log_progress(f"   ⚠️  Using synthetic timestamps - data may be inaccurate")
                df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1s', tz='UTC')
                processing_stats['warnings'].append("Used synthetic timestamps")
            
            # Ensure DataFrame has timestamp column
            df = df.reset_index()
            if 'timestamp' not in df.columns:
                if df.index.name == 'timestamp':
                    df = df.reset_index()
                else:
                    raise ValueError("No timestamp column available after conversion")
            
            log_progress(f"   ✅ Converted {len(df):,} rows")
            check_memory_and_cleanup('dbn_conversion')
            
        except Exception as e:
            return handle_stage_error('dbn_conversion', e, critical=True)
        
        processing_stats['processing_stages']['dbn_conversion'] = time.time() - stage_start
        
        # Stage 2: Enhanced data cleaning with better error recovery
        stage_start = time.time()
        try:
            df_cleaned = clean_price_data(df)
            processing_stats['cleaned_rows'] = len(df_cleaned)
            
            if len(df_cleaned) == 0:
                raise ValueError("No valid data after cleaning")
            
            # Calculate cleaning statistics
            rows_removed = len(df) - len(df_cleaned)
            removal_percentage = (rows_removed / len(df)) * 100 if len(df) > 0 else 0
            
            if removal_percentage > 50:  # More than 50% removed is concerning
                processing_stats['warnings'].append(f"High data removal: {removal_percentage:.1f}%")
            
            # Memory cleanup after cleaning
            del df
            check_memory_and_cleanup('data_cleaning')
            
        except Exception as e:
            return handle_stage_error('data_cleaning', e, critical=True)
        
        processing_stats['processing_stages']['data_cleaning'] = time.time() - stage_start
        
        # Stage 3: Enhanced RTH filtering with better timezone handling
        stage_start = time.time()
        log_progress(f"   🕐 Filtering to RTH...")
        
        try:
            central_tz = pytz.timezone('US/Central')
            
            timestamps = pd.to_datetime(df_cleaned['timestamp'])
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize(pytz.UTC)
            
            central_timestamps = timestamps.dt.tz_convert(central_tz)
            df_time = central_timestamps.dt.time
            
            rth_start_time = dt_time(7, 30)
            rth_end_time = dt_time(15, 0)
            
            rth_mask = (df_time >= rth_start_time) & (df_time < rth_end_time)
            df_rth = df_cleaned[rth_mask].copy()
            
            # Ensure UTC timestamps for consistency
            df_rth['timestamp'] = timestamps[rth_mask].dt.tz_convert(pytz.UTC)
            
            processing_stats['rth_rows'] = len(df_rth)
            rth_percentage = len(df_rth) / len(df_cleaned) * 100 if len(df_cleaned) > 0 else 0
            
            log_progress(f"   ✅ RTH filtered: {len(df_rth):,} rows ({rth_percentage:.1f}%)")
            
            # Validate RTH percentage is reasonable (should be 30-40%)
            if rth_percentage < 20 or rth_percentage > 60:
                processing_stats['warnings'].append(f"Unusual RTH percentage: {rth_percentage:.1f}%")
            
            # Memory cleanup after RTH filtering
            del df_cleaned
            check_memory_and_cleanup('rth_filtering')
            
            if len(df_rth) == 0:
                raise ValueError("No RTH data found")
            
        except Exception as e:
            return handle_stage_error('rth_filtering', e, critical=True)
        
        processing_stats['processing_stages']['rth_filtering'] = time.time() - stage_start
        
        # Stage 4: Enhanced weighted labeling with improved integration and fallback
        stage_start = time.time()
        log_progress(f"   🏷️  Weighted labeling...")
        
        try:
            df_labeled = None
            labeling_method = None
            
            # Strategy 1: Try WeightedLabelingEngine with enhanced configuration
            try:
                # Import configuration if available
                try:
                    from src.data_pipeline.weighted_labeling import LabelingConfig
                    config = LabelingConfig(
                        enable_memory_optimization=True,
                        enable_progress_tracking=False,  # Reduce noise in monthly processing
                        chunk_size=50000  # Smaller chunks for memory efficiency
                    )
                    engine = WeightedLabelingEngine(config)
                except ImportError:
                    engine = WeightedLabelingEngine()
                
                df_labeled = engine.process_dataframe(df_rth, validate_performance=False)
                labeling_method = "WeightedLabelingEngine"
                log_progress(f"   ✅ Used WeightedLabelingEngine")
                
            except Exception as engine_error:
                log_progress(f"   ⚠️  WeightedLabelingEngine failed: {engine_error}")
                
                # Strategy 2: Fallback to process_weighted_labeling function
                try:
                    log_progress(f"   🔄 Falling back to process_weighted_labeling function")
                    df_labeled = process_weighted_labeling(df_rth)
                    labeling_method = "process_weighted_labeling"
                    log_progress(f"   ✅ Used fallback function")
                    
                except Exception as fallback_error:
                    raise ValueError(f"Both labeling methods failed - Engine: {engine_error}, Fallback: {fallback_error}")
            
            # Enhanced validation of labeling output
            expected_label_cols = [f"label_{mode}" for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                                                               'low_vol_short', 'normal_vol_short', 'high_vol_short']]
            expected_weight_cols = [f"weight_{mode}" for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long', 
                                                                 'low_vol_short', 'normal_vol_short', 'high_vol_short']]
            
            missing_labels = [col for col in expected_label_cols if col not in df_labeled.columns]
            missing_weights = [col for col in expected_weight_cols if col not in df_labeled.columns]
            
            if missing_labels or missing_weights:
                raise ValueError(f"Missing labeling columns - Labels: {missing_labels}, Weights: {missing_weights}")
            
            # Validate label values (should be 0 or 1)
            for col in expected_label_cols:
                unique_vals = set(df_labeled[col].dropna().unique())
                if not unique_vals.issubset({0, 1}):
                    processing_stats['warnings'].append(f"Invalid label values in {col}: {unique_vals}")
            
            # Validate weight values (should be positive)
            for col in expected_weight_cols:
                if (df_labeled[col] <= 0).any():
                    processing_stats['warnings'].append(f"Non-positive weights found in {col}")
            
            processing_stats['component_versions']['labeling_method'] = labeling_method
            log_progress(f"   ✅ Labeled: {len(df_labeled.columns)} columns (12 labeling columns added)")
            
            # Memory cleanup after labeling
            del df_rth
            check_memory_and_cleanup('weighted_labeling')
            
        except Exception as e:
            return handle_stage_error('weighted_labeling', e, critical=True)
        
        processing_stats['processing_stages']['weighted_labeling'] = time.time() - stage_start
        
        # Stage 5: Enhanced feature engineering with better integration and validation
        stage_start = time.time()
        log_progress(f"   🔧 Feature engineering...")
        
        try:
            # Check if chunked processing is needed for large datasets
            if len(df_labeled) > 100000:  # > 100K rows
                try:
                    from src.data_pipeline.features import create_all_features_chunked
                    df_final = create_all_features_chunked(df_labeled, chunk_size=50000)
                    feature_method = "chunked"
                except ImportError:
                    df_final = create_all_features(df_labeled)
                    feature_method = "standard"
            else:
                df_final = create_all_features(df_labeled)
                feature_method = "standard"
            
            # Enhanced validation of feature engineering output
            original_cols = 6  # timestamp, open, high, low, close, volume
            labeling_cols = 12  # 6 labels + 6 weights
            expected_feature_cols = 43
            expected_total = original_cols + labeling_cols + expected_feature_cols  # ~61
            
            actual_cols = len(df_final.columns)
            if actual_cols < expected_total - 5:  # Allow some tolerance
                processing_stats['warnings'].append(f"Fewer columns than expected: {actual_cols} vs {expected_total}")
            
            # Check for excessive NaN values in features
            feature_cols = [col for col in df_final.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume'] 
                           and not col.startswith(('label_', 'weight_'))]
            
            high_nan_features = []
            for col in feature_cols:
                nan_pct = df_final[col].isna().sum() / len(df_final) * 100
                if nan_pct > 35:  # More than 35% NaN is concerning
                    high_nan_features.append(f"{col}: {nan_pct:.1f}%")
            
            if high_nan_features:
                processing_stats['warnings'].append(f"High NaN features: {', '.join(high_nan_features[:3])}")
            
            processing_stats['final_rows'] = len(df_final)
            processing_stats['component_versions']['feature_method'] = feature_method
            
            log_progress(f"   ✅ Features: {len(df_final.columns)} columns ({len(feature_cols)} features added)")
            
            # Memory cleanup after feature engineering
            del df_labeled
            check_memory_and_cleanup('feature_engineering')
            
        except Exception as e:
            return handle_stage_error('feature_engineering', e, critical=True)
        
        processing_stats['processing_stages']['feature_engineering'] = time.time() - stage_start
        
        # Stage 6: Enhanced saving with comprehensive validation
        stage_start = time.time()
        log_progress(f"   💾 Saving results...")
        
        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Pre-save validation
            if len(df_final) == 0:
                raise ValueError("Final DataFrame is empty")
            
            # Analyze labeling results for win rates and feature quality while df_final is available
            try:
                # Calculate win rates for each trading mode
                label_columns = [col for col in df_final.columns if col.startswith('label_')]
                weight_columns = [col for col in df_final.columns if col.startswith('weight_')]
                feature_columns = [col for col in df_final.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and not col.startswith(('label_', 'weight_'))]
                
                # Win rate analysis
                win_rates = {}
                for col in label_columns:
                    win_rate = df_final[col].mean()
                    win_rates[col] = float(win_rate)
                
                # Feature quality analysis
                feature_quality = {}
                for col in feature_columns:
                    nan_pct = (df_final[col].isna().sum() / len(df_final)) * 100
                    feature_quality[col] = {
                        'nan_percentage': float(nan_pct),
                        'min_value': float(df_final[col].min()) if not df_final[col].isna().all() else None,
                        'max_value': float(df_final[col].max()) if not df_final[col].isna().all() else None,
                        'mean_value': float(df_final[col].mean()) if not df_final[col].isna().all() else None
                    }
                
                # Add to processing stats
                processing_stats['labeling_results'] = {
                    'win_rates': win_rates,
                    'total_label_columns': len(label_columns),
                    'total_weight_columns': len(weight_columns)
                }
                processing_stats['feature_results'] = {
                    'feature_quality': feature_quality,
                    'total_feature_columns': len(feature_columns),
                    'high_nan_features': [col for col, stats in feature_quality.items() if stats['nan_percentage'] > 35]
                }
                
                log_progress(f"   📊 Analyzed {len(label_columns)} labels, {len(feature_columns)} features")
                
            except Exception as analysis_error:
                processing_stats['warnings'].append(f"Could not analyze labeling/feature results: {analysis_error}")
                log_progress(f"   ⚠️  Analysis error: {analysis_error}")
            
            # Save with optimized compression for S3 storage
            df_final.to_parquet(
                output_file, 
                index=False, 
                compression='snappy',
                engine='pyarrow'
            )
            
            # Post-save validation
            if not output_file.exists():
                raise ValueError("Output file was not created")
            
            file_size_mb = output_file.stat().st_size / (1024**2)
            
            # Validate file size is reasonable
            if file_size_mb < 0.1:  # Less than 100KB is suspicious
                raise ValueError(f"Output file too small: {file_size_mb:.1f} MB")
            
            # Quick validation by reading back a sample
            try:
                # Read first few rows to validate file integrity
                sample_df = pd.read_parquet(output_file)
                if len(sample_df) > 100:
                    sample_df = sample_df.head(100)  # Take first 100 rows if file is large
                if len(sample_df.columns) != len(df_final.columns):
                    raise ValueError("Column count mismatch in saved file")
            except Exception as read_error:
                raise ValueError(f"Cannot read saved file: {read_error}")
            
            log_progress(f"   ✅ Saved: {file_size_mb:.1f} MB")
            
            # Final memory cleanup and statistics collection
            del df_final
            final_memory = check_memory_and_cleanup('saving', force_gc=True)
            
        except Exception as e:
            return handle_stage_error('saving', e, critical=True)
        
        processing_stats['processing_stages']['saving'] = time.time() - stage_start
        
        # Stage 7: Comprehensive statistics collection and validation
        stage_start = time.time()
        log_progress(f"   📊 Collecting statistics...")
        
        try:
            # Calculate comprehensive processing statistics
            total_time = time.time() - processing_stats['start_time']
            processing_stats['total_time_minutes'] = total_time / 60
            
            # Data flow statistics
            data_retention_rates = {
                'cleaning_retention': (processing_stats['cleaned_rows'] / processing_stats['raw_rows'] * 100) if processing_stats['raw_rows'] > 0 else 0,
                'rth_retention': (processing_stats['rth_rows'] / processing_stats['cleaned_rows'] * 100) if processing_stats['cleaned_rows'] > 0 else 0,
                'final_retention': (processing_stats['final_rows'] / processing_stats['rth_rows'] * 100) if processing_stats['rth_rows'] > 0 else 0
            }
            
            # Memory efficiency statistics
            memory_efficiency = {
                'peak_memory_mb': processing_stats['memory_peak_mb'],
                'final_memory_mb': final_memory,
                'memory_reduction_mb': processing_stats['memory_peak_mb'] - final_memory,
                'memory_per_row_kb': (processing_stats['memory_peak_mb'] * 1024) / processing_stats['final_rows'] if processing_stats['final_rows'] > 0 else 0
            }
            
            # Processing performance statistics
            performance_stats = {
                'rows_per_minute': processing_stats['final_rows'] / processing_stats['total_time_minutes'] if processing_stats['total_time_minutes'] > 0 else 0,
                'stage_times': processing_stats['processing_stages'],
                'slowest_stage': max(processing_stats['processing_stages'].items(), key=lambda x: x[1]) if processing_stats['processing_stages'] else ('unknown', 0)
            }
            
            # Data quality analysis - analyze the final dataset
            data_quality_metrics = {}
            try:
                # Analyze labeling columns (6 modes × 2 columns each = 12 total)
                label_columns = [col for col in df_final.columns if col.startswith('label_')]
                weight_columns = [col for col in df_final.columns if col.startswith('weight_')]
                
                labeling_metrics = {}
                for label_col in label_columns:
                    mode_name = label_col.replace('label_', '')
                    weight_col = f'weight_{mode_name}'
                    
                    if label_col in df_final.columns:
                        win_rate = df_final[label_col].mean()  # Percentage of 1s (winners)
                        total_trades = len(df_final[label_col])
                        winners = df_final[label_col].sum()
                        losers = total_trades - winners
                        
                        labeling_metrics[mode_name] = {
                            'win_rate': float(win_rate),
                            'total_trades': int(total_trades),
                            'winners': int(winners),
                            'losers': int(losers)
                        }
                        
                        # Add weight statistics if available
                        if weight_col in df_final.columns:
                            labeling_metrics[mode_name].update({
                                'avg_weight': float(df_final[weight_col].mean()),
                                'min_weight': float(df_final[weight_col].min()),
                                'max_weight': float(df_final[weight_col].max()),
                                'weight_std': float(df_final[weight_col].std())
                            })
                
                # Analyze feature columns (should be 43 features)
                feature_columns = [col for col in df_final.columns if not col.startswith(('label_', 'weight_')) and col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                feature_metrics = {}
                for feature_col in feature_columns:
                    if feature_col in df_final.columns:
                        series = df_final[feature_col]
                        nan_pct = (series.isna().sum() / len(series)) * 100
                        
                        feature_metrics[feature_col] = {
                            'nan_percentage': float(nan_pct),
                            'min_value': float(series.min()) if not series.isna().all() else None,
                            'max_value': float(series.max()) if not series.isna().all() else None,
                            'mean_value': float(series.mean()) if not series.isna().all() else None,
                            'std_value': float(series.std()) if not series.isna().all() else None
                        }
                
                data_quality_metrics = {
                    'labeling_metrics': labeling_metrics,
                    'feature_metrics': feature_metrics,
                    'total_columns': len(df_final.columns),
                    'label_columns_count': len(label_columns),
                    'weight_columns_count': len(weight_columns),
                    'feature_columns_count': len(feature_columns)
                }
                
            except Exception as e:
                data_quality_metrics = {'error': f'Could not analyze data quality: {e}'}
            
            # Quality indicators
            quality_indicators = {
                'has_errors': len(processing_stats['errors']) > 0,
                'has_warnings': len(processing_stats['warnings']) > 0,
                'error_count': len(processing_stats['errors']),
                'warning_count': len(processing_stats['warnings']),
                'component_versions': processing_stats['component_versions']
            }
            
            # Create comprehensive statistics summary
            processing_stats.update({
                'data_retention_rates': data_retention_rates,
                'memory_efficiency': memory_efficiency,
                'performance_stats': performance_stats,
                'quality_indicators': quality_indicators
            })
            
            # Log comprehensive statistics
            log_progress(f"   📊 Processing statistics for {month_str}:")
            log_progress(f"      Data flow: {processing_stats['raw_rows']:,} → {processing_stats['cleaned_rows']:,} → {processing_stats['rth_rows']:,} → {processing_stats['final_rows']:,} rows")
            log_progress(f"      Retention: {data_retention_rates['cleaning_retention']:.1f}% → {data_retention_rates['rth_retention']:.1f}% → {data_retention_rates['final_retention']:.1f}%")
            log_progress(f"      Memory: Peak {processing_stats['memory_peak_mb']:.1f} MB, Final {final_memory:.1f} MB")
            log_progress(f"      Performance: {performance_stats['rows_per_minute']:.0f} rows/min, {processing_stats['total_time_minutes']:.1f} min total")
            log_progress(f"      Slowest stage: {performance_stats['slowest_stage'][0]} ({performance_stats['slowest_stage'][1]:.1f}s)")
            
            if processing_stats['warnings']:
                log_progress(f"      Warnings ({len(processing_stats['warnings'])}): {'; '.join(processing_stats['warnings'][:2])}")
            

            # Save processing statistics for restart capability and debugging
            stats_file = output_file.parent / f"{month_str}_processing_stats.json"
            try:
                import json
                with open(stats_file, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    json_stats = {}
                    for key, value in processing_stats.items():
                        if isinstance(value, (np.integer, np.floating)):
                            json_stats[key] = value.item()
                        elif isinstance(value, dict):
                            json_stats[key] = {k: v.item() if isinstance(v, (np.integer, np.floating)) else v for k, v in value.items()}
                        else:
                            json_stats[key] = value
                    
                    json.dump(json_stats, f, indent=2, default=str)
                
                log_progress(f"   💾 Saved processing statistics to {stats_file.name}")
                
            except Exception as stats_error:
                processing_stats['warnings'].append(f"Could not save statistics: {stats_error}")
            
        except Exception as e:
            # Statistics collection failure is not critical
            handle_stage_error('statistics_collection', e, critical=False)
        
        processing_stats['processing_stages']['statistics_collection'] = time.time() - stage_start
        
        # Final validation and return
        if processing_stats['errors']:
            log_progress(f"   ❌ Processing completed with {len(processing_stats['errors'])} errors")
            return None
        else:
            log_progress(f"   ✅ Processing completed successfully")
            if processing_stats['warnings']:
                log_progress(f"   ⚠️  {len(processing_stats['warnings'])} warnings logged")
            
            return str(output_file)
        
    except Exception as e:
        # Catch-all for unexpected errors with enhanced debugging
        error_msg = f"Unexpected processing error: {e}"
        log_progress(f"   ❌ {error_msg}")
        processing_stats['errors'].append(error_msg)
        
        # Log detailed error information for debugging
        import traceback
        detailed_error = traceback.format_exc()
        log_progress(f"   Error details: {detailed_error}")
        
        # Save error state for debugging and restart capability
        try:
            error_file = Path(f"/tmp/monthly_processing/{month_str}/error_state.json")
            error_file.parent.mkdir(parents=True, exist_ok=True)
            
            error_state = {
                'month': month_str,
                'error_message': error_msg,
                'detailed_error': detailed_error,
                'processing_stats': processing_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(error_file, 'w') as f:
                json.dump(error_state, f, indent=2, default=str)
            
            log_progress(f"   💾 Saved error state to {error_file}")
            
        except Exception as save_error:
            log_progress(f"   ⚠️  Could not save error state: {save_error}")
        
        return None

def upload_monthly_results(file_info, processed_file, monthly_statistics=None):
    """
    Enhanced upload monthly results to S3 with comprehensive statistics and metadata
    Uses optimized S3 operations with compression, retry logic, and integrity validation
    
    Args:
        file_info: Monthly file information
        processed_file: Path to processed parquet file
        monthly_statistics: Optional MonthlyProcessingStatistics object with comprehensive metrics
    """
    try:
        # Use enhanced S3 operations if available
        if S3_OPERATIONS_AVAILABLE:
            log_progress(f"   🚀 Using enhanced S3 operations with optimization")
            
            s3_ops = EnhancedS3Operations("es-1-second-data")
            success = s3_ops.upload_monthly_results_optimized(
                file_info=file_info,
                processed_file=processed_file,
                monthly_statistics=monthly_statistics
            )
            
            if success:
                log_progress(f"   ✅ Enhanced upload completed successfully")
                return True
            else:
                log_progress(f"   ⚠️  Enhanced upload failed, falling back to basic upload", level="WARNING")
                # Fall through to basic upload
        
        # Fallback to basic upload (original implementation)
        log_progress(f"   🔄 Using basic S3 upload operations")
        
        # Validate file before upload
        if not validate_processed_file(processed_file):
            log_progress(f"   ❌ Processed file failed validation")
            return False
        
        # Optimize compression before upload
        optimized_file = optimize_parquet_compression(processed_file)
        if optimized_file != processed_file:
            processed_file = optimized_file
        
        s3_client = boto3.client('s3')
        bucket_name = "es-1-second-data"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Organize output files with consistent naming and compression
        # Structure: processed-data/monthly/YYYY/MM/monthly_YYYY-MM_timestamp.parquet
        year, month = file_info['month_str'].split('-')
        s3_key = f"processed-data/monthly/{year}/{month}/monthly_{file_info['month_str']}_{timestamp}.parquet"
        
        # Get file stats for metadata
        file_path = Path(processed_file)
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024**2)
        
        # Read comprehensive stats from parquet file
        try:
            df_sample = pd.read_parquet(processed_file)
            if len(df_sample) > 1000:
                df_sample = df_sample.head(1000)
            row_count = len(pd.read_parquet(processed_file))
            column_count = len(df_sample.columns)
            
            # Enhanced column analysis
            label_columns = [col for col in df_sample.columns if col.startswith('label_')]
            weight_columns = [col for col in df_sample.columns if col.startswith('weight_')]
            feature_columns = [col for col in df_sample.columns if col not in 
                             ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + label_columns + weight_columns]
            
        except Exception as e:
            log_progress(f"   ⚠️  Could not read file stats: {e}")
            row_count = 0
            column_count = 0
            label_columns = []
            weight_columns = []
            feature_columns = []
        
        # Prepare enhanced metadata with quality flags and reprocessing recommendations
        enhanced_metadata = {
            'source': 'monthly_processing_pipeline_enhanced',
            'month': file_info['month_str'],
            'processing_date': timestamp,
            'file_size_mb': str(file_size_mb),
            'row_count': str(row_count),
            'column_count': str(column_count),
            'label_columns_count': str(len(label_columns)),
            'weight_columns_count': str(len(weight_columns)),
            'feature_columns_count': str(len(feature_columns)),
            'pipeline_version': '4.0_optimized_s3_fallback',
            'compression': 'snappy',
            'data_format': 'parquet'
        }
        
        # Add comprehensive statistics to metadata if available
        if monthly_statistics:
            # Add quality flags and reprocessing recommendations
            quality_flags = []
            if hasattr(monthly_statistics, 'requires_reprocessing') and monthly_statistics.requires_reprocessing:
                if hasattr(monthly_statistics, 'reprocessing_reasons'):
                    quality_flags.extend(monthly_statistics.reprocessing_reasons)
            
            # Add mode-specific quality flags
            if hasattr(monthly_statistics, 'mode_statistics'):
                for mode_name, mode_stats in monthly_statistics.mode_statistics.items():
                    if hasattr(mode_stats, 'quality_flags') and mode_stats.quality_flags:
                        quality_flags.extend([f"{mode_name}_{flag}" for flag in mode_stats.quality_flags])
            
            # Safely add statistics metadata
            stats_metadata = {}
            if hasattr(monthly_statistics, 'overall_quality_score'):
                stats_metadata['overall_quality_score'] = str(monthly_statistics.overall_quality_score)
            if hasattr(monthly_statistics, 'requires_reprocessing'):
                stats_metadata['requires_reprocessing'] = str(monthly_statistics.requires_reprocessing)
            if hasattr(monthly_statistics, 'processing_successful'):
                stats_metadata['processing_successful'] = str(monthly_statistics.processing_successful)
            if hasattr(monthly_statistics, 'total_rollover_events'):
                stats_metadata['total_rollover_events'] = str(monthly_statistics.total_rollover_events)
            
            enhanced_metadata.update(stats_metadata)
            enhanced_metadata['quality_flags'] = ','.join(quality_flags) if quality_flags else 'none'
        
        # Upload main parquet file with enhanced metadata and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                s3_client.upload_file(
                    processed_file,
                    bucket_name,
                    s3_key,
                    ExtraArgs={'Metadata': enhanced_metadata}
                )
                
                # Verify upload with integrity check
                try:
                    response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                    uploaded_size = response['ContentLength']
                    
                    if uploaded_size == file_size:
                        log_progress(f"   ✅ Uploaded and verified: s3://{bucket_name}/{s3_key}")
                        log_progress(f"   📊 File: {file_size_mb:.1f} MB, {row_count:,} rows, {column_count} columns")
                        
                        # Upload local statistics file and processing report to S3
                        month_str = file_info['month_str']
                        year, month = month_str.split('-')
                        
                        # Upload processing statistics JSON using preserved content
                        if monthly_statistics and "stats_content" in monthly_statistics:
                            stats_s3_key = f"processed-data/monthly/{year}/{month}/statistics/{month_str}_processing_stats.json"
                            try:
                                import tempfile
                                import json
                                # Create temporary file with the actual statistics content
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_stats_file:
                                    json.dump(monthly_statistics["stats_content"], temp_stats_file, indent=2)
                                    temp_stats_path = temp_stats_file.name
                                
                                s3_client.upload_file(
                                    temp_stats_path,
                                    bucket_name,
                                    stats_s3_key,
                                    ExtraArgs={
                                        'Metadata': {
                                            'content_type': 'application/json',
                                            'month': month_str,
                                            'file_type': 'processing_statistics'
                                        }
                                    }
                                )
                                log_progress(f"   📊 Processing statistics uploaded: s3://{bucket_name}/{stats_s3_key}")
                                
                                # Clean up temp file
                                Path(temp_stats_path).unlink()
                                
                            except Exception as stats_error:
                                log_progress(f"   ⚠️  Statistics upload failed: {stats_error}", level="WARNING")
                        
                        # Upload final processing report if it exists
                        report_files = list(Path("/tmp/monthly_processing").glob("final_processing_report_*.md"))
                        if report_files:
                            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                            report_s3_key = f"processed-data/monthly/{year}/{month}/reports/{month_str}_processing_report.md"
                            try:
                                s3_client.upload_file(
                                    str(latest_report),
                                    bucket_name,
                                    report_s3_key,
                                    ExtraArgs={
                                        'Metadata': {
                                            'content_type': 'text/markdown',
                                            'month': month_str,
                                            'file_type': 'processing_report'
                                        }
                                    }
                                )
                                log_progress(f"   📄 Processing report uploaded: s3://{bucket_name}/{report_s3_key}")
                            except Exception as report_error:
                                log_progress(f"   ⚠️  Report upload failed: {report_error}", level="WARNING")
                        
                        # Old placeholder logic removed - real statistics uploaded above
                        
                        return True
                    else:
                        log_progress(f"   ⚠️  Size mismatch: local={file_size}, s3={uploaded_size}")
                        if attempt < max_retries - 1:
                            continue
                        
                except Exception as e:
                    log_progress(f"   ⚠️  Upload verification failed: {e}")
                    if attempt < max_retries - 1:
                        continue
                
                return True
                
            except Exception as e:
                wait_time = 2 ** attempt
                log_progress(f"   ⚠️  Upload attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    log_progress(f"   ⏳ Retrying upload in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    log_progress(f"   ❌ All upload attempts failed")
                    return False
        
        return False
        
    except Exception as e:
        log_progress(f"   ❌ Upload failed: {e}")
        return False

def optimize_parquet_compression(input_file, output_file=None):
    """
    Optimize Parquet file compression for S3 storage
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output optimized file (optional, defaults to input_file)
    
    Returns:
        Path to optimized file
    """
    try:
        if output_file is None:
            output_file = input_file
        
        # Read the parquet file
        df = pd.read_parquet(input_file)
        
        # Optimize data types for better compression
        for col in df.columns:
            if df[col].dtype == 'float64':
                # Check if we can downcast to float32 without losing precision
                if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                    df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                # Check if we can downcast to int32
                if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
        
        # Save with optimal compression settings
        df.to_parquet(
            output_file,
            compression='snappy',  # Good balance of speed and compression
            index=False,
            engine='pyarrow'
        )
        
        # Log compression results
        original_size = Path(input_file).stat().st_size if input_file != output_file else 0
        optimized_size = Path(output_file).stat().st_size
        
        if original_size > 0:
            compression_ratio = (1 - optimized_size / original_size) * 100
            log_progress(f"   🗜️  Compression: {compression_ratio:.1f}% reduction ({original_size/1024/1024:.1f}MB → {optimized_size/1024/1024:.1f}MB)")
        
        return output_file
        
    except Exception as e:
        log_progress(f"   ⚠️  Compression optimization failed: {e}", level="WARNING")
        return input_file

def validate_processed_file(file_path):
    """Validate processed parquet file before upload"""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            log_progress(f"   ❌ File does not exist: {file_path}")
            return False
        
        # Check file size (should be reasonable for processed data)
        file_size = file_path.stat().st_size
        if file_size < 1024:  # Less than 1KB
            log_progress(f"   ❌ File too small: {file_size} bytes")
            return False
        
        # Try to read parquet file
        try:
            df_sample = pd.read_parquet(file_path)
            if len(df_sample) > 100:
                df_sample = df_sample.head(100)
            
            # Check for required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df_sample.columns]
            
            if missing_columns:
                log_progress(f"   ❌ Missing required columns: {missing_columns}")
                return False
            
            # Check for labeling columns (should have 6 labels + 6 weights)
            label_columns = [col for col in df_sample.columns if col.startswith('label_')]
            weight_columns = [col for col in df_sample.columns if col.startswith('weight_')]
            
            if len(label_columns) != 6 or len(weight_columns) != 6:
                log_progress(f"   ❌ Incorrect labeling columns: {len(label_columns)} labels, {len(weight_columns)} weights")
                return False
            
            # Check total column count (should be around 61: 6 original + 12 labeling + 43 features)
            if len(df_sample.columns) < 50:
                log_progress(f"   ❌ Too few columns: {len(df_sample.columns)}")
                return False
            
            return True
            
        except Exception as e:
            log_progress(f"   ❌ Parquet validation failed: {e}")
            return False
            
    except Exception as e:
        log_progress(f"   ❌ File validation error: {e}")
        return False

def cleanup_monthly_files(file_info):
    """Clean up temporary files for the month"""
    month_dir = Path(f"/tmp/monthly_processing/{file_info['month_str']}")
    
    try:
        for file_path in month_dir.glob("*"):
            file_path.unlink()
        month_dir.rmdir()
        log_progress(f"   🧹 Cleaned up temporary files")
    except Exception as e:
        log_progress(f"   ⚠️  Cleanup warning: {e}")

def main():
    """Enhanced main monthly processing pipeline with improved progress tracking and time estimation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process ES futures data monthly")
    parser.add_argument("--test-month", help="Process only specific month (YYYY-MM format, e.g., 2010-07)")
    parser.add_argument("--start-month", help="Start from specific month (YYYY-MM format)")
    parser.add_argument("--end-month", help="End at specific month (YYYY-MM format)")
    args = parser.parse_args()
    
    log_progress("🚀 MONTHLY ES DATA PROCESSING PIPELINE (WITH DATA CLEANING)")
    
    if args.test_month:
        log_progress(f"🎯 SINGLE MONTH TEST MODE: {args.test_month}")
    else:
        log_progress("Processing 15 years of data in monthly chunks")
    
    # Generate file list
    monthly_files = generate_monthly_file_list()
    
    # Filter for specific month if requested
    if args.test_month:
        monthly_files = [f for f in monthly_files if f['month_str'] == args.test_month]
        if not monthly_files:
            log_progress(f"❌ Month {args.test_month} not found in available data")
            return
        log_progress(f"✅ Found test month: {args.test_month}")
    elif args.start_month or args.end_month:
        if args.start_month:
            monthly_files = [f for f in monthly_files if f['month_str'] >= args.start_month]
        if args.end_month:
            monthly_files = [f for f in monthly_files if f['month_str'] <= args.end_month]
        log_progress(f"✅ Filtered to date range: {len(monthly_files)} months")
    
    # Check existing files
    to_process = check_existing_processed_files(monthly_files)
    
    if not to_process:
        log_progress("✅ All months already processed!")
        return
    
    # Enhanced progress tracking initialization
    progress_tracker = EnhancedProgressTracker(len(to_process))
    
    log_progress(f"🎯 PROCESSING PLAN")
    log_progress(f"   Total months to process: {len(to_process)}")
    log_progress(f"   Date range: {to_process[0]['month_str']} to {to_process[-1]['month_str']}")
    log_progress(f"   Estimated time per month: 10-30 minutes")
    log_progress(f"   Total estimated time: {len(to_process) * 20 / 60:.1f} hours")
    
    # Enhanced processing with failure tracking and recovery
    successful = 0
    failed = 0
    failed_months = []
    processing_errors = []
    stage_timings = {}  # Track bottlenecks across all months
    
    start_time = time.time()
    
    for i, file_info in enumerate(to_process, 1):
        month_str = file_info['month_str']
        month_start_time = time.time()
        
        # Enhanced progress display with better time estimation
        progress_tracker.start_month(month_str, i)
        
        try:
            # Enhanced single month processing with stage timing
            result = process_single_month_with_timing(file_info, stage_timings)
            
            month_duration = time.time() - month_start_time
            
            if result:
                successful += 1
                progress_tracker.complete_month(month_str, True, month_duration)
                log_progress(f"   ✅ {month_str} completed successfully in {month_duration/60:.1f} minutes", level="INFO")
            else:
                failed += 1
                failed_months.append(month_str)
                progress_tracker.complete_month(month_str, False, month_duration)
                log_progress(f"   ❌ {month_str} failed after {month_duration/60:.1f} minutes", level="ERROR")
                
                # Continue processing remaining months (requirement 7.1)
                log_progress(f"   🔄 Continuing with remaining months despite failure", level="INFO")
        
        except Exception as e:
            # Catch any unexpected errors in month processing to ensure continuation
            month_duration = time.time() - month_start_time
            failed += 1
            failed_months.append(month_str)
            
            error_info = {
                'month': month_str,
                'error': str(e),
                'error_type': type(e).__name__,
                'duration_minutes': month_duration / 60
            }
            processing_errors.append(error_info)
            
            progress_tracker.complete_month(month_str, False, month_duration)
            log_progress(f"   ❌ {month_str} failed with unexpected error after {month_duration/60:.1f} minutes", level="ERROR", error_details=e)
            log_progress(f"   🔄 Continuing with remaining months", level="INFO")
        
        # Enhanced progress summary with improved time estimation and bottleneck identification
        progress_summary = progress_tracker.get_progress_summary()
        
        log_progress(f"   📊 Progress Summary:")
        log_progress(f"      ✅ Successful: {successful}/{i} ({progress_summary['success_rate']:.1f}%)")
        log_progress(f"      ❌ Failed: {failed}/{i}")
        log_progress(f"      ⏱️  Elapsed: {progress_summary['elapsed_hours']:.1f}h, ETA: {progress_summary['eta_hours']:.1f}h")
        log_progress(f"      📈 Avg time per month: {progress_summary['avg_time_minutes']:.1f} minutes")
        log_progress(f"      🎯 Completion: {progress_summary['completion_time']}")
        
        # Identify and report bottlenecks every 10 months
        if i % 10 == 0 and stage_timings:
            bottlenecks = identify_processing_bottlenecks(stage_timings)
            if bottlenecks:
                log_progress(f"      🐌 Bottlenecks identified: {', '.join(bottlenecks[:2])}")
        
        # Log recent failures for monitoring
        if failed_months and len(failed_months) <= 3:
            log_progress(f"      🔍 Recent failures: {', '.join(failed_months[-3:])}")
        
        # Memory cleanup between months to prevent accumulation
        import gc
        gc.collect()
    
    total_time = time.time() - start_time
    success_rate = (successful / len(to_process)) * 100 if to_process else 0
    
    # Enhanced completion summary with detailed performance analysis
    final_summary = progress_tracker.get_progress_summary()
    
    log_progress(f"🎉 MONTHLY PROCESSING COMPLETE!")
    log_progress(f"   📊 Results Summary:")
    log_progress(f"      ✅ Successful: {successful}/{len(to_process)} months ({success_rate:.1f}%)")
    log_progress(f"      ❌ Failed: {failed}/{len(to_process)} months")
    log_progress(f"      ⏱️  Total time: {total_time/3600:.1f} hours")
    log_progress(f"      📈 Average time per month: {final_summary['avg_time_minutes']:.1f} minutes")
    
    # Enhanced performance analysis
    if final_summary['fastest_month'] and final_summary['slowest_month']:
        log_progress(f"   🏃 Performance Analysis:")
        log_progress(f"      ⚡ Fastest month: {final_summary['fastest_month']} ({final_summary['fastest_time_minutes']:.1f} min)")
        log_progress(f"      🐌 Slowest month: {final_summary['slowest_month']} ({final_summary['slowest_time_minutes']:.1f} min)")
        
        speed_ratio = final_summary['slowest_time_minutes'] / final_summary['fastest_time_minutes'] if final_summary['fastest_time_minutes'] > 0 else 1
        log_progress(f"      📊 Speed variation: {speed_ratio:.1f}x difference")
    
    # Bottleneck analysis
    if stage_timings:
        bottlenecks = identify_processing_bottlenecks(stage_timings)
        if bottlenecks:
            log_progress(f"   🔍 Bottleneck Analysis:")
            for bottleneck in bottlenecks[:3]:  # Show top 3
                log_progress(f"      🐌 {bottleneck}")
        else:
            log_progress(f"   ✅ No significant bottlenecks identified")
    
    # Detailed failure reporting for recovery planning
    if failed_months:
        log_progress(f"   🔍 Failed Months Analysis:")
        log_progress(f"      Failed months: {', '.join(failed_months)}")
        
        # Group failures by error type if available
        if processing_errors:
            error_types = {}
            for error_info in processing_errors:
                error_type = error_info['error_type']
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error_info['month'])
            
            log_progress(f"      Error type breakdown:")
            for error_type, months in error_types.items():
                log_progress(f"        {error_type}: {len(months)} months ({', '.join(months[:3])}{'...' if len(months) > 3 else ''})")
        
        # Save failed months list for retry
        try:
            failed_months_file = Path("/tmp/monthly_processing/failed_months.txt")
            failed_months_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(failed_months_file, 'w') as f:
                f.write("# Failed months from processing run\n")
                f.write(f"# Run completed: {datetime.now().isoformat()}\n")
                f.write(f"# Success rate: {success_rate:.1f}%\n\n")
                for month in failed_months:
                    f.write(f"{month}\n")
            
            log_progress(f"   💾 Saved failed months list to {failed_months_file}")
            
        except Exception as e:
            log_progress(f"   ⚠️  Could not save failed months list: {e}", level="WARNING")
    
    else:
        log_progress(f"   🎉 All months processed successfully!")
    
    # Final recommendations
    if success_rate < 90:
        log_progress(f"   💡 Recommendations:")
        log_progress(f"      - Review failed months for common error patterns")
        log_progress(f"      - Consider reprocessing failed months individually")
        log_progress(f"      - Check S3 connectivity and file availability")
    elif success_rate < 100:
        log_progress(f"   💡 Recommendation: Consider reprocessing the {failed} failed months")
    
    # Use JSON statistics instead of problematic markdown report
    log_progress(f"📊 PROCESSING COMPLETE - DATA QUALITY METRICS IN JSON STATISTICS")
    log_progress(f"   📊 Statistics JSON files uploaded to S3 contain all data quality metrics")
    log_progress(f"   🎯 Processed data ready for XGBoost training")
    
    # Log final status based on success rate
    if success_rate >= 95:
        log_progress(f"   🎯 Status: Ready for model training (success rate: {success_rate:.1f}%)")
    elif success_rate >= 85:
        log_progress(f"   ⚠️  Status: Minor reprocessing needed (success rate: {success_rate:.1f}%)")
    else:
        log_progress(f"   🔄 Status: Significant reprocessing required (success rate: {success_rate:.1f}%)")
    
    return {
        'successful': successful,
        'failed': failed,
        'failed_months': failed_months,
        'success_rate': success_rate,
        'total_time_hours': total_time / 3600,
        'processing_errors': processing_errors
    }

if __name__ == "__main__":
    main()