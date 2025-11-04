"""
Enhanced Logging and Monitoring System

This module provides comprehensive logging and monitoring capabilities for monthly processing
as specified in requirement 9.3 and 7.7.

Features:
- Detailed timestamp and processing time capture
- Processing start/end times for each month and stage
- Comprehensive processing log with success/failure status
- Memory usage and performance monitoring
- Structured logging for analysis and debugging
"""

import json
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
import threading
import traceback


class ProcessingStageTracker:
    """
    Tracks processing stages with detailed timing and performance metrics
    """
    
    def __init__(self):
        """Initialize stage tracker"""
        self.stage_start_times = {}
        self.stage_end_times = {}
        self.stage_durations = defaultdict(list)
        self.stage_memory_usage = defaultdict(list)
        self.stage_success_rates = defaultdict(list)
        self.current_stages = set()
        
        # Performance thresholds
        self.warning_thresholds = {
            'download': 300,  # 5 minutes
            'processing': 1800,  # 30 minutes
            'upload': 600,  # 10 minutes
            'cleanup': 60,  # 1 minute
            'month_processing': 2700,  # 45 minutes
            'feature_engineering': 900,  # 15 minutes
            'weighted_labeling': 1200,  # 20 minutes
            'data_quality': 300,  # 5 minutes
            'statistics_collection': 180  # 3 minutes
        }
    
    def start_stage(self, stage_name: str, context: Optional[Dict] = None) -> float:
        """
        Start tracking a processing stage
        
        Args:
            stage_name: Name of the processing stage
            context: Additional context information
            
        Returns:
            Start timestamp
        """
        start_time = time.time()
        self.stage_start_times[stage_name] = start_time
        self.current_stages.add(stage_name)
        
        # Record memory usage at stage start
        try:
            memory_mb = psutil.Process().memory_info().rss / (1024**2)
            self.stage_memory_usage[f"{stage_name}_start"].append(memory_mb)
        except Exception:
            pass
        
        return start_time
    
    def end_stage(self, stage_name: str, success: bool = True, context: Optional[Dict] = None) -> Optional[float]:
        """
        End tracking a processing stage
        
        Args:
            stage_name: Name of the processing stage
            success: Whether the stage completed successfully
            context: Additional context information
            
        Returns:
            Stage duration in seconds, or None if stage wasn't started
        """
        end_time = time.time()
        
        if stage_name not in self.stage_start_times:
            return None
        
        # Calculate duration
        duration = end_time - self.stage_start_times[stage_name]
        
        # Record stage completion
        self.stage_end_times[stage_name] = end_time
        self.stage_durations[stage_name].append(duration)
        self.stage_success_rates[stage_name].append(success)
        
        # Record memory usage at stage end
        try:
            memory_mb = psutil.Process().memory_info().rss / (1024**2)
            self.stage_memory_usage[f"{stage_name}_end"].append(memory_mb)
        except Exception:
            pass
        
        # Clean up tracking data
        if stage_name in self.current_stages:
            self.current_stages.remove(stage_name)
        
        return duration
    
    def get_stage_statistics(self, stage_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a processing stage
        
        Args:
            stage_name: Name of the processing stage
            
        Returns:
            Dictionary with stage statistics
        """
        durations = self.stage_durations.get(stage_name, [])
        success_rates = self.stage_success_rates.get(stage_name, [])
        
        if not durations:
            return {
                'stage_name': stage_name,
                'executions': 0,
                'avg_duration_seconds': 0,
                'min_duration_seconds': 0,
                'max_duration_seconds': 0,
                'success_rate': 0,
                'is_slow': False
            }
        
        avg_duration = sum(durations) / len(durations)
        success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Check if stage is consistently slow
        warning_threshold = self.warning_thresholds.get(stage_name, 600)  # Default 10 minutes
        is_slow = avg_duration > warning_threshold
        
        return {
            'stage_name': stage_name,
            'executions': len(durations),
            'avg_duration_seconds': avg_duration,
            'min_duration_seconds': min(durations),
            'max_duration_seconds': max(durations),
            'total_duration_seconds': sum(durations),
            'success_rate': success_rate,
            'successful_executions': sum(success_rates),
            'failed_executions': len(success_rates) - sum(success_rates),
            'is_slow': is_slow,
            'warning_threshold': warning_threshold
        }
    
    def get_all_stage_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tracked stages"""
        all_stages = set(self.stage_durations.keys()) | set(self.stage_success_rates.keys())
        return {stage: self.get_stage_statistics(stage) for stage in all_stages}


class MemoryMonitor:
    """
    Monitors memory usage throughout processing with detailed tracking
    """
    
    def __init__(self, monitoring_interval: float = 30.0):
        """
        Initialize memory monitor
        
        Args:
            monitoring_interval: Interval between memory snapshots in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.memory_snapshots = []
        self.peak_memory_mb = 0.0
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Memory thresholds
        self.warning_threshold_mb = 6000  # 6GB
        self.critical_threshold_mb = 7500  # 7.5GB
    
    def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                self.take_snapshot()
                time.sleep(self.monitoring_interval)
            except Exception:
                pass
    
    def take_snapshot(self, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Take a memory usage snapshot
        
        Args:
            context: Context information for the snapshot
            
        Returns:
            Memory snapshot data
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            snapshot = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'context': context,
                'process_memory_mb': memory_info.rss / (1024**2),
                'process_virtual_mb': memory_info.vms / (1024**2),
                'system_memory_percent': system_memory.percent,
                'system_available_gb': system_memory.available / (1024**3),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
            
            self.memory_snapshots.append(snapshot)
            
            # Update peak memory
            if snapshot['process_memory_mb'] > self.peak_memory_mb:
                self.peak_memory_mb = snapshot['process_memory_mb']
            
            # Check thresholds
            if snapshot['process_memory_mb'] > self.critical_threshold_mb:
                return snapshot, 'CRITICAL'
            elif snapshot['process_memory_mb'] > self.warning_threshold_mb:
                return snapshot, 'WARNING'
            
            return snapshot, 'OK'
            
        except Exception as e:
            return {'error': str(e), 'timestamp': time.time()}, 'ERROR'
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics"""
        if not self.memory_snapshots:
            return {
                'peak_memory_mb': 0,
                'avg_memory_mb': 0,
                'min_memory_mb': 0,
                'snapshots_count': 0,
                'monitoring_duration_minutes': 0
            }
        
        memory_values = [s['process_memory_mb'] for s in self.memory_snapshots if 'process_memory_mb' in s]
        
        if not memory_values:
            return {
                'peak_memory_mb': 0,
                'avg_memory_mb': 0,
                'min_memory_mb': 0,
                'snapshots_count': len(self.memory_snapshots),
                'monitoring_duration_minutes': 0
            }
        
        first_snapshot = self.memory_snapshots[0]['timestamp']
        last_snapshot = self.memory_snapshots[-1]['timestamp']
        
        return {
            'peak_memory_mb': max(memory_values),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'min_memory_mb': min(memory_values),
            'current_memory_mb': memory_values[-1],
            'snapshots_count': len(self.memory_snapshots),
            'monitoring_duration_minutes': (last_snapshot - first_snapshot) / 60,
            'memory_growth_mb': memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0,
            'exceeded_warning_count': sum(1 for v in memory_values if v > self.warning_threshold_mb),
            'exceeded_critical_count': sum(1 for v in memory_values if v > self.critical_threshold_mb)
        }


class EnhancedLogger:
    """
    Enhanced logging system with comprehensive monitoring and structured output
    """
    
    def __init__(self, log_directory: Union[str, Path] = "/tmp/monthly_processing"):
        """
        Initialize enhanced logger
        
        Args:
            log_directory: Directory for log files
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.stage_tracker = ProcessingStageTracker()
        self.memory_monitor = MemoryMonitor()
        
        # Log files
        self.main_log_file = self.log_directory / "enhanced_processing.log"
        self.structured_log_file = self.log_directory / "structured_processing.jsonl"
        self.performance_log_file = self.log_directory / "performance_metrics.jsonl"
        self.error_log_file = self.log_directory / "error_details.log"
        
        # Processing session tracking
        self.session_start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        # Log session start
        self.log("Enhanced logging system initialized", level="INFO", 
               context={'session_id': self.session_id, 'log_directory': str(self.log_directory)})
    
    def log(self, message: str, level: str = "INFO", error_details: Optional[Exception] = None, 
            context: Optional[Dict] = None, stage: Optional[str] = None, 
            stage_event: Optional[str] = None) -> None:
        """
        Enhanced logging with comprehensive information capture
        
        Args:
            message: Main log message
            level: Log level (INFO, WARNING, ERROR, CRITICAL)
            error_details: Exception object for error logging
            context: Additional context information
            stage: Processing stage name
            stage_event: Stage event type ('start', 'end', 'progress')
        """
        timestamp = datetime.now()
        processing_time = time.time()
        
        # Handle stage events
        stage_duration = None
        if stage and stage_event:
            if stage_event == 'start':
                self.stage_tracker.start_stage(stage, context)
            elif stage_event == 'end':
                success = level not in ['ERROR', 'CRITICAL']
                stage_duration = self.stage_tracker.end_stage(stage, success, context)
        
        # Take memory snapshot for important events
        memory_snapshot = None
        memory_status = 'OK'
        if level in ['ERROR', 'CRITICAL'] or stage_event in ['start', 'end']:
            memory_snapshot, memory_status = self.memory_monitor.take_snapshot(
                context=f"{stage}_{stage_event}" if stage and stage_event else level
            )
        
        # Create structured log entry
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'processing_time': processing_time,
            'session_id': self.session_id,
            'level': level,
            'message': message,
            'stage': stage,
            'stage_event': stage_event,
            'stage_duration': stage_duration,
            'context': context or {},
            'memory_status': memory_status,
            'has_error': error_details is not None
        }
        
        # Add memory information if available
        if memory_snapshot and 'process_memory_mb' in memory_snapshot:
            log_entry['memory_mb'] = memory_snapshot['process_memory_mb']
            log_entry['cpu_percent'] = memory_snapshot.get('cpu_percent', 0)
        
        # Format main log message
        stage_info = f"[{stage}]" if stage else ""
        main_message = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [{level}] {stage_info} {message}"
        
        # Console output
        print(main_message)
        
        # Write to main log file
        try:
            with open(self.main_log_file, "a", encoding='utf-8') as f:
                f.write(main_message + "\n")
                
                # Add stage duration if available
                if stage_duration:
                    f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [TIMING] {stage} completed in {stage_duration:.2f} seconds\n")
                
                # Add memory information for high usage or errors
                if memory_snapshot and 'process_memory_mb' in memory_snapshot:
                    if memory_status != 'OK' or level in ['ERROR', 'CRITICAL']:
                        f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [MEMORY] {memory_snapshot['process_memory_mb']:.1f} MB RSS, {memory_snapshot['system_memory_percent']:.1f}% system\n")
                
                # Add context information
                if context:
                    for key, value in context.items():
                        f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [CONTEXT] {key}: {value}\n")
                
                f.flush()
        except Exception:
            pass
        
        # Write structured log entry
        try:
            with open(self.structured_log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
                f.flush()
        except Exception:
            pass
        
        # Handle error details
        if error_details:
            self._log_error_details(timestamp, error_details, context, stage)
        
        # Log performance metrics for stage events
        if stage and stage_event == 'end' and stage_duration:
            self._log_performance_metrics(timestamp, stage, stage_duration, memory_snapshot, context)
    
    def _log_error_details(self, timestamp: datetime, error: Exception, 
                          context: Optional[Dict], stage: Optional[str]) -> None:
        """Log detailed error information"""
        try:
            error_entry = {
                'timestamp': timestamp.isoformat(),
                'session_id': self.session_id,
                'stage': stage,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {}
            }
            
            with open(self.error_log_file, "a", encoding='utf-8') as f:
                f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] ERROR DETAILS:\n")
                f.write(f"Stage: {stage}\n")
                f.write(f"Error Type: {type(error).__name__}\n")
                f.write(f"Error Message: {str(error)}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
                if context:
                    f.write(f"Context: {json.dumps(context, default=str, indent=2)}\n")
                f.write("-" * 80 + "\n")
                f.flush()
        except Exception:
            pass
    
    def _log_performance_metrics(self, timestamp: datetime, stage: str, duration: float,
                                memory_snapshot: Optional[Dict], context: Optional[Dict]) -> None:
        """Log performance metrics for analysis"""
        try:
            performance_entry = {
                'timestamp': timestamp.isoformat(),
                'session_id': self.session_id,
                'stage': stage,
                'duration_seconds': duration,
                'context': context or {}
            }
            
            # Add memory metrics if available
            if memory_snapshot and 'process_memory_mb' in memory_snapshot:
                performance_entry.update({
                    'memory_mb': memory_snapshot['process_memory_mb'],
                    'cpu_percent': memory_snapshot.get('cpu_percent', 0),
                    'system_memory_percent': memory_snapshot.get('system_memory_percent', 0)
                })
            
            with open(self.performance_log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(performance_entry, default=str) + "\n")
                f.flush()
        except Exception:
            pass
    
    def start_stage(self, stage_name: str, context: Optional[Dict] = None) -> None:
        """Start tracking a processing stage"""
        self.log(f"Starting {stage_name}", level="INFO", stage=stage_name, 
                stage_event="start", context=context)
    
    def end_stage(self, stage_name: str, success: bool = True, context: Optional[Dict] = None) -> None:
        """End tracking a processing stage"""
        level = "INFO" if success else "ERROR"
        status = "completed successfully" if success else "failed"
        self.log(f"{stage_name} {status}", level=level, stage=stage_name, 
                stage_event="end", context=context)
    
    def log_progress(self, stage_name: str, message: str, context: Optional[Dict] = None) -> None:
        """Log progress within a stage"""
        self.log(message, level="INFO", stage=stage_name, stage_event="progress", context=context)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        session_duration = time.time() - self.session_start_time
        
        # Get stage statistics
        stage_stats = self.stage_tracker.get_all_stage_statistics()
        
        # Get memory statistics
        memory_stats = self.memory_monitor.get_memory_statistics()
        
        # Calculate overall performance metrics
        total_stage_time = sum(
            stats['total_duration_seconds'] 
            for stats in stage_stats.values()
        )
        
        successful_stages = sum(
            stats['successful_executions'] 
            for stats in stage_stats.values()
        )
        
        failed_stages = sum(
            stats['failed_executions'] 
            for stats in stage_stats.values()
        )
        
        return {
            'session_id': self.session_id,
            'session_duration_minutes': session_duration / 60,
            'total_stage_time_minutes': total_stage_time / 60,
            'stage_statistics': stage_stats,
            'memory_statistics': memory_stats,
            'performance_summary': {
                'successful_stages': successful_stages,
                'failed_stages': failed_stages,
                'success_rate': successful_stages / (successful_stages + failed_stages) if (successful_stages + failed_stages) > 0 else 0,
                'avg_stage_duration_minutes': (total_stage_time / len(stage_stats)) / 60 if stage_stats else 0
            },
            'log_files': {
                'main_log': str(self.main_log_file),
                'structured_log': str(self.structured_log_file),
                'performance_log': str(self.performance_log_file),
                'error_log': str(self.error_log_file)
            }
        }
    
    def generate_session_report(self) -> str:
        """Generate a comprehensive session report"""
        summary = self.get_session_summary()
        
        report_lines = []
        report_lines.append(f"# Processing Session Report")
        report_lines.append(f"**Session ID:** {summary['session_id']}")
        report_lines.append(f"**Duration:** {summary['session_duration_minutes']:.1f} minutes")
        report_lines.append(f"**Total Stage Time:** {summary['total_stage_time_minutes']:.1f} minutes")
        report_lines.append("")
        
        # Performance Summary
        perf = summary['performance_summary']
        report_lines.append("## Performance Summary")
        report_lines.append(f"- **Successful Stages:** {perf['successful_stages']}")
        report_lines.append(f"- **Failed Stages:** {perf['failed_stages']}")
        report_lines.append(f"- **Success Rate:** {perf['success_rate']:.1%}")
        report_lines.append(f"- **Average Stage Duration:** {perf['avg_stage_duration_minutes']:.1f} minutes")
        report_lines.append("")
        
        # Memory Statistics
        mem = summary['memory_statistics']
        report_lines.append("## Memory Usage")
        report_lines.append(f"- **Peak Memory:** {mem['peak_memory_mb']:.1f} MB")
        report_lines.append(f"- **Average Memory:** {mem['avg_memory_mb']:.1f} MB")
        report_lines.append(f"- **Memory Growth:** {mem['memory_growth_mb']:.1f} MB")
        report_lines.append(f"- **Warning Threshold Exceeded:** {mem['exceeded_warning_count']} times")
        report_lines.append(f"- **Critical Threshold Exceeded:** {mem['exceeded_critical_count']} times")
        report_lines.append("")
        
        # Stage Details
        report_lines.append("## Stage Performance")
        for stage_name, stats in summary['stage_statistics'].items():
            status = "⚠️" if stats['is_slow'] else "✅"
            report_lines.append(f"- **{stage_name}** {status}: {stats['executions']} executions, {stats['avg_duration_seconds']:.1f}s avg")
            if stats['failed_executions'] > 0:
                report_lines.append(f"  - Failed: {stats['failed_executions']}/{stats['executions']}")
        
        report_lines.append("")
        
        # Log Files
        report_lines.append("## Log Files")
        for log_type, log_path in summary['log_files'].items():
            report_lines.append(f"- **{log_type.replace('_', ' ').title()}:** {log_path}")
        
        return "\n".join(report_lines)
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.memory_monitor.stop_monitoring()
        
        # Log session end
        self.log("Enhanced logging system shutting down", level="INFO",
               context={'session_duration_minutes': (time.time() - self.session_start_time) / 60})


# Global enhanced logger instance
enhanced_logger = None

def get_enhanced_logger(log_directory: Union[str, Path] = "/tmp/monthly_processing") -> EnhancedLogger:
    """Get or create the global enhanced logger instance"""
    global enhanced_logger
    if enhanced_logger is None:
        enhanced_logger = EnhancedLogger(log_directory)
    return enhanced_logger

def log_enhanced(message: str, level: str = "INFO", error_details: Optional[Exception] = None,
                context: Optional[Dict] = None, stage: Optional[str] = None,
                stage_event: Optional[str] = None) -> None:
    """
    Enhanced logging function that can be used as a drop-in replacement
    
    Args:
        message: Main log message
        level: Log level (INFO, WARNING, ERROR, CRITICAL)
        error_details: Exception object for error logging
        context: Additional context information
        stage: Processing stage name
        stage_event: Stage event type ('start', 'end', 'progress')
    """
    logger = get_enhanced_logger()
    logger.log(message, level, error_details, context, stage, stage_event)