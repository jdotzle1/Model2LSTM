"""
Comprehensive Performance Monitoring System

This module provides comprehensive performance monitoring for all processing stages,
bottleneck identification, memory usage pattern analysis, and optimization recommendations.
"""

import time
import psutil
import numpy as np
import pandas as pd
import gc
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

@dataclass
class StageMetrics:
    """Metrics for a single processing stage"""
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    cpu_percent_avg: float = 0.0
    rows_processed: int = 0
    rows_per_second: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class BottleneckAnalysis:
    """Analysis of processing bottlenecks"""
    slowest_stage: str
    slowest_duration: float
    memory_intensive_stage: str
    peak_memory_usage: float
    cpu_intensive_stage: str
    peak_cpu_usage: float
    recommendations: List[str]
    optimization_opportunities: List[str]

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    total_duration: float
    total_rows_processed: int
    overall_rows_per_second: float
    peak_memory_mb: float
    avg_memory_mb: float
    memory_efficiency_score: float 
   stage_metrics: List[StageMetrics]
    bottleneck_analysis: BottleneckAnalysis
    memory_usage_pattern: Dict[str, Any]
    optimization_recommendations: List[str]
    quality_flags: List[str]
    timestamp: datetime

class ComprehensivePerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, enable_memory_tracking: bool = True, 
                 memory_sampling_interval: float = 1.0,
                 enable_detailed_logging: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_detailed_logging = enable_detailed_logging
        self.monitoring_active = False
        self.start_time = None
        self.end_time = None
        self.total_rows_processed = 0
        self.stage_timers = {}
        self.completed_stages = []
        self.performance_events = []
        self.quality_flags = []
        self.memory_samples = []
        self.performance_targets = {
            'min_rows_per_second': 1000,
            'max_memory_usage_gb': 8.0,
            'max_stage_duration_minutes': 30,
            'memory_efficiency_threshold': 70.0
        }
    
    def start_monitoring(self, expected_rows: int = 0):
        """Start comprehensive performance monitoring"""
        if self.monitoring_active:
            self.stop_monitoring()
        
        self.monitoring_active = True
        self.start_time = time.time()
        self.end_time = None
        self.total_rows_processed = 0
        self.performance_events = []
        self.quality_flags = []
        self.completed_stages = []
        self.memory_samples = []
        
        if self.enable_detailed_logging:
            self._log_event("monitoring_started", {
                'expected_rows': expected_rows,
                'memory_tracking': self.enable_memory_tracking,
                'start_time': datetime.now().isoformat()
            })    

    def stop_monitoring(self) -> PerformanceReport:
        """Stop monitoring and generate comprehensive performance report"""
        if not self.monitoring_active:
            raise RuntimeError("Monitoring is not active")
        
        self.end_time = time.time()
        self.monitoring_active = False
        
        report = self._generate_performance_report()
        
        if self.enable_detailed_logging:
            self._log_event("monitoring_completed", {
                'total_duration': report.total_duration,
                'total_rows': report.total_rows_processed,
                'peak_memory_mb': report.peak_memory_mb,
                'bottlenecks_found': len(report.bottleneck_analysis.recommendations)
            })
        
        return report
    
    def start_stage(self, stage_name: str, context: Optional[Dict] = None) -> str:
        """Start monitoring a processing stage"""
        if not self.monitoring_active:
            raise RuntimeError("Monitoring is not active")
        
        stage_id = f"{stage_name}_{int(time.time() * 1000)}"
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            cpu_percent = process.cpu_percent()
        except:
            memory_mb = 0.0
            cpu_percent = 0.0
        
        stage_metrics = StageMetrics(
            stage_name=stage_name,
            start_time=time.time(),
            memory_start_mb=memory_mb,
            cpu_percent_avg=cpu_percent,
            context=context or {}
        )
        
        self.stage_timers[stage_id] = stage_metrics
        
        if self.enable_detailed_logging:
            self._log_event("stage_started", {
                'stage_name': stage_name,
                'stage_id': stage_id,
                'context': context
            })
        
        return stage_id    
   
 def end_stage(self, stage_id: str, rows_processed: int = 0, 
                  success: bool = True, error_message: Optional[str] = None) -> StageMetrics:
        """End monitoring for a processing stage"""
        if not self.monitoring_active:
            raise RuntimeError("Monitoring is not active")
        
        if stage_id not in self.stage_timers:
            raise ValueError(f"Stage ID {stage_id} not found in active stages")
        
        stage_metrics = self.stage_timers[stage_id]
        stage_metrics.end_time = time.time()
        stage_metrics.duration = stage_metrics.end_time - stage_metrics.start_time
        stage_metrics.rows_processed = rows_processed
        stage_metrics.success = success
        stage_metrics.error_message = error_message
        
        if stage_metrics.duration > 0 and rows_processed > 0:
            stage_metrics.rows_per_second = rows_processed / stage_metrics.duration
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            stage_metrics.memory_end_mb = memory_mb
            stage_metrics.memory_delta_mb = memory_mb - stage_metrics.memory_start_mb
        except:
            pass
        
        self.completed_stages.append(stage_metrics)
        del self.stage_timers[stage_id]
        
        self.total_rows_processed += rows_processed
        
        self._check_stage_performance(stage_metrics)
        
        if self.enable_detailed_logging:
            self._log_event("stage_completed", {
                'stage_name': stage_metrics.stage_name,
                'duration': stage_metrics.duration,
                'rows_processed': rows_processed,
                'rows_per_second': stage_metrics.rows_per_second,
                'memory_delta_mb': stage_metrics.memory_delta_mb,
                'success': success
            })
        
        return stage_metrics
    
    def add_quality_flag(self, flag: str, severity: str = "warning"):
        """Add a data quality flag"""
        self.quality_flags.append({
            'flag': flag,
            'severity': severity,
            'timestamp': time.time()
        })    
 
   def get_live_performance_stats(self) -> Dict[str, Any]:
        """Get live performance statistics during monitoring"""
        if not self.monitoring_active:
            return {}
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        try:
            process = psutil.Process()
            current_memory_mb = process.memory_info().rss / (1024**2)
        except:
            current_memory_mb = 0.0
        
        return {
            'elapsed_time': elapsed_time,
            'total_rows_processed': self.total_rows_processed,
            'current_rows_per_second': self.total_rows_processed / elapsed_time if elapsed_time > 0 else 0,
            'completed_stages': len(self.completed_stages),
            'active_stages': len(self.stage_timers),
            'current_memory_mb': current_memory_mb,
            'quality_flags_count': len(self.quality_flags),
            'performance_events_count': len(self.performance_events)
        }
    
    def _generate_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report"""
        if not self.end_time or not self.start_time:
            raise RuntimeError("Monitoring not properly completed")
        
        total_duration = self.end_time - self.start_time
        
        memory_values = []
        for stage in self.completed_stages:
            if stage.memory_start_mb > 0:
                memory_values.append(stage.memory_start_mb)
            if stage.memory_end_mb > 0:
                memory_values.append(stage.memory_end_mb)
        
        if memory_values:
            peak_memory_mb = max(memory_values)
            avg_memory_mb = np.mean(memory_values)
        else:
            peak_memory_mb = 0.0
            avg_memory_mb = 0.0
        
        overall_rows_per_second = self.total_rows_processed / total_duration if total_duration > 0 else 0
        memory_efficiency_score = self._calculate_memory_efficiency_score(peak_memory_mb)
        bottleneck_analysis = self._analyze_bottlenecks()
        optimization_recommendations = self._generate_optimization_recommendations(peak_memory_mb, bottleneck_analysis)
        
        memory_usage_pattern = {
            'peak_memory_mb': peak_memory_mb,
            'avg_memory_mb': avg_memory_mb,
            'pattern': 'analyzed' if memory_values else 'no_data'
        }
        
        return PerformanceReport(
            total_duration=total_duration,
            total_rows_processed=self.total_rows_processed,
            overall_rows_per_second=overall_rows_per_second,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            memory_efficiency_score=memory_efficiency_score,
            stage_metrics=self.completed_stages,
            bottleneck_analysis=bottleneck_analysis,
            memory_usage_pattern=memory_usage_pattern,
            optimization_recommendations=optimization_recommendations,
            quality_flags=[f['flag'] for f in self.quality_flags],
            timestamp=datetime.now()
        )  
  
    def _analyze_bottlenecks(self) -> BottleneckAnalysis:
        """Analyze performance data to identify bottlenecks"""
        if not self.completed_stages:
            return BottleneckAnalysis(
                slowest_stage="none",
                slowest_duration=0.0,
                memory_intensive_stage="none",
                peak_memory_usage=0.0,
                cpu_intensive_stage="none",
                peak_cpu_usage=0.0,
                recommendations=[],
                optimization_opportunities=[]
            )
        
        slowest_stage = max(self.completed_stages, key=lambda s: s.duration or 0)
        memory_intensive_stage = max(self.completed_stages, key=lambda s: s.memory_delta_mb)
        cpu_intensive_stage = max(self.completed_stages, key=lambda s: s.cpu_percent_avg)
        
        recommendations = []
        optimization_opportunities = []
        
        slow_stages = [s for s in self.completed_stages if (s.duration or 0) > 30.0]
        if slow_stages:
            recommendations.append(f"Optimize {len(slow_stages)} slow stages taking >30s")
            for stage in slow_stages:
                if stage.rows_per_second < 1000:
                    optimization_opportunities.append(f"Improve {stage.stage_name} throughput ({stage.rows_per_second:.0f} rows/sec)")
        
        memory_spikes = [s for s in self.completed_stages if s.memory_delta_mb > 500]
        if memory_spikes:
            recommendations.append(f"Fix {len(memory_spikes)} stages with large memory increases")
            for stage in memory_spikes:
                optimization_opportunities.append(f"Reduce memory usage in {stage.stage_name} (+{stage.memory_delta_mb:.0f}MB)")
        
        return BottleneckAnalysis(
            slowest_stage=slowest_stage.stage_name,
            slowest_duration=slowest_stage.duration or 0,
            memory_intensive_stage=memory_intensive_stage.stage_name,
            peak_memory_usage=memory_intensive_stage.memory_delta_mb,
            cpu_intensive_stage=cpu_intensive_stage.stage_name,
            peak_cpu_usage=cpu_intensive_stage.cpu_percent_avg,
            recommendations=recommendations,
            optimization_opportunities=optimization_opportunities
        )
    
    def _calculate_memory_efficiency_score(self, peak_memory_mb: float) -> float:
        """Calculate overall memory efficiency score (0-100)"""
        if peak_memory_mb <= 0:
            return 100.0
        
        peak_memory_gb = peak_memory_mb / 1024
        target_memory_gb = self.performance_targets['max_memory_usage_gb']
        usage_score = max(0, 100 - (peak_memory_gb / target_memory_gb * 100))
        
        return usage_score 
   
    def _generate_optimization_recommendations(self, peak_memory_mb: float,
                                            bottleneck_analysis: BottleneckAnalysis) -> List[str]:
        """Generate specific optimization recommendations"""
        recommendations = []
        
        recommendations.extend(bottleneck_analysis.recommendations)
        recommendations.extend(bottleneck_analysis.optimization_opportunities)
        
        if peak_memory_mb > self.performance_targets['max_memory_usage_gb'] * 1024:
            recommendations.append("Consider reducing chunk size to lower peak memory usage")
        
        overall_throughput = sum(s.rows_per_second for s in self.completed_stages if s.rows_per_second > 0)
        if overall_throughput < self.performance_targets['min_rows_per_second']:
            recommendations.append("Overall throughput below target - consider vectorization or parallel processing")
        
        slow_stages = [s for s in self.completed_stages if (s.duration or 0) > self.performance_targets['max_stage_duration_minutes'] * 60]
        if slow_stages:
            for stage in slow_stages:
                recommendations.append(f"Optimize {stage.stage_name} - taking {stage.duration/60:.1f} minutes")
        
        return list(set(recommendations))
    
    def _check_stage_performance(self, stage_metrics: StageMetrics):
        """Check stage performance against targets and add quality flags"""
        if stage_metrics.duration and stage_metrics.duration > self.performance_targets['max_stage_duration_minutes'] * 60:
            self.add_quality_flag(
                f"Stage {stage_metrics.stage_name} exceeded duration target ({stage_metrics.duration/60:.1f} min)",
                "warning"
            )
        
        if stage_metrics.rows_per_second > 0 and stage_metrics.rows_per_second < self.performance_targets['min_rows_per_second']:
            self.add_quality_flag(
                f"Stage {stage_metrics.stage_name} below throughput target ({stage_metrics.rows_per_second:.0f} rows/sec)",
                "warning"
            )
        
        if stage_metrics.memory_delta_mb > 1000:
            self.add_quality_flag(
                f"Stage {stage_metrics.stage_name} high memory usage (+{stage_metrics.memory_delta_mb:.0f}MB)",
                "warning"
            )
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a performance event"""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'data': data
        }
        self.performance_events.append(event)
        
        if len(self.performance_events) > 1000:
            self.performance_events = self.performance_events[-1000:]

def create_performance_monitor(enable_memory_tracking: bool = True,
                             memory_sampling_interval: float = 1.0,
                             enable_detailed_logging: bool = True) -> ComprehensivePerformanceMonitor:
    """Factory function to create a comprehensive performance monitor"""
    return ComprehensivePerformanceMonitor(
        enable_memory_tracking=enable_memory_tracking,
        memory_sampling_interval=memory_sampling_interval,
        enable_detailed_logging=enable_detailed_logging
    )

def generate_performance_report_json(report: PerformanceReport, output_path: Optional[str] = None) -> str:
    """Generate JSON performance report"""
    report_dict = asdict(report)
    report_dict['timestamp'] = report.timestamp.isoformat()
    json_str = json.dumps(report_dict, indent=2, default=str)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(json_str)
    
    return json_str