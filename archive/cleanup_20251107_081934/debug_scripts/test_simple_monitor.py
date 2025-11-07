#!/usr/bin/env python3
"""
Simple test for comprehensive performance monitoring system
"""

import time
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class StageMetrics:
    """Metrics for a single processing stage"""
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_delta_mb: float = 0.0
    rows_processed: int = 0
    rows_per_second: float = 0.0
    success: bool = True

@dataclass
class BottleneckAnalysis:
    """Analysis of processing bottlenecks"""
    slowest_stage: str
    slowest_duration: float
    memory_intensive_stage: str
    peak_memory_usage: float
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
    optimization_recommendations: List[str]
    quality_flags: List[str]
    timestamp: datetime

class ComprehensivePerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self):
        self.monitoring_active = False
        self.start_time = None
        self.end_time = None
        self.total_rows_processed = 0
        self.stage_timers = {}
        self.completed_stages = []
        self.quality_flags = []
    
    def start_monitoring(self, expected_rows: int = 0):
        """Start comprehensive performance monitoring"""
        self.monitoring_active = True
        self.start_time = time.time()
        self.end_time = None
        self.total_rows_processed = 0
        self.completed_stages = []
        self.quality_flags = []
        print(f"🚀 Started monitoring (expecting {expected_rows:,} rows)")
    
    def stop_monitoring(self) -> PerformanceReport:
        """Stop monitoring and generate comprehensive performance report"""
        if not self.monitoring_active:
            raise RuntimeError("Monitoring is not active")
        
        self.end_time = time.time()
        self.monitoring_active = False
        
        report = self._generate_performance_report()
        print(f"📊 Monitoring completed - {report.total_duration:.2f}s, {report.total_rows_processed:,} rows")
        
        return report
    
    def start_stage(self, stage_name: str, context: Optional[Dict] = None) -> str:
        """Start monitoring a processing stage"""
        if not self.monitoring_active:
            raise RuntimeError("Monitoring is not active")
        
        stage_id = f"{stage_name}_{int(time.time() * 1000)}"
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
        except:
            memory_mb = 0.0
        
        stage_metrics = StageMetrics(
            stage_name=stage_name,
            start_time=time.time(),
            memory_start_mb=memory_mb
        )
        
        self.stage_timers[stage_id] = stage_metrics
        return stage_id
    
    def end_stage(self, stage_id: str, rows_processed: int = 0, 
                  success: bool = True) -> StageMetrics:
        """End monitoring for a processing stage"""
        if stage_id not in self.stage_timers:
            raise ValueError(f"Stage ID {stage_id} not found")
        
        stage_metrics = self.stage_timers[stage_id]
        stage_metrics.end_time = time.time()
        stage_metrics.duration = stage_metrics.end_time - stage_metrics.start_time
        stage_metrics.rows_processed = rows_processed
        stage_metrics.success = success
        
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
        
        return stage_metrics
    
    def add_quality_flag(self, flag: str, severity: str = "warning"):
        """Add a data quality flag"""
        self.quality_flags.append(flag)
    
    def _generate_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report"""
        total_duration = self.end_time - self.start_time
        
        memory_values = []
        for stage in self.completed_stages:
            if stage.memory_start_mb > 0:
                memory_values.append(stage.memory_start_mb)
            if stage.memory_end_mb > 0:
                memory_values.append(stage.memory_end_mb)
        
        peak_memory_mb = max(memory_values) if memory_values else 0.0
        avg_memory_mb = np.mean(memory_values) if memory_values else 0.0
        
        overall_rows_per_second = self.total_rows_processed / total_duration if total_duration > 0 else 0
        memory_efficiency_score = max(0, 100 - (peak_memory_mb / 8192 * 100))  # 8GB target
        
        bottleneck_analysis = self._analyze_bottlenecks()
        optimization_recommendations = self._generate_optimization_recommendations(bottleneck_analysis)
        
        return PerformanceReport(
            total_duration=total_duration,
            total_rows_processed=self.total_rows_processed,
            overall_rows_per_second=overall_rows_per_second,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            memory_efficiency_score=memory_efficiency_score,
            stage_metrics=self.completed_stages,
            bottleneck_analysis=bottleneck_analysis,
            optimization_recommendations=optimization_recommendations,
            quality_flags=self.quality_flags,
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
                recommendations=[],
                optimization_opportunities=[]
            )
        
        slowest_stage = max(self.completed_stages, key=lambda s: s.duration or 0)
        memory_intensive_stage = max(self.completed_stages, key=lambda s: s.memory_delta_mb)
        
        recommendations = []
        optimization_opportunities = []
        
        slow_stages = [s for s in self.completed_stages if (s.duration or 0) > 5.0]
        if slow_stages:
            recommendations.append(f"Optimize {len(slow_stages)} slow stages taking >5s")
        
        memory_spikes = [s for s in self.completed_stages if s.memory_delta_mb > 100]
        if memory_spikes:
            recommendations.append(f"Fix {len(memory_spikes)} stages with large memory increases")
        
        return BottleneckAnalysis(
            slowest_stage=slowest_stage.stage_name,
            slowest_duration=slowest_stage.duration or 0,
            memory_intensive_stage=memory_intensive_stage.stage_name,
            peak_memory_usage=memory_intensive_stage.memory_delta_mb,
            recommendations=recommendations,
            optimization_opportunities=optimization_opportunities
        )
    
    def _generate_optimization_recommendations(self, bottleneck_analysis: BottleneckAnalysis) -> List[str]:
        """Generate specific optimization recommendations"""
        recommendations = []
        recommendations.extend(bottleneck_analysis.recommendations)
        recommendations.extend(bottleneck_analysis.optimization_opportunities)
        return recommendations

def simulate_processing_stage(monitor, stage_name, duration_seconds, rows_processed):
    """Simulate a processing stage for testing"""
    print(f"  🔄 Starting {stage_name}...")
    
    stage_id = monitor.start_stage(stage_name)
    
    # Simulate processing work
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        time.sleep(0.1)
    
    stage_metrics = monitor.end_stage(stage_id, rows_processed, success=True)
    
    print(f"    ✅ Completed {stage_name} in {stage_metrics.duration:.2f}s "
          f"({stage_metrics.rows_per_second:.0f} rows/sec, "
          f"{stage_metrics.memory_delta_mb:+.1f}MB)")
    
    return stage_metrics

def main():
    """Test the comprehensive performance monitoring system"""
    print("🧪 Testing Comprehensive Performance Monitor")
    print("=" * 50)
    
    # Create monitor
    monitor = ComprehensivePerformanceMonitor()
    
    # Start monitoring
    expected_rows = 50000
    monitor.start_monitoring(expected_rows)
    
    try:
        # Simulate various processing stages
        simulate_processing_stage(monitor, "data_loading", 1.0, 10000)
        simulate_processing_stage(monitor, "data_cleaning", 0.8, 8000)
        simulate_processing_stage(monitor, "feature_engineering", 2.0, 15000)
        simulate_processing_stage(monitor, "weighted_labeling", 3.0, 12000)
        simulate_processing_stage(monitor, "validation", 0.5, 5000)
        
        # Add some quality flags
        monitor.add_quality_flag("Test quality flag 1")
        monitor.add_quality_flag("Test quality flag 2")
        
        # Stop monitoring and get report
        report = monitor.stop_monitoring()
        
        print(f"\n📈 Performance Report Summary:")
        print(f"  - Total duration: {report.total_duration:.2f}s")
        print(f"  - Total rows: {report.total_rows_processed:,}")
        print(f"  - Overall throughput: {report.overall_rows_per_second:.0f} rows/sec")
        print(f"  - Peak memory: {report.peak_memory_mb:.1f}MB")
        print(f"  - Memory efficiency: {report.memory_efficiency_score:.1f}%")
        print(f"  - Stages completed: {len(report.stage_metrics)}")
        
        # Show bottleneck analysis
        bottlenecks = report.bottleneck_analysis
        print(f"\n🔍 Bottleneck Analysis:")
        print(f"  - Slowest stage: {bottlenecks.slowest_stage} ({bottlenecks.slowest_duration:.2f}s)")
        print(f"  - Memory intensive: {bottlenecks.memory_intensive_stage} (+{bottlenecks.peak_memory_usage:.1f}MB)")
        
        # Show recommendations
        if report.optimization_recommendations:
            print(f"\n💡 Optimization Recommendations:")
            for i, rec in enumerate(report.optimization_recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Show quality flags
        if report.quality_flags:
            print(f"\n⚠️  Quality Flags:")
            for flag in report.quality_flags:
                print(f"  - {flag}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)