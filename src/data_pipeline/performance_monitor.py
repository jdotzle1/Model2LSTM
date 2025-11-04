"""
Performance Monitoring and Optimization Module

This module provides performance tracking, memory monitoring, and optimization
utilities for the weighted labeling system to meet the 60-minute target for 10M rows.
"""

import time
import psutil
import numpy as np
import pandas as pd
import gc
import threading
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot"""
    timestamp: float
    rss_mb: float
    vms_mb: float
    available_mb: float
    percent_used: float
    
    @property
    def rss_gb(self) -> float:
        return self.rss_mb / 1024.0


@dataclass
class PerformanceMetrics:
    """Container for performance metrics with enhanced memory tracking"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    rows_processed: int = 0
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)
    processing_stages: Dict[str, float] = field(default_factory=dict)
    gc_events: List[Tuple[float, str]] = field(default_factory=list)  # (timestamp, reason)
    memory_cleanup_events: List[Tuple[float, float, float]] = field(default_factory=list)  # (timestamp, before_mb, after_mb)
    
    @property
    def elapsed_time(self) -> float:
        """Total elapsed time in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def rows_per_minute(self) -> float:
        """Processing speed in rows per minute"""
        if self.elapsed_time == 0:
            return 0.0
        return (self.rows_processed / self.elapsed_time) * 60
    
    @property
    def peak_memory_mb(self) -> float:
        """Peak memory usage in MB"""
        return max(snapshot.rss_mb for snapshot in self.memory_snapshots) if self.memory_snapshots else 0.0
    
    @property
    def peak_memory_gb(self) -> float:
        """Peak memory usage in GB"""
        return self.peak_memory_mb / 1024.0
    
    @property
    def memory_usage_mb(self) -> List[float]:
        """Legacy compatibility - return RSS memory usage"""
        return [snapshot.rss_mb for snapshot in self.memory_snapshots]
    
    @property
    def current_memory_mb(self) -> float:
        """Current memory usage in MB"""
        return self.memory_snapshots[-1].rss_mb if self.memory_snapshots else 0.0
    
    @property
    def memory_growth_rate_mb_per_minute(self) -> float:
        """Memory growth rate in MB per minute"""
        if len(self.memory_snapshots) < 2:
            return 0.0
        
        first = self.memory_snapshots[0]
        last = self.memory_snapshots[-1]
        time_diff = (last.timestamp - first.timestamp) / 60.0  # minutes
        
        if time_diff == 0:
            return 0.0
        
        return (last.rss_mb - first.rss_mb) / time_diff


class MemoryManager:
    """Enhanced memory management with automatic cleanup triggers"""
    
    def __init__(self, memory_limit_gb: float = 8.0, cleanup_threshold: float = 0.8):
        """
        Initialize memory manager
        
        Args:
            memory_limit_gb: Memory usage limit in GB
            cleanup_threshold: Trigger cleanup when memory usage exceeds this fraction of limit
        """
        self.memory_limit_gb = memory_limit_gb
        self.cleanup_threshold = cleanup_threshold
        self.cleanup_threshold_mb = memory_limit_gb * 1024 * cleanup_threshold
        self._process = psutil.Process()
        self.cleanup_callbacks: List[Callable] = []
        
    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register a callback to be called during memory cleanup"""
        self.cleanup_callbacks.append(callback)
    
    def check_memory_and_cleanup(self, force: bool = False) -> Tuple[bool, float, float]:
        """
        Check memory usage and trigger cleanup if needed
        
        Args:
            force: Force cleanup regardless of threshold
            
        Returns:
            Tuple of (cleanup_triggered, memory_before_mb, memory_after_mb)
        """
        memory_before = self.get_current_memory_mb()
        
        if force or memory_before > self.cleanup_threshold_mb:
            # Call registered cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Warning: Cleanup callback failed: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Additional cleanup for numpy/pandas
            try:
                import numpy as np
                # Clear numpy cache
                np.seterr(all='ignore')  # Suppress warnings during cleanup
            except:
                pass
            
            memory_after = self.get_current_memory_mb()
            return True, memory_before, memory_after
        
        return False, memory_before, memory_before
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        memory_info = self._process.memory_info()
        return memory_info.rss / (1024 ** 2)
    
    def get_memory_snapshot(self) -> MemorySnapshot:
        """Get detailed memory snapshot"""
        memory_info = self._process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / (1024 ** 2),
            vms_mb=memory_info.vms / (1024 ** 2),
            available_mb=system_memory.available / (1024 ** 2),
            percent_used=system_memory.percent
        )
    
    def optimize_processing_order(self, chunk_indices: List[int]) -> List[int]:
        """
        Optimize processing order to minimize memory fragmentation
        
        Args:
            chunk_indices: List of chunk indices to process
            
        Returns:
            Optimized order of chunk indices
        """
        # For now, use sequential order but could implement more sophisticated strategies
        # like processing smaller chunks first or interleaving different types of operations
        return sorted(chunk_indices)


class PerformanceMonitor:
    """Enhanced performance monitoring with automatic memory management"""
    
    def __init__(self, target_rows_per_minute: int = 167_000, 
                 memory_limit_gb: float = 8.0,
                 enable_auto_cleanup: bool = True,
                 cleanup_frequency: int = 10):
        """
        Initialize performance monitor with enhanced memory management
        
        Args:
            target_rows_per_minute: Target processing speed (167K for 10M in 60 min)
            memory_limit_gb: Memory usage limit in GB
            enable_auto_cleanup: Enable automatic memory cleanup
            cleanup_frequency: Trigger cleanup check every N progress updates
        """
        self.target_rows_per_minute = target_rows_per_minute
        self.memory_limit_gb = memory_limit_gb
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_frequency = cleanup_frequency
        
        self.metrics = PerformanceMetrics()
        self.memory_manager = MemoryManager(memory_limit_gb)
        self._process = psutil.Process()
        self._update_counter = 0
    
    def start_monitoring(self, total_rows: int) -> None:
        """
        Start performance monitoring with enhanced memory tracking
        
        Args:
            total_rows: Total number of rows to process
        """
        self.metrics = PerformanceMetrics()
        self.metrics.rows_processed = 0
        self.metrics.start_time = time.time()
        self._update_counter = 0
        
        # Record initial memory snapshot
        self._record_memory_snapshot()
        
        # Initial memory cleanup to start fresh
        if self.enable_auto_cleanup:
            cleanup_triggered, before_mb, after_mb = self.memory_manager.check_memory_and_cleanup(force=True)
            if cleanup_triggered:
                self.metrics.memory_cleanup_events.append((time.time(), before_mb, after_mb))
                self.metrics.gc_events.append((time.time(), "initial_cleanup"))
        
        print(f"Enhanced performance monitoring started for {total_rows:,} rows")
        print(f"Target: {self.target_rows_per_minute:,} rows/minute")
        print(f"Memory limit: {self.memory_limit_gb:.1f} GB")
        print(f"Auto cleanup: {'enabled' if self.enable_auto_cleanup else 'disabled'}")
        
        initial_memory = self.get_current_memory_gb()
        print(f"Initial memory usage: {initial_memory:.2f} GB")
    
    def update_progress(self, rows_processed: int, stage: str = None) -> None:
        """
        Update processing progress with enhanced memory monitoring
        
        Args:
            rows_processed: Number of rows processed so far
            stage: Optional stage name for detailed tracking
        """
        self.metrics.rows_processed = rows_processed
        self._update_counter += 1
        
        # Record memory snapshot
        self._record_memory_snapshot()
        
        # Record stage timing if provided
        if stage:
            self.metrics.processing_stages[stage] = time.time()
        
        # Check for automatic memory cleanup
        if self.enable_auto_cleanup and self._update_counter % self.cleanup_frequency == 0:
            cleanup_triggered, before_mb, after_mb = self.memory_manager.check_memory_and_cleanup()
            if cleanup_triggered:
                self.metrics.memory_cleanup_events.append((time.time(), before_mb, after_mb))
                self.metrics.gc_events.append((time.time(), f"auto_cleanup_update_{self._update_counter}"))
                
                memory_saved = before_mb - after_mb
                if memory_saved > 100:  # Only report significant cleanups
                    print(f"🧹 Memory cleanup: {before_mb:.0f} MB → {after_mb:.0f} MB "
                          f"(saved {memory_saved:.0f} MB)")
        
        # Check memory limit and warn
        current_memory_gb = self.get_current_memory_gb()
        if current_memory_gb > self.memory_limit_gb:
            print(f"⚠ Memory usage ({current_memory_gb:.1f} GB) exceeds limit "
                  f"({self.memory_limit_gb:.1f} GB)")
            
            # Force cleanup if significantly over limit
            if current_memory_gb > self.memory_limit_gb * 1.2:
                print("🚨 Forcing emergency memory cleanup...")
                cleanup_triggered, before_mb, after_mb = self.memory_manager.check_memory_and_cleanup(force=True)
                if cleanup_triggered:
                    self.metrics.memory_cleanup_events.append((time.time(), before_mb, after_mb))
                    self.metrics.gc_events.append((time.time(), "emergency_cleanup"))
                    print(f"   Emergency cleanup: {before_mb:.0f} MB → {after_mb:.0f} MB")
    
    def finish_monitoring(self) -> PerformanceMetrics:
        """
        Finish monitoring and return final metrics with cleanup
        
        Returns:
            Final performance metrics
        """
        self.metrics.end_time = time.time()
        
        # Final memory cleanup
        if self.enable_auto_cleanup:
            cleanup_triggered, before_mb, after_mb = self.memory_manager.check_memory_and_cleanup(force=True)
            if cleanup_triggered:
                self.metrics.memory_cleanup_events.append((time.time(), before_mb, after_mb))
                self.metrics.gc_events.append((time.time(), "final_cleanup"))
        
        # Record final memory snapshot
        self._record_memory_snapshot()
        
        return self.metrics
    
    def get_current_memory_gb(self) -> float:
        """Get current memory usage in GB"""
        memory_info = self._process.memory_info()
        return memory_info.rss / (1024 ** 3)  # Convert bytes to GB
    
    def _record_memory_snapshot(self) -> None:
        """Record detailed memory snapshot"""
        snapshot = self.memory_manager.get_memory_snapshot()
        self.metrics.memory_snapshots.append(snapshot)
    
    def get_memory_manager(self) -> MemoryManager:
        """Get the memory manager instance"""
        return self.memory_manager
    
    def validate_performance_target(self, total_rows: int = 10_000_000) -> Dict[str, bool]:
        """
        Validate performance against targets
        
        Args:
            total_rows: Total rows for projection (default 10M)
            
        Returns:
            Dictionary with validation results
        """
        current_speed = self.metrics.rows_per_minute
        projected_time_minutes = total_rows / current_speed if current_speed > 0 else float('inf')
        
        results = {
            'speed_target_met': current_speed >= self.target_rows_per_minute,
            'memory_target_met': self.metrics.peak_memory_gb <= self.memory_limit_gb,
            'time_target_met': projected_time_minutes <= 60.0,
            'current_speed': current_speed,
            'projected_time_minutes': projected_time_minutes,
            'peak_memory_gb': self.metrics.peak_memory_gb
        }
        
        return results
    
    def print_performance_report(self) -> None:
        """Print comprehensive performance report with enhanced memory analysis"""
        print("\n" + "="*70)
        print("ENHANCED PERFORMANCE REPORT")
        print("="*70)
        
        # Basic metrics
        print(f"Rows processed: {self.metrics.rows_processed:,}")
        print(f"Elapsed time: {self.metrics.elapsed_time:.1f} seconds")
        print(f"Processing speed: {self.metrics.rows_per_minute:,.0f} rows/minute")
        
        # Enhanced memory metrics
        print(f"\nMemory Analysis:")
        print(f"  Peak memory usage: {self.metrics.peak_memory_gb:.2f} GB")
        print(f"  Current memory usage: {self.metrics.current_memory_mb:.0f} MB")
        print(f"  Memory growth rate: {self.metrics.memory_growth_rate_mb_per_minute:.1f} MB/min")
        print(f"  Memory snapshots recorded: {len(self.metrics.memory_snapshots)}")
        
        # Memory cleanup statistics
        if self.metrics.memory_cleanup_events:
            total_cleaned = sum(before - after for _, before, after in self.metrics.memory_cleanup_events)
            print(f"  Memory cleanup events: {len(self.metrics.memory_cleanup_events)}")
            print(f"  Total memory cleaned: {total_cleaned:.0f} MB")
            print(f"  GC events: {len(self.metrics.gc_events)}")
        
        # Target validation
        validation = self.validate_performance_target()
        print(f"\nTarget Validation:")
        print(f"  Speed target (167K/min): {'✓' if validation['speed_target_met'] else '❌'} "
              f"({validation['current_speed']:,.0f})")
        print(f"  Memory target (8GB): {'✓' if validation['memory_target_met'] else '❌'} "
              f"({validation['peak_memory_gb']:.2f} GB)")
        print(f"  Time target (60 min for 10M): {'✓' if validation['time_target_met'] else '❌'} "
              f"({validation['projected_time_minutes']:.1f} min projected)")
        
        # Stage breakdown if available
        if self.metrics.processing_stages:
            print(f"\nStage Breakdown:")
            prev_time = self.metrics.start_time
            for stage, stage_time in self.metrics.processing_stages.items():
                duration = stage_time - prev_time
                print(f"  {stage}: {duration:.1f}s")
                prev_time = stage_time
        
        # Memory optimization recommendations
        self._print_memory_recommendations()
    
    def _print_memory_recommendations(self) -> None:
        """Print memory optimization recommendations"""
        print(f"\nMemory Optimization Recommendations:")
        
        # Analyze memory growth
        if self.metrics.memory_growth_rate_mb_per_minute > 50:
            print("  ⚠ High memory growth rate detected - check for memory leaks")
        
        # Analyze cleanup effectiveness
        if self.metrics.memory_cleanup_events:
            avg_cleanup = sum(before - after for _, before, after in self.metrics.memory_cleanup_events) / len(self.metrics.memory_cleanup_events)
            if avg_cleanup < 50:
                print("  ⚠ Low cleanup effectiveness - consider more aggressive cleanup strategies")
            else:
                print(f"  ✓ Good cleanup effectiveness: {avg_cleanup:.0f} MB average per cleanup")
        
        # Peak memory analysis
        if self.metrics.peak_memory_gb > self.memory_limit_gb * 0.9:
            print(f"  ⚠ Peak memory usage near limit - consider reducing chunk size")
        else:
            print(f"  ✓ Peak memory usage within safe limits")
        
        # Processing order recommendations
        if len(self.metrics.memory_snapshots) > 10:
            memory_variance = np.var([s.rss_mb for s in self.metrics.memory_snapshots])
            if memory_variance > 10000:  # High variance
                print("  💡 Consider optimizing processing order to reduce memory fragmentation")


@contextmanager
def performance_context(monitor: PerformanceMonitor, total_rows: int):
    """
    Context manager for performance monitoring
    
    Args:
        monitor: PerformanceMonitor instance
        total_rows: Total number of rows to process
    """
    monitor.start_monitoring(total_rows)
    try:
        yield monitor
    finally:
        monitor.finish_monitoring()
        monitor.print_performance_report()


class OptimizedCalculations:
    """Optimized numerical computations using numpy vectorization"""
    
    @staticmethod
    def vectorized_price_hits(highs: np.ndarray, lows: np.ndarray, 
                             target_prices: np.ndarray, stop_prices: np.ndarray,
                             directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized calculation of target and stop hits for multiple entries
        
        Args:
            highs: Array of high prices
            lows: Array of low prices  
            target_prices: Array of target prices for each entry
            stop_prices: Array of stop prices for each entry
            directions: Array of directions (1 for long, -1 for short)
            
        Returns:
            Tuple of (target_hits, stop_hits) boolean arrays
        """
        # Long trades: target hit when high >= target, stop hit when low <= stop
        long_mask = directions == 1
        long_target_hits = np.zeros_like(highs, dtype=bool)
        long_stop_hits = np.zeros_like(lows, dtype=bool)
        
        if long_mask.any():
            long_target_hits[long_mask] = highs[long_mask] >= target_prices[long_mask]
            long_stop_hits[long_mask] = lows[long_mask] <= stop_prices[long_mask]
        
        # Short trades: target hit when low <= target, stop hit when high >= stop
        short_mask = directions == -1
        short_target_hits = np.zeros_like(lows, dtype=bool)
        short_stop_hits = np.zeros_like(highs, dtype=bool)
        
        if short_mask.any():
            short_target_hits[short_mask] = lows[short_mask] <= target_prices[short_mask]
            short_stop_hits[short_mask] = highs[short_mask] >= stop_prices[short_mask]
        
        # Combine results
        target_hits = long_target_hits | short_target_hits
        stop_hits = long_stop_hits | short_stop_hits
        
        return target_hits, stop_hits
    
    @staticmethod
    def vectorized_adverse_moves(entry_prices: np.ndarray, highs: np.ndarray, 
                                lows: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation of adverse moves for MAE tracking
        
        Args:
            entry_prices: Array of entry prices
            highs: Array of high prices
            lows: Array of low prices
            directions: Array of directions (1 for long, -1 for short)
            
        Returns:
            Array of adverse moves (always positive)
        """
        # Long trades: adverse move is entry - low (when low < entry)
        long_mask = directions == 1
        long_adverse = np.zeros_like(entry_prices)
        if long_mask.any():
            long_adverse[long_mask] = np.maximum(0, entry_prices[long_mask] - lows[long_mask])
        
        # Short trades: adverse move is high - entry (when high > entry)
        short_mask = directions == -1
        short_adverse = np.zeros_like(entry_prices)
        if short_mask.any():
            short_adverse[short_mask] = np.maximum(0, highs[short_mask] - entry_prices[short_mask])
        
        return long_adverse + short_adverse
    
    @staticmethod
    def vectorized_weight_calculations(mae_ticks: np.ndarray, seconds_to_target: np.ndarray,
                                     stop_ticks: int, months_ago: np.ndarray,
                                     decay_rate: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized calculation of quality, velocity, and time decay weights
        
        Args:
            mae_ticks: Array of MAE values in ticks
            seconds_to_target: Array of seconds to target
            stop_ticks: Stop distance in ticks for quality calculation
            months_ago: Array of months ago for each sample
            decay_rate: Decay rate for time decay calculation
            
        Returns:
            Tuple of (quality_weights, velocity_weights, time_decay_weights)
        """
        # Quality weights: 2.0 - (1.5 × mae_ratio), clipped to [0.5, 2.0]
        mae_ratio = mae_ticks / stop_ticks
        quality_weights = np.clip(2.0 - (1.5 * mae_ratio), 0.5, 2.0)
        
        # Velocity weights: 2.0 - (1.5 × (seconds - 300) / 600), clipped to [0.5, 2.0]
        velocity_weights = np.clip(2.0 - (1.5 * (seconds_to_target - 300) / 600), 0.5, 2.0)
        
        # Time decay weights: exp(-decay_rate × months_ago)
        time_decay_weights = np.exp(-decay_rate * months_ago)
        
        return quality_weights, velocity_weights, time_decay_weights
    
    @staticmethod
    def batch_process_entries(df_chunk: pd.DataFrame, mode_configs: Dict,
                             timeout_seconds: int = 900) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Batch process multiple trading modes simultaneously for a chunk
        
        Args:
            df_chunk: DataFrame chunk to process
            mode_configs: Dictionary of mode configurations
            timeout_seconds: Maximum timeout for lookforward
            
        Returns:
            Dictionary with results for each mode
        """
        n_bars = len(df_chunk)
        opens = df_chunk['open'].values
        highs = df_chunk['high'].values
        lows = df_chunk['low'].values
        timestamps = df_chunk['timestamp'].values
        
        results = {}
        
        # Process each mode
        for mode_name, mode_config in mode_configs.items():
            labels = np.zeros(n_bars, dtype=int)
            mae_ticks = np.full(n_bars, np.nan)
            seconds_to_target = np.full(n_bars, np.nan)
            
            # Calculate target and stop prices for all entries
            entry_prices = opens[1:]  # Next bar's open
            if mode_config['direction'] == 'long':
                target_prices = entry_prices + (mode_config['target_ticks'] * 0.25)
                stop_prices = entry_prices - (mode_config['stop_ticks'] * 0.25)
                directions = np.ones(len(entry_prices))
            else:
                target_prices = entry_prices - (mode_config['target_ticks'] * 0.25)
                stop_prices = entry_prices + (mode_config['stop_ticks'] * 0.25)
                directions = -np.ones(len(entry_prices))
            
            # Process each entry (vectorized where possible)
            for i in range(n_bars - 1):
                entry_price = entry_prices[i]
                target_price = target_prices[i]
                stop_price = stop_prices[i]
                direction = directions[i]
                
                # Look forward up to timeout
                start_idx = i + 1
                end_idx = min(start_idx + timeout_seconds, n_bars)
                
                if start_idx >= n_bars:
                    continue
                
                # Get price arrays for lookforward period
                forward_highs = highs[start_idx:end_idx]
                forward_lows = lows[start_idx:end_idx]
                forward_times = timestamps[start_idx:end_idx]
                
                # Calculate adverse moves for this entry
                adverse_moves = OptimizedCalculations.vectorized_adverse_moves(
                    np.full(len(forward_highs), entry_price),
                    forward_highs, forward_lows,
                    np.full(len(forward_highs), direction)
                )
                
                # Find target and stop hits
                if direction == 1:  # Long
                    target_hits = forward_highs >= target_price
                    stop_hits = forward_lows <= stop_price
                else:  # Short
                    target_hits = forward_lows <= target_price
                    stop_hits = forward_highs >= stop_price
                
                # Determine outcome
                target_hit_indices = np.where(target_hits)[0]
                stop_hit_indices = np.where(stop_hits)[0]
                
                if len(target_hit_indices) > 0 and len(stop_hit_indices) > 0:
                    # Both hit - check which came first
                    first_target = target_hit_indices[0]
                    first_stop = stop_hit_indices[0]
                    
                    if first_target <= first_stop:
                        # Target hit first (or same bar - conservative: assume stop)
                        if first_target < first_stop:
                            labels[i] = 1
                            mae_ticks[i] = np.max(adverse_moves[:first_target + 1]) / 0.25
                            time_diff = forward_times[first_target] - timestamps[start_idx]
                            seconds_to_target[i] = float(time_diff / np.timedelta64(1, 's'))
                elif len(target_hit_indices) > 0:
                    # Only target hit
                    first_target = target_hit_indices[0]
                    labels[i] = 1
                    mae_ticks[i] = np.max(adverse_moves[:first_target + 1]) / 0.25
                    time_diff = forward_times[first_target] - timestamps[start_idx]
                    seconds_to_target[i] = float(time_diff / np.timedelta64(1, 's'))
                # else: only stop hit or timeout - label remains 0
            
            results[mode_name] = {
                'labels': labels,
                'mae_ticks': mae_ticks,
                'seconds_to_target': seconds_to_target
            }
        
        return results


def validate_performance_requirements(monitor: PerformanceMonitor, 
                                    total_rows: int = 10_000_000,
                                    skip_for_small_datasets: bool = True) -> bool:
    """
    Validate that performance requirements are met
    
    Args:
        monitor: PerformanceMonitor with completed metrics
        total_rows: Total rows for validation (default 10M)
        skip_for_small_datasets: Skip validation for datasets < 10K rows
        
    Returns:
        True if all requirements are met
        
    Raises:
        PerformanceError: If requirements are not met
    """
    from .weighted_labeling import PerformanceError
    
    # Skip validation for very small datasets where overhead dominates
    if skip_for_small_datasets and monitor.metrics.rows_processed < 10_000:
        print(f"Skipping performance validation for small dataset ({monitor.metrics.rows_processed:,} rows)")
        return True
    
    validation = monitor.validate_performance_target(total_rows)
    
    if not validation['speed_target_met']:
        raise PerformanceError(
            f"Speed requirement not met: {validation['current_speed']:,.0f} < "
            f"{monitor.target_rows_per_minute:,} rows/minute"
        )
    
    if not validation['memory_target_met']:
        raise PerformanceError(
            f"Memory requirement not met: {validation['peak_memory_gb']:.2f} > "
            f"{monitor.memory_limit_gb:.1f} GB"
        )
    
    if not validation['time_target_met']:
        raise PerformanceError(
            f"Time requirement not met: {validation['projected_time_minutes']:.1f} > "
            f"60 minutes for 10M rows"
        )
    
    return True