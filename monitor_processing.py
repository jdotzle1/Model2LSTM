#!/usr/bin/env python3
"""
Real-time Processing Monitor for Data Processing Pipeline
Monitors processing performance, data quality, and system resources
"""

import time
import json
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import boto3
from typing import Dict, List, Optional

class ProcessingMonitor:
    def __init__(self, config_file: str = "monitor_config.json"):
        self.config = self._load_config(config_file)
        self.alerts = []
        self.metrics_history = []
        self.setup_logging()
        
    def _load_config(self, config_file: str) -> Dict:
        """Load monitoring configuration"""
        default_config = {
            "thresholds": {
                "processing_time_minutes": 30,
                "memory_usage_gb": 8,
                "cpu_usage_percent": 80,
                "disk_free_gb": 100,
                "quality_score": 0.8,
                "win_rate_min": 0.05,
                "win_rate_max": 0.50,
                "feature_nan_percent": 0.35
            },
            "alert_settings": {
                "email_enabled": True,
                "email_recipients": ["data-eng@company.com"],
                "log_alerts": True,
                "dashboard_updates": True
            },
            "monitoring_intervals": {
                "real_time_seconds": 30,
                "quality_check_minutes": 15,
                "system_health_minutes": 5
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            return default_config
    
    def setup_logging(self):
        """Setup monitoring logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/tmp/monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ProcessingMonitor')
    
    def collect_system_metrics(self) -> Dict:
        """Collect current system metrics"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get processing-specific metrics
        processing_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                if 'process_monthly' in proc.info['name']:
                    processing_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'cpu_percent': cpu_percent
            },
            'processing': {
                'active_processes': len(processing_processes),
                'total_memory_mb': sum(p.get('memory_info', {}).get('rss', 0) for p in processing_processes) / (1024**2),
                'processes': processing_processes
            }
        }
        
        return metrics
    
    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        self.logger.info("Starting monitoring cycle")
        
        # Collect all metrics
        system_metrics = self.collect_system_metrics()
        
        # Combine metrics
        all_metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': system_metrics.get('system', {}),
            'processing': system_metrics.get('processing', {})
        }
        
        # Save metrics to file
        metrics_file = Path('/tmp/current_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        self.logger.info("Monitoring cycle completed.")
        return all_metrics

def main():
    """Main monitoring function"""
    monitor = ProcessingMonitor()
    metrics = monitor.run_monitoring_cycle()
    print("Monitoring completed.")

if __name__ == "__main__":
    main()