#!/bin/bash
# Resource Monitoring for Weighted Labeling Pipeline

echo "=== PIPELINE RESOURCE MONITORING ==="
echo "Timestamp: $(date)"
echo

# System resources
echo "System Resources:"
echo "  CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "  Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
echo

# Pipeline processes
echo "Pipeline Processes:"
pgrep -f "ec2_weighted_labeling_pipeline" | while read pid; do
    if [ -n "$pid" ]; then
        echo "  PID $pid: $(ps -p $pid -o %cpu,%mem,etime,cmd --no-headers)"
    fi
done

# Working directory size
if [ -d "/tmp/es_weighted_pipeline" ]; then
    echo "Working Directory:"
    echo "  Size: $(du -sh /tmp/es_weighted_pipeline | cut -f1)"
    echo "  Files: $(find /tmp/es_weighted_pipeline -type f | wc -l)"
fi

# Recent log entries
if [ -f "/tmp/es_weighted_pipeline/pipeline.log" ]; then
    echo
    echo "Recent Log Entries:"
    tail -5 /tmp/es_weighted_pipeline/pipeline.log | sed 's/^/  /'
fi

echo
echo "=== END MONITORING ==="
