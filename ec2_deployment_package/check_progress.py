#!/usr/bin/env python3
"""
Pipeline Progress Checker

Monitors the weighted labeling pipeline progress and provides estimates.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta

def check_pipeline_progress():
    """Check current pipeline progress"""
    
    log_file = "/tmp/es_weighted_pipeline/pipeline.log"
    
    if not os.path.exists(log_file):
        print("❌ Pipeline log file not found")
        print(f"Expected: {log_file}")
        return
    
    print("=== PIPELINE PROGRESS CHECK ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Read log file
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find progress indicators
    current_step = "Unknown"
    rows_processed = 0
    total_rows = 0
    start_time = None
    
    for line in lines:
        if "STEP" in line and "===" in line:
            current_step = line.strip()
        elif "rows processed" in line.lower():
            # Extract numbers from progress lines
            words = line.split()
            for i, word in enumerate(words):
                if word.replace(',', '').isdigit():
                    rows_processed = int(word.replace(',', ''))
                    break
        elif "total rows" in line.lower() or "dataset:" in line.lower():
            words = line.split()
            for word in words:
                if word.replace(',', '').isdigit():
                    total_rows = max(total_rows, int(word.replace(',', '')))
        elif "Pipeline" in line and "complete" in line.lower():
            current_step = "✅ COMPLETE"
    
    print(f"Current Step: {current_step}")
    
    if rows_processed > 0 and total_rows > 0:
        progress_pct = (rows_processed / total_rows) * 100
        print(f"Progress: {rows_processed:,} / {total_rows:,} rows ({progress_pct:.1f}%)")
        
        # Estimate remaining time
        if progress_pct > 5:  # Only estimate after 5% progress
            # Assume linear processing rate
            elapsed_lines = [l for l in lines if "elapsed" in l.lower() or "time:" in l.lower()]
            if elapsed_lines:
                # Simple time estimation based on progress
                remaining_pct = 100 - progress_pct
                estimated_remaining = (remaining_pct / progress_pct) * 60  # Rough estimate in minutes
                print(f"Estimated remaining: {estimated_remaining:.0f} minutes")
    
    # Show recent activity
    print(f"\nRecent Activity (last 5 lines):")
    for line in lines[-5:]:
        print(f"  {line.strip()}")
    
    # Check for errors
    error_lines = [l for l in lines if "error" in l.lower() or "failed" in l.lower()]
    if error_lines:
        print(f"\n⚠️  Potential Issues Found:")
        for line in error_lines[-3:]:  # Show last 3 errors
            print(f"  {line.strip()}")

if __name__ == "__main__":
    check_pipeline_progress()
