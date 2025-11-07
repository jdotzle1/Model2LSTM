#!/usr/bin/env python3
"""
Re-run June 2011 Processing

Let's re-process the problematic month (June 2011) to verify 
if the bug fix actually resolved the high short win rates.
"""

import subprocess
import sys
import os
import json
import pandas as pd
from datetime import datetime

def clear_existing_results():
    """Clear existing June 2011 results from S3"""
    
    print("🗑️  CLEARING EXISTING JUNE 2011 RESULTS")
    print("=" * 50)
    
    commands = [
        # Clear processed data
        "aws s3 rm s3://es-1-second-data/processed-data/monthly/2011/06/ --recursive --region us-east-1",
        
        # Clear any cached results
        "aws s3 rm s3://es-1-second-data/cache/monthly/2011/06/ --recursive --region us-east-1"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Success")
            else:
                print(f"⚠️  Warning: {result.stderr.strip()}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()

def run_june_2011_processing():
    """Run the June 2011 processing with current (fixed) code"""
    
    print("🚀 RUNNING JUNE 2011 PROCESSING")
    print("=" * 50)
    
    cmd = "python process_monthly_chunks_fixed.py --test-month 2011-06"
    
    print(f"Command: {cmd}")
    print("This will take several minutes...")
    print()
    
    try:
        # Run the processing
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        print()
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            print()
        
        if result.returncode == 0:
            print("✅ Processing completed successfully")
            return True
        else:
            print(f"❌ Processing failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running processing: {e}")
        return False

def download_and_analyze_results():
    """Download and analyze the new results"""
    
    print("📥 DOWNLOADING AND ANALYZING RESULTS")
    print("=" * 50)
    
    # Create temp directory for results
    temp_dir = f"temp_june_2011_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download statistics
    stats_cmd = f"aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/06/statistics/ {temp_dir}/stats/ --recursive --region us-east-1"
    
    print(f"Downloading stats: {stats_cmd}")
    
    try:
        result = subprocess.run(stats_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Statistics downloaded")
        else:
            print(f"❌ Failed to download stats: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return None
    
    # Find and analyze statistics file
    stats_files = []
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('_statistics.json'):
                stats_files.append(os.path.join(root, file))
    
    if not stats_files:
        print("❌ No statistics files found")
        return None
    
    # Analyze the latest statistics file
    latest_stats = max(stats_files, key=os.path.getmtime)
    print(f"📊 Analyzing: {latest_stats}")
    
    try:
        with open(latest_stats, 'r') as f:
            stats = json.load(f)
        
        print("\n🎯 WIN RATE ANALYSIS:")
        print("-" * 30)
        
        # Extract win rates
        win_rates = {}
        if 'label_statistics' in stats:
            for label, data in stats['label_statistics'].items():
                if 'win_rate' in data:
                    win_rates[label] = data['win_rate']
                    print(f"{label}: {data['win_rate']:.1%}")
        
        # Focus on short trades
        short_rates = {k: v for k, v in win_rates.items() if 'short' in k}
        
        print(f"\n📊 SHORT TRADE SUMMARY:")
        print("-" * 30)
        
        if short_rates:
            avg_short_rate = sum(short_rates.values()) / len(short_rates)
            print(f"Average short win rate: {avg_short_rate:.1%}")
            
            if avg_short_rate > 0.5:
                print("🚨 HIGH WIN RATES STILL PRESENT - BUG NOT FIXED!")
                return "BUG_STILL_EXISTS"
            elif avg_short_rate > 0.4:
                print("⚠️  Elevated win rates - needs investigation")
                return "ELEVATED_RATES"
            else:
                print("✅ Normal win rates - bug appears fixed")
                return "BUG_FIXED"
        else:
            print("❌ No short trade data found")
            return "NO_DATA"
            
    except Exception as e:
        print(f"❌ Error analyzing stats: {e}")
        return None

def compare_with_previous_results():
    """Compare with previous problematic results if available"""
    
    print("\n📋 COMPARISON WITH PREVIOUS RESULTS")
    print("=" * 50)
    
    # Check if we have the downloaded statistics file
    downloaded_file = "/C:/Users/jdotzler/Downloads/monthly_2011-06_20251106_213935_statistics.json"
    
    if os.path.exists(downloaded_file):
        print(f"Found previous results: {downloaded_file}")
        
        try:
            with open(downloaded_file, 'r') as f:
                old_stats = json.load(f)
            
            print("Previous (problematic) win rates:")
            if 'label_statistics' in old_stats:
                for label, data in old_stats['label_statistics'].items():
                    if 'short' in label and 'win_rate' in data:
                        print(f"  {label}: {data['win_rate']:.1%}")
            
        except Exception as e:
            print(f"Error reading previous results: {e}")
    else:
        print("No previous results file found for comparison")

def main():
    """Main execution"""
    
    print("🔍 RE-RUNNING JUNE 2011 TO VERIFY BUG FIX")
    print("=" * 60)
    print("This will:")
    print("1. Clear existing June 2011 results from S3")
    print("2. Re-process June 2011 with current (supposedly fixed) code")
    print("3. Download and analyze the new results")
    print("4. Determine if the bug is actually fixed")
    print()
    
    # Step 1: Clear existing results
    clear_existing_results()
    
    # Step 2: Run processing
    success = run_june_2011_processing()
    
    if not success:
        print("❌ Processing failed - cannot verify fix")
        return
    
    # Step 3: Download and analyze
    result = download_and_analyze_results()
    
    # Step 4: Compare with previous
    compare_with_previous_results()
    
    # Final conclusion
    print("\n" + "="*60)
    print("🎯 FINAL VERIFICATION RESULT")
    print("="*60)
    
    if result == "BUG_FIXED":
        print("✅ SUCCESS: Bug appears to be fixed!")
        print("   Short win rates are now normal (~30-40%)")
        print("   Ready to proceed with model training")
    elif result == "BUG_STILL_EXISTS":
        print("🚨 FAILURE: Bug still exists!")
        print("   Short win rates are still >50%")
        print("   Need to continue debugging")
    elif result == "ELEVATED_RATES":
        print("⚠️  PARTIAL: Win rates elevated but not extreme")
        print("   May need further investigation")
    else:
        print("❓ INCONCLUSIVE: Unable to determine result")
        print("   Check processing logs and try again")

if __name__ == "__main__":
    main()