#!/usr/bin/env python3
"""
Check June 2011 Results

Download and analyze the re-processed June 2011 results to see if the bug is fixed
"""

import subprocess
import json
import os
from datetime import datetime

def download_latest_results():
    """Download the latest June 2011 results"""
    
    print("📥 DOWNLOADING LATEST JUNE 2011 RESULTS")
    print("=" * 50)
    
    # Create directory for new results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"june_2011_rerun_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Download statistics
    cmd = f"aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/06/statistics/ {results_dir}/ --recursive --region us-east-1"
    
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Download successful")
            print(f"Files saved to: {results_dir}/")
            return results_dir
        else:
            print(f"❌ Download failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def analyze_results(results_dir):
    """Analyze the downloaded results"""
    
    if not results_dir or not os.path.exists(results_dir):
        print("❌ No results directory to analyze")
        return
    
    print(f"\n📊 ANALYZING RESULTS FROM: {results_dir}")
    print("=" * 50)
    
    # Find statistics files
    stats_files = []
    for file in os.listdir(results_dir):
        if file.endswith('_statistics.json'):
            stats_files.append(os.path.join(results_dir, file))
    
    if not stats_files:
        print("❌ No statistics files found")
        return
    
    # Use the most recent statistics file
    latest_stats = max(stats_files, key=os.path.getmtime)
    print(f"📄 Latest statistics file: {os.path.basename(latest_stats)}")
    
    try:
        with open(latest_stats, 'r') as f:
            stats = json.load(f)
        
        print("\n🎯 WIN RATE ANALYSIS:")
        print("-" * 30)
        
        # Extract and display win rates
        if 'label_statistics' in stats:
            short_rates = []
            long_rates = []
            
            for label, data in stats['label_statistics'].items():
                if 'win_rate' in data:
                    win_rate = data['win_rate']
                    print(f"{label}: {win_rate:.1%}")
                    
                    if 'short' in label:
                        short_rates.append(win_rate)
                    elif 'long' in label:
                        long_rates.append(win_rate)
            
            # Summary analysis
            print(f"\n📊 SUMMARY:")
            print("-" * 20)
            
            if short_rates:
                avg_short = sum(short_rates) / len(short_rates)
                print(f"Average short win rate: {avg_short:.1%}")
                
                if avg_short > 0.6:
                    print("🚨 BUG STILL EXISTS: Short rates >60%")
                    verdict = "BUG_STILL_EXISTS"
                elif avg_short > 0.45:
                    print("⚠️  ELEVATED: Short rates elevated but not extreme")
                    verdict = "ELEVATED"
                else:
                    print("✅ NORMAL: Short rates appear normal")
                    verdict = "NORMAL"
            
            if long_rates:
                avg_long = sum(long_rates) / len(long_rates)
                print(f"Average long win rate: {avg_long:.1%}")
            
            return verdict, stats
            
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return None, None

def compare_with_old_results(new_stats):
    """Compare with the old problematic results"""
    
    print(f"\n📋 COMPARISON WITH OLD RESULTS")
    print("=" * 50)
    
    old_file = "/C:/Users/jdotzler/Downloads/monthly_2011-06_20251106_213935_statistics.json"
    
    if os.path.exists(old_file):
        try:
            with open(old_file, 'r') as f:
                old_stats = json.load(f)
            
            print("BEFORE (Problematic):")
            if 'label_statistics' in old_stats:
                for label, data in old_stats['label_statistics'].items():
                    if 'short' in label and 'win_rate' in data:
                        print(f"  {label}: {data['win_rate']:.1%}")
            
            print("\nAFTER (Re-processed):")
            if new_stats and 'label_statistics' in new_stats:
                for label, data in new_stats['label_statistics'].items():
                    if 'short' in label and 'win_rate' in data:
                        print(f"  {label}: {data['win_rate']:.1%}")
                        
        except Exception as e:
            print(f"Error reading old results: {e}")
    else:
        print("Old results file not found for comparison")

def main():
    """Main execution"""
    
    print("🔍 CHECKING JUNE 2011 RE-PROCESSING RESULTS")
    print("=" * 60)
    
    # Download latest results
    results_dir = download_latest_results()
    
    if not results_dir:
        print("❌ Could not download results")
        return
    
    # Analyze results
    verdict, new_stats = analyze_results(results_dir)
    
    # Compare with old results
    compare_with_old_results(new_stats)
    
    # Final conclusion
    print("\n" + "="*60)
    print("🎯 FINAL VERDICT")
    print("="*60)
    
    if verdict == "BUG_STILL_EXISTS":
        print("🚨 THE BUG IS STILL THERE!")
        print("   Short win rates are still >60%")
        print("   The 'fix' did not work")
        print("   Need to continue debugging")
    elif verdict == "ELEVATED":
        print("⚠️  PARTIALLY IMPROVED")
        print("   Win rates are lower but still elevated")
        print("   May need further investigation")
    elif verdict == "NORMAL":
        print("✅ BUG APPEARS FIXED!")
        print("   Short win rates are now normal")
        print("   Ready to proceed with model training")
    else:
        print("❓ INCONCLUSIVE")
        print("   Unable to determine if bug is fixed")
        print("   Check the results manually")

if __name__ == "__main__":
    main()