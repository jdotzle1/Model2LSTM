#!/usr/bin/env python3
"""
Verify Current State

The analysis shows normal win rates (~10%) instead of the high rates (66%) 
we were investigating. Let's figure out what changed.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def check_all_available_data():
    """Check all available data files to understand the discrepancy"""
    
    print("🔍 VERIFYING CURRENT STATE")
    print("=" * 60)
    print("Expected: High short win rates (66%) from previous investigation")
    print("Observed: Normal win rates (10%) in current analysis")
    print()
    
    # List all parquet files
    parquet_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    print(f"📁 FOUND {len(parquet_files)} PARQUET FILES:")
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            print(f"  {file}: {len(df):,} rows")
        except:
            print(f"  {file}: ERROR reading")
    print()
    
    # Check each file for win rates
    print("📊 WIN RATE COMPARISON:")
    print("-" * 40)
    
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            
            if 'label_normal_vol_short' in df.columns:
                short_win_rate = df['label_normal_vol_short'].mean()
                long_win_rate = df.get('label_normal_vol_long', pd.Series([0])).mean()
                
                print(f"{os.path.basename(file)}:")
                print(f"  Short win rate: {short_win_rate:.1%}")
                print(f"  Long win rate: {long_win_rate:.1%}")
                print(f"  Rows: {len(df):,}")
                
                # Check if this matches the high win rate pattern
                if short_win_rate > 0.5:
                    print(f"  🚨 HIGH WIN RATE DETECTED!")
                elif short_win_rate > 0.2:
                    print(f"  ⚠️  Elevated win rate")
                else:
                    print(f"  ✅ Normal win rate")
                print()
                
        except Exception as e:
            print(f"  Error analyzing {file}: {e}")
    
    return parquet_files

def check_recent_processing():
    """Check for recent processing results"""
    
    print("🕐 RECENT PROCESSING CHECK:")
    print("-" * 40)
    
    # Look for recent log files or results
    recent_files = []
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(keyword in file.lower() for keyword in ['2011-06', 'monthly', 'statistics', 'debug']):
                if file.endswith(('.json', '.log', '.md', '.parquet')):
                    file_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(file_path)
                        recent_files.append((file_path, mtime))
                    except:
                        pass
    
    # Sort by modification time (most recent first)
    recent_files.sort(key=lambda x: x[1], reverse=True)
    
    print("Recent files (last 10):")
    for file_path, mtime in recent_files[:10]:
        mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {mod_time}: {file_path}")
    
    print()
    
    return recent_files

def analyze_discrepancy():
    """Analyze why we're seeing different results"""
    
    print("🤔 DISCREPANCY ANALYSIS:")
    print("-" * 40)
    
    print("POSSIBLE EXPLANATIONS:")
    print()
    
    print("1. 🔧 BUG WAS FIXED:")
    print("   - Previous investigation identified and fixed the issue")
    print("   - Current data reflects corrected labeling logic")
    print("   - Win rates are now realistic (~10%)")
    print()
    
    print("2. 📊 DIFFERENT DATA SOURCE:")
    print("   - Current analysis uses different data file")
    print("   - Original high win rates were from specific processing run")
    print("   - Need to find the original problematic data")
    print()
    
    print("3. ⚙️  PROCESSING PARAMETERS:")
    print("   - Different processing parameters used")
    print("   - Original issue was environment-specific")
    print("   - Current processing uses corrected parameters")
    print()
    
    print("4. 🎯 SAMPLE SIZE EFFECT:")
    print("   - Original analysis was on smaller sample")
    print("   - Full dataset shows more realistic patterns")
    print("   - High win rates were statistical anomaly")
    print()

def check_investigation_files():
    """Check the investigation files for clues"""
    
    print("🔍 INVESTIGATION FILES CHECK:")
    print("-" * 40)
    
    investigation_files = [
        'SHORT_WIN_RATE_INVESTIGATION_SUMMARY.md',
        'FINAL_BUG_FIX_SUMMARY.md',
        'FINAL_ROLLOVER_FIX.md'
    ]
    
    for file in investigation_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if '66%' in content or '60%' in content:
                        print(f"  Contains high win rate references")
                    if 'fixed' in content.lower() or 'resolved' in content.lower():
                        print(f"  Contains fix references")
            except:
                pass
        else:
            print(f"❌ Missing: {file}")
    
    print()

def recommend_next_action():
    """Recommend what to do next"""
    
    print("🎯 RECOMMENDED NEXT ACTION:")
    print("-" * 40)
    
    print("IMMEDIATE STEPS:")
    print()
    
    print("1. 🔍 VERIFY THE FIX:")
    print("   - Current win rates (~10%) appear normal")
    print("   - This suggests the bug was successfully fixed")
    print("   - Need to confirm this is the intended result")
    print()
    
    print("2. 📋 DOCUMENT THE RESOLUTION:")
    print("   - Update investigation summary with current findings")
    print("   - Mark the issue as resolved if win rates are correct")
    print("   - Archive debugging files")
    print()
    
    print("3. ✅ VALIDATE CORRECTNESS:")
    print("   - Run manual verification on current data")
    print("   - Ensure labeling logic is working as expected")
    print("   - Test with multiple months to confirm consistency")
    print()
    
    print("4. 🚀 PROCEED WITH MODEL TRAINING:")
    print("   - If win rates are validated as correct")
    print("   - Begin XGBoost model training with current data")
    print("   - Use the 6 volatility modes as designed")
    print()

def main():
    """Main execution"""
    
    parquet_files = check_all_available_data()
    recent_files = check_recent_processing()
    analyze_discrepancy()
    check_investigation_files()
    recommend_next_action()
    
    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "status": "Win rates appear normal (~10%)",
        "previous_issue": "High short win rates (66%)",
        "current_observation": "Normal win rates suggest bug was fixed",
        "parquet_files_found": len(parquet_files),
        "recommendation": "Verify fix and proceed with model training"
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"current_state_verification_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Verification saved to: current_state_verification_{timestamp}.json")

if __name__ == "__main__":
    main()