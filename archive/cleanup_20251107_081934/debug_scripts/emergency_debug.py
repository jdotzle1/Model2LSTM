#!/usr/bin/env python3
"""
Emergency Debug

This is getting ridiculous. Let me check EVERYTHING to find out
why the fix isn't taking effect.
"""

import sys
import os
from pathlib import Path

def emergency_debug():
    """Emergency debugging to find the real issue"""
    
    print("🚨 EMERGENCY DEBUG - FINDING THE REAL ISSUE")
    print("=" * 60)
    
    print("The win rates are STILL 66% even after:")
    print("✅ Removing the inversion code")
    print("✅ Clearing Python cache")
    print("✅ Reprocessing the data")
    print("✅ Confirming fix works in tests")
    print()
    print("This suggests one of these scenarios:")
    print("1. 🔄 The processing script is using a DIFFERENT weighted_labeling file")
    print("2. 📦 There's a deployment package overriding our changes")
    print("3. 🐍 Python is importing from a different location")
    print("4. 💾 The data is being cached somewhere else")
    print("5. 🤖 The processing script has its own copy of the logic")
    print()
    
    print("🔍 INVESTIGATION COMMANDS FOR EC2:")
    print("-" * 40)
    
    commands = [
        "# 1. Find ALL weighted_labeling.py files",
        "find /home/ec2-user -name 'weighted_labeling.py' -type f 2>/dev/null",
        "",
        "# 2. Check what the processing script actually imports",
        "cd /home/ec2-user/Model2LSTM",
        "python3 -c \"",
        "import sys",
        "sys.path.insert(0, '.')",
        "import src.data_pipeline.weighted_labeling as wl",
        "print('File location:', wl.__file__)",
        "print('File size:', os.path.getsize(wl.__file__))",
        "import inspect",
        "print('WeightedLabelingEngine location:', inspect.getfile(wl.WeightedLabelingEngine))",
        "\"",
        "",
        "# 3. Check the actual _process_single_mode method",
        "python3 -c \"",
        "import sys",
        "sys.path.insert(0, '.')",
        "from src.data_pipeline.weighted_labeling import WeightedLabelingEngine",
        "import inspect",
        "source = inspect.getsource(WeightedLabelingEngine._process_single_mode)",
        "if '1 - labels' in source:",
        "    print('❌ INVERSION CODE STILL IN _process_single_mode!')",
        "    print('Lines with inversion:')",
        "    for i, line in enumerate(source.split('\\\\n')):",
        "        if '1 - labels' in line:",
        "            print(f'  Line {i+1}: {line.strip()}')",
        "else:",
        "    print('✅ No inversion code found in _process_single_mode')",
        "\"",
        "",
        "# 4. Check if there's a deployment override",
        "ls -la deployment/ec2/ec2_deployment_package/project/project/data_pipeline/weighted_labeling.py 2>/dev/null || echo 'No deployment file'",
        "",
        "# 5. Check the process_monthly_chunks_fixed.py import path",
        "grep -n 'weighted_labeling' process_monthly_chunks_fixed.py",
        "",
        "# 6. Check if there's any other labeling logic",
        "grep -r '1 - labels' . --include='*.py' 2>/dev/null || echo 'No inversion code found'",
        "",
        "# 7. Check Python path when running",
        "python3 -c \"import sys; print('\\\\n'.join(sys.path))\"",
    ]
    
    for cmd in commands:
        print(cmd)
    
    print(f"\n🔍 ALTERNATIVE THEORIES:")
    print("-" * 40)
    print("1. 📊 DATA ISSUE: The June 2011 data itself has some characteristic")
    print("   that makes short trades genuinely win 66% of the time")
    print()
    print("2. 🔄 DIFFERENT ALGORITHM: The processing script might be using")
    print("   a completely different labeling algorithm")
    print()
    print("3. 💾 S3 CACHE: AWS S3 might be serving cached results")
    print("   even though we deleted the files")
    print()
    print("4. 🤖 HARDCODED VALUES: The statistics might be hardcoded")
    print("   somewhere instead of calculated")
    print()
    
    print("🚀 NUCLEAR DEBUGGING OPTIONS:")
    print("-" * 40)
    print("1. 📝 ADD PRINT STATEMENTS: Modify _process_single_mode to print")
    print("   exactly what it's doing for each trade")
    print()
    print("2. 🔍 TRACE EXECUTION: Add logging to see which code path is taken")
    print()
    print("3. 📊 MANUAL CALCULATION: Download the raw data and manually")
    print("   calculate what the win rates should be")
    print()
    print("4. 🆕 FRESH START: Create a completely new processing script")
    print("   that bypasses all existing code")

def generate_trace_code():
    """Generate code to trace what's actually happening"""
    
    print(f"\n🔍 TRACE CODE TO ADD")
    print("=" * 60)
    print("Add this to _process_single_mode in weighted_labeling.py:")
    print()
    
    trace_code = '''
    def _process_single_mode(self, df: pd.DataFrame, mode_name: str) -> None:
        """Process a single trading mode and add results to DataFrame"""
        mode = TRADING_MODES[mode_name]
        
        # TRACE: Print what we're doing
        print(f"🔍 TRACE: Processing {mode_name}, direction: {mode.direction}")
        
        # Calculate labels and tracking data
        labels, mae_ticks, seconds_to_target = self.label_calculators[mode_name].calculate_labels(df)
        
        # TRACE: Print original labels
        print(f"🔍 TRACE: Original labels for {mode_name}: {labels[:5]}... (first 5)")
        print(f"🔍 TRACE: Original win rate: {labels.mean():.1%}")
        
        # Calculate weights
        weights = self.weight_calculators[mode_name].calculate_weights(
            labels, mae_ticks, seconds_to_target, df['timestamp']
        )
        
        # TRACE: Check for any inversion
        if mode.direction == 'short':
            print(f"🔍 TRACE: This is a SHORT mode - checking for inversion...")
            print(f"🔍 TRACE: About to add labels to DataFrame without inversion")
        
        # Add columns to DataFrame
        df[mode.label_column] = labels
        df[mode.weight_column] = weights
        
        # TRACE: Print final labels
        final_labels = df[mode.label_column].values
        print(f"🔍 TRACE: Final labels in DataFrame: {final_labels[:5]}... (first 5)")
        print(f"🔍 TRACE: Final win rate: {final_labels.mean():.1%}")
        
        if self.config.enable_progress_tracking:
            win_rate = labels.mean()
            avg_weight = weights.mean()
            print(f"  {mode_name}: {win_rate:.1%} win rate, avg weight: {avg_weight:.3f}")
    '''
    
    print(trace_code)

def main():
    """Main execution"""
    
    emergency_debug()
    generate_trace_code()
    
    print(f"\n🎯 IMMEDIATE ACTION PLAN:")
    print("1. Run the investigation commands on EC2")
    print("2. Add the trace code to see what's actually happening")
    print("3. If that doesn't work, we need to create a completely new processing script")
    print()
    print("This is definitely a mystery - the fix should be working!")


if __name__ == "__main__":
    main()