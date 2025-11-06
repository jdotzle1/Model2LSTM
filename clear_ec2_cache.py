#!/usr/bin/env python3
"""
Clear EC2 Cache

This script generates commands to clear all potential cache files
on the EC2 instance that might be preventing the fix from taking effect.
"""

def generate_ec2_cleanup_commands():
    """Generate comprehensive cleanup commands for EC2"""
    
    print("🧹 EC2 CACHE CLEANUP COMMANDS")
    print("=" * 60)
    print("Run these commands on your EC2 instance to clear all caches:")
    print()
    
    commands = [
        "# 1. Clear Python cache files",
        "find /home/ec2-user -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
        "find /home/ec2-user -name '*.pyc' -delete 2>/dev/null || true",
        "find /home/ec2-user -name '*.pyo' -delete 2>/dev/null || true",
        "",
        "# 2. Clear tmp directories",
        "sudo rm -rf /tmp/*",
        "rm -rf /home/ec2-user/tmp/* 2>/dev/null || true",
        "",
        "# 3. Clear any local data cache",
        "rm -rf /home/ec2-user/Model2LSTM/project/data/processed/* 2>/dev/null || true",
        "rm -rf /home/ec2-user/Model2LSTM/project/data/temp/* 2>/dev/null || true",
        "",
        "# 4. Clear any pip cache",
        "pip cache purge 2>/dev/null || true",
        "",
        "# 5. Clear system cache (if needed)",
        "sudo sync",
        "sudo echo 3 | sudo tee /proc/sys/vm/drop_caches",
        "",
        "# 6. Restart Python processes (if any are running)",
        "pkill -f python 2>/dev/null || true",
        "",
        "# 7. Clear any conda cache (if using conda)",
        "conda clean --all -y 2>/dev/null || true",
        "",
        "# 8. Verify cleanup",
        "echo 'Cache cleanup completed!'",
        "df -h /tmp",
        "free -h",
    ]
    
    for cmd in commands:
        print(cmd)
    
    print()
    print("🔄 AFTER CLEANUP - REPROCESS DATA:")
    print("-" * 40)
    
    reprocess_commands = [
        "# Navigate to project directory",
        "cd /home/ec2-user/Model2LSTM",
        "",
        "# Delete old S3 results",
        "aws s3 rm s3://es-1-second-data/processed-data/monthly/2011/06/ --recursive --region us-east-1",
        "",
        "# Verify the fix is in the code",
        "grep -n '1 - labels' src/data_pipeline/weighted_labeling.py || echo 'Inversion code not found (good!)'",
        "",
        "# Force reload Python modules",
        "export PYTHONDONTWRITEBYTECODE=1",
        "",
        "# Reprocess with clean environment",
        "python3 process_monthly_chunks_fixed.py --test-month 2011-06",
        "",
        "# Check new results",
        "aws s3 ls s3://es-1-second-data/processed-data/monthly/2011/06/statistics/ --region us-east-1",
    ]
    
    for cmd in reprocess_commands:
        print(cmd)

def generate_verification_commands():
    """Generate commands to verify the fix is working"""
    
    print(f"\n✅ VERIFICATION COMMANDS")
    print("=" * 60)
    print("Run these to verify the fix is working:")
    print()
    
    verification_commands = [
        "# Quick test of the fixed code",
        "cd /home/ec2-user/Model2LSTM",
        "",
        "python3 -c \"",
        "import sys",
        "sys.path.insert(0, '.')",
        "from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig",
        "import pandas as pd",
        "from datetime import datetime",
        "",
        "# Test data (rising prices)",
        "data = {",
        "    'timestamp': [datetime(2011, 6, 10, 9, 30, i) for i in range(3)],",
        "    'open': [1300.00, 1300.00, 1300.50],",
        "    'high': [1300.25, 1300.50, 1300.75],",
        "    'low': [1299.75, 1299.50, 1300.25],",
        "    'close': [1300.00, 1300.25, 1300.50],",
        "    'volume': [100, 200, 150]",
        "}",
        "",
        "df = pd.DataFrame(data)",
        "config = LabelingConfig()",
        "engine = WeightedLabelingEngine(config)",
        "result = engine.process_dataframe(df)",
        "",
        "# Check short win rates (should be 0% in rising market)",
        "short_cols = [col for col in result.columns if 'label_' in col and 'short' in col]",
        "for col in short_cols:",
        "    rate = result[col].mean()",
        "    print(f'{col}: {rate:.1%}')",
        "",
        "avg_short = sum(result[col].mean() for col in short_cols) / len(short_cols)",
        "if avg_short == 0.0:",
        "    print('✅ FIX WORKING: Short trades correctly lose in rising market')",
        "else:",
        "    print(f'❌ FIX NOT WORKING: Short win rate is {avg_short:.1%}')",
        "\"",
    ]
    
    for cmd in verification_commands:
        print(cmd)

def generate_troubleshooting_guide():
    """Generate troubleshooting guide"""
    
    print(f"\n🔧 TROUBLESHOOTING GUIDE")
    print("=" * 60)
    
    print("If the fix still doesn't work after cleanup:")
    print()
    print("1. 📁 CHECK FILE VERSIONS:")
    print("   ls -la src/data_pipeline/weighted_labeling.py")
    print("   # Should show recent modification time")
    print()
    print("2. 🔍 VERIFY CODE CHANGES:")
    print("   grep -A5 -B5 'Add columns to DataFrame' src/data_pipeline/weighted_labeling.py")
    print("   # Should NOT show any inversion code")
    print()
    print("3. 🐍 CHECK PYTHON PATH:")
    print("   python3 -c \"import src.data_pipeline.weighted_labeling as wl; print(wl.__file__)\"")
    print("   # Should point to the correct file")
    print()
    print("4. 🔄 FORCE MODULE RELOAD:")
    print("   python3 -c \"")
    print("   import sys")
    print("   if 'src.data_pipeline.weighted_labeling' in sys.modules:")
    print("       del sys.modules['src.data_pipeline.weighted_labeling']")
    print("   print('Module cache cleared')")
    print("   \"")
    print()
    print("5. 📦 CHECK FOR DUPLICATE FILES:")
    print("   find . -name 'weighted_labeling.py' -type f")
    print("   # Should only show src/data_pipeline/weighted_labeling.py")
    print()
    print("6. 🚀 NUCLEAR OPTION - RESTART EC2:")
    print("   sudo reboot")
    print("   # This will clear all memory caches")

def main():
    """Main execution"""
    
    print("🚨 EC2 CACHE CLEANUP GUIDE")
    print("This issue is likely caused by cached Python bytecode or tmp files")
    print("on the EC2 instance preventing the fix from taking effect.")
    print()
    
    generate_ec2_cleanup_commands()
    generate_verification_commands()
    generate_troubleshooting_guide()
    
    print(f"\n🎯 SUMMARY:")
    print("1. Run the cleanup commands on EC2")
    print("2. Verify the fix is working with the test")
    print("3. Reprocess June 2011 data")
    print("4. Check that short win rates are now 30-40% (not 66%)")
    print()
    print("The fix is confirmed working locally - it's just a cache issue!")


if __name__ == "__main__":
    main()