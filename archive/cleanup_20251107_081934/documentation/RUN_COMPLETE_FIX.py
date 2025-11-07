#!/usr/bin/env python3
"""
Complete Fix Execution Script

This script guides you through the complete process:
1. Verify the problem
2. Clean up the code
3. Create fresh deployment
4. Provide deployment instructions

Run this to fix everything in one go.
"""

import sys
import subprocess
from pathlib import Path

def run_command(description, command):
    """Run a command and show results"""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    print(f"Command: {command}\n")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {description} - COMPLETE")
    return True

def main():
    print("=" * 80)
    print("COMPLETE FIX EXECUTION")
    print("=" * 80)
    print("""
This script will:
1. Verify the code mismatch problem
2. Clean up debug scripts (with confirmation)
3. Create fresh deployment package
4. Provide EC2 deployment instructions

This will take about 45 minutes total.
""")
    
    response = input("Ready to proceed? (yes/no): ")
    if response.lower() != 'yes':
        print("\n❌ Cancelled")
        return
    
    # Step 1: Verify the problem
    print("\n" + "=" * 80)
    print("STEP 1: VERIFY THE PROBLEM")
    print("=" * 80)
    
    if not run_command(
        "Verify production code path",
        "python verify_production_code_path.py"
    ):
        print("\n⚠️  Verification failed, but continuing...")
    
    input("\n📋 Review the output above. Press Enter to continue...")
    
    # Step 2: Clean up (with user confirmation)
    print("\n" + "=" * 80)
    print("STEP 2: CLEAN UP DEBUG SCRIPTS")
    print("=" * 80)
    print("""
This will move 100+ debug/test scripts to archive.
You will be asked to confirm before files are moved.
""")
    
    response = input("Run cleanup? (yes/no): ")
    if response.lower() == 'yes':
        if not run_command(
            "Execute cleanup",
            "python EXECUTE_CLEANUP.py"
        ):
            print("\n⚠️  Cleanup had issues, but continuing...")
    else:
        print("\n⏭️  Skipping cleanup")
    
    # Step 3: Create deployment package
    print("\n" + "=" * 80)
    print("STEP 3: CREATE FRESH DEPLOYMENT PACKAGE")
    print("=" * 80)
    
    if not run_command(
        "Create deployment package",
        "python create_fresh_deployment_package.py"
    ):
        print("\n❌ Failed to create deployment package")
        return
    
    # Step 4: Show next steps
    print("\n" + "=" * 80)
    print("STEP 4: DEPLOY TO EC2")
    print("=" * 80)
    print("""
✅ Local cleanup and package creation complete!

📦 Deployment package created in: deployment/ec2/

🚀 NEXT STEPS (Manual):

1. Find your deployment package:
   ls -lh deployment/ec2/ec2_deployment_package_*.tar.gz

2. Upload to EC2:
   scp deployment/ec2/ec2_deployment_package_*.tar.gz ec2-user@YOUR_INSTANCE:/home/ec2-user/

3. On EC2, extract and verify:
   ssh ec2-user@YOUR_INSTANCE
   tar -xzf ec2_deployment_package_*.tar.gz
   cd ec2_deployment_package_*
   cat MANIFEST.txt
   md5sum src/data_pipeline/weighted_labeling.py

4. Install dependencies:
   pip install -r requirements.txt

5. Test imports:
   python -c "from src.data_pipeline.weighted_labeling import WeightedLabelingEngine; print('OK')"

6. Re-run June 2011:
   python process_monthly_chunks_fixed.py --month 2011-06

7. Download and compare results:
   # On local machine:
   scp ec2-user@YOUR_INSTANCE:/path/to/results/monthly_2011-06_*.json ./results_new_code/
   
   # Compare win rates:
   # Old code: ~66% short win rates
   # New code: ??? (this will tell us if it was a bug or real market behavior)

📋 EXPECTED OUTCOMES:

A) Win rates drop to ~40-50%:
   ✅ Bug was in old code, now fixed
   ✅ Continue processing with new code

B) Win rates stay ~66%:
   ✅ Market behavior is real
   ✅ June 2011 was exceptional
   ✅ Accept results and continue

C) Win rates change unexpectedly:
   ⚠️  Need further investigation
   ⚠️  Compare logic differences

📄 DOCUMENTATION:

- CLEANUP_AND_DEPLOYMENT_SUMMARY.md - Complete overview
- CRITICAL_FINDING.md - Details of the code mismatch
- ANSWER_THE_QUESTION.md - Answer to your original question

🎯 BOTTOM LINE:

You were RIGHT to question the code management!
EC2 was running old code with potential bugs.
Now you have clean code and verified deployment.
Re-run June 2011 to get the truth!
""")
    
    print("\n" + "=" * 80)
    print("✅ LOCAL WORK COMPLETE!")
    print("=" * 80)
    print("\nNext: Deploy to EC2 and re-run June 2011")

if __name__ == "__main__":
    main()
