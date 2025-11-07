#!/usr/bin/env python3
"""
Execute Code Cleanup

This script will:
1. Move all debug/test/investigation scripts to archive
2. Keep only production code in root
3. Organize test files properly
4. Create a clean directory structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Patterns for files to archive
DEBUG_PATTERNS = [
    'debug_*.py',
    'investigate_*.py',
    'trace_*.py',
    'emergency_*.py',
    'quick_*.py',
    'check_*.py',
    'verify_*.py',
    'validate_*.py',
    'test_*.py',
    'analyze_*.py',
    'inspect_*.py',
    'manual_*.py',
    'deep_*.py',
    'confirm_*.py',
    'continue_*.py',
    'rerun_*.py',
    'get_*.py',
]

DOC_PATTERNS = [
    '*_SUMMARY.md',
    '*_FIX*.md',
    '*_INVESTIGATION*.md',
    '*_GUIDE.md',
    '*_PROCEDURES.md',
    '*_CHECKLIST.md',
    '*_IMPLEMENTATION*.md',
    'TASK_*.md',
    'DEBUGGING_*.md',
    'ROLLOVER_*.md',
    'SHORT_WIN_*.md',
]

# Files to KEEP in root
KEEP_IN_ROOT = [
    'main.py',
    'requirements.txt',
    'README.md',
    '.gitignore',
    'process_monthly_chunks_fixed.py',  # EC2 production script
    'CLEANUP_PLAN.md',
    'EXECUTE_CLEANUP.py',
    'verify_production_code_path.py',
]

# Directories to KEEP
KEEP_DIRS = [
    'src',
    'tests',
    'archive',
    'docs',
    'scripts',
    'deployment',
    'aws_setup',
    '.git',
    '.kiro',
    '__pycache__',
]

def should_archive_file(filename):
    """Check if a file should be archived"""
    if filename in KEEP_IN_ROOT:
        return False
    
    # Check debug patterns
    for pattern in DEBUG_PATTERNS:
        if filename.startswith(pattern.replace('*', '')):
            return True
        if pattern.startswith('*') and filename.endswith(pattern[1:]):
            return True
    
    # Check doc patterns
    for pattern in DOC_PATTERNS:
        if pattern.startswith('*') and pattern.endswith('*'):
            # Contains pattern
            middle = pattern[1:-1]
            if middle in filename:
                return True
        elif pattern.startswith('*'):
            # Ends with pattern
            if filename.endswith(pattern[1:]):
                return True
        elif pattern.endswith('*'):
            # Starts with pattern
            if filename.startswith(pattern[:-1]):
                return True
    
    return False

def main():
    print("=" * 80)
    print("CODE CLEANUP EXECUTION")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    
    # Create archive structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_root = project_root / 'archive' / f'cleanup_{timestamp}'
    
    archive_dirs = {
        'debug_scripts': archive_root / 'debug_scripts',
        'test_scripts': archive_root / 'test_scripts',
        'investigation_scripts': archive_root / 'investigation_scripts',
        'validation_scripts': archive_root / 'validation_scripts',
        'documentation': archive_root / 'documentation',
        'other': archive_root / 'other',
    }
    
    for dir_path in archive_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Created archive structure: {archive_root}")
    
    # Scan root directory
    print("\n🔍 Scanning root directory...")
    
    files_to_move = []
    for item in project_root.iterdir():
        if item.is_file() and should_archive_file(item.name):
            files_to_move.append(item)
    
    print(f"\n📋 Found {len(files_to_move)} files to archive:")
    
    # Categorize files
    categorized = {
        'debug_scripts': [],
        'test_scripts': [],
        'investigation_scripts': [],
        'validation_scripts': [],
        'documentation': [],
        'other': [],
    }
    
    for file_path in files_to_move:
        filename = file_path.name
        
        if filename.startswith('debug_') or filename.startswith('deep_') or filename.startswith('emergency_'):
            categorized['debug_scripts'].append(file_path)
        elif filename.startswith('test_'):
            categorized['test_scripts'].append(file_path)
        elif filename.startswith('investigate_') or filename.startswith('trace_') or filename.startswith('analyze_'):
            categorized['investigation_scripts'].append(file_path)
        elif filename.startswith('validate_') or filename.startswith('verify_') or filename.startswith('check_'):
            categorized['validation_scripts'].append(file_path)
        elif filename.endswith('.md'):
            categorized['documentation'].append(file_path)
        else:
            categorized['other'].append(file_path)
    
    # Show categorization
    for category, files in categorized.items():
        if files:
            print(f"\n  {category.upper()}: {len(files)} files")
            for file_path in files[:5]:  # Show first 5
                print(f"    - {file_path.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
    
    # Ask for confirmation
    print("\n" + "=" * 80)
    print("⚠️  WARNING: This will move files to archive!")
    print("=" * 80)
    response = input("\nProceed with cleanup? (yes/no): ")
    
    if response.lower() != 'yes':
        print("\n❌ Cleanup cancelled")
        return
    
    # Execute move
    print("\n🚀 Executing cleanup...")
    
    moved_count = 0
    for category, files in categorized.items():
        if files:
            print(f"\n  Moving {len(files)} files to {category}...")
            for file_path in files:
                dest = archive_dirs[category] / file_path.name
                try:
                    shutil.move(str(file_path), str(dest))
                    moved_count += 1
                except Exception as e:
                    print(f"    ❌ Error moving {file_path.name}: {e}")
    
    print(f"\n✅ Moved {moved_count} files to archive")
    
    # Show remaining files in root
    print("\n📋 Remaining files in root directory:")
    remaining_files = [f for f in project_root.iterdir() if f.is_file()]
    for file_path in sorted(remaining_files):
        print(f"  ✓ {file_path.name}")
    
    # Create cleanup summary
    summary_path = archive_root / 'CLEANUP_SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write(f"# Cleanup Summary - {timestamp}\n\n")
        f.write(f"## Files Moved: {moved_count}\n\n")
        
        for category, files in categorized.items():
            if files:
                f.write(f"### {category.replace('_', ' ').title()}\n")
                f.write(f"Moved {len(files)} files:\n\n")
                for file_path in sorted(files):
                    f.write(f"- {file_path.name}\n")
                f.write("\n")
        
        f.write("## Production Code Structure\n\n")
        f.write("```\n")
        f.write("root/\n")
        f.write("├── main.py                          # Main entry point\n")
        f.write("├── process_monthly_chunks_fixed.py  # EC2 production script\n")
        f.write("├── requirements.txt                 # Dependencies\n")
        f.write("├── README.md                        # Documentation\n")
        f.write("├── src/                             # Production source code\n")
        f.write("│   └── data_pipeline/\n")
        f.write("│       ├── weighted_labeling.py     # ACTIVE labeling system\n")
        f.write("│       ├── features.py              # Feature engineering\n")
        f.write("│       ├── pipeline.py              # Main pipeline\n")
        f.write("│       └── ...\n")
        f.write("├── tests/                           # Test files\n")
        f.write("├── scripts/                         # Utility scripts\n")
        f.write("└── archive/                         # Archived code\n")
        f.write("```\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. ✅ Root directory cleaned\n")
        f.write("2. ⏭️ Verify EC2 is using process_monthly_chunks_fixed.py\n")
        f.write("3. ⏭️ Check if src/data_pipeline/labeling.py should be removed\n")
        f.write("4. ⏭️ Verify the 66% short win rates are from correct code\n")
    
    print(f"\n📄 Cleanup summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 80)
    print("\n📋 Next steps:")
    print("  1. Review the cleanup summary")
    print("  2. Verify EC2 is using the correct code")
    print("  3. Re-run the labeling to confirm results")
    print("  4. Consider removing src/data_pipeline/labeling.py (old system)")

if __name__ == "__main__":
    main()
