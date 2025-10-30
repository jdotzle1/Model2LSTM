#!/usr/bin/env python3
"""
Utility script to update import paths after directory reorganization

This script helps update remaining import statements from 'project.' to 'src.'
"""

import os
import re
import sys
from pathlib import Path


def update_imports_in_file(file_path):
    """Update import statements in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update import statements
        content = re.sub(r'from project\.', 'from src.', content)
        content = re.sub(r'import project\.', 'import src.', content)
        
        # Update path insertions for relative imports
        content = re.sub(
            r"project_root = os\.path\.join\(os\.path\.dirname\(__file__\), '\.\.'\)",
            "project_root = os.path.join(os.path.dirname(__file__), '..', '..')",
            content
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Updated: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False


def main():
    """Update imports in all Python files"""
    root_dir = Path(__file__).parent.parent.parent
    
    # Directories to search
    search_dirs = [
        root_dir / "tests",
        root_dir / "scripts", 
        root_dir / "deployment"
    ]
    
    updated_files = []
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for py_file in search_dir.rglob("*.py"):
                if update_imports_in_file(py_file):
                    updated_files.append(py_file)
    
    print(f"\n📊 Summary:")
    print(f"Updated {len(updated_files)} files")
    
    if updated_files:
        print("\nUpdated files:")
        for file_path in updated_files:
            print(f"  - {file_path}")


if __name__ == "__main__":
    main()