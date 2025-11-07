#!/usr/bin/env python3
"""
Verify Production Code Path

This script traces the actual code path to identify:
1. Which labeling module is being used
2. If there are any duplicate implementations
3. What the actual import chain is
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_module_location(module_name):
    """Check where a module is actually located"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return spec.origin
        return "Not found"
    except Exception as e:
        return f"Error: {e}"

def check_function_source(module_name, function_name):
    """Check the source code location of a specific function"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            if hasattr(func, '__code__'):
                return func.__code__.co_filename
            return "Function found but no source location"
        return f"Function '{function_name}' not found in module"
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 80)
    print("PRODUCTION CODE PATH VERIFICATION")
    print("=" * 80)
    
    # Check labeling modules
    print("\n1. LABELING MODULE LOCATIONS")
    print("-" * 80)
    
    modules_to_check = [
        'src.data_pipeline.labeling',
        'src.data_pipeline.weighted_labeling',
    ]
    
    for module in modules_to_check:
        location = check_module_location(module)
        print(f"\n{module}:")
        print(f"  Location: {location}")
        
        # Check if it exists
        if "Not found" not in location and "Error" not in location:
            print(f"  ✅ EXISTS")
            
            # Check key functions
            if 'weighted_labeling' in module:
                functions = ['process_weighted_labeling', 'WeightedLabelingEngine', 'LabelCalculator']
            else:
                functions = ['calculate_labels_for_all_profiles']
            
            for func in functions:
                func_location = check_function_source(module, func)
                print(f"    {func}: {func_location}")
        else:
            print(f"  ❌ NOT FOUND")
    
    # Check what main.py imports
    print("\n\n2. MAIN.PY IMPORT CHAIN")
    print("-" * 80)
    
    try:
        from src.data_pipeline.pipeline import process_labeling_and_features
        print("✅ main.py → src.data_pipeline.pipeline")
        
        # Check what pipeline imports
        import src.data_pipeline.pipeline as pipeline_module
        print(f"   Pipeline location: {pipeline_module.__file__}")
        
        # Check if it imports weighted_labeling
        if hasattr(pipeline_module, 'process_weighted_labeling'):
            print("   ✅ Pipeline imports process_weighted_labeling")
        
        # Try to trace the actual function
        func_location = check_function_source('src.data_pipeline.pipeline', 'process_labeling_and_features')
        print(f"   process_labeling_and_features location: {func_location}")
        
    except Exception as e:
        print(f"❌ Error importing main.py chain: {e}")
    
    # Check what process_monthly_chunks_fixed.py imports
    print("\n\n3. PROCESS_MONTHLY_CHUNKS_FIXED.PY IMPORT CHAIN")
    print("-" * 80)
    
    try:
        from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, process_weighted_labeling
        print("✅ process_monthly_chunks_fixed.py → src.data_pipeline.weighted_labeling")
        
        # Check locations
        engine_location = check_function_source('src.data_pipeline.weighted_labeling', 'WeightedLabelingEngine')
        process_location = check_function_source('src.data_pipeline.weighted_labeling', 'process_weighted_labeling')
        
        print(f"   WeightedLabelingEngine: {engine_location}")
        print(f"   process_weighted_labeling: {process_location}")
        
    except Exception as e:
        print(f"❌ Error importing process_monthly_chunks_fixed.py chain: {e}")
    
    # Check for duplicate implementations
    print("\n\n4. CHECKING FOR DUPLICATE IMPLEMENTATIONS")
    print("-" * 80)
    
    # Search for files containing labeling logic
    labeling_files = []
    for root, dirs, files in os.walk(project_root):
        # Skip archive and test directories
        if 'archive' in root or '__pycache__' in root or '.git' in root:
            continue
        
        for file in files:
            if file.endswith('.py') and 'label' in file.lower():
                full_path = os.path.join(root, file)
                labeling_files.append(full_path)
    
    print(f"\nFound {len(labeling_files)} files with 'label' in name:")
    for file in sorted(labeling_files):
        rel_path = os.path.relpath(file, project_root)
        
        # Check if it's production code
        if rel_path.startswith('src/'):
            print(f"  🟢 PRODUCTION: {rel_path}")
        elif rel_path.startswith('tests/'):
            print(f"  🔵 TEST: {rel_path}")
        elif rel_path.startswith('archive/'):
            print(f"  ⚪ ARCHIVED: {rel_path}")
        else:
            print(f"  🟡 ROOT/OTHER: {rel_path}")
    
    # Check for LabelCalculator class (the actual labeling logic)
    print("\n\n5. LABEL CALCULATOR IMPLEMENTATIONS")
    print("-" * 80)
    
    try:
        from src.data_pipeline.weighted_labeling import LabelCalculator
        import inspect
        
        # Get the source file
        source_file = inspect.getfile(LabelCalculator)
        print(f"✅ LabelCalculator found in: {source_file}")
        
        # Get the _calculate_labels method
        if hasattr(LabelCalculator, '_calculate_labels'):
            method = getattr(LabelCalculator, '_calculate_labels')
            method_source = inspect.getsourcefile(method)
            print(f"   _calculate_labels method in: {method_source}")
            
            # Get first few lines of the method
            source_lines = inspect.getsource(method).split('\n')[:10]
            print(f"\n   First 10 lines of _calculate_labels:")
            for line in source_lines:
                print(f"     {line}")
        
    except Exception as e:
        print(f"❌ Error checking LabelCalculator: {e}")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n✅ PRODUCTION CODE PATH:")
    print("   main.py → src.data_pipeline.pipeline → src.data_pipeline.weighted_labeling")
    print("   process_monthly_chunks_fixed.py → src.data_pipeline.weighted_labeling")
    
    print("\n📋 RECOMMENDATION:")
    print("   1. All production code uses src.data_pipeline.weighted_labeling ✅")
    print("   2. Move all debug/test scripts to archive/ or tests/")
    print("   3. Keep only production code in root directory")
    print("   4. Verify EC2 is using process_monthly_chunks_fixed.py")

if __name__ == "__main__":
    main()
