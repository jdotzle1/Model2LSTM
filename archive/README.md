# Archive Directory

## Overview
Deprecated files and earlier implementation attempts that are no longer in active use but kept for reference.

## Contents

### Earlier Optimization Attempts
- `optimized_labeling.py` - First optimization attempt (had bugs)
- `pandas_optimized_labeling.py` - Second optimization attempt (had bugs)

### Analysis & Verification Scripts
- `analyze_results.py` - Results analysis utilities
- `check_mae_ties.py` - MAE tie-breaking analysis
- `count_sequences.py` - Sequence counting utilities
- `verify_one_per_sequence.py` - Sequence validation
- `check_columns.py` - Data column verification
- `convert_to_csv.py` - CSV conversion utilities
- `verify_csv.py` - CSV verification

## Why Archived

These files were part of the development process but are no longer needed because:

1. **Superseded by better implementations**: `simple_optimized_labeling.py` replaced earlier attempts
2. **One-time analysis**: Many were used for specific debugging/analysis tasks
3. **Format changes**: CSV utilities no longer needed (using Parquet)
4. **Functionality moved**: Features incorporated into main codebase

## Usage

These files are kept for:
- Historical reference
- Understanding development process
- Potential future debugging
- Code archaeology if needed

**Note**: Do not use these files in production. Use the current implementations in the main project directories.