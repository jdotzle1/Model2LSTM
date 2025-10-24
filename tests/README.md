# Tests Directory

## Overview
Testing and validation utilities for the ES Trading Model project.

## Structure

### validation/
Scripts to validate algorithm correctness and performance:

- `validate_optimization.py` - **Main validation script**
  - Compares optimized vs original labeling algorithms
  - Validates identical results with performance metrics
  - Run before using optimized version on full dataset

- `test_labeling.py` - Basic labeling functionality tests
- `quick_validation.py` - Fast validation on small samples

### debug/
Debugging utilities used during optimization development:

- `debug_remaining_issue.py` - Final debugging of edge cases
- `debug_mae_filter_fix.py` - MAE filter bug analysis
- `detailed_difference_analysis.py` - Comprehensive difference analysis
- `debug_validation_discrepancy.py` - Validation discrepancy investigation
- `debug_tie_breaking.py` - Tie-breaking logic debugging
- `check_duplicate_timestamps.py` - Duplicate timestamp handling
- Other debug utilities for specific issues

## Usage

### Before Production Use
Always validate optimizations:
```bash
# Validate optimized algorithm
python tests/validation/validate_optimization.py

# Quick check on small sample
python tests/validation/quick_validation.py
```

### Debugging
If issues arise, use appropriate debug scripts:
```bash
# General debugging
python tests/debug/debug_remaining_issue.py

# Specific issue analysis
python tests/debug/detailed_difference_analysis.py
```

## Validation Results

Latest validation (1000 bars):
- ✅ All outcomes match exactly
- ✅ All MAE values match exactly  
- ✅ All hold times match exactly
- ✅ All labels match exactly
- ✅ 207x performance improvement
- ✅ Ready for production use

## Notes

- Debug files are kept for reference and future troubleshooting
- Always run validation after any algorithm changes
- Test on small samples before scaling to full dataset