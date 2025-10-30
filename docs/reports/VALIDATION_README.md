# Weighted Labeling Validation Utilities

This document describes the validation and quality assurance utilities for the weighted labeling system.

## Overview

The validation suite includes four main components:

1. **Label Distribution Validation** - Validates label distributions per trading mode
2. **Weight Distribution Validation** - Validates weight distributions per trading mode  
3. **Data Quality Checks** - Checks for NaN/infinite values and data integrity
4. **Original System Comparison** - Compares with the original labeling system

## Requirements Addressed

- **10.1**: Validate that all label values are exactly 0 or 1
- **10.2**: Validate that all weight values are positive
- **10.3**: Report label distribution statistics for each mode
- **10.4**: Report weight distribution statistics for each mode
- **10.5**: Validate that winning percentages are reasonable (5-50% range)
- **10.6**: Check for any NaN or infinite values in output columns
- **10.7**: Provide summary statistics including win rates, MAE, timing, and weight distributions

## Validation Scripts

### 1. Label Distribution Validation

```bash
python validate_label_distributions.py [input_file.parquet]
```

**Purpose**: Validates that labels contain only 0 or 1 values and win rates are in reasonable range (5-50%).

**Options**:
- `--sample-size N`: Validate only N samples
- `--output-file FILE`: Save results to JSON file

### 2. Weight Distribution Validation

```bash
python validate_weight_distributions.py [input_file.parquet]
```

**Purpose**: Validates that weights are positive and analyzes distribution statistics.

**Options**:
- `--sample-size N`: Validate only N samples  
- `--output-file FILE`: Save results to JSON file

### 3. Data Quality Validation

```bash
python validate_data_quality.py [input_file.parquet]
```

**Purpose**: Performs comprehensive data quality checks for NaN/infinite values.

**Options**:
- `--sample-size N`: Validate only N samples
- `--output-file FILE`: Save results to JSON file
- `--detailed`: Show detailed column-by-column analysis

### 4. Labeling Systems Comparison

```bash
python compare_labeling_systems.py [weighted_file.parquet]
```

**Purpose**: Compares weighted labeling results with original labeling system.

**Options**:
- `--original FILE`: Specify original labeling file (optional)
- `--sample-size N`: Compare only N samples
- `--output-file FILE`: Save results to JSON file
- `--generate-original`: Generate original labeling for comparison

### 5. Comprehensive Validation Suite

```bash
python run_comprehensive_validation.py [input_file.parquet]
```

**Purpose**: Runs all validation checks in a single command.

**Options**:
- `--original FILE`: Include original labeling comparison
- `--sample-size N`: Validate only N samples
- `--output-dir DIR`: Directory to save results (default: validation_results)
- `--no-reports`: Skip detailed reports, show summary only
- `--save-individual`: Save individual validation results as separate files

## Usage Examples

### Quick Validation
```bash
# Run comprehensive validation on test data
python run_comprehensive_validation.py project/data/test/weighted_labeling_test_results.parquet

# Validate just labels with sample
python validate_label_distributions.py project/data/processed/full_dataset.parquet --sample-size 10000
```

### Production Validation
```bash
# Full validation with all reports and individual files
python run_comprehensive_validation.py project/data/processed/weighted_labeled_dataset.parquet \
    --output-dir production_validation \
    --save-individual

# Compare with original system
python compare_labeling_systems.py project/data/processed/weighted_labeled_dataset.parquet \
    --original project/data/processed/original_labeled_dataset.parquet
```

### Data Quality Check
```bash
# Detailed data quality analysis
python validate_data_quality.py project/data/processed/weighted_labeled_dataset.parquet \
    --detailed \
    --output-file quality_report.json
```

## Output Files

### Validation Results Directory
When using `run_comprehensive_validation.py`, results are saved to:

- `comprehensive_validation_YYYYMMDD_HHMMSS.json` - Complete validation results
- `validation_summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary
- Individual validation files (if `--save-individual` is used)

### JSON Output Format
All validation scripts can save results in JSON format for programmatic analysis:

```json
{
  "mode_name": {
    "total_samples": 1000,
    "win_rate": 0.15,
    "validation_passed": true,
    "mean_weight": 2.5,
    "weight_std": 1.2
  }
}
```

## Validation Criteria

### Label Validation
- ✅ **Pass**: Labels are 0 or 1, win rate 5-50%
- ❌ **Fail**: Invalid label values or unrealistic win rates

### Weight Validation  
- ✅ **Pass**: All weights positive, no NaN/infinite values
- ❌ **Fail**: Non-positive weights or invalid values

### Data Quality
- ✅ **Pass**: No high-severity issues (NaN in labels/weights)
- ❌ **Fail**: High-severity data quality issues found

### XGBoost Compatibility
- ✅ **Ready**: All required columns present with valid data types
- ❌ **Not Ready**: Missing columns or invalid data

## Integration with Pipeline

### Before Training
Always run comprehensive validation before XGBoost training:

```bash
# Validate production dataset
python run_comprehensive_validation.py project/data/processed/final_dataset.parquet

# Only proceed if validation passes (exit code 0)
if [ $? -eq 0 ]; then
    echo "✅ Validation passed - proceeding with training"
    python train_xgboost_models.py
else
    echo "❌ Validation failed - fix issues before training"
    exit 1
fi
```

### During Development
Use individual validation scripts for focused testing:

```bash
# Quick label check during development
python validate_label_distributions.py test_output.parquet --sample-size 1000

# Weight distribution analysis
python validate_weight_distributions.py test_output.parquet --output-file weights_analysis.json
```

## Troubleshooting

### Common Issues

1. **High Win Rates (>50%)**
   - Check if using test/synthetic data
   - Verify target/stop parameters are realistic
   - Review market conditions in data period

2. **Low Win Rates (<5%)**
   - Check if stop/target ratios are too aggressive
   - Verify entry logic is working correctly
   - Review market volatility during data period

3. **Non-positive Weights**
   - Check weight calculation logic
   - Verify time decay calculation
   - Review quality/velocity weight formulas

4. **Missing Columns**
   - Ensure weighted labeling completed successfully
   - Check column naming conventions
   - Verify all 6 modes were processed

### Getting Help

For validation issues:
1. Run with `--detailed` flag for more information
2. Check individual validation scripts for specific issues
3. Review the comprehensive JSON output for detailed diagnostics
4. Compare with original labeling system to identify discrepancies

## Files Created by Task 9

- `validate_label_distributions.py` - Label distribution validation script
- `validate_weight_distributions.py` - Weight distribution validation script  
- `validate_data_quality.py` - Data quality validation script
- `compare_labeling_systems.py` - Original vs weighted system comparison
- `run_comprehensive_validation.py` - Comprehensive validation suite
- `project/data_pipeline/validation_utils.py` - Core validation utilities (already existed)
- `VALIDATION_README.md` - This documentation file