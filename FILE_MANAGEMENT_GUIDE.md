# File Management Guide - ES Trading Model

## The Problem We Had

We had **multiple scripts doing similar things** scattered across the project:
- `main.py` - Generic entry point
- `process_monthly_chunks_fixed.py` - 2,783-line monster for S3 monthly processing
- `process_oct2025_final.py` - One-off testing script
- Various scripts in `aws_setup/` for EC2

**Result:** Confusion about which script to use for what purpose.

## The Solution: Clear Separation of Concerns

### Production Code Structure

```
src/data_pipeline/          # Core modules (reusable components)
├── pipeline.py             # Main pipeline: labeling + features
├── weighted_labeling.py    # 6 volatility-based modes
├── features.py             # 43 engineered features
├── corrected_contract_filtering.py  # Contract filtering + gap filling
├── s3_operations.py        # S3 download/upload operations
├── monthly_processor.py    # Monthly batch processing orchestration
└── validation_utils.py     # Data quality validation

scripts/                    # CLI scripts (use the modules above)
├── process_monthly_batches.py  # PRODUCTION: Process 15 years from S3
└── process_single_file.py      # Quick: Process one local file

tests/                      # Testing and validation
├── unit/                   # Unit tests
├── integration/            # Integration tests
└── validation/             # Validation scripts
```

## Which Script Should I Use?

### For Production: Monthly S3 Processing

**Use:** `scripts/process_monthly_batches.py`

```bash
# Process all 15 years (2010-2025)
python scripts/process_monthly_batches.py

# Process specific date range
python scripts/process_monthly_batches.py --start-year 2024 --start-month 1

# Skip already processed months
python scripts/process_monthly_batches.py --skip-existing
```

**What it does:**
1. Downloads monthly DBN files from S3
2. Applies corrected contract filtering + gap filling
3. Adds weighted labeling (6 modes)
4. Adds feature engineering (43 features)
5. Uploads processed Parquet to S3
6. Cleans up temporary files

### For Testing: Single Local File

**Use:** `main.py` (for local files)

```bash
# Process a single local file
python main.py --input data.parquet --output processed.parquet

# With validation
python main.py --input data.parquet --output processed.parquet --validate
```

**What it does:**
1. Loads local Parquet file
2. Adds weighted labeling (6 modes)
3. Adds feature engineering (43 features)
4. Saves processed file locally

### For Quick Testing: October 2025

**Use:** `process_oct2025_final.py` (one-off script)

```bash
python process_oct2025_final.py
```

**What it does:**
- Hardcoded for October 2025 testing
- Uses corrected pipeline
- Saves to `oct2025_processed_FINAL.parquet`

## What to Archive

### Move to `archive/`

1. **`process_monthly_chunks_fixed.py`** - 2,783-line monster
   - Reason: Replaced by modular `monthly_processor.py` + `process_monthly_batches.py`
   - Keep for reference, but don't use

2. **Scripts in `aws_setup/`** - Old EC2 deployment scripts
   - Reason: Superseded by new modular structure
   - Keep for reference

3. **One-off test scripts** - Various `test_*.py` in root
   - Reason: Should be in `tests/` directory
   - Move to `tests/validation/` or `archive/`

### Keep in Root

1. **`main.py`** - Simple local file processing
2. **`process_oct2025_final.py`** - Quick testing script
3. **`README.md`** - Project documentation
4. **`requirements.txt`** - Dependencies
5. **`STATUS.md`** - Current status

## Module Responsibilities

### `src/data_pipeline/monthly_processor.py` (NEW)

**Purpose:** Orchestrate monthly batch processing

**Key methods:**
- `generate_monthly_file_list()` - Create list of months to process
- `check_existing_processed()` - Skip already processed months
- `process_single_month()` - Process one month end-to-end
- `process_all_months()` - Process entire batch with progress tracking

**Uses:**
- `s3_operations.py` for S3 download/upload
- `corrected_contract_filtering.py` for data cleaning
- `pipeline.py` for labeling + features

### `src/data_pipeline/s3_operations.py` (EXISTS)

**Purpose:** Handle all S3 interactions

**Key methods:**
- `download_monthly_file_optimized()` - Download with retry logic
- `upload_monthly_results()` - Upload processed data + statistics
- `list_processed_months()` - Check what's already done

### `src/data_pipeline/pipeline.py` (EXISTS)

**Purpose:** Core data processing pipeline

**Key functions:**
- `process_labeling_and_features()` - Main pipeline
- `process_weighted_labeling()` - 6 volatility modes
- `create_all_features()` - 43 features

### `scripts/process_monthly_batches.py` (NEW)

**Purpose:** Clean CLI for production batch processing

**Features:**
- Command-line arguments for date range
- Progress tracking
- Summary statistics
- Error handling

## Migration Plan

### Phase 1: Test New Structure (NOW)

```bash
# Test on October 2025
python scripts/process_monthly_batches.py --start-year 2025 --start-month 10 --end-month 10
```

### Phase 2: Archive Old Scripts

```bash
# Move old scripts to archive
mkdir -p archive/old_processing_scripts
mv process_monthly_chunks_fixed.py archive/old_processing_scripts/
mv aws_setup/*.py archive/old_processing_scripts/
```

### Phase 3: Full Production Run

```bash
# Process all 15 years
python scripts/process_monthly_batches.py --skip-existing
```

## Benefits of New Structure

1. **Modular:** Each module has one clear responsibility
2. **Testable:** Easy to unit test individual components
3. **Maintainable:** Changes in one area don't affect others
4. **Reusable:** Modules can be used in different contexts
5. **Clear:** Obvious which script to use for what purpose

## Quick Reference

| Task | Script | Location |
|------|--------|----------|
| Process 15 years from S3 | `process_monthly_batches.py` | `scripts/` |
| Process local file | `main.py` | Root |
| Quick test October 2025 | `process_oct2025_final.py` | Root |
| Unit tests | `pytest` | `tests/unit/` |
| Integration tests | `pytest` | `tests/integration/` |

## Next Steps

1. ✅ Created modular structure
2. ⏳ Test on October 2025
3. ⏳ Archive old scripts
4. ⏳ Run full production batch
5. ⏳ Update documentation

---

**Key Principle:** One script, one purpose. No more 2,783-line monsters.
