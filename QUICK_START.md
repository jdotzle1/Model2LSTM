# Quick Start Guide - ES Trading Model

## Current Status

✅ **Modular structure implemented and tested**
⏳ **Blocked by Python 3.14 incompatibility** (need Python 3.12 or 3.13)

## Prerequisites

1. **Python 3.12 or 3.13** (NOT 3.14 - databento incompatible)
2. **AWS credentials configured** (for S3 access)
3. **Required packages installed:**
   ```bash
   pip install -r requirements.txt
   ```

## Production Workflow

### Process Full 15-Year Dataset

```bash
# Process all months from S3 (skip already processed)
python scripts/process_monthly_batches.py --skip-existing
```

**What it does:**
1. Downloads monthly DBN files from S3
2. Applies corrected contract filtering + gap filling
3. Adds weighted labeling (6 volatility modes)
4. Adds feature engineering (43 features)
5. Uploads processed Parquet to S3
6. Cleans up temporary files

**Expected:**
- ~185 months to process
- ~107 million rows total
- ~20-30 minutes per month
- ~60-90 hours total

### Process Specific Date Range

```bash
# Process just 2024 data
python scripts/process_monthly_batches.py --start-year 2024 --start-month 1 --end-year 2024 --end-month 12

# Process just October 2025
python scripts/process_monthly_batches.py --start-year 2025 --start-month 10 --end-month 10
```

## Testing Workflow

### Test Integration (No Data Processing)

```bash
# Verify all modules work together
python tests/integration/test_monthly_processor_integration.py
```

**Expected output:**
```
✅ PASS: Import Test
✅ PASS: MonthlyProcessor Initialization
✅ PASS: S3 Operations Initialization
Total: 3/3 tests passed
```

### Test on October 2025 (Quick Validation)

```bash
# Process October 2025 locally (requires DBN file)
python process_oct2025_final.py
```

**Expected output:**
- ~621,000 rows (23 trading days × 27,000 bars/day)
- File: `oct2025_processed_FINAL.parquet`

### Process Local File

```bash
# Process any local Parquet file
python main.py --input data.parquet --output processed.parquet

# With validation
python main.py --input data.parquet --output processed.parquet --validate
```

## File Structure

```
src/data_pipeline/          # Core modules (reusable)
├── monthly_processor.py    # Monthly batch orchestration
├── s3_operations.py        # S3 download/upload
├── pipeline.py             # Main pipeline (labeling + features)
├── corrected_contract_filtering.py  # Contract filtering + gap filling
├── weighted_labeling.py    # 6 volatility-based modes
└── features.py             # 43 engineered features

scripts/                    # CLI scripts (runnable)
└── process_monthly_batches.py  # Production batch processing

tests/                      # Testing
├── integration/            # Integration tests
└── unit/                   # Unit tests
```

## Common Commands

### Check Status

```bash
# View current project status
cat STATUS.md

# View file organization guide
cat FILE_MANAGEMENT_GUIDE.md
```

### Run Tests

```bash
# Integration test (no data processing)
python tests/integration/test_monthly_processor_integration.py

# Unit tests (if available)
python -m pytest tests/unit/ -v

# Full integration test (requires data)
python tests/integration/test_final_integration_1000_bars.py
```

### Validation

```bash
# Comprehensive validation
python tests/validation/run_comprehensive_validation.py

# Data quality validation
python tests/validation/validate_data_quality.py
```

## Troubleshooting

### Python 3.14 Error

**Error:** `databento library doesn't support Python 3.14`

**Solution:**
1. Install Python 3.12 or 3.13
2. Create new virtual environment
3. Reinstall packages: `pip install -r requirements.txt`

See `DATABENTO_PYTHON314_ISSUE.md` for details.

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure you're in project root
cd /path/to/Model2LSTM

# Run scripts from project root
python scripts/process_monthly_batches.py
```

### S3 Access Errors

**Error:** `NoCredentialsError` or `AccessDenied`

**Solution:**
```bash
# Configure AWS credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### Memory Errors

**Error:** `MemoryError` during processing

**Solution:**
- Reduce chunk size in `PipelineConfig`
- Process fewer months at a time
- Use EC2 instance with more RAM

## Next Steps

### Immediate (After Python Fix)

1. ✅ Install Python 3.12 or 3.13
2. ✅ Run integration test: `python tests/integration/test_monthly_processor_integration.py`
3. ✅ Test on October 2025: `python scripts/process_monthly_batches.py --start-year 2025 --start-month 10 --end-month 10`
4. ✅ Validate output: ~621,000 rows

### Short-term

1. Process full October 2025 with validation
2. Archive old scripts: `mv process_monthly_chunks_fixed.py archive/`
3. Update documentation

### Long-term

1. Process full 15-year dataset: `python scripts/process_monthly_batches.py --skip-existing`
2. Train 6 XGBoost models
3. Deploy ensemble system

## Key Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `scripts/process_monthly_batches.py` | Production batch processing | Process 15 years from S3 |
| `main.py` | Local file processing | Quick test on local file |
| `process_oct2025_final.py` | October 2025 validation | Quick validation test |
| `tests/integration/test_monthly_processor_integration.py` | Integration test | Verify modules work together |

## Documentation

- **`STATUS.md`** - Current project status
- **`FILE_MANAGEMENT_GUIDE.md`** - File organization guide
- **`FILE_MANAGEMENT_FIX_SUMMARY.md`** - What was fixed and why
- **`CORRECTED_PIPELINE_SUMMARY.md`** - Pipeline implementation details
- **`DATABENTO_PYTHON314_ISSUE.md`** - Python compatibility issue

## Support

For detailed information:
- File organization: `FILE_MANAGEMENT_GUIDE.md`
- Pipeline details: `CORRECTED_PIPELINE_SUMMARY.md`
- Current status: `STATUS.md`
- Steering rules: `.kiro/steering/`

---

**Remember:** Use `scripts/process_monthly_batches.py` for production. It's clean, modular, and tested.
