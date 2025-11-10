# ES Trading Model - Production Pipeline

Machine learning pipeline for E-mini S&P 500 futures trading using 6 specialized XGBoost models with volatility-adaptive selection.

## Quick Start

### Prerequisites

- **Python 3.12 or 3.13** (NOT 3.14 - databento incompatible)
- AWS credentials configured (for S3 access)
- Required packages: `pip install -r requirements.txt`

### Production: Process 15 Years from S3

```bash
# Process all months (skip already processed)
python scripts/process_monthly_batches.py --skip-existing

# Process specific date range
python scripts/process_monthly_batches.py --start-year 2024 --start-month 1
```

### Testing: Process Local File

```bash
# Process single local file
python main.py --input data.parquet --output processed.parquet

# With validation
python main.py --input data.parquet --output processed.parquet --validate
```

## Project Structure

```
Model2LSTM/
├── src/                    # Production code (reusable modules)
│   └── data_pipeline/
│       ├── monthly_processor.py        # Monthly batch orchestration
│       ├── s3_operations.py            # S3 download/upload
│       ├── pipeline.py                 # Main pipeline (labeling + features)
│       ├── corrected_contract_filtering.py  # Contract filtering + gap filling
│       ├── weighted_labeling.py        # 6 volatility-based modes
│       └── features.py                 # 43 engineered features
├── scripts/                # CLI scripts (runnable)
│   └── process_monthly_batches.py      # Production batch processing
├── tests/                  # Testing
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── validation/        # Validation scripts
├── docs/                   # Generated documentation
├── archive/                # Archived old code (reference only)
├── main.py                 # Simple local file processor
├── requirements.txt        # Python dependencies
├── STATUS.md              # Current project status
└── QUICK_START.md         # Quick reference guide
```

## Pipeline Overview

### 1. Data Processing
- **Contract Filtering:** Volume-based front month selection
- **RTH Filtering:** 07:30-15:00 CT with DST handling
- **Gap Filling:** True 1-second resolution (27,000 bars/day)

### 2. Weighted Labeling (6 Volatility Modes)
- **Low Vol Long/Short:** 6-tick stop, 12-tick target
- **Normal Vol Long/Short:** 8-tick stop, 16-tick target
- **High Vol Long/Short:** 10-tick stop, 20-tick target

**Weighting System:**
- Quality weights (MAE-based): [0.5, 2.0]
- Velocity weights (speed-based): [0.5, 2.0]
- Time decay weights (recency-based): exp(-0.05 × months_ago)

### 3. Feature Engineering (43 Features)
- Volume features (4)
- Price context features (5)
- Consolidation features (10)
- Return features (5)
- Volatility features (6)
- Microstructure features (6)
- Time features (7)

### 4. Output Format
- **61 columns total:**
  - 6 original OHLCV columns
  - 12 labeling columns (6 labels + 6 weights)
  - 43 engineered features
- **XGBoost ready:** Binary labels with sample weights

## Common Commands

### Production

```bash
# Process all 15 years from S3
python scripts/process_monthly_batches.py --skip-existing

# Process specific year
python scripts/process_monthly_batches.py --start-year 2024 --start-month 1 --end-year 2024 --end-month 12

# Process single month
python scripts/process_monthly_batches.py --start-year 2025 --start-month 10 --end-month 10
```

### Testing

```bash
# Integration test (no data processing)
python tests/integration/test_monthly_processor_integration.py

# Process local file
python main.py --input data.parquet --output processed.parquet

# Run unit tests
python -m pytest tests/unit/ -v
```

### Validation

```bash
# Comprehensive validation
python tests/validation/run_comprehensive_validation.py

# Data quality validation
python tests/validation/validate_data_quality.py
```

## Expected Output

### Single Month (e.g., October 2025)
- **Rows:** ~621,000 (23 trading days × 27,000 bars/day)
- **Columns:** 61 (6 OHLCV + 12 labeling + 43 features)
- **Size:** ~50-100 MB (Parquet compressed)

### Full 15 Years (2010-2025)
- **Rows:** ~107 million
- **Months:** ~185
- **Processing time:** ~60-90 hours (20-30 min/month)

## Current Status

See `STATUS.md` for:
- Current project status
- What's working
- What's blocked
- Next steps

## Documentation

- **`STATUS.md`** - Current status and next steps
- **`QUICK_START.md`** - Quick reference guide
- **`MIGRATION_COMPLETE.md`** - Recent migration details
- **`.kiro/steering/`** - AI assistant guidance

## Troubleshooting

### Python 3.14 Error

**Error:** `databento library doesn't support Python 3.14`

**Solution:**
1. Install Python 3.12 or 3.13
2. Create new virtual environment
3. Reinstall: `pip install -r requirements.txt`

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Run scripts from project root:
```bash
cd /path/to/Model2LSTM
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
```

## Development

### Adding New Features

1. **Check existing modules first** - Don't duplicate functionality
2. **Extend existing modules** - Add to `src/data_pipeline/`
3. **Keep files <500 lines** - Break into modules if larger
4. **Write tests** - Add to `tests/unit/` or `tests/integration/`
5. **Update documentation** - Update `STATUS.md`

### File Organization Rules

- **Modules** → `src/` (reusable components)
- **Scripts** → `scripts/` (CLI entry points)
- **Tests** → `tests/` (testing code)
- **Archive** → `archive/` (old code, reference only)

See `.kiro/steering/file-organization.md` for detailed rules.

## Next Steps

1. **Install Python 3.12/3.13** (if not already)
2. **Test integration:** `python tests/integration/test_monthly_processor_integration.py`
3. **Test on one month:** `python scripts/process_monthly_batches.py --start-year 2025 --start-month 10 --end-month 10`
4. **Run full production:** `python scripts/process_monthly_batches.py --skip-existing`
5. **Train XGBoost models** (6 specialized models)
6. **Deploy ensemble system** (volatility-adaptive selection)

## License

[Your License Here]

## Contact

[Your Contact Info Here]

---

**Production Script:** `python scripts/process_monthly_batches.py`

**Status:** Ready for production (pending Python 3.12/3.13 installation)
