# Project Structure - CURRENT (Updated Nov 2025)

## ⚠️ CRITICAL: Keep Repository Clean

**The repository was recently cleaned up from chaos (25+ files in root) to professional structure (9 files in root).**

**DO NOT let it become messy again!**

## Current Directory Organization

```
Model2LSTM/
├── .kiro/                      # Kiro configuration and steering
│   └── steering/              # AI assistant guidance documents
├── src/                        # Production source code (REUSABLE MODULES)
│   ├── data_pipeline/         # Core pipeline modules
│   │   ├── monthly_processor.py        # Monthly batch orchestration
│   │   ├── s3_operations.py            # S3 download/upload operations
│   │   ├── pipeline.py                 # Main pipeline orchestration
│   │   ├── corrected_contract_filtering.py  # Contract filtering + gap filling
│   │   ├── weighted_labeling.py        # Weighted labeling (6 volatility modes)
│   │   ├── features.py                 # Feature engineering (43 features)
│   │   ├── gap_filling.py              # Gap filling logic
│   │   ├── contract_filtering.py       # Contract filtering logic
│   │   ├── validation_utils.py         # Validation and quality assurance
│   │   └── final_processing_report.py  # Processing reports
│   └── config/                # Configuration files
│       └── config.py          # Project constants and settings
├── scripts/                    # CLI scripts (RUNNABLE ENTRY POINTS)
│   └── process_monthly_batches.py      # Production: Process 15 years from S3
├── tests/                      # Testing code
│   ├── unit/                  # Unit tests
│   │   ├── test_weighted_labeling_comprehensive.py
│   │   └── test_features_comprehensive.py
│   ├── integration/           # Integration tests
│   │   ├── test_monthly_processor_integration.py
│   │   └── test_final_integration_1000_bars.py
│   └── validation/            # Validation scripts
│       ├── run_comprehensive_validation.py
│       └── validate_*.py
├── docs/                       # Generated documentation
│   └── reports/               # Generated reports and analysis
├── archive/                    # Archived old code (REFERENCE ONLY)
│   ├── old_scripts/           # Old scripts (including 2,783-line monster)
│   ├── old_docs/              # Old documentation
│   ├── old_aws_setup/         # Old EC2 scripts
│   ├── old_deployment/        # Old deployment
│   ├── old_runs/              # Old run data
│   ├── old_validation/        # Old validation results
│   ├── old_test_models/       # Old test models
│   ├── old_examples/          # Old examples
│   └── test_data/             # Old test data files
├── project/                    # Legacy directory (TO BE REVIEWED)
├── .gitignore                  # Git configuration
├── README.md                   # Professional project overview
├── STATUS.md                   # Current status (SINGLE SOURCE OF TRUTH)
├── QUICK_START.md              # Quick reference guide
├── main.py                     # Simple local file processor
└── requirements.txt            # Python dependencies
```

## File Count Rules

### Root Directory: EXACTLY 6 Files

**Current: 6 files** ✅

**Allowed in root (ONLY THESE):**
1. `.gitignore` - Git configuration
2. `README.md` - Project overview
3. `STATUS.md` - Current status (SINGLE SOURCE OF TRUTH)
4. `QUICK_START.md` - Quick reference
5. `main.py` - Simple local file processor
6. `requirements.txt` - Dependencies

**NO OTHER FILES ALLOWED IN ROOT**

**NOT allowed in root:**
- ❌ Test scripts (→ `tests/`)
- ❌ Analysis scripts (→ `scripts/` or `archive/`)
- ❌ Data files (→ `archive/test_data/`)
- ❌ Multiple similar scripts (consolidate or archive)
- ❌ Old documentation (→ `archive/old_docs/`)
- ❌ Temporary migration docs (→ `archive/old_docs/`)
- ❌ Investigation scripts (→ `archive/`)
- ❌ Any other .md, .py, or data files

### When Root Exceeds 6 Files: STOP AND CLEAN IMMEDIATELY

If root directory has >6 files:
1. STOP what you're doing
2. Identify what doesn't belong (only 6 files allowed)
3. Archive extras immediately
4. Update this document

## Production Entry Points

### For Production: Monthly S3 Processing

**Use:** `scripts/process_monthly_batches.py`

```bash
# Process all 15 years from S3
python scripts/process_monthly_batches.py --skip-existing

# Process specific date range
python scripts/process_monthly_batches.py --start-year 2024 --start-month 1
```

**What it does:**
1. Downloads monthly DBN files from S3
2. Applies corrected contract filtering + gap filling
3. Adds weighted labeling (6 volatility modes)
4. Adds feature engineering (43 features)
5. Uploads processed Parquet to S3
6. Cleans up temporary files

### For Testing: Local File Processing

**Use:** `main.py`

```bash
# Process single local file
python main.py --input data.parquet --output processed.parquet

# With validation
python main.py --input data.parquet --output processed.parquet --validate
```

**What it does:**
1. Loads local Parquet file
2. Adds weighted labeling (6 modes)
3. Adds feature engineering (43 features)
4. Saves processed file locally

## Module Responsibilities

### `src/data_pipeline/monthly_processor.py` (NEW)
**Purpose:** Orchestrate monthly batch processing from S3

**Key methods:**
- `generate_monthly_file_list()` - Create list of months to process
- `check_existing_processed()` - Skip already processed months
- `process_single_month()` - Process one month end-to-end
- `process_all_months()` - Process entire batch with progress tracking

**Uses:**
- `s3_operations.py` for S3 download/upload
- `corrected_contract_filtering.py` for data cleaning
- `pipeline.py` for labeling + features

### `src/data_pipeline/s3_operations.py`
**Purpose:** Handle all S3 interactions

**Key methods:**
- `download_monthly_file_optimized()` - Download with retry logic
- `upload_monthly_results_optimized()` - Upload processed data + statistics
- Progress tracking and error handling

### `src/data_pipeline/pipeline.py`
**Purpose:** Core data processing pipeline

**Key functions:**
- `process_labeling_and_features()` - Main pipeline
- `process_weighted_labeling()` - 6 volatility modes
- `create_all_features()` - 43 features

### `src/data_pipeline/corrected_contract_filtering.py`
**Purpose:** Contract filtering and gap filling

**Key function:**
- `process_complete_pipeline()` - Volume-based filtering + RTH + gap filling

**Features:**
1. Volume-based contract filtering (not 5-point gaps)
2. RTH filtering (07:30-15:00 CT with DST handling)
3. Gap filling (true 1-second resolution, 27,000 bars/day)

### `src/data_pipeline/weighted_labeling.py`
**Purpose:** Weighted labeling system

**Features:**
- Defines 6 volatility-based trading modes (Low/Normal/High vol × Long/Short)
- Implements binary labeling (0=loss, 1=win) for each mode
- Calculates three-component weights:
  - **Quality weights**: Based on MAE (Maximum Adverse Excursion)
  - **Velocity weights**: Based on speed to target
  - **Time decay weights**: Based on data recency
- Generates 12 columns for XGBoost training

### `src/data_pipeline/features.py`
**Purpose:** Feature engineering

**Features (43 total):**
- Volume features (4): ratios, slopes, exhaustion patterns
- Price context features (5): VWAP, distances, slopes
- Consolidation features (10): range identification, retouch counting
- Return features (5): momentum at multiple timeframes
- Volatility features (6): ATR, regime detection, breakouts
- Microstructure features (6): bar characteristics, tick flow
- Time features (7): session period identification

## Data Flow

1. **Raw Data**: DBN.ZST files from S3
2. **Contract Filtering**: Volume-based front month selection
3. **RTH Filtering**: 07:30-15:00 CT with DST handling
4. **Gap Filling**: True 1-second resolution (27,000 bars/day)
5. **Weighted Labeling**: 12 columns (6 labels + 6 weights) for 6 volatility modes
6. **Feature Engineering**: 43 engineered features
7. **Output**: 61 total columns (6 original + 12 labeling + 43 features)
8. **Upload**: Processed Parquet to S3

## Naming Conventions

### Files
- **Modules (src/)**: `snake_case.py` (e.g., `monthly_processor.py`)
- **Scripts (scripts/)**: `snake_case.py` (e.g., `process_monthly_batches.py`)
- **Tests (tests/)**: `test_*.py` (e.g., `test_monthly_processor_integration.py`)

### Code
- **Functions**: `snake_case` (e.g., `process_single_month`)
- **Classes**: `PascalCase` (e.g., `MonthlyProcessor`)
- **Variables**: `snake_case` (e.g., `month_str`, `file_info`)
- **Constants**: `UPPER_CASE` (e.g., `TICK_SIZE`, `DEFAULT_CHUNK_SIZE`)
- **Column names**: `snake_case` with descriptive suffixes (e.g., `volume_ratio_30s`, `label_low_vol_long`)

## Import Patterns

### From Scripts
```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from src.data_pipeline.monthly_processor import MonthlyProcessor
from src.data_pipeline.s3_operations import EnhancedS3Operations
```

### From Modules
```python
# Standard imports
import pandas as pd
import numpy as np
from pathlib import Path

# Local imports (relative)
from .pipeline import process_labeling_and_features, PipelineConfig
from .s3_operations import EnhancedS3Operations
from .corrected_contract_filtering import process_complete_pipeline
```

## Archive Strategy

### What Goes in Archive

**Old scripts** → `archive/old_scripts/`
- Replaced scripts (e.g., `process_monthly_chunks_fixed.py`)
- One-off testing scripts
- Deprecated utilities

**Old documentation** → `archive/old_docs/`
- Superseded documentation
- Historical investigations
- One-off validation reports
- Temporary issue docs

**Test data** → `archive/test_data/`
- Old test output files
- Sample data files
- Validation data

**Old directories** → `archive/old_*/`
- Old deployment scripts
- Old AWS setup
- Old validation results
- Old test models

### What NEVER Goes in Archive

- ❌ Current production code
- ❌ Current documentation (README, STATUS, QUICK_START)
- ❌ Active tests
- ❌ Configuration files

## Red Flags: When to Clean Up

### 🚨 Immediate Action Required

1. **Root directory has >10 files**
   - Move extras to appropriate directories
   - Archive old files

2. **Multiple similar scripts in root**
   - Consolidate into one
   - Archive old versions

3. **Test scripts in root**
   - Move to `tests/`
   - Archive if one-off

4. **Data files in root**
   - Move to `archive/test_data/`

5. **Multiple documentation files with overlapping content**
   - Consolidate into STATUS.md or README.md
   - Archive old versions

6. **File >500 lines**
   - Break into modules
   - Separate concerns

## Documentation Strategy

### Single Source of Truth: STATUS.md

**All current information goes here:**
- Current status
- What's working
- What's blocked
- Next steps
- Key files

**Update STATUS.md whenever:**
- Project status changes
- New features are added
- Blockers are resolved
- Structure changes

### Quick Reference: QUICK_START.md

**Common commands and workflows:**
- How to run production
- How to test
- Troubleshooting
- Common commands

### Professional Overview: README.md

**Project overview for new users:**
- Quick start
- Project structure
- Pipeline overview
- Common commands
- Troubleshooting

### Everything Else: Archive

**Historical information → archive/old_docs/**
- Old investigations
- Temporary docs
- One-off validations
- Superseded documentation

## Maintenance Checklist

### Weekly
- [ ] Check root directory file count (<10)
- [ ] Review and archive temporary files
- [ ] Update STATUS.md with current status

### Monthly
- [ ] Review archive directory
- [ ] Clean up old test data
- [ ] Update documentation if needed

### After Major Changes
- [ ] Update STATUS.md
- [ ] Update README.md if structure changed
- [ ] Archive old scripts/docs
- [ ] Run integration tests
- [ ] Update steering documents

## Current Status

**Last Cleanup:** November 2025 (Final)
**Root Files:** 6 (EXACTLY as required) ✅
**Structure:** Clean and professional ✅
**Documentation:** Up to date ✅

**Root directory contains ONLY:**
1. .gitignore
2. README.md
3. STATUS.md
4. QUICK_START.md
5. main.py
6. requirements.txt

**Next Review:** After Python 3.12/3.13 installation and production testing

---

**Remember:** Keep it clean. Archive aggressively. One file, one purpose.
