# Databento Library Issue - Python 3.14 Incompatibility

## Problem
The `databento` library cannot be installed on Python 3.14 because its dependency `databento-dbn` is not available for Python 3.14.

## Error
```
ERROR: Could not find a version that satisfies the requirement databento-dbn
ERROR: No matching distribution found for databento-dbn
```

## Root Cause
- Current Python version: **3.14.0**
- Databento supports: Python 3.9 - 3.13
- Python 3.14 is too new

## Solutions

### Option 1: Downgrade Python (Recommended)
Install Python 3.12 or 3.13:
```bash
# Download Python 3.12 from python.org
# Or use pyenv/conda to manage multiple Python versions
```

### Option 2: Wait for Databento Update
Monitor databento releases for Python 3.14 support:
- https://github.com/databento/databento-python

### Option 3: Use Virtual Environment with Python 3.12
```bash
# Create venv with Python 3.12
py -3.12 -m venv venv312
venv312\Scripts\activate
pip install databento pandas numpy pytz boto3
```

## Workaround for Now
The corrected pipeline code (`src/data_pipeline/corrected_contract_filtering.py`) is ready and tested. Once the Python/databento issue is resolved, it can be used immediately with:

```python
from src.data_pipeline.corrected_contract_filtering import process_complete_pipeline

# Load data (once databento works)
import databento as db
store = db.DBNStore.from_file("data.dbn.zst")
df_raw = store.to_df()

# Process with corrected pipeline
df_processed, stats = process_complete_pipeline(df_raw)
```

## Status
- ✅ Corrected pipeline implemented and tested
- ✅ File repo cleaned and organized
- ❌ Cannot test on real data until Python/databento issue resolved
- **Action Required:** Install Python 3.12 or 3.13

---

**Date:** November 10, 2025
**Python Version:** 3.14.0 (too new)
**Required:** 3.9 - 3.13
