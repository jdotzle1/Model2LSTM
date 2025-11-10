# File Organization Rules

## Core Principle: Modular, Not Monolithic

**Rule:** If a script is >500 lines, it's doing too much. Break it into modules.

## Directory Structure

```
src/                        # Reusable modules (import these)
├── data_pipeline/         # Core pipeline modules
│   ├── pipeline.py        # Main pipeline orchestration
│   ├── weighted_labeling.py  # Labeling logic
│   ├── features.py        # Feature engineering
│   ├── s3_operations.py   # S3 interactions
│   └── monthly_processor.py  # Batch processing
└── config/                # Configuration

scripts/                   # CLI entry points (run these)
├── process_monthly_batches.py  # Production batch processing
└── process_single_file.py      # Quick single file processing

tests/                     # Testing
├── unit/                  # Unit tests
├── integration/           # Integration tests
└── validation/            # Validation scripts

archive/                   # Deprecated code (reference only)
```

## When to Create a New File

### Create a Module (`src/`) When:
- You need reusable functionality
- Logic will be used by multiple scripts
- You want to unit test the functionality
- It's a core component of the system

**Example:** `src/data_pipeline/monthly_processor.py`

### Create a Script (`scripts/`) When:
- You need a CLI entry point
- It's a production workflow
- It orchestrates multiple modules
- Users will run it directly

**Example:** `scripts/process_monthly_batches.py`

### Create a Test (`tests/`) When:
- You're testing functionality
- You're validating data quality
- You're debugging an issue

**Example:** `tests/integration/test_monthly_processing.py`

## When NOT to Create a New File

### Don't Create When:
- You're just testing something quickly (use Jupyter notebook)
- It's a one-time analysis (use `archive/investigations/`)
- It duplicates existing functionality (extend existing module)
- It's temporary debugging code (delete after use)

## File Naming Conventions

### Modules (src/)
- `snake_case.py`
- Descriptive noun: `monthly_processor.py`, `s3_operations.py`
- No prefixes like `test_`, `debug_`, `temp_`

### Scripts (scripts/)
- `snake_case.py`
- Verb phrase: `process_monthly_batches.py`, `train_models.py`
- Clear action: what does it do?

### Tests (tests/)
- `test_*.py` prefix required
- Matches module name: `test_monthly_processor.py`
- Descriptive: `test_weighted_labeling_comprehensive.py`

## Code Organization Rules

### Module Structure

```python
"""
Module docstring: What does this module do?
"""

# Imports (standard, third-party, local)
import pandas as pd
from pathlib import Path
from .pipeline import process_labeling_and_features

# Constants
DEFAULT_CHUNK_SIZE = 500_000

# Classes (if needed)
class MonthlyProcessor:
    """Class docstring"""
    pass

# Functions (if no class needed)
def process_single_month(file_info):
    """Function docstring"""
    pass
```

### Script Structure

```python
#!/usr/bin/env python3
"""
Script docstring: What does this script do?

Usage:
    python scripts/process_monthly_batches.py
    python scripts/process_monthly_batches.py --start-year 2024
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from src.data_pipeline.monthly_processor import MonthlyProcessor

def main():
    """Main entry point"""
    # CLI logic here
    pass

if __name__ == "__main__":
    sys.exit(main())
```

## Integration Rules

### When Adding New Functionality

1. **Check if module exists:**
   - Does `s3_operations.py` already handle S3?
   - Does `pipeline.py` already handle processing?
   - Don't duplicate functionality

2. **Extend existing module:**
   ```python
   # Add to existing src/data_pipeline/s3_operations.py
   def new_s3_function(self):
       pass
   ```

3. **Create new module only if:**
   - Functionality is distinct and substantial
   - It's a new major component
   - It doesn't fit in existing modules

4. **Update CLI script:**
   ```python
   # Use new functionality in scripts/process_monthly_batches.py
   from src.data_pipeline.s3_operations import new_s3_function
   ```

## Red Flags

### 🚨 Warning Signs You're Doing It Wrong

1. **File is >1000 lines**
   - Break into multiple modules
   - Separate concerns

2. **Copying code between files**
   - Create shared module
   - Import, don't duplicate

3. **Multiple scripts doing similar things**
   - Consolidate into one script
   - Use command-line arguments for variations

4. **Unclear which script to use**
   - Document in README
   - Create clear naming
   - Archive old scripts

5. **Script has "fixed", "v2", "new" in name**
   - Replace old version
   - Archive old version
   - Don't keep both

## Migration Checklist

When replacing an old script with new modular structure:

- [ ] Create new modules in `src/`
- [ ] Create new CLI script in `scripts/`
- [ ] Test new structure thoroughly
- [ ] Update documentation (README, STATUS)
- [ ] Archive old script to `archive/`
- [ ] Update imports in other files
- [ ] Update steering rules if needed

## Examples

### ✅ Good Structure

```
src/data_pipeline/monthly_processor.py  (200 lines - orchestration)
src/data_pipeline/s3_operations.py      (300 lines - S3 logic)
scripts/process_monthly_batches.py      (100 lines - CLI)
```

**Why good:**
- Clear separation of concerns
- Each file has one purpose
- Easy to test and maintain
- Obvious which script to run

### ❌ Bad Structure

```
process_monthly_chunks_fixed.py  (2,783 lines - everything)
process_monthly_chunks_v2.py     (2,500 lines - similar)
process_monthly_new.py           (2,200 lines - also similar)
```

**Why bad:**
- Monolithic files
- Duplicated functionality
- Unclear which to use
- Hard to test and maintain

## Quick Decision Tree

```
Need to add functionality?
├─ Does it fit in existing module?
│  ├─ Yes → Add to existing module
│  └─ No → Is it substantial (>200 lines)?
│     ├─ Yes → Create new module in src/
│     └─ No → Add to most relevant existing module
│
Need to run a workflow?
├─ Does script exist?
│  ├─ Yes → Use existing script
│  └─ No → Create new script in scripts/
│
Need to test something?
├─ Is it a unit test?
│  ├─ Yes → Create in tests/unit/
│  └─ No → Create in tests/integration/
```

## Summary

**Golden Rules:**
1. Modules in `src/` (reusable)
2. Scripts in `scripts/` (runnable)
3. Tests in `tests/` (testable)
4. One file, one purpose
5. <500 lines per file
6. No duplication
7. Clear naming
8. Archive old versions

**When in doubt:** Ask "Is this reusable?" → Module. "Is this runnable?" → Script.
