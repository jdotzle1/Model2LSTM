# Code Management Cleanup Plan

## Problem Identified
**100+ files in root directory**, mostly debug/test scripts created during troubleshooting. This creates:
- Confusion about which code is actually running
- Risk of importing wrong modules
- Impossible to maintain
- Can't identify the actual bug source

## Production Code (KEEP)
```
src/
├── data_pipeline/
│   ├── labeling.py              # ORIGINAL labeling (deprecated?)
│   ├── weighted_labeling.py     # NEW weighted labeling system
│   ├── features.py              # Feature engineering
│   ├── pipeline.py              # Main pipeline
│   ├── s3_operations.py         # S3 utilities
│   ├── session_utils.py         # Session detection
│   └── validation_utils.py      # Validation helpers
├── config/
│   └── config.py                # Configuration
└── convert_dbn.py               # DBN conversion

main.py                          # Main entry point
requirements.txt                 # Dependencies
README.md                        # Documentation
```

## Debug Files to Archive (MOVE)
All files matching these patterns:
- `debug_*.py` (20+ files)
- `test_*.py` in root (30+ files)
- `investigate_*.py` (5+ files)
- `validate_*.py` in root (10+ files)
- `check_*.py` (5+ files)
- `trace_*.py`, `emergency_*.py`, `quick_*.py`
- `*_investigation_*.md`, `*_FIX_*.md`, `*_SUMMARY.md`

## Critical Question: Which Labeling Module is Actually Used?

### Option 1: `src/data_pipeline/labeling.py`
- Original implementation
- May be deprecated

### Option 2: `src/data_pipeline/weighted_labeling.py`
- New weighted system with 6 volatility modes
- All recent code imports this

**WE NEED TO VERIFY WHICH ONE IS ACTUALLY RUNNING ON EC2!**

## Cleanup Steps

### Step 1: Create Archive Structure
```
archive/
├── debug_scripts/          # All debug_*.py files
├── test_scripts/           # All test_*.py files  
├── investigation_scripts/  # All investigate_*.py files
├── validation_scripts/     # All validate_*.py files
├── documentation/          # All .md files except README
└── old_processing/         # Old processing scripts
```

### Step 2: Identify Active Production Code
1. Check what EC2 is actually running
2. Check what `main.py` imports
3. Check what `process_monthly_chunks_fixed.py` imports
4. Verify the import chain

### Step 3: Move Non-Production Files
Move all debug/test/investigation files to archive

### Step 4: Create Clean Test Structure
```
tests/
├── unit/                   # Unit tests
├── integration/            # Integration tests
└── validation/             # Validation scripts
```

### Step 5: Document Production Code Path
Create clear documentation of:
- What runs on EC2
- What the import chain is
- Which labeling module is used
- How to run production code

## Immediate Action Required

**Before any cleanup, we MUST answer:**
1. Which labeling module is EC2 actually using?
2. Is there a duplicate/conflicting implementation?
3. Are we debugging the wrong file?

## Next Steps
1. Trace EC2 execution to find actual code path
2. Verify no duplicate labeling logic
3. Clean up root directory
4. Establish clear production vs development separation
