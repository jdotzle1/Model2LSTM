# Repository Cleanup Plan

## Current State: DISASTER 🔥

The repository is cluttered with:
- Multiple similar scripts in root
- Duplicate documentation files
- Test files scattered everywhere
- Old investigation directories
- Unclear what's production vs testing

## Goal: Clean, Professional Structure

Keep only what's needed for production. Archive everything else.

## What to Keep in Root

### Essential Production Files
- ✅ `README.md` - Main documentation
- ✅ `requirements.txt` - Dependencies
- ✅ `main.py` - Simple local file processor
- ✅ `.gitignore` - Git configuration

### Essential Documentation (Keep Only These)
- ✅ `STATUS.md` - Current status (SINGLE SOURCE OF TRUTH)
- ✅ `QUICK_START.md` - Quick reference guide
- ⚠️  `MIGRATION_COMPLETE.md` - Keep temporarily, archive after migration verified

### Essential Directories
- ✅ `src/` - Production code
- ✅ `scripts/` - Production CLI scripts
- ✅ `tests/` - Testing code
- ✅ `.kiro/` - Kiro configuration

## What to Archive

### Scripts to Archive (Root → archive/old_scripts/)
- ❌ `process_monthly_chunks_fixed.py` - Replaced by modular structure
- ❌ `process_oct2025_final.py` - One-off testing script
- ❌ `test_oct2025_labeling.py` - One-off testing script
- ❌ `analyze_gap_distribution.py` - One-off analysis
- ❌ `check_test_file.py` - One-off testing

### Documentation to Archive (Root → archive/old_docs/)
- ❌ `CORRECTED_PIPELINE_SUMMARY.md` - Superseded by STATUS.md
- ❌ `DATABENTO_PYTHON314_ISSUE.md` - Temporary issue doc
- ❌ `FILE_MANAGEMENT_FIX_SUMMARY.md` - Superseded by MIGRATION_COMPLETE.md
- ❌ `FILE_MANAGEMENT_GUIDE.md` - Superseded by QUICK_START.md
- ❌ `FINAL_INVESTIGATION_SUMMARY.md` - Historical investigation
- ❌ `INTEGRATION_VERIFICATION.md` - One-off verification
- ❌ `OCTOBER_2025_VALIDATION_RESULTS.md` - One-off validation
- ❌ `PIPELINE_FIX_SUMMARY.md` - Historical fix summary
- ❌ `QUICK_ANSWER.md` - Temporary answer doc
- ❌ `EC2_DEPLOYMENT_INSTRUCTIONS.md` - Old deployment instructions

### Data Files to Archive (Root → archive/test_data/)
- ❌ `oct2025_demo_processed.parquet` - Test output
- ❌ `oct2025_processed_FINAL.parquet` - Test output

### Directories to Archive
- ❌ `aws_setup/` → `archive/old_aws_setup/` - Old EC2 scripts
- ❌ `deployment/` → `archive/old_deployment/` - Old deployment
- ❌ `june_2011_rerun_20251106_155409/` → `archive/old_runs/` - Old run data
- ❌ `validation_results/` → `archive/old_validation/` - Old validation
- ❌ `test_models/` → `archive/old_test_models/` - Old test models
- ❌ `examples/` → `archive/old_examples/` - Old examples
- ❌ `project/` → Review and consolidate into `src/`

## Final Clean Structure

```
Model2LSTM/
├── .git/
├── .kiro/
│   └── steering/
│       ├── file-organization.md
│       ├── tech.md
│       ├── structure.md
│       └── ...
├── src/
│   ├── data_pipeline/
│   │   ├── monthly_processor.py
│   │   ├── s3_operations.py
│   │   ├── pipeline.py
│   │   ├── corrected_contract_filtering.py
│   │   ├── weighted_labeling.py
│   │   ├── features.py
│   │   └── ...
│   └── config/
├── scripts/
│   └── process_monthly_batches.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── validation/
├── docs/
│   └── (generated documentation)
├── archive/
│   ├── old_scripts/
│   ├── old_docs/
│   ├── old_aws_setup/
│   ├── old_deployment/
│   ├── old_runs/
│   ├── old_validation/
│   ├── old_test_models/
│   ├── old_examples/
│   └── test_data/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── STATUS.md
├── QUICK_START.md
└── MIGRATION_COMPLETE.md (temporary)
```

## Cleanup Commands

```bash
# Create archive directories
mkdir -p archive/old_scripts
mkdir -p archive/old_docs
mkdir -p archive/old_aws_setup
mkdir -p archive/old_deployment
mkdir -p archive/old_runs
mkdir -p archive/old_validation
mkdir -p archive/old_test_models
mkdir -p archive/old_examples
mkdir -p archive/test_data

# Archive scripts
mv process_monthly_chunks_fixed.py archive/old_scripts/
mv process_oct2025_final.py archive/old_scripts/
mv test_oct2025_labeling.py archive/old_scripts/
mv analyze_gap_distribution.py archive/old_scripts/
mv check_test_file.py archive/old_scripts/

# Archive documentation
mv CORRECTED_PIPELINE_SUMMARY.md archive/old_docs/
mv DATABENTO_PYTHON314_ISSUE.md archive/old_docs/
mv FILE_MANAGEMENT_FIX_SUMMARY.md archive/old_docs/
mv FILE_MANAGEMENT_GUIDE.md archive/old_docs/
mv FINAL_INVESTIGATION_SUMMARY.md archive/old_docs/
mv INTEGRATION_VERIFICATION.md archive/old_docs/
mv OCTOBER_2025_VALIDATION_RESULTS.md archive/old_docs/
mv PIPELINE_FIX_SUMMARY.md archive/old_docs/
mv QUICK_ANSWER.md archive/old_docs/
mv EC2_DEPLOYMENT_INSTRUCTIONS.md archive/old_docs/

# Archive data files
mv oct2025_demo_processed.parquet archive/test_data/
mv oct2025_processed_FINAL.parquet archive/test_data/

# Archive directories
mv aws_setup archive/old_aws_setup/
mv deployment archive/old_deployment/
mv june_2011_rerun_20251106_155409 archive/old_runs/
mv validation_results archive/old_validation/
mv test_models archive/old_test_models/
mv examples archive/old_examples/

# Update .gitignore
echo "" >> .gitignore
echo "# Archived files" >> .gitignore
echo "archive/" >> .gitignore
```

## After Cleanup: Root Directory

```
Model2LSTM/
├── .git/
├── .kiro/
├── src/
├── scripts/
├── tests/
├── docs/
├── archive/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── STATUS.md
├── QUICK_START.md
└── MIGRATION_COMPLETE.md
```

**Total files in root: 7** (down from 25+)

## Documentation Strategy

### Single Source of Truth: STATUS.md

All current information goes in `STATUS.md`:
- Current status
- What's working
- What's blocked
- Next steps
- Key files

### Quick Reference: QUICK_START.md

Common commands and workflows:
- How to run production
- How to test
- Troubleshooting

### Everything Else: Archive

Historical information, old investigations, temporary docs → `archive/old_docs/`

## Benefits

1. **Clear root directory** - Only essential files
2. **Obvious what's production** - `src/` and `scripts/`
3. **Single source of truth** - `STATUS.md`
4. **Easy to navigate** - Clean structure
5. **Nothing lost** - Everything archived, not deleted

## Execution Plan

1. ✅ Create cleanup plan (this document)
2. ⏳ Execute cleanup commands
3. ⏳ Update README.md to reflect new structure
4. ⏳ Update STATUS.md with cleanup completion
5. ⏳ Verify everything still works
6. ⏳ Commit clean structure

---

**Status:** Ready to execute
**Risk:** Low (everything archived, not deleted)
**Time:** 5 minutes
