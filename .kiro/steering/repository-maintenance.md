# Repository Maintenance Rules

## Critical: Prevent Repository Chaos

**The repository was cleaned up in November 2025 from 25+ files in root to 9 files.**

**These rules exist to prevent it from becoming a mess again.**

## The Problem We Had

### Before Cleanup (The Chaos)
- 25+ files in root directory
- Multiple similar scripts (`process_monthly_chunks_fixed.py`, `process_oct2025_final.py`, etc.)
- 10+ documentation files with overlapping content
- Test data files in root
- Unclear what's production vs testing
- 6+ directories with old/unused code
- Confusion about which script to use

### After Cleanup (Professional)
- 9 files in root directory
- Clear separation: `src/` (modules), `scripts/` (CLI), `tests/` (testing)
- Single source of truth: `STATUS.md`
- Professional README
- Everything archived, nothing lost

## Golden Rules

### Rule 1: Root Directory Maximum 6 Files

**Current: 6 files** ✅

**Allowed (ONLY THESE):**
1. `.gitignore` - Git configuration
2. `README.md` - Professional project overview
3. `STATUS.md` - Current status (SINGLE SOURCE OF TRUTH)
4. `QUICK_START.md` - Quick reference guide
5. `main.py` - Simple local file processor
6. `requirements.txt` - Python dependencies

**NO OTHER FILES ALLOWED IN ROOT**

**If root exceeds 6 files:**
1. STOP immediately
2. Identify what doesn't belong (only 6 files allowed)
3. Move to appropriate directory or archive
4. Update steering documents

**Any file not in the allowed list above must be archived immediately.**

### Rule 2: One File, One Purpose

**Maximum file size: 500 lines**

If a file exceeds 500 lines:
1. Break into multiple modules
2. Separate concerns
3. Create clear interfaces

**Example of what NOT to do:**
- ❌ `process_monthly_chunks_fixed.py` (2,783 lines) - Everything in one file

**Example of what TO do:**
- ✅ `monthly_processor.py` (200 lines) - Orchestration
- ✅ `s3_operations.py` (300 lines) - S3 logic
- ✅ `process_monthly_batches.py` (100 lines) - CLI

### Rule 3: No Duplicate Functionality

**Before creating new code:**
1. Check if module already exists
2. Check if functionality already exists
3. Extend existing module instead of duplicating

**Example:**
- ❌ Creating new S3 download function when `s3_operations.py` exists
- ✅ Adding method to existing `EnhancedS3Operations` class

### Rule 4: Archive Aggressively

**When to archive:**
- Old scripts that have been replaced
- One-off testing scripts
- Superseded documentation
- Historical investigations
- Test data files
- Old deployment scripts

**Where to archive:**
- `archive/old_scripts/` - Old scripts
- `archive/old_docs/` - Old documentation
- `archive/test_data/` - Test data files
- `archive/old_*/` - Old directories

**Never delete, always archive.**

### Rule 5: Single Source of Truth for Documentation

**STATUS.md is the single source of truth for current information:**
- Current status
- What's working
- What's blocked
- Next steps
- Key files

**Other documentation:**
- `README.md` - Professional overview for new users
- `QUICK_START.md` - Quick reference for common tasks
- Everything else → Archive

**No more:**
- ❌ Multiple docs with overlapping content
- ❌ Temporary issue docs in root
- ❌ One-off validation reports in root

## File Organization Decision Tree

```
Need to create a file?
│
├─ Is it reusable code?
│  ├─ Yes → Create in src/data_pipeline/
│  └─ No → Continue
│
├─ Is it a CLI script?
│  ├─ Yes → Create in scripts/
│  └─ No → Continue
│
├─ Is it a test?
│  ├─ Yes → Create in tests/unit/ or tests/integration/
│  └─ No → Continue
│
├─ Is it documentation?
│  ├─ Current info → Update STATUS.md
│  ├─ Quick reference → Update QUICK_START.md
│  ├─ Overview → Update README.md
│  └─ Historical → Archive in archive/old_docs/
│
├─ Is it test data?
│  └─ Yes → Save in archive/test_data/
│
└─ Is it temporary/one-off?
   └─ Yes → Use and delete, or archive immediately
```

## Red Flags: When to Take Action

### 🚨 Critical - Act Immediately

1. **Root directory has >6 files**
   ```bash
   # Check file count (Windows)
   (Get-ChildItem -File).Count
   
   # Should be exactly 6 files
   # If >6, identify and archive extras immediately
   ```

2. **Multiple similar scripts**
   - Example: `process_v1.py`, `process_v2.py`, `process_fixed.py`
   - Action: Consolidate into one, archive old versions

3. **File exceeds 500 lines**
   ```bash
   # Check line count
   wc -l filename.py
   
   # If >500, break into modules
   ```

4. **Test scripts in root**
   - Action: Move to `tests/` or archive

5. **Data files in root**
   - Action: Move to `archive/test_data/`

### ⚠️ Warning - Review Soon

1. **Multiple documentation files**
   - Review for overlapping content
   - Consolidate into STATUS.md or README.md
   - Archive old versions

2. **Unclear script purpose**
   - Add clear docstring
   - Update README.md
   - Consider renaming

3. **Unused directories**
   - Review contents
   - Archive if no longer needed

## Maintenance Schedule

### Daily (When Working on Project)
- [ ] Check root directory file count
- [ ] Archive temporary files after use
- [ ] Update STATUS.md if status changes

### Weekly
- [ ] Review root directory (should be <10 files)
- [ ] Check for duplicate functionality
- [ ] Archive one-off scripts/docs

### Monthly
- [ ] Review archive directory
- [ ] Clean up old test data
- [ ] Update documentation if needed
- [ ] Review steering documents

### After Major Changes
- [ ] Update STATUS.md
- [ ] Update README.md if structure changed
- [ ] Archive old scripts/docs
- [ ] Run integration tests
- [ ] Update steering documents

## Cleanup Checklist

When you notice the repository getting messy:

### Step 1: Assess
- [ ] Count files in root directory
- [ ] Identify duplicates
- [ ] Identify one-off scripts
- [ ] Identify old documentation

### Step 2: Create Archive Directories
```bash
mkdir -p archive/old_scripts
mkdir -p archive/old_docs
mkdir -p archive/test_data
```

### Step 3: Archive
```bash
# Archive old scripts
mv old_script.py archive/old_scripts/

# Archive old docs
mv OLD_DOC.md archive/old_docs/

# Archive test data
mv test_output.parquet archive/test_data/
```

### Step 4: Update .gitignore
```bash
echo "archive/" >> .gitignore
```

### Step 5: Verify
- [ ] Root directory has exactly 6 files (no more, no less)
- [ ] Only allowed files in root (.gitignore, README.md, STATUS.md, QUICK_START.md, main.py, requirements.txt)
- [ ] All production code in `src/`
- [ ] All CLI scripts in `scripts/`
- [ ] All tests in `tests/`
- [ ] Nothing lost (everything archived)

### Step 6: Document
- [ ] Update STATUS.md
- [ ] Update README.md if needed
- [ ] Create cleanup summary document

## Examples

### ✅ Good Repository State

```
Model2LSTM/
├── src/                    # Production code
├── scripts/                # CLI scripts
├── tests/                  # Testing
├── archive/                # Old stuff
├── README.md              # Overview
├── STATUS.md              # Current status
├── QUICK_START.md         # Quick reference
├── main.py                # Simple processor
└── requirements.txt       # Dependencies
```

**Root files: 6** ✅

### ❌ Bad Repository State

```
Model2LSTM/
├── process_v1.py
├── process_v2.py
├── process_fixed.py
├── process_monthly_chunks_fixed.py
├── test_oct2025.py
├── analyze_gaps.py
├── check_data.py
├── DOC1.md
├── DOC2.md
├── INVESTIGATION.md
├── VALIDATION.md
├── ISSUE.md
├── test_output.parquet
├── oct2025_processed.parquet
├── README.md
├── STATUS.md
├── main.py
└── requirements.txt
```

**Root files: 18** ❌

## Recovery Plan

If the repository becomes messy again:

### 1. Stop and Assess
- Don't add more files
- Identify the problem
- Review this document

### 2. Create Cleanup Plan
- List files to archive
- List files to consolidate
- List files to delete (if truly temporary)

### 3. Execute Cleanup
- Create archive directories
- Move files systematically
- Update .gitignore

### 4. Verify
- Check root file count
- Run integration tests
- Verify nothing broken

### 5. Document
- Update STATUS.md
- Create cleanup summary
- Update steering documents

### 6. Prevent Recurrence
- Review what went wrong
- Update rules if needed
- Set up maintenance schedule

## Current Status

**Last Cleanup:** November 2025 (Final)
**Root Files:** 6 (EXACTLY as required) ✅
**Status:** Clean and professional ✅

**Allowed files in root:**
1. .gitignore
2. README.md
3. STATUS.md
4. QUICK_START.md
5. main.py
6. requirements.txt

**Any other file in root must be archived immediately.**

**Maintenance Schedule:**
- Daily: Check during active development
- Weekly: Review and archive
- Monthly: Comprehensive review

**Next Review:** After Python 3.12/3.13 installation and production testing

---

**Remember:** 
- Root directory EXACTLY 6 files (no more, no less)
- Only these 6 files allowed: .gitignore, README.md, STATUS.md, QUICK_START.md, main.py, requirements.txt
- One file, one purpose
- Archive aggressively
- Single source of truth (STATUS.md)
- Keep it clean!
