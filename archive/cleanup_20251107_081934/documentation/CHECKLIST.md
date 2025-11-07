# Complete Fix Checklist

## Phase 1: Understand the Problem ✅

- [x] Identified 100+ debug scripts in root
- [x] Found TWO labeling modules (old and new)
- [x] Discovered EC2 code mismatch
- [x] Verified production code path
- [x] Documented findings

## Phase 2: Local Cleanup (30 min)

- [ ] Read START_HERE.md
- [ ] Run `python RUN_COMPLETE_FIX.py`
- [ ] Confirm cleanup when prompted
- [ ] Verify root directory is clean
- [ ] Check archive directory created

**Verification:**
```bash
# Should see <10 files
ls -la

# Should see archived files
ls -la archive/cleanup_*/
```

## Phase 3: Create Deployment (15 min)

- [ ] Deployment package created automatically
- [ ] Review MANIFEST.txt
- [ ] Verify file hashes
- [ ] Note package location

**Verification:**
```bash
# Find package
ls -lh deployment/ec2/ec2_deployment_package_*.tar.gz

# Check manifest
tar -xzf deployment/ec2/ec2_deployment_package_*.tar.gz -O */MANIFEST.txt
```

## Phase 4: Deploy to EC2 (30 min)

- [ ] Upload package to EC2
  ```bash
  scp deployment/ec2/ec2_deployment_package_*.tar.gz ec2-user@YOUR_INSTANCE:/home/ec2-user/
  ```

- [ ] SSH to EC2
  ```bash
  ssh ec2-user@YOUR_INSTANCE
  ```

- [ ] Extract package
  ```bash
  tar -xzf ec2_deployment_package_*.tar.gz
  cd ec2_deployment_package_*
  ```

- [ ] Verify hashes match
  ```bash
  cat MANIFEST.txt
  md5sum src/data_pipeline/weighted_labeling.py
  # Compare to local hash
  ```

- [ ] Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Test imports
  ```bash
  python -c "from src.data_pipeline.weighted_labeling import WeightedLabelingEngine; print('OK')"
  ```

**Verification:**
- [ ] All hashes match local
- [ ] Imports work without errors
- [ ] No missing dependencies

## Phase 5: Re-run June 2011 (2-3 hours)

- [ ] Start processing
  ```bash
  python process_monthly_chunks_fixed.py --month 2011-06
  ```

- [ ] Monitor progress
  ```bash
  tail -f processing.log
  ```

- [ ] Wait for completion

- [ ] Verify output files created
  ```bash
  ls -lh *2011-06*.parquet
  ls -lh *2011-06*.json
  ```

**Verification:**
- [ ] Processing completed without errors
- [ ] Output files exist
- [ ] Statistics JSON created

## Phase 6: Download and Compare (30 min)

- [ ] Download new results
  ```bash
  # On local machine
  scp ec2-user@YOUR_INSTANCE:/path/to/monthly_2011-06_*.json ./results_new_code/
  scp ec2-user@YOUR_INSTANCE:/path/to/monthly_2011-06_*.parquet ./results_new_code/
  ```

- [ ] Compare statistics
  ```bash
  # Old results (from previous run)
  cat ~/Downloads/monthly_2011-06_*_statistics.json | grep -A 20 "win_rates"
  
  # New results
  cat results_new_code/monthly_2011-06_*_statistics.json | grep -A 20 "win_rates"
  ```

- [ ] Document differences

**Key Metrics to Compare:**
- [ ] Short win rates (old: ~66%, new: ???)
- [ ] Long win rates
- [ ] Total trades
- [ ] Average weights

## Phase 7: Analyze Results (30 min)

### Scenario A: Win Rates Drop Significantly (e.g., 66% → 45%)

- [ ] ✅ Bug was in old code
- [ ] ✅ New code is correct
- [ ] ✅ Document the fix
- [ ] ✅ Continue processing remaining months
- [ ] ✅ Update documentation

**Action:** Proceed with confidence

### Scenario B: Win Rates Stay Similar (e.g., 66% → 64%)

- [ ] ✅ Market behavior is real
- [ ] ✅ June 2011 was exceptional
- [ ] ✅ Document market conditions
- [ ] ✅ Continue processing remaining months
- [ ] ✅ Accept the results

**Action:** Proceed with confidence

### Scenario C: Win Rates Change Unexpectedly (e.g., 66% → 80%)

- [ ] ⚠️ Investigate new code changes
- [ ] ⚠️ Compare logic differences
- [ ] ⚠️ Test on other months
- [ ] ⚠️ Review recent fixes
- [ ] ⚠️ May need further debugging

**Action:** Investigate before proceeding

## Phase 8: Document and Continue

- [ ] Update README.md with findings
- [ ] Document production code path
- [ ] Create deployment checklist
- [ ] Establish verification process
- [ ] Process remaining months

**Best Practices Going Forward:**
- [ ] Always verify EC2 code matches local
- [ ] Use file hashes for verification
- [ ] Keep root directory clean
- [ ] Archive debug scripts immediately
- [ ] Document deployment process

## Success Criteria

✅ **Complete when:**
- [ ] Root directory cleaned
- [ ] Deployment package created
- [ ] EC2 code verified
- [ ] June 2011 re-run complete
- [ ] Results compared and explained
- [ ] Decision made on next steps

## Time Estimate

- Phase 1: ✅ Complete
- Phase 2: 30 minutes
- Phase 3: 15 minutes
- Phase 4: 30 minutes
- Phase 5: 2-3 hours (automated)
- Phase 6: 30 minutes
- Phase 7: 30 minutes
- Phase 8: 30 minutes

**Total:** ~5-6 hours (mostly automated)

## Notes

### Important Files
- `START_HERE.md` - Overview
- `RUN_COMPLETE_FIX.py` - Automation script
- `ANSWER_THE_QUESTION.md` - Detailed answer
- `CRITICAL_FINDING.md` - Code mismatch details
- `CLEANUP_AND_DEPLOYMENT_SUMMARY.md` - Complete plan

### Backup Plan
If anything goes wrong:
- Old code is in `deployment/ec2/ec2_deployment_package/`
- Debug scripts are in `archive/cleanup_*/`
- Can restore from archives if needed

### Questions During Process
- Check documentation files
- Review error messages carefully
- Verify file hashes at each step
- Don't skip verification steps

## Ready to Start?

```bash
python RUN_COMPLETE_FIX.py
```

Good luck! 🚀
