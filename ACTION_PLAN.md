# Action Plan - Data Quality Resolution

## Current Status: ROOT CAUSE IDENTIFIED ✅

The Databento OHLCV-1s data is **NOT providing true 1-second bars**. Instead, it's providing approximately **4-second bars** (3.86s mode).

## Immediate Actions

### 1. Contact Databento Support (PRIORITY 1)
**File:** `DATABENTO_SUPPORT_EMAIL.md` contains the complete email template

**Key Points to Include:**
- File shows 0% of bars are 1-second apart (mode: 3.86s)
- 67.8% of bars have zero range (H=L)
- 78.9% of bars are flat (O=C)
- Request verification and guidance

**Expected Response:**
- Confirmation if this is expected for 2010 data
- Alternative schema or parameters to get true 1-second bars
- Explanation of data availability for historical periods

### 2. Test October 2025 Data (PRIORITY 2)
**File:** `C:\Users\jdotzler\Downloads\glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst`

**Why:** Determine if this is a historical data issue or affects all data

**Steps:**
1. Install databento (resolve dependency conflicts)
2. Convert October 2025 file to Parquet
3. Run timestamp sequencing analysis
4. Compare with July 2010 results

**If October 2025 has same issue:** Problem is with OHLCV-1s schema itself
**If October 2025 is fine:** Problem is specific to historical data from 2010

### 3. Review Contract Filtering (COMPLETED ✅)
**Status:** Contract filtering module created and tested

**Results:**
- Successfully removed 69,563 rows (32.1%)
- Removed 134 out of 150 price gaps (89.3%)
- Module is working correctly and should be kept

**Note:** Contract filtering helped but didn't fix the fundamental data quality issue

## Pending Databento Response

### Scenario A: Databento Confirms Issue
**Action:** Request proper 1-second data or refund
**Timeline:** Wait for Databento to provide corrected data
**Backup:** Consider alternative data providers

### Scenario B: This is Expected for 2010 Data
**Action:** Adjust trading parameters for 4-second bars
**Changes Needed:**
- Recalculate stops/targets for 4-second resolution
- Adjust lookforward period (currently 900 bars = 15 min)
- Update feature engineering for 4-second timeframe

### Scenario C: Need Different Schema
**Action:** Re-download with correct schema/parameters
**Steps:**
1. Get schema recommendation from Databento
2. Re-download all historical data
3. Re-process with new data

## Technical Debt to Address

### Files Created During Investigation (Keep)
1. `src/data_pipeline/contract_filtering.py` - ✅ KEEP (still valuable)
2. `verify_timestamp_sequencing.py` - ✅ KEEP (useful diagnostic)
3. `analyze_ohlc_corruption.py` - ✅ KEEP (useful diagnostic)
4. `DATA_QUALITY_INVESTIGATION_SUMMARY.md` - ✅ KEEP (documentation)
5. `DATABENTO_SUPPORT_EMAIL.md` - ✅ KEEP (reference)

### Files to Clean Up (After Resolution)
1. `investigate_processed_data.py` - Can archive
2. `investigate_remaining_gaps.py` - Can archive
3. `test_contract_filtering.py` - Can archive
4. `test_relabel_filtered_data.py` - Can archive
5. `diagnose_dbn_conversion.py` - Can archive

## Pipeline Updates Needed

### If Data is Fixed
1. Re-download all historical data with correct schema
2. Re-run full pipeline with contract filtering enabled
3. Verify win rates are reasonable (15-30%)
4. Proceed with model training

### If Using 4-Second Bars
1. Update stop/target parameters:
   - Current: 6/8/10 tick stops, 12/16/20 tick targets
   - Adjusted: Scale proportionally for 4-second bars
2. Update lookforward period:
   - Current: 900 bars (15 min at 1-second)
   - Adjusted: 225 bars (15 min at 4-second)
3. Update feature engineering:
   - Adjust rolling windows (30s → 8 bars, 300s → 75 bars)
   - Verify features still make sense at 4-second resolution

## Success Criteria

### Data Quality Validation
- [ ] Timestamp intervals are consistent (1-second or documented resolution)
- [ ] Zero-range bars <10% (currently 67.8%)
- [ ] Flat bars <15% (currently 78.9%)
- [ ] Average bar range >0.25 points (currently 0.09)

### Labeling Validation
- [ ] Long win rates: 15-30% (currently 3%)
- [ ] Short win rates: 15-30% (currently 87%)
- [ ] Manual verification matches labels
- [ ] No extreme directional bias

### Pipeline Validation
- [ ] Contract filtering working correctly
- [ ] RTH filtering working correctly
- [ ] Feature engineering producing valid features
- [ ] No data corruption or gaps

## Timeline

**Week 1 (Current):**
- [x] Identify root cause
- [ ] Contact Databento support
- [ ] Test October 2025 data

**Week 2:**
- [ ] Receive Databento response
- [ ] Implement solution based on response
- [ ] Re-process sample data

**Week 3:**
- [ ] Validate solution works
- [ ] Re-process full historical dataset
- [ ] Verify win rates are reasonable

**Week 4:**
- [ ] Begin model training with corrected data
- [ ] Deploy to production

## Contact Information

**Databento Support:**
- Email: support@databento.com
- Documentation: https://docs.databento.com/
- Status Page: https://status.databento.com/

## Notes

- Keep all investigation files until issue is resolved
- Document any responses from Databento
- Update this action plan as situation evolves
- Consider alternative data providers if Databento can't resolve

---

**Last Updated:** November 7, 2025  
**Status:** Awaiting Databento support response  
**Next Action:** Send support email and test October 2025 data
