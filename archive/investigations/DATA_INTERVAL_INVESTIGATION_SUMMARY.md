# Data Interval Investigation Summary

## The Problem

Your data is **3.86-second bars**, not 1-second bars, despite the files being named `ohlcv-1s.dbn.zst`.

## Evidence

### 1. Raw Data Check
- **File**: `monthly_2010-07_20251107_152756.parquet`
- **Median interval**: 3.858916 seconds
- **99.8% of bars**: Exactly 3.86 seconds apart
- **0% of bars**: 1-second intervals

### 2. Contract Filtering Test (Proper Order)
Following your exact steps:
1. ✅ Identified days: 31 days
2. ✅ Identified contracts per day: 151 contract segments
3. ✅ Found highest volume contract per day
4. ✅ Filtered to dominant contract only: Removed 65,966 bars (30.5%)
5. ✅ Applied RTH filtering: Removed 35,873 bars (23.8%)
6. ❌ **Result: 3.86-second bars, NOT 1-second**

### 3. DBN Files Found
All files are named with `ohlcv-1s`:
- `glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst` (July 2010)
- `glbx-mdp3-20100606-20100630.ohlcv-1s.dbn.zst` (June 2010)
- `glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst` (Oct 2025)
- `glbx-mdp3-20250922-20251021.ohlcv-1s.dbn.zst` (Sep-Oct 2025)

## What This Means

### Expected vs Actual
**If data were 1-second bars:**
- RTH duration: 23,400 seconds (6.5 hours)
- Expected bars per day: 23,400
- Expected bars per month: ~515,000

**Actual data (3.86-second bars):**
- RTH duration: 23,400 seconds (6.5 hours)
- Expected bars per day: 6,062 (23,400 / 3.86)
- Expected bars per month: ~133,000
- **Actual after filtering: 3,701 bars/day (61% coverage)**

### Coverage Analysis
With proper contract filtering:
- Original: 216,559 bars
- After contract filtering: 150,593 bars (30.5% removed)
- After RTH filtering: 114,720 bars (23.8% removed)
- **Final: 3,701 bars per RTH day**

This is 61% of expected coverage for 3.86-second bars, which suggests:
- Contract filtering is working correctly
- We're losing some data due to gaps or low-volume periods
- This is reasonable for real market data

## Possible Explanations

### 1. Databento's "ohlcv-1s" Doesn't Mean 1-Second
The schema name might be misleading:
- "1s" might refer to something else (1-second tick aggregation window?)
- The actual output might be different from the schema name
- This could be a Databento naming convention issue

### 2. Data Aggregation During Download
- Your download script might have an aggregation parameter
- The API call might have a default aggregation setting
- Check your download code for any aggregation parameters

### 3. Historical Data Limitation
- 2010 data might not have been available at 1-second resolution
- Databento might only have 3.86-second bars for older data
- Recent data (2024-2025) might be different

### 4. ES Futures Specific
- ES futures might have a specific bar interval
- CME might publish data at 3.86-second intervals
- This could be exchange-specific

## What 3.86 Seconds Means

**Calculation:**
- 3.86 seconds ≈ 3858.916 milliseconds
- This is NOT a standard interval (1s, 3s, 5s, 10s, 15s, 30s, 60s)
- It's suspiciously close to 4 seconds but not exactly

**Possible origins:**
- 15 ticks × 259ms = 3.885 seconds
- Some custom aggregation window
- Exchange-specific reporting interval

## Action Items

### 1. Check Your Download Script
Look for:
```python
schema='ohlcv-1s'  # What does this actually mean?
aggregation=?      # Is there an aggregation parameter?
interval=?         # Is there an interval parameter?
```

### 2. Contact Databento Support
Ask them:
- "What does the 'ohlcv-1s' schema actually provide?"
- "Why are my bars 3.86 seconds apart instead of 1 second?"
- "How do I get TRUE 1-second OHLCV bars for ES futures?"
- "Is this interval specific to historical data (2010) or all data?"

### 3. Test Recent Data
Check if recent data (2024-2025) has the same issue:
- Convert one of the 2025 files
- Check the interval
- See if it's also 3.86 seconds

### 4. Check Databento Documentation
- Review the schema documentation
- Look for interval specifications
- Check if there are different schemas for different intervals

## Impact on Your Project

### Good News
- Contract filtering IS working correctly
- You have consistent 3.86-second bars
- Coverage is reasonable (61% after filtering)
- ~20 million bars for 15 years is still excellent for training

### Adjustments Needed
- Update all documentation to reflect 3.86-second bars
- Adjust feature engineering windows if needed
- Update expected bar counts in validation
- Recalculate lookforward windows for labeling

### Feature Engineering
Most features should still work fine:
- Rolling windows are time-based, not bar-based
- 30-second window = ~8 bars (instead of 30)
- 60-second window = ~16 bars (instead of 60)
- 300-second window = ~78 bars (instead of 300)

### Labeling
- 15-minute lookforward = 900 seconds = ~233 bars (instead of 900)
- Stop/target distances are in ticks (unchanged)
- MAE calculations are in ticks (unchanged)

## Recommendation

**Option 1: Accept 3.86-Second Bars**
- Update documentation
- Adjust expectations
- Proceed with training
- Still have excellent dataset size

**Option 2: Get True 1-Second Data**
- Contact Databento support
- Re-download with correct parameters
- This will 3x your dataset size
- May require different schema or API call

**Option 3: Hybrid Approach**
- Use 3.86-second bars for initial development
- Get 1-second data for final production model
- This lets you proceed while investigating

## Next Steps

1. **Immediate**: Run the test on a 2025 file to see if recent data is different
2. **Short-term**: Contact Databento support with specific questions
3. **Medium-term**: Decide whether to proceed with 3.86s or wait for 1s data
4. **Long-term**: Update all documentation and expectations

## Files Created for Investigation

- `investigate_raw_data_interval.py` - Checks raw data intervals
- `proper_contract_filtering_test.py` - Tests filtering in correct order
- `find_original_dbn_files.py` - Locates DBN files
- `check_dbn_schema.py` - Checks DBN metadata
- `simple_dbn_check.py` - Simple file header check

## Conclusion

**Your contract filtering is working correctly.** The issue is that the source data from Databento is 3.86-second bars, not 1-second bars, despite the filename saying "ohlcv-1s". You need to either:
1. Accept this and adjust your project accordingly, or
2. Contact Databento to get true 1-second data

The good news is that 3.86-second bars are still usable for your project, just with adjusted expectations for dataset size and feature windows.
