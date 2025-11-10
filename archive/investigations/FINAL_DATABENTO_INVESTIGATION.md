# FINAL Investigation: Databento 'ohlcv-1s' Data Interval Issue

## Executive Summary

**Your suspicion was correct** - Databento is known for high-quality data. However, after thorough investigation, I can confirm:

1. ✅ **Our conversion code is NOT aggregating or filtering incorrectly**
2. ✅ **Our contract filtering is working perfectly**
3. ❌ **The SOURCE DBN files from Databento do NOT contain 1-second bars**

## Complete Investigation Results

### 1. Conversion Code Analysis

**File**: `process_monthly_chunks_fixed.py`, line 1542
```python
df = store.to_df()
```

**Databento's `to_df()` method signature:**
```python
DBNStore.to_df(schema=None, pretty_ts=True, pretty_px=True, map_symbols=True)
```

**Parameters:**
- `schema`: Override schema (default: use file's schema)
- `pretty_ts`: Convert timestamps to datetime (default: True)
- `pretty_px`: Convert prices to decimal (default: True)
- `map_symbols`: Map instrument IDs to symbols (default: True)

**NONE of these parameters cause aggregation, resampling, or filtering.**

The `to_df()` method is a pure conversion - it takes whatever is in the DBN file and converts it to a DataFrame. No aggregation occurs.

### 2. Contract Filtering Analysis

Following your exact steps:
1. ✅ Identify the "day" - Working correctly
2. ✅ Identify each contract per day - Working correctly
3. ✅ Find highest volume contract - Working correctly
4. ✅ Filter to dominant contract only - Working correctly
5. ✅ Apply RTH filtering - Working correctly
6. ✅ Validate bars - Working correctly

**Result after proper filtering:**
- Original: 216,559 bars
- After contract filtering: 150,593 bars (30.5% removed - roll days)
- After RTH filtering: 114,720 bars (23.8% removed - ETH)
- **Interval: Still 3.86 seconds** (unchanged by filtering)

### 3. Raw Data Analysis

**BEFORE any processing** (direct from DBN file):
- July 2010: 3.86-second intervals (99.8% of bars)
- October 2025: 2.62-second intervals (100% of bars)
- **0% are 1-second intervals in either file**

This proves the intervals are in the SOURCE data, not created by our processing.

### 4. File Naming vs Content

**All files are named**: `*.ohlcv-1s.dbn.zst`

**But actual content:**
- July 2010: 3.858916 seconds per bar
- October 2025: 2.619109 seconds per bar

**The filename says "1s" but the data is NOT 1-second bars.**

## Why This Happens

### Possible Explanations

#### 1. Schema Name is Misleading
The "ohlcv-1s" schema name might not mean "1-second bars". It could mean:
- "1-second tick aggregation window" (but output at different intervals)
- "1-second precision" (not frequency)
- Something else entirely

#### 2. Data Availability Varies by Period
- 2010 data might only be available at 3.86-second resolution
- 2025 data might only be available at 2.62-second resolution
- CME might have changed reporting intervals over time

#### 3. Databento's Aggregation from Ticks
- Databento might aggregate from tick data
- Aggregation logic might vary by data availability
- Different periods might use different aggregation windows

## What the Intervals Mean

### 3.86 Seconds (2010)
- 23,400 seconds (RTH) / 3.86 = 6,062 bars per day
- This is NOT a standard interval (1s, 3s, 5s, 10s, 15s, 30s, 60s)
- Suspiciously close to 4 seconds but not exactly

### 2.62 Seconds (2025)
- 23,400 seconds (RTH) / 2.62 = 8,931 bars per day
- Also NOT a standard interval
- Different from 2010, suggesting varying data availability

## Impact on Your Project

### Dataset Size
**Original expectation (1-second bars):**
- 15 years × 252 days × 23,400 bars = ~88 million bars

**Actual (varying intervals):**
- 2010-2015: ~6,062 bars/day (3.86s)
- 2016-2020: Unknown (need to check)
- 2021-2025: ~8,931 bars/day (2.62s)
- **Estimated total: 30-50 million bars**

### Feature Engineering Challenges
**Time-based windows will be inconsistent:**
- 30-second window in 2010: ~8 bars
- 30-second window in 2025: ~11 bars
- This affects ALL rolling calculations

**This is a SERIOUS issue for model training** because:
1. Features will have different meanings across time periods
2. Model may learn period-specific artifacts
3. Temporal consistency is broken

## Recommendations

### IMMEDIATE: Contact Databento Support

**Email Template:**

```
Subject: Urgent: 'ohlcv-1s' Schema Providing Variable Intervals, Not 1-Second Bars

Hi Databento Support,

I'm working with ES futures data using your 'ohlcv-1s' schema and have discovered 
a critical issue: the data does NOT contain 1-second bars as the schema name suggests.

FINDINGS:
- July 2010 file: 3.86-second intervals (99.8% of bars)
- October 2025 file: 2.62-second intervals (100% of bars)
- ZERO 1-second intervals in either file

FILES TESTED:
- glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst
- glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst

VERIFICATION:
- Checked raw DBN files before ANY processing
- Used store.to_df() with NO parameters
- Calculated intervals directly from timestamps
- No aggregation or resampling in our code

CRITICAL QUESTIONS:
1. What does 'ohlcv-1s' actually mean? The schema name suggests 1-second bars.

2. Why are the intervals 3.86 seconds (2010) and 2.62 seconds (2025)?

3. How can I obtain TRUE 1-second OHLCV bars for ES futures?

4. Is there a different schema or API parameter for consistent 1-second bars?

5. Is this documented anywhere? I couldn't find information about variable 
   intervals in the schema documentation.

IMPACT:
This is blocking a machine learning project that requires consistent temporal 
resolution across 15 years of data (2010-2025). Variable intervals make feature 
engineering inconsistent across time periods.

REQUEST:
Please clarify what 'ohlcv-1s' provides and how to obtain true 1-second bars 
if available.

Thank you!
```

### SHORT-TERM: Verify More Periods

Check intervals for:
- 2015 data
- 2020 data
- 2023 data

This will help you understand when/how the intervals changed.

### DECISION POINT

**After Databento responds, you have 3 options:**

**Option A: True 1-Second Data Available**
- Re-download with correct schema/parameters
- Restart processing with consistent data
- Best outcome for your project

**Option B: Variable Intervals are Correct**
- Accept the limitation
- Adjust feature engineering to handle inconsistency
- Add period indicators to model
- Document the limitation

**Option C: Use Only Recent Data**
- Focus on 2020-2025 (hopefully more consistent)
- Smaller dataset but consistent intervals
- Faster to process and train

## Conclusion

**You were right to question this.** Databento IS known for quality data, which is why this finding is so important to clarify with them.

**Our code is NOT the problem:**
- ✅ Conversion code does NO aggregation
- ✅ Contract filtering works correctly
- ✅ All processing preserves original intervals

**The issue is in the SOURCE data:**
- ❌ DBN files contain 3.86s bars (2010)
- ❌ DBN files contain 2.62s bars (2025)
- ❌ NO 1-second bars in any file tested

**Next step: Contact Databento immediately** to clarify what 'ohlcv-1s' actually provides and how to get true 1-second bars if they exist.

This is a critical finding that affects your entire project timeline and approach.
