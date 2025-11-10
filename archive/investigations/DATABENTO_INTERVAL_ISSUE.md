# Databento 'ohlcv-1s' Interval Issue - CRITICAL FINDING

## Executive Summary

**Databento's 'ohlcv-1s' schema does NOT provide 1-second bars.** The interval varies by time period:
- **July 2010**: 3.86 seconds per bar
- **October 2025**: 2.62 seconds per bar
- **Neither period**: 1-second bars

## Evidence

### July 2010 Data
- File: `glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst`
- Total rows: 216,559
- **Median interval: 3.858916 seconds**
- 99.8% of bars are exactly 3.86 seconds apart
- 0% are 1-second intervals

### October 2025 Data (Very Recent)
- File: `glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst`
- Total rows: 1,022,639
- **Median interval: 2.619109 seconds**
- 100% of bars are exactly 2.62 seconds apart
- 0% are 1-second intervals

### Comparison
- **Ratio**: 3.86 / 2.62 = 1.47x
- **Difference**: 1.24 seconds
- **Pattern**: Both are consistent within their period but different across periods

## What This Means

### The 'ohlcv-1s' Schema is Misleading
The schema name suggests "1-second OHLCV bars" but actually provides:
- Variable intervals depending on time period
- NOT 1-second bars in any period tested
- Consistent intervals within each period

### Possible Explanations

#### 1. Schema Name Doesn't Mean What We Think
- "1s" might refer to tick aggregation window, not bar interval
- "1s" might mean "1-second precision" not "1-second frequency"
- Databento might use different terminology than expected

#### 2. Exchange Data Availability
- CME might not have published 1-second bars in 2010
- CME might have changed reporting intervals over time
- Different periods might have different native resolutions

#### 3. Databento Aggregation Logic
- Databento might aggregate from tick data
- Aggregation logic might vary by data availability
- Older data might be aggregated differently than recent data

#### 4. Data Compression or Sampling
- Databento might be downsampling to reduce file size
- Different compression ratios for different periods
- This would explain the varying intervals

## Impact on Your Project

### Dataset Size Implications

**If data were 1-second bars:**
- RTH: 23,400 bars/day
- 15 years: ~86 million bars

**Actual with varying intervals:**
- 2010: ~6,062 bars/day (3.86s)
- 2025: ~8,935 bars/day (2.62s)
- 15 years: ~30-40 million bars (estimated)

### Feature Engineering Impact

**Time-based windows need adjustment:**
- 30-second window:
  - 2010: ~8 bars
  - 2025: ~11 bars
- 60-second window:
  - 2010: ~16 bars
  - 2025: ~23 bars
- 300-second window:
  - 2010: ~78 bars
  - 2025: ~115 bars

**This creates inconsistency across time periods!**

### Model Training Concerns

1. **Inconsistent temporal resolution** across training data
2. **Feature values will differ** between 2010 and 2025 data
3. **Rolling windows** will capture different time spans
4. **Lookforward periods** will have different bar counts

## Critical Questions for Databento

### Email Template

```
Subject: Clarification on 'ohlcv-1s' Schema - Variable Intervals

Hi Databento Support,

I'm using your 'ohlcv-1s' schema for ES futures data and have discovered 
that the bar intervals are NOT 1-second as the schema name suggests.

My findings:
- July 2010 data: 3.86-second intervals (99.8% of bars)
- October 2025 data: 2.62-second intervals (100% of bars)
- Neither period has 1-second intervals

Files tested:
- glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst
- glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst

Questions:
1. What does 'ohlcv-1s' actually mean? Does "1s" refer to something 
   other than 1-second bar intervals?

2. Why do the intervals vary by time period (3.86s in 2010, 2.62s in 2025)?

3. How can I obtain TRUE 1-second OHLCV bars for ES futures across 
   the full 2010-2025 period?

4. Is there a different schema or parameter I should use for consistent 
   1-second bars?

5. Is this behavior documented anywhere? I couldn't find information 
   about variable intervals in the schema documentation.

This is critical for my project as I need consistent temporal resolution 
across the entire 15-year dataset for machine learning model training.

Thank you for your help!
```

## Immediate Action Items

### 1. Contact Databento (URGENT)
- Send the email above
- Request clarification on schema behavior
- Ask for true 1-second data if available

### 2. Decide on Project Direction

**Option A: Wait for Databento Response**
- Pause processing until you get clarification
- May get access to true 1-second data
- Risk: Delays project timeline

**Option B: Proceed with Variable Intervals**
- Accept the varying intervals
- Adjust feature engineering to handle inconsistency
- Document the limitation
- Risk: Model may not generalize well across time periods

**Option C: Use Only Recent Data**
- Focus on 2025 data (2.62s intervals)
- Consistent intervals but smaller dataset
- May be sufficient for initial model
- Risk: Less training data

### 3. Technical Adjustments if Proceeding

**If you decide to proceed with current data:**

1. **Normalize features by time, not bars**
   - Use actual time windows, not bar counts
   - Calculate features based on seconds, not bar indices

2. **Add temporal metadata**
   - Include year/period as a feature
   - Let model learn period-specific patterns

3. **Separate validation by period**
   - Train on mixed periods
   - Validate on each period separately
   - Check for period-specific performance

4. **Document the limitation**
   - Note in all documentation
   - Include in model cards
   - Warn about temporal inconsistency

## Recommendations

### Short-term (This Week)
1. ✅ **Contact Databento immediately** with the questions above
2. ⏸️ **Pause full dataset processing** until you get clarification
3. 📊 **Analyze a few more months** to see if there's a pattern
   - Check 2015, 2020 data if available
   - Map out when intervals changed

### Medium-term (Next 2 Weeks)
1. **Based on Databento response**, decide on path forward
2. **If true 1-second data exists**, re-download everything
3. **If not**, adjust project scope and feature engineering

### Long-term (Project Completion)
1. **Document the data limitation** in all materials
2. **Test model performance** across different time periods
3. **Consider period-specific models** if performance varies significantly

## Files for Reference

Investigation scripts created:
- `check_oct2025_interval.py` - Checks October 2025 intervals
- `compare_all_intervals.py` - Compares intervals across periods
- `proper_contract_filtering_test.py` - Validates filtering logic
- `investigate_raw_data_interval.py` - Checks raw data intervals

## Conclusion

**Your contract filtering is working perfectly.** The issue is that Databento's 'ohlcv-1s' schema provides variable-interval data:
- 3.86 seconds in 2010
- 2.62 seconds in 2025
- NOT 1-second in any period

**You MUST contact Databento support** to clarify this before proceeding with full dataset processing. This is a fundamental data quality issue that affects your entire project.

The good news: Once you understand what data is actually available, you can make an informed decision about how to proceed.
