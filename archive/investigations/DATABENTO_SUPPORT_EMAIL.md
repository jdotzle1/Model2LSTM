# Email to Databento Support

## Subject: OHLCV-1s Schema Provides 2.6-3.9 Second Bars, Not 1-Second Bars (Affects All Time Periods)

## Email Body:

Hello Databento Support,

I'm experiencing an issue with historical ES futures data downloaded using the OHLCV-1s schema. The data does not appear to be true 1-second aggregated bars as expected.

### Issue Summary

**Files Tested:**
1. `glbx-mdp3-20100701-20100731.ohlcv-1s.dbn.zst` (July 2010)
2. `glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst` (October 2025)

**Schema:** `ohlcv-1s`  
**Symbol:** `ES.FUT` (parent)  
**Issue:** Neither file provides true 1-second bars as expected

### Problem Description

When analyzing the converted data, I found:

1. **Timestamp Intervals Are NOT 1-Second:**
   - Expected: 1.0 second between consecutive bars
   - Actual: 3.86 seconds (mode), 12.09 seconds (mean)
   - **0% of bars are 1-second apart**

2. **Excessive Zero-Range Bars:**
   - 67.8% of bars have H=L (zero range)
   - This is impossible for properly aggregated 1-second OHLC data
   - Suggests the data is "last price" snapshots rather than aggregated bars

3. **Excessive Flat Bars:**
   - 78.9% of bars have O=C (flat bars)
   - Normal 1-second ES data should have ~5-10% flat bars
   - This indicates most "bars" are single price points, not aggregated data

### Analysis Results

**July 2010 Data:**
```
Timestamp Analysis:
- Mode time difference: 3.86 seconds
- Mean time difference: 12.09 seconds  
- 1-second intervals: 0 (0.0%)
- Total rows: 216,559

OHLC Quality:
- Zero range bars (H=L): 67.8%
- Flat bars (O=C): 78.9%
- Median bar range: 0.00 points
- Average bar range: 0.09 points
```

**October 2025 Data:**
```
Timestamp Analysis:
- Mode time difference: 2.62 seconds (uniform)
- Mean time difference: 2.62 seconds
- 1-second intervals: 0 (0.0%)
- Total rows: 1,022,639

Pattern:
- ALL bars exactly 2.62 seconds apart
- No variation in intervals
- Appears to be systematic under-sampling
```

**Comparison:**
| Metric | July 2010 | October 2025 | Expected |
|--------|-----------|--------------|----------|
| Interval | 3.86s | 2.62s | 1.00s |
| 1-sec bars | 0% | 0% | 100% |

### Expected vs Actual

**Expected for OHLCV-1s:**
- Consecutive bars exactly 1 second apart
- Each bar aggregates ALL trades in that 1-second window
- H = highest trade price in that second
- L = lowest trade price in that second
- O = first trade price in that second
- C = last trade price in that second
- Most bars should have some range (H > L)

**What I'm Getting:**
- Bars approximately 4 seconds apart (3.86s mode)
- Most bars appear to be single price snapshots
- Very few bars have any range
- Looks like "last price every ~4 seconds" rather than aggregated OHLC

### Questions

1. **Is OHLCV-1s supposed to provide 1-second bars?**
   - The schema name suggests 1-second resolution
   - But we're getting 2.62-3.86 second intervals instead
   - Is this documented behavior?

2. **Why different intervals for different time periods?**
   - July 2010: 3.86 seconds (variable)
   - October 2025: 2.62 seconds (uniform)
   - Both are wrong, but why different?

3. **Is there a different schema for true 1-second resolution?**
   - Should I be using a different schema?
   - Is there a parameter I'm missing?
   - How do I get actual 1-second aggregated OHLC bars?

4. **Is this a known issue?**
   - Has this been reported before?
   - Is there a fix or workaround?
   - Is this documented anywhere?

5. **Can you provide data with true 1-second resolution?**
   - If OHLCV-1s doesn't provide 1-second bars, what does?
   - Can you verify your data and confirm the actual resolution?
   - Is there an alternative approach to get what I need?

### Download Parameters

```json
{
    "dataset": "GLBX.MDP3",
    "schema": "ohlcv-1s",
    "symbols": ["ES.FUT"],
    "stype_in": "parent",
    "stype_out": "instrument_id",
    "start": "2010-07-01",
    "end": "2010-07-31",
    "encoding": "dbn",
    "compression": "zstd"
}
```

### Impact

This data quality issue is preventing me from:
- Training machine learning models on 1-second bars
- Backtesting trading strategies with proper resolution
- Using the data for its intended purpose

### Request

Could you please:
1. Verify if this is expected behavior or a data quality issue
2. Provide guidance on how to obtain true 1-second aggregated OHLC bars
3. Confirm if recent data (2024-2025) has the same issue
4. Suggest alternative schemas or parameters if OHLCV-1s isn't appropriate

I'm happy to provide additional analysis or data samples if helpful.

Thank you for your assistance!

---

**Attachments to Include:**
- Sample of the converted data showing timestamp intervals
- Statistics showing zero-range and flat bar percentages
- Comparison of expected vs actual data characteristics

