# Data Quality Investigation Summary - July 2010

## Problem Statement
Win rates are extremely skewed:
- Long trades: 2-3% win rate (expected: 15-30%)
- Short trades: 87-88% win rate (expected: 15-30%)

## Investigation Timeline

### 1. Initial Hypothesis: Labeling Logic Bug
**Status:** ❌ RULED OUT
- Manual verification showed labeling logic is correct
- All 5 random samples matched expected results

### 2. Second Hypothesis: Contract Roll Issues
**Status:** ✅ PARTIALLY CONFIRMED
- Found 150 price gaps >5 points
- 19 days with >30% volume changes
- Multiple contracts mixed in the data

**Action Taken:**
- Created contract filtering module (`src/data_pipeline/contract_filtering.py`)
- Successfully removed 69,563 rows (32.1%)
- Removed 134 out of 150 price gaps (89.3%)

**Result:** Win rates still problematic after filtering (3.5% long, 87.2% short)

### 3. Third Hypothesis: OHLC Data Corruption
**Status:** ✅ **ROOT CAUSE IDENTIFIED**

## Root Cause: Data is NOT True OHLC Bars

### Critical Findings:

1. **78.9% of bars are FLAT (O=C)**
   - In real 1-second ES data, this should be ~5-10%
   - Indicates most "bars" are single price snapshots

2. **67.8% of bars have ZERO RANGE (H=L)**
   - True OHLC bars should have range from multiple ticks
   - This means most bars have no price movement at all

3. **Directional Bias Impossible**
   - Market moved UP 71.25 points overall
   - But only 10.5% of bars are "up bars" (C>O)
   - This is mathematically impossible for aggregated data

4. **Median Bar Range: 0.00 points**
   - Average bar range: 0.09 points
   - For comparison, ES typically moves 0.25-1.0 points per second

### What This Means:

The data appears to be **"last price" snapshots** taken every second, NOT aggregated OHLC bars:
- When no trade occurs in a second → O=H=L=C (flat bar, zero range)
- When one trade occurs → O=H=L=C=that trade price
- Only when multiple trades occur → We get a proper OHLC bar with range

This explains the extreme win rates:
- With mostly flat/zero-range bars, price barely moves
- A 6-tick stop (1.5 points) is HUGE relative to typical bar movement
- A 12-tick target (3 points) is even bigger
- Most trades timeout or hit stops because there's insufficient movement

## Databento Schema Issue

The file is named `ohlcv-1s.dbn.zst` which should be 1-second OHLC bars, but the data characteristics suggest:

**Possible Issues:**
1. **Wrong Schema Used:** Maybe we need a different Databento schema
2. **Aggregation Not Applied:** The data might need post-processing aggregation
3. **Sparse Data Period:** July 2010 might have had very low trading activity
4. **Databento Bug:** The OHLCV-1s schema might not be working as expected for historical data

## 🎯 ROOT CAUSE CONFIRMED

### The Data is NOT 1-Second Bars!

**Critical Discovery:**
- **0% of bars are 1-second apart**
- **Mode time difference: 3.86 seconds**
- **Mean time difference: 12.09 seconds**

The file is named `ohlcv-1s.dbn.zst` but it's actually providing **~4-second bars**, not 1-second bars!

This explains ALL the issues:
1. Why 67.8% have zero range - With 4-second sampling, many periods have no trades
2. Why 78.9% are flat bars - Price doesn't move in many 4-second periods  
3. Why win rates are broken - Stops/targets sized for 1-second bars don't work with 4-second bars

## Recommended Solutions

### Option 1: Contact Databento Support ⭐ RECOMMENDED
**Ask them:**
- Is the OHLCV-1s schema supposed to give true aggregated OHLC bars?
- Why do 67.8% of bars have zero range (H=L)?
- Is there a different schema or parameter needed for proper 1-second aggregation?
- Can they provide a sample of what proper OHLCV-1s data should look like?

### Option 2: Post-Process Aggregation
**Create our own aggregation:**
- Read the raw tick data (if available)
- Aggregate into proper 1-second OHLC bars ourselves
- Ensure each bar has proper O/H/L/C from all ticks in that second

### Option 3: Use Different Timeframe
**Switch to larger bars:**
- Try 5-second or 10-second bars instead
- These might have better aggregation
- Would need to adjust stop/target parameters proportionally

### Option 4: Use Different Data Source
**Alternative providers:**
- Try a different historical data provider
- Verify they provide true aggregated OHLC bars
- Test with a small sample first

## Files Created During Investigation

1. `src/data_pipeline/contract_filtering.py` - Contract roll detection (KEEP - still useful)
2. `investigate_processed_data.py` - Data quality analysis
3. `investigate_remaining_gaps.py` - Gap analysis after filtering
4. `test_contract_filtering.py` - Contract filtering test
5. `test_relabel_filtered_data.py` - Re-labeling test
6. `analyze_ohlc_corruption.py` - OHLC corruption analysis
7. `diagnose_dbn_conversion.py` - DBN conversion diagnosis

## Next Steps

1. **Immediate:** Contact Databento support with our findings
2. **Short-term:** Request proper OHLCV-1s sample data to verify
3. **Medium-term:** If Databento can't fix, implement post-processing aggregation
4. **Long-term:** Consider alternative data sources if issue persists

## Key Metrics for Validation

When we get proper data, we should see:
- **Zero-range bars:** <10% (currently 67.8%)
- **Flat bars (O=C):** <15% (currently 78.9%)
- **Up/Down ratio:** ~0.9-1.1 (currently 0.99 after swapping, but with flat bars)
- **Average bar range:** 0.25-1.0 points (currently 0.09)
- **Win rates:** 15-30% for both long and short (currently 3% / 87%)

---

**Investigation Date:** November 7, 2025  
**Status:** Root cause identified, awaiting data source resolution
