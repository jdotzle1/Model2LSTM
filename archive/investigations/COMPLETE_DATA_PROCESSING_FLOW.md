# Complete Data Processing Flow - Detailed Specification

## Overview

This document details the EXACT flow from compressed DBN files through to final ML-ready data with weighted labels and features.

## Processing Pipeline

```
DBN.ZST → DataFrame → Contract Filter → Gap Fill → RTH Filter → Weighted Labeling → Feature Engineering → Final Dataset
```

---

## STEP 1: DBN Decompression & Conversion

### Input
- **File**: `glbx-mdp3-YYYYMM-YYYYMM.ohlcv-1s.dbn.zst`
- **Format**: Compressed Databento binary format
- **Content**: 1-second OHLCV bars with gaps (zero-volume bars omitted)
- **Contracts**: Multiple ES contracts (ESZ5, ESH6, ESM6, etc.)

### Process
```python
import databento as db
store = db.DBNStore.from_file(dbn_path)
df = store.to_df()
```

### Output
- **DataFrame** with columns:
  - `timestamp` (or `ts_event` as index)
  - `open`, `high`, `low`, `close`, `volume`
  - `symbol` (contract identifier)
  - `instrument_id`, `publisher_id`, `rtype`

### Data Characteristics
- **Rows**: Variable (gaps where volume = 0)
- **Intervals**: Mostly 1 second, with gaps (2s, 3s, 4s, etc.)
- **Contracts**: ALL contracts present (ESZ5, ESH6, spreads, etc.)
- **Time coverage**: Full 24-hour period (ETH + RTH)

### Example
```
timestamp                    symbol  open     close   volume
2025-11-09 23:00:00+00:00   ESZ5    6788.00  6788.00  730
2025-11-09 23:00:00+00:00   ESH6    6850.00  6850.00  1
2025-11-09 23:00:01+00:00   ESZ5    6787.25  6787.25  219
2025-11-09 23:00:02+00:00   ESZ5    6789.50  6789.50  133
[GAP - no row for 23:00:03 ESZ5]
2025-11-09 23:00:04+00:00   ESZ5    6792.25  6792.25  184
```

**Row Count Example (July 2010)**: ~216,559 rows (all contracts, all hours, with gaps)

---

## STEP 2: Contract Filtering

### Purpose
Keep ONLY the dominant (highest volume) contract for each trading day to avoid contract roll artifacts.

### Input
- DataFrame from Step 1 with multiple contracts

### Process
```python
from src.data_pipeline.contract_filtering import detect_and_filter_contracts

df_filtered, stats = detect_and_filter_contracts(df, min_daily_volume=50000)
```

### Algorithm
1. **Group by trading day** (Central Time)
2. **Detect contract segments** using price gaps (>5 points)
3. **Calculate volume per segment** for each day
4. **Identify dominant contract** (highest volume) per day
5. **Filter**: Keep ONLY bars from dominant contract
6. **Remove low-volume days** (total volume < 50K)

### What Gets Removed
- **Non-dominant contracts** on roll days
- **Spread contracts** (ESZ5-ESH6, etc.)
- **Far-month contracts** (ESM6, ESU6, etc.)
- **Low-volume days** (holidays, data issues)

### Output
- **DataFrame** with ONLY primary contract bars
- **Still has gaps** (zero-volume bars still omitted)
- **Still has ETH** (Extended Trading Hours)

### Data Characteristics
- **Rows**: Reduced by ~30-40% (contract filtering)
- **Intervals**: Still has gaps (1s, 2s, 3s, etc.)
- **Contracts**: Single contract per day
- **Time coverage**: Full 24-hour period

### Example
```
timestamp                    symbol  open     close   volume
2025-11-09 23:00:00+00:00   ESZ5    6788.00  6788.00  730
2025-11-09 23:00:01+00:00   ESZ5    6787.25  6787.25  219
2025-11-09 23:00:02+00:00   ESZ5    6789.50  6789.50  133
[GAP - no row for 23:00:03]
2025-11-09 23:00:04+00:00   ESZ5    6792.25  6792.25  184
```

**Row Count Example (July 2010)**: ~150,593 rows (30.5% removed)

---

## STEP 3: Gap Filling

### Purpose
Fill gaps where Databento omitted zero-volume bars to create perfect 1-second intervals.

### Input
- DataFrame from Step 2 with gaps

### Process
```python
from src.data_pipeline.gap_filling import fill_1second_gaps

df_complete = fill_1second_gaps(df_filtered, forward_fill_price=True)
```

### Algorithm
1. **Identify time range**: min to max timestamp
2. **Generate complete 1-second range**: Every second in range
3. **Merge with original data**: Left join on timestamp
4. **Fill missing values**:
   - `volume` = 0 (no trading activity)
   - `open` = last close (forward fill)
   - `high` = last close (forward fill)
   - `low` = last close (forward fill)
   - `close` = last close (forward fill)

### What Gets Added
- **Zero-volume bars** for every second with no trading
- **Forward-filled prices** (last traded price)
- **Typically 40-50%** of final rows are filled gaps

### Output
- **DataFrame** with PERFECT 1-second intervals
- **100% 1-second spacing** (no gaps)
- **Still has ETH** (Extended Trading Hours)

### Data Characteristics
- **Rows**: Increased by ~40-50% (gap filling)
- **Intervals**: EXACTLY 1 second (100%)
- **Contracts**: Single contract per day
- **Time coverage**: Full 24-hour period, every second

### Example
```
timestamp                    symbol  open     close   volume
2025-11-09 23:00:00+00:00   ESZ5    6788.00  6788.00  730
2025-11-09 23:00:01+00:00   ESZ5    6787.25  6787.25  219
2025-11-09 23:00:02+00:00   ESZ5    6789.50  6789.50  133
2025-11-09 23:00:03+00:00   ESZ5    6789.50  6789.50  0    ← FILLED
2025-11-09 23:00:04+00:00   ESZ5    6792.25  6792.25  184
```

**Row Count Example (July 2010)**: ~216,000 rows (gap filling added ~65,000 rows)

---

## STEP 4: RTH Filtering

### Purpose
Keep ONLY Regular Trading Hours (9:30 AM - 4:00 PM ET) for model training.

### Input
- DataFrame from Step 3 with complete 1-second bars (ETH + RTH)

### Process
```python
import pytz
from datetime import time as dt_time

# Convert to Central Time
central_tz = pytz.timezone('US/Central')
df['timestamp_ct'] = df['timestamp'].dt.tz_convert(central_tz)
df['time'] = df['timestamp_ct'].dt.time

# Filter to RTH
rth_start = dt_time(9, 30)  # 9:30 AM ET = 8:30 AM CT
rth_end = dt_time(16, 0)    # 4:00 PM ET = 3:00 PM CT

df_rth = df[(df['time'] >= rth_start) & (df['time'] < rth_end)].copy()
```

### What Gets Removed
- **Pre-market** (before 9:30 AM ET)
- **After-hours** (after 4:00 PM ET)
- **Overnight** (4:00 PM - 9:30 AM next day)
- **Typically 60-70%** of rows removed

### Output
- **DataFrame** with ONLY RTH bars
- **Perfect 1-second intervals** within RTH
- **23,400 bars per day** (6.5 hours × 3600 seconds)

### Data Characteristics
- **Rows**: Reduced by ~60-70% (RTH filtering)
- **Intervals**: EXACTLY 1 second (100%)
- **Contracts**: Single contract per day
- **Time coverage**: 9:30 AM - 4:00 PM ET only

### Example
```
timestamp                    symbol  open     close   volume
2025-11-09 14:30:00+00:00   ESZ5    6788.00  6788.00  730   ← 9:30 AM ET
2025-11-09 14:30:01+00:00   ESZ5    6787.25  6787.25  219
2025-11-09 14:30:02+00:00   ESZ5    6789.50  6789.50  133
2025-11-09 14:30:03+00:00   ESZ5    6789.50  6789.50  0
...
2025-11-09 21:00:00+00:00   ESZ5    6792.25  6792.25  184   ← 4:00 PM ET
```

**Row Count Example (July 2010)**: ~114,720 rows (23,400 per day × ~5 days)

---

## STEP 5: Weighted Labeling

### Purpose
Generate binary labels (0/1) and quality weights for 6 volatility-based trading modes.

### Input
- DataFrame from Step 4 with RTH bars only

### Process
```python
from src.data_pipeline.weighted_labeling import process_weighted_labeling

df_labeled = process_weighted_labeling(df_rth)
```

### Algorithm
For each bar, for each of 6 modes:
1. **Simulate trade** with mode-specific stop/target
2. **Look forward 15 minutes** (900 seconds = 900 bars)
3. **Determine outcome**:
   - Label = 1 if target hit before stop
   - Label = 0 if stop hit before target or timeout
4. **Calculate weights** (for winners only):
   - Quality weight: Based on MAE (lower MAE = higher weight)
   - Velocity weight: Based on speed to target (faster = higher weight)
   - Time decay weight: Based on data recency (recent = higher weight)
5. **Final weight**: quality × velocity × time_decay

### 6 Trading Modes
- **Low Vol Long**: 6-tick stop, 12-tick target
- **Normal Vol Long**: 8-tick stop, 16-tick target
- **High Vol Long**: 10-tick stop, 20-tick target
- **Low Vol Short**: 6-tick stop, 12-tick target
- **Normal Vol Short**: 8-tick stop, 16-tick target
- **High Vol Short**: 10-tick stop, 20-tick target

### Output Columns Added (12 total)
- `label_low_vol_long`, `weight_low_vol_long`
- `label_normal_vol_long`, `weight_normal_vol_long`
- `label_high_vol_long`, `weight_high_vol_long`
- `label_low_vol_short`, `weight_low_vol_short`
- `label_normal_vol_short`, `weight_normal_vol_short`
- `label_high_vol_short`, `weight_high_vol_short`

### Data Characteristics
- **Rows**: Same as input (no rows added/removed)
- **Columns**: +12 (6 labels + 6 weights)
- **Labels**: Binary (0 or 1)
- **Weights**: Positive floats (typically 0.5 to 4.0)

### Example
```
timestamp    close  volume  label_low_vol_long  weight_low_vol_long  label_normal_vol_long  ...
14:30:00     6788   730     1                   2.45                 1                      ...
14:30:01     6787   219     0                   0.85                 0                      ...
14:30:02     6790   133     1                   1.92                 1                      ...
```

**Row Count**: Same as input (~114,720 rows)
**Column Count**: Original + 12 = ~18 columns

---

## STEP 6: Feature Engineering

### Purpose
Calculate 43 technical features for ML model input.

### Input
- DataFrame from Step 5 with labels and weights

### Process
```python
from src.data_pipeline.features import create_all_features

df_final = create_all_features(df_labeled)
```

### Feature Categories (43 total)
1. **Volume Features (4)**: ratios, slopes, exhaustion
2. **Price Context (5)**: VWAP, distances, slopes
3. **Consolidation (10)**: range identification, retouches
4. **Returns (5)**: momentum at multiple timeframes
5. **Volatility (6)**: ATR, regime detection, breakouts
6. **Microstructure (6)**: bar characteristics, tick flow
7. **Time Features (7)**: session period indicators

### Rolling Window Calculations
- **30-second windows**: ~30 bars (with 1-second data)
- **60-second windows**: ~60 bars
- **300-second windows**: ~300 bars

### Output Columns Added (43 total)
All 43 feature columns added to DataFrame

### Data Characteristics
- **Rows**: Same as input (no rows added/removed)
- **Columns**: +43 features
- **NaN handling**: First N rows have NaN for rolling calculations
- **Typical NaN**: ~300 rows at start of each session

### Example
```
timestamp    close  volume  label_...  weight_...  volume_ratio_30s  vwap     atr_30s  ...
14:30:00     6788   730     1          2.45        1.85              6787.5   2.3      ...
14:30:01     6787   219     0          0.85        0.92              6787.3   2.2      ...
14:30:02     6790   133     1          1.92        1.12              6787.8   2.4      ...
```

**Row Count**: Same as input (~114,720 rows)
**Column Count**: Original + 12 + 43 = ~61 columns

---

## STEP 7: Final Dataset

### Output
- **Complete ML-ready dataset** with:
  - Original OHLCV data (6 columns)
  - Weighted labels (12 columns: 6 labels + 6 weights)
  - Engineered features (43 columns)
  - **Total: 61 columns**

### Data Characteristics
- **Perfect 1-second intervals** (RTH only)
- **23,400 bars per trading day**
- **~515,000 bars per month** (22 trading days)
- **~93 million bars for 15 years** (3,960 trading days)

### File Format
- **Parquet** for efficient storage and fast loading
- **Compression**: ~500 MB per month
- **15 years**: ~9 GB total

---

## Complete Flow Summary

### Row Count Progression (July 2010 Example)

| Step | Description | Rows | Change | Notes |
|------|-------------|------|--------|-------|
| 1 | DBN Conversion | 216,559 | - | All contracts, all hours, with gaps |
| 2 | Contract Filter | 150,593 | -30.5% | Single contract, all hours, with gaps |
| 3 | Gap Filling | 216,000 | +43.5% | Single contract, all hours, NO gaps |
| 4 | RTH Filter | 114,720 | -46.9% | Single contract, RTH only, NO gaps |
| 5 | Weighted Labeling | 114,720 | 0% | +12 columns (labels + weights) |
| 6 | Feature Engineering | 114,720 | 0% | +43 columns (features) |
| **FINAL** | **ML-Ready** | **114,720** | - | **61 total columns** |

### Time Coverage Progression

| Step | Time Coverage | Intervals |
|------|---------------|-----------|
| 1 | 24 hours (ETH + RTH) | Variable (gaps) |
| 2 | 24 hours (ETH + RTH) | Variable (gaps) |
| 3 | 24 hours (ETH + RTH) | Perfect 1-second |
| 4 | 6.5 hours (RTH only) | Perfect 1-second |
| 5 | 6.5 hours (RTH only) | Perfect 1-second |
| 6 | 6.5 hours (RTH only) | Perfect 1-second |

### Column Progression

| Step | Columns | Added |
|------|---------|-------|
| 1 | 10 | timestamp, OHLCV, symbol, etc. |
| 2 | 10 | (same) |
| 3 | 10 | (same) |
| 4 | 10 | (same) |
| 5 | 22 | +12 (6 labels + 6 weights) |
| 6 | 65 | +43 (features) |

---

## Critical Ordering Rules

### ✅ CORRECT Order
1. **Contract Filter FIRST** (before gap filling)
   - Reason: Don't fill gaps for contracts we'll discard
   
2. **Gap Fill SECOND** (before RTH filter)
   - Reason: Need complete data to identify session boundaries
   
3. **RTH Filter THIRD** (after gap filling)
   - Reason: Clean session boundaries with complete data
   
4. **Weighted Labeling FOURTH** (after RTH filter)
   - Reason: Only label RTH bars, lookforward needs complete data
   
5. **Feature Engineering LAST** (after labeling)
   - Reason: Rolling windows need complete 1-second data

### ❌ WRONG Order Examples

**DON'T: RTH filter before gap filling**
```
Contract Filter → RTH Filter → Gap Fill → Labeling → Features
```
Problem: Session boundaries unclear, gaps at session edges

**DON'T: Gap fill before contract filtering**
```
Gap Fill → Contract Filter → RTH Filter → Labeling → Features
```
Problem: Wasted computation filling gaps for contracts we'll discard

**DON'T: Labeling before RTH filter**
```
Contract Filter → Gap Fill → Labeling → RTH Filter → Features
```
Problem: Labeling ETH bars we'll discard, lookforward crosses sessions

---

## Performance Considerations

### Memory Usage
- **Peak**: ~2-3 GB per month during processing
- **Final**: ~500 MB per month (Parquet compressed)

### Processing Time (per month)
- DBN Conversion: ~30 seconds
- Contract Filtering: ~10 seconds
- Gap Filling: ~5 seconds
- RTH Filtering: ~2 seconds
- Weighted Labeling: ~5 minutes
- Feature Engineering: ~1 minute
- **Total**: ~7 minutes per month

### Parallelization
- Process months independently
- 15 years = 180 months
- With 10 parallel workers: ~2 hours total

---

## Validation Checkpoints

### After Each Step

**Step 1 - DBN Conversion:**
- ✓ Timestamp column exists
- ✓ OHLCV columns present
- ✓ Multiple contracts present

**Step 2 - Contract Filtering:**
- ✓ Single contract per day
- ✓ No large price gaps (>5 points)
- ✓ Reasonable row reduction (20-40%)

**Step 3 - Gap Filling:**
- ✓ 100% 1-second intervals
- ✓ Volume = 0 for filled bars
- ✓ No NaN in OHLCV
- ✓ Row increase (30-50%)

**Step 4 - RTH Filtering:**
- ✓ All timestamps within 9:30 AM - 4:00 PM ET
- ✓ ~23,400 bars per day
- ✓ Row reduction (60-70%)

**Step 5 - Weighted Labeling:**
- ✓ 12 new columns added
- ✓ Labels are binary (0 or 1)
- ✓ Weights are positive
- ✓ No rows added/removed

**Step 6 - Feature Engineering:**
- ✓ 43 new columns added
- ✓ NaN only in first ~300 rows per session
- ✓ No rows added/removed
- ✓ Total 61 columns

---

## Next Steps

1. **Update processing pipeline** with correct order
2. **Test on single month** to validate flow
3. **Process all 180 months** with parallelization
4. **Train 6 XGBoost models** on final dataset
5. **Deploy ensemble** with volatility regime detection

