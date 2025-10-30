# Weighted Labeling Logic - Beginner's Guide

## What Are We Labeling?
For each 1-second bar, we generate **12 columns** (6 labels + 6 weights) for training 6 specialized XGBoost models based on volatility regimes.

## The New Weighted System

### 6 Volatility-Based Trading Modes
Instead of one-size-fits-all, we now have 6 specialized modes:

**Long Modes:**
- **Low Vol Long:** 6-tick stop, 12-tick target (2:1 R/R)
- **Normal Vol Long:** 8-tick stop, 16-tick target (2:1 R/R)  
- **High Vol Long:** 10-tick stop, 20-tick target (2:1 R/R)

**Short Modes:**
- **Low Vol Short:** 6-tick stop, 12-tick target (2:1 R/R)
- **Normal Vol Short:** 8-tick stop, 16-tick target (2:1 R/R)
- **High Vol Short:** 10-tick stop, 20-tick target (2:1 R/R)

### Binary Labels (0 or 1)
Each mode gets a binary label:
- **1 (Win):** Price hits target before stop within 15 minutes
- **0 (Loss):** Price hits stop before target, or timeout

**No more optimal/suboptimal distinction** - the weighting system handles quality.

## The Weighting System

### Three Weight Components

#### 1. Quality Weights (MAE-Based)
**Formula:** `quality_weight = 2.0 - (1.5 × mae_ratio)`
**Range:** [0.5, 2.0]

```
MAE = 0 ticks (perfect entry) → Weight = 2.0
MAE = 3 ticks (50% of stop) → Weight = 1.25  
MAE = 6 ticks (at stop level) → Weight = 0.5
```

**Lower MAE = Higher quality weight**

#### 2. Velocity Weights (Speed-Based)
**Formula:** `velocity_weight = 2.0 - (1.5 × (seconds_to_target - 300) / 600)`
**Range:** [0.5, 2.0]
**Optimal Time:** 300 seconds (5 minutes)

```
60 seconds (very fast) → Weight = 1.4
300 seconds (optimal) → Weight = 2.0
600 seconds (slow) → Weight = 1.25
900 seconds (timeout) → Weight = 0.5
```

**Faster to target = Higher velocity weight**

#### 3. Time Decay Weights (Recency-Based)
**Formula:** `time_decay = exp(-0.05 × months_ago)`

```
Current month → Weight = 1.0
6 months ago → Weight = 0.74
12 months ago → Weight = 0.55
24 months ago → Weight = 0.30
```

**More recent data = Higher time decay weight**

### Final Weight Calculation

**For Winners:** `final_weight = quality_weight × velocity_weight × time_decay`
**For Losers:** `final_weight = time_decay` (quality=1.0, velocity=1.0)

## Example Calculation

**Scenario:** Long Normal Vol trade from 6 months ago
- Entry: 4750.00, Target: 4754.00 (+16 ticks), Stop: 4748.00 (-8 ticks)
- Result: Hit target in 240 seconds with MAE of 2 ticks

**Weight Components:**
1. **Quality:** MAE ratio = 2/8 = 0.25 → `2.0 - (1.5 × 0.25) = 1.625`
2. **Velocity:** `2.0 - (1.5 × (240-300)/600) = 2.15` → clipped to 2.0
3. **Time Decay:** `exp(-0.05 × 6) = 0.74`

**Final Weight:** `1.625 × 2.0 × 0.74 = 2.405`

## Output Format (12 Columns)

```
label_low_vol_long: 0 or 1
weight_low_vol_long: 0.5 to 4.0 (typically)
label_normal_vol_long: 0 or 1  
weight_normal_vol_long: 0.5 to 4.0
label_high_vol_long: 0 or 1
weight_high_vol_long: 0.5 to 4.0
label_low_vol_short: 0 or 1
weight_low_vol_short: 0.5 to 4.0
label_normal_vol_short: 0 or 1
weight_normal_vol_short: 0.5 to 4.0
label_high_vol_short: 0 or 1
weight_high_vol_short: 0.5 to 4.0
```

## Why This Approach?

### Advantages Over Old System
- **Volatility-Adaptive:** Different stop/target sizes for different market conditions
- **Quality-Weighted:** Better entries get higher weights automatically
- **Speed-Rewarded:** Fast winners get higher weights than slow winners
- **Recency-Biased:** Recent data matters more than old data
- **XGBoost-Ready:** Direct binary classification with sample weights

### Model Training Strategy
Train 6 separate XGBoost models:
- Each model specializes in one volatility regime
- Use corresponding label/weight columns for each model
- Deploy ensemble that selects appropriate model based on current volatility

## Expected Distributions
From 1000 bars:
- **Win rates:** 13-53% (varies by mode and market direction)
- **Average weights:** 1.2-1.9 (winners get higher weights)
- **Weight range:** 0.5-4.0 (theoretical max ~8.0 for perfect recent entries)

## Code Location
`src/data_pipeline/weighted_labeling.py`

## Key Classes/Functions
- `WeightedLabelingEngine` - Main processing engine
- `WeightCalculator` - Handles weight calculations
- `process_weighted_labeling()` - Main entry point
- `TRADING_MODES` - Configuration for all 6 modes

## Usage
```bash
# Main production entry point
python main.py --input raw_data.parquet --output processed_data.parquet

# With validation
python main.py --input raw_data.parquet --output processed_data.parquet --validate
```