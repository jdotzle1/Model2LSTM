# Labeling Logic - Beginner's Guide

## What Are We Labeling?
For each 1-second bar, we want to know: "If I entered a trade here, would it be optimal, suboptimal, or a loss?"

## The Process (Step-by-Step)

### Step 1: Check Target and Stop
Look forward up to 15 minutes (900 seconds) from entry bar.

**For a Long trade:**
- **Win:** Price hits target (+12, +16, or +20 ticks) before stop
- **Loss:** Price hits stop (-6, -8, or -10 ticks) before target
- **Timeout:** Neither hits within 15 minutes

**Example (Long 2:1 Small):**
```
Entry at bar 100: close = 4750.00
Target: 4753.00 (+12 ticks)
Stop: 4748.50 (-6 ticks)

Bar 101: high = 4750.50, low = 4749.00 (no hit)
Bar 102: high = 4751.00, low = 4749.50 (no hit)
Bar 103: high = 4753.25, low = 4750.00 (TARGET HIT! ✓ Win)
```

### Step 2: Calculate MAE for Winners
MAE = Maximum Adverse Excursion = Worst point before winning

**Example from above:**
```
Entry: 4750.00
Path to target:
  Bar 101 low: 4749.00 (adverse: -1.00 point = -4 ticks)
  Bar 102 low: 4749.50 (adverse: -0.50 point = -2 ticks)
  Bar 103: Hit target

MAE = Worst adverse = -4 ticks
```

**Lower MAE = Better entry timing!**

### Step 3: Find Consecutive Winners
Sometimes multiple bars in a row would all win. These form a "sequence."

**Example:**
```
Bar 100: Win (MAE = 5 ticks)
Bar 101: Win (MAE = 3 ticks) ← Best timing!
Bar 102: Win (MAE = 4 ticks)
Bar 103: Win (MAE = 6 ticks)
Bar 104: Loss
```

Bars 100-103 are a consecutive winning sequence.

### Step 4: Mark Only the BEST Entry in Each Sequence
Within each sequence, only the bar with **lowest MAE** gets labeled as Optimal (+1).

**From example above:**
```
Bar 100: Label = 0 (suboptimal, MAE too high)
Bar 101: Label = +1 (OPTIMAL, lowest MAE = 3 ticks) ✓
Bar 102: Label = 0 (suboptimal)
Bar 103: Label = 0 (suboptimal)
Bar 104: Label = -1 (loss)
```

## Why This Approach?
- **Optimal entries are rare** - Most bars are not good entries
- **Model learns selectivity** - Wait for the best moment
- **MAE filter removes luck** - Winning isn't enough, timing matters
- **Production-ready** - Model learns what we actually want to trade

## Label Distribution (Expected)
From 1000 bars:
- **Optimal (+1):** ~1-2% (7-20 bars)
- **Suboptimal (0):** ~40-45% (400-450 bars)
- **Loss (-1):** ~50-55% (500-550 bars)
- **Timeout (NaN):** Excluded from training

## Code Location
`project/data_pipeline/labeling.py`

## Key Functions
- `calculate_labels_for_all_profiles()` - Main entry point
- `check_target_stop()` - Determines win/loss/timeout
- `calculate_mae()` - Calculates drawdown for winners
- `apply_mae_filter()` - Marks optimal vs suboptimal