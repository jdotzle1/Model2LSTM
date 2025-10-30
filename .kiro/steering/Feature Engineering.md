# Feature Engineering - Beginner's Guide

## What Are Features?
Features are calculated values that give the model context about market conditions for the 6 XGBoost models.

**Raw data only tells part of the story:**
```
timestamp: 2025-01-15 10:30:05
close: 4750.75
volume: 1250
```

**Features add context:**
```
Is 4750.75 high or low? → distance_from_vwap_pct = -0.05% (below average)
Is volume high? → volume_ratio_30s = 1.8 (80% above recent avg)
Is price moving fast? → return_30s = 0.0015 (0.15% up in 30 seconds)
Which volatility regime? → volatility_regime = 1.2 (high volatility)
```

## Feature Categories (43 Total)

### 1. Volume Features (4 features)
**Purpose:** Understand volume context and patterns

- `volume_ratio_30s` - Volume vs last 30 seconds average
  - 1.0 = normal, 2.0 = twice normal, 0.5 = half normal
  
- `volume_slope_30s` - Volume trend over 30 seconds
  - Positive = increasing volume, negative = decreasing
  
- `volume_slope_5s` - Short-term volume momentum
  - Captures immediate volume changes
  
- `volume_exhaustion` - Combined volume ratio and slope
  - High values indicate volume exhaustion patterns

**Why transformed volume, not raw?**
Raw volume changes over time (2010 vs 2025). Ratios and slopes stay consistent.

### 2. Price Context Features (5 features)
**Purpose:** Where is price relative to key levels?

- `vwap` - Volume-weighted average price (fair value)
  - Uses 5-minute rolling window for efficiency
  
- `distance_from_vwap_pct` - Percentage away from VWAP
  - Positive = above VWAP, negative = below
  
- `vwap_slope` - VWAP trend direction
  - Positive = uptrending, negative = downtrending
  
- `distance_from_rth_high` - How far from session high
  - Near 0 = at highs (potential resistance)
  
- `distance_from_rth_low` - How far from session low
  - Near 0 = at lows (potential support)

### 3. Consolidation Features (10 features)
**Purpose:** Identify range-bound periods and breakout potential

**What's a consolidation range?**
A period where price trades within defined high/low boundaries.

**Short-term ranges (2 minutes):**
- `short_range_high` - Highest price in 120-bar window
- `short_range_low` - Lowest price in 120-bar window  
- `short_range_size` - Range size in points
- `position_in_short_range` - Where price sits in range (0-1)
- `short_range_retouches` - Binary: touching range boundaries

**Medium-term ranges (5 minutes):**
- `medium_range_high` - Highest price in 300-bar window
- `medium_range_low` - Lowest price in 300-bar window
- `medium_range_size` - Range size in points
- `range_compression_ratio` - Short range / Medium range
- `medium_range_retouches` - Binary: touching range boundaries

**Why important?**
Breakouts from consolidation ranges often lead to strong directional moves. Range compression indicates building pressure.

### 4. Return Features (5 features)
**Purpose:** Measure momentum at different timeframes

- `return_30s` - 30-second price change (%)
- `return_60s` - 1-minute price change (%)
- `return_300s` - 5-minute price change (%)
- `momentum_acceleration` - Change in momentum (return_60s - return_30s)
- `momentum_consistency` - Standard deviation of recent returns

**Example:**
```
30 seconds ago: 4750.00
Now: 4751.00
return_30s = (4751 - 4750) / 4750 = 0.00021 = 0.021%
```

**Why these timeframes?**
Aligned with volatility mode timeframes and trading decision horizons.

### 5. Volatility Features (6 features)
**Purpose:** Measure market volatility for regime detection

- `atr_30s` - Average True Range over 30 seconds
  - Higher = more volatile, used for volatility regime classification
  
- `atr_300s` - Average True Range over 5 minutes
  - Longer-term volatility context
  
- `volatility_regime` - Short-term ATR / Long-term ATR
  - >1.2 = high volatility, <0.8 = low volatility, else normal
  
- `volatility_acceleration` - Change in volatility over time
  - Positive = volatility increasing, negative = decreasing
  
- `volatility_breakout` - Z-score of current volatility
  - >2 = volatility breakout, <-2 = volatility compression
  
- `atr_percentile` - Where current ATR ranks in recent history
  - 0.9 = 90th percentile (very high volatility)

**Why critical for this system?**
Volatility regime determines which of the 6 XGBoost models to use. Proper volatility classification is essential for model selection.

### 6. Microstructure Features (6 features)
**Purpose:** Bar-level price action and tick flow

- `bar_range` - High - Low (absolute range)
- `relative_bar_size` - Current bar range / recent average range
  - >1.5 = large bar, <0.5 = small bar
  
- `uptick_pct_30s` - Percentage of upticks in last 30 seconds
  - >0.6 = buying pressure, <0.4 = selling pressure
  
- `uptick_pct_60s` - Percentage of upticks in last 60 seconds
  - Longer-term tick flow direction
  
- `bar_flow_consistency` - Consistency of tick direction within bars
  - High = consistent flow, low = choppy action
  
- `directional_strength` - Net directional pressure
  - Combines tick flow with bar characteristics

**Why important?**
Microstructure features help identify the quality of price moves and potential continuation vs reversal patterns.

### 7. Time Features (7 features)
**Purpose:** Market behavior changes throughout the day

**Session Period Indicators (Binary 0/1):**
- `is_eth` - Extended Trading Hours (outside RTH)
- `is_pre_open` - Pre-market period (7:30-9:30 CT)
- `is_rth_open` - Regular Trading Hours opening (9:30-10:30 CT)
- `is_morning` - Morning session (10:30-12:00 CT)
- `is_lunch` - Lunch period (12:00-13:30 CT)
- `is_afternoon` - Afternoon session (13:30-15:00 CT)
- `is_rth_close` - Regular Trading Hours closing (15:00-16:00 CT)

**Why binary encoding?**
XGBoost handles binary features efficiently. Each session has distinct volatility and directional characteristics that affect the success of different trading modes.

**Session Characteristics:**
- **Opening:** High volatility, good for high-vol modes
- **Morning:** Trending moves, good for directional trades
- **Lunch:** Low volatility, good for low-vol modes
- **Afternoon:** Mixed conditions, normal-vol modes
- **Close:** High volatility, reversal patterns

## Feature Checklist
Before XGBoost training, ensure:
- [ ] All 43 features calculated correctly
- [ ] NaN values within acceptable limits (≤35% for rolling calculations)
- [ ] Volume features use ratios/slopes (not raw counts)
- [ ] Volatility regime feature working for model selection
- [ ] Time features properly identify session periods
- [ ] Features validated against realistic market data ranges

## Integration with Weighted Labeling
- **43 features** work with **12 weighted labeling columns** (6 labels + 6 weights)
- **Total output:** 61 columns (6 original + 12 labeling + 43 features)
- **XGBoost ready:** Binary labels with sample weights for 6 specialized models

## Code Location
`src/data_pipeline/features.py`

## Testing
- **Integration test:** `tests/integration/test_final_integration_1000_bars.py`
- **Feature tests:** `tests/unit/test_features_comprehensive.py`
- **Performance validation:** `tests/validation/validate_performance.py`

## Model Usage
Each of the 6 XGBoost models uses:
- **Same 43 features** as input
- **Corresponding label column** as target (e.g., `label_low_vol_long`)
- **Corresponding weight column** for sample weighting (e.g., `weight_low_vol_long`)
- **Volatility regime feature** determines which model to use for prediction