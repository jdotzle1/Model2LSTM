# Feature Engineering - Beginner's Guide

## What Are Features?
Features are calculated values that give the model context about market conditions.

**Raw data only tells part of the story:**
```
timestamp: 2025-01-15 10:30:05
close: 4750.75
volume: 1250
```

**Features add context:**
```
Is 4750.75 high or low? → distance_from_vwap = -2.5 (below average)
Is volume high? → volume_ratio_30s = 1.8 (80% above recent avg)
Is price moving fast? → return_5s = 0.0015 (0.15% up in 5 seconds)
```

## Feature Categories (55 Total)

### 1. Volume Features (5 features)
**Purpose:** Understand volume context

- `volume_ratio_30s` - Volume vs last 30 seconds average
  - 1.0 = normal, 2.0 = twice normal, 0.5 = half normal
  
- `volume_ratio_300s` - Volume vs last 5 minutes average
  - Longer-term volume context
  
- `volume_pct_rank` - Where volume ranks today (0-1)
  - 0.9 = 90th percentile for the day
  
- `volume_change_5s` - Volume momentum
  - Positive = increasing, negative = decreasing
  
- `volume_zscore_300s` - Statistical outlier detection
  - >2 = unusually high volume, <-2 = unusually low

**Why transformed volume, not raw?**
Raw volume changes over time (2010 vs 2025). Ratios stay consistent.

### 2. Price Context Features (8 features)
**Purpose:** Where is price relative to key levels?

- `vwap` - Volume-weighted average price (fair value for the day)
- `distance_from_vwap` - Points away from VWAP
  - Positive = above VWAP, negative = below
  
- `distance_from_vwap_pct` - Percentage away from VWAP
  
- `rth_high` / `rth_low` - Session high/low so far today
  
- `distance_from_rth_high` - How far from session high
  - Near 0 = at highs (potential resistance)
  
- `distance_from_rth_low` - How far from session low
  - Near 0 = at lows (potential support)
  
- `position_in_rth_range` - Position in today's range (0-1)
  - 0 = at low, 0.5 = middle, 1 = at high

### 3. Swing High/Low Features (10 features)
**Purpose:** Identify recent peaks and troughs for reversals

**What's a swing high?**
A bar where price is highest compared to bars around it.

**Example (5-second lookback):**
```
Bar 95: high = 4750.00
Bar 96: high = 4750.50
Bar 97: high = 4751.00 ← Swing high!
Bar 98: high = 4750.75
Bar 99: high = 4750.25
```

Bar 97 is a swing high because its high (4751.00) is the highest in the 5-bar window.

- `last_swing_high_5s` - Most recent 5-second swing high price
- `distance_from_swing_high_5s` - How far from that swing high
- `last_swing_high_60s` - Most recent 60-second swing high
- `distance_from_swing_high_60s` - How far from that swing high
- (Same for swing lows)
- `position_in_swing_range` - Where in swing range (0-1)
- `bars_since_swing_high` - Time since last swing high

**Why important?**
Reversals often happen near swing highs/lows.

### 4. Return Features (6 features)
**Purpose:** Measure momentum at different timeframes

- `return_1s` - 1-second price change (%)
- `return_5s` - 5-second price change (%)
- `return_10s` - 10-second price change (%)
- `return_30s` - 30-second price change (%)
- `return_60s` - 1-minute price change (%)
- `return_300s` - 5-minute price change (%)

**Example:**
```
5 seconds ago: 4750.00
Now: 4751.00
return_5s = (4751 - 4750) / 4750 = 0.00021 = 0.021%
```

### 5. Volatility Features (8 features)
**Purpose:** How much is price moving?

- `atr_30s` - Average True Range over 30 seconds
  - Higher = more volatile
  
- `atr_60s` / `atr_300s` - ATR at longer timeframes
  
- `realized_vol_30s` - Standard deviation of returns (30s)
  
- `realized_vol_60s` / `realized_vol_300s` - Longer-term vol
  
- `vol_ratio_30_300` - Short-term vol / long-term vol
  - >1 = volatility increasing, <1 = decreasing
  
- `atr_pct` - ATR as % of price (normalized)

**Why important?**
Stop distances should adjust to volatility. High vol = wider stops needed.

### 6. Microstructure Features (8 features)
**Purpose:** Bar-level price action

- `bar_range` - High - Low
- `bar_range_pct` - Range as % of price
- `bar_body` - |Close - Open| (size of colored part of candle)
- `body_ratio` - Body / Range (conviction indicator)
  - 0.9 = strong directional move
  - 0.1 = indecisive/choppy
  
- `tick_direction` - +1 (up), 0 (flat), -1 (down)
- `consecutive_up` - How many up ticks in a row
- `consecutive_down` - How many down ticks in a row
- `net_ticks_60s` - Net directional ticks over 60 seconds
  - +40 = strong buying, -40 = strong selling

### 7. Time Features (10 features)
**Purpose:** Market behavior changes throughout the day

- `hour` / `minute` - Time of day
- `day_of_week` - Monday = 0, Friday = 4
- `seconds_since_open` - Seconds since 9:30 AM
  - 0-1800 = first 30 minutes (opening volatility)
  
- `seconds_until_close` - Seconds until 4:00 PM
  - <1800 = last 30 minutes (closing volatility)
  
- `is_opening` - First 30 minutes flag (1 or 0)
- `is_lunch` - Lunch period flag (11:30-1:30)
- `is_close` - Last hour flag
- `is_monday` / `is_friday` - Day of week flags

**Why important?**
Market is different at open vs lunch vs close. Model needs to know.

## Feature Checklist
Before training, ensure:
- [ ] No NaN values (except first ~300 bars from rolling calcs)
- [ ] Features are in reasonable ranges
- [ ] Volume features use ratios (not raw counts)
- [ ] Time features are correct for timezone (UTC for ES)
- [ ] Swing features forward-fill properly

## Code Location
`project/data_pipeline/features.py`

## Testing
Run on small sample first (1000 bars) to verify calculations before scaling to full dataset.