# Feature Engineering Definitions - Correct Timing

## Core Principle: Predict Bar N Using Only Historical Data
**At the close of bar N-1, predict if bar N will be optimal**
**Features for bar N can ONLY use data from bars 0 to N-1 (NOT including N)**
**We CANNOT use any data from bar N or future bars**

## Timing Example:
```
Bar 98: [complete] - can use this data
Bar 99: [complete] - can use this data  
Bar 100: [just starting] - PREDICT if this will be optimal
Bar 101+: [future] - cannot use this data
```

**Features for predicting bar 100 use only bars 0-99**

## Feature Categories (55 Total Features)

### 1. Volume Features (5 features)
**Purpose:** Recent volume context to predict bar N

- `volume_ratio_30s` = volume[N-1] / rolling_mean(volume[N-30:N-1])
  - Last bar's volume vs 30-bar historical average
  - Uses bars [N-30 to N-1] to calculate mean

- `volume_ratio_300s` = volume[N-1] / rolling_mean(volume[N-300:N-1])
  - Last bar's volume vs 300-bar historical average
  - Uses bars [N-300 to N-1] to calculate mean

- `volume_pct_rank` = percentile rank of volume[N-1] within session
  - Rank last bar's volume against [session_start to N-1]
  - Uses only completed bars from same trading day

- `volume_change_5s` = (volume[N-1] - volume[N-6]) / volume[N-6]
  - 5-bar volume momentum ending at N-1
  - Uses only historical bars

- `volume_zscore_300s` = (volume[N-1] - rolling_mean[N-300:N-1]) / rolling_std[N-300:N-1]
  - Z-score of last bar's volume vs 300-bar history

### 2. Price Context Features (8 features)
**Purpose:** Where was recent price relative to key levels

- `vwap` = cumulative VWAP from session start to bar N-1
  - VWAP = sum(price[session_start:N-1] * volume[session_start:N-1]) / sum(volume[session_start:N-1])
  - Uses only completed bars up to N-1

- `distance_from_vwap` = close[N-1] - vwap[N-1]
  - Last bar's close vs VWAP calculated through N-1

- `distance_from_vwap_pct` = (close[N-1] - vwap[N-1]) / vwap[N-1] * 100
  - Percentage distance from VWAP

- `rth_high` = max(high[session_start:N-1])
  - Session high through completed bars only

- `rth_low` = min(low[session_start:N-1])
  - Session low through completed bars only

- `distance_from_rth_high` = close[N-1] - rth_high
  - How far last close was below session high

- `distance_from_rth_low` = close[N-1] - rth_low
  - How far last close was above session low

- `position_in_rth_range` = (close[N-1] - rth_low) / (rth_high - rth_low)
  - Where last close was in session range (0 = at low, 1 = at high)

### 3. Swing High/Low Features (10 features)
**Purpose:** Historical swing points for reversal context

**✅ Swing Detection Logic (Historical Only):**
To predict bar N, we identify swing points that occurred in the past and are now confirmed.

**5-Second Swing High Detection:**
```python
# At bar N, look back 5 bars to check if bar N-5 was a swing high
check_idx = N - 5
if check_idx >= 5:  # Need enough history
    # Check if bar N-5 was highest in 10-bar window [N-10 to N-1]
    window_start = max(0, check_idx - 5)
    window_end = N  # Up to but not including current bar N
    if high[check_idx] == max(high[window_start:window_end]):
        swing_high_5s[N] = high[check_idx]
```

**Features:**
- `last_swing_high_5s` = most recent confirmed 5s swing high (forward fill)
- `distance_from_swing_high_5s` = close[N-1] - last_swing_high_5s
- `bars_since_swing_high_5s` = bars elapsed since last swing high
- `position_in_swing_range_5s` = (close[N-1] - last_swing_low_5s) / (last_swing_high_5s - last_swing_low_5s)

**Same logic for:**
- `last_swing_low_5s` and related features
- 60-second swing highs/lows (using 60-bar lookback)

**Key Point:** All swing points are confirmed using only historical data, with a natural delay for confirmation.

### 4. Return Features (6 features)
**Purpose:** Recent momentum at multiple timeframes

- `return_1s` = (close[N-1] - close[N-2]) / close[N-2]
- `return_5s` = (close[N-1] - close[N-6]) / close[N-6]
- `return_10s` = (close[N-1] - close[N-11]) / close[N-11]
- `return_30s` = (close[N-1] - close[N-31]) / close[N-31]
- `return_60s` = (close[N-1] - close[N-61]) / close[N-61]
- `return_300s` = (close[N-1] - close[N-301]) / close[N-301]

**All use only completed historical bars to predict bar N**

### 5. Volatility Features (8 features)
**Purpose:** Recent volatility context

- `atr_30s` = rolling_mean(true_range, 30 bars) using bars [N-30:N-1]
- `atr_60s` = rolling_mean(true_range, 60 bars) using bars [N-60:N-1]
- `atr_300s` = rolling_mean(true_range, 300 bars) using bars [N-300:N-1]

- `realized_vol_30s` = rolling_std(return_1s, 30 bars) using bars [N-30:N-1]
- `realized_vol_60s` = rolling_std(return_1s, 60 bars) using bars [N-60:N-1]
- `realized_vol_300s` = rolling_std(return_1s, 300 bars) using bars [N-300:N-1]

- `vol_ratio_30_300` = realized_vol_30s / realized_vol_300s
- `atr_pct` = atr_30s / close[N-1] * 100

### 6. Microstructure Features (8 features)
**Purpose:** Recent bar characteristics and price action

- `bar_range` = high[N-1] - low[N-1]
- `bar_range_pct` = (high[N-1] - low[N-1]) / close[N-1] * 100
- `bar_body` = abs(close[N-1] - open[N-1])
- `body_ratio` = bar_body / bar_range (last bar's conviction)

- `tick_direction` = sign(close[N-1] - close[N-2])
- `consecutive_up` = count consecutive +1 tick_directions ending at N-1
- `consecutive_down` = count consecutive -1 tick_directions ending at N-1
- `net_ticks_60s` = sum(tick_direction[N-60:N-1])

### 7. Time Features (10 features)
**Purpose:** Time context for when we're making the prediction

- `hour` = timestamp[N].hour (when bar N starts)
- `minute` = timestamp[N].minute
- `day_of_week` = timestamp[N].dayofweek
- `seconds_since_open` = (timestamp[N] - session_open_time).total_seconds()
- `seconds_until_close` = (session_close_time - timestamp[N]).total_seconds()
- `is_opening` = 1 if bar N is within first 30 minutes of session
- `is_lunch` = 1 if bar N is between 11:30-13:30 CT
- `is_close` = 1 if bar N is within last hour of session
- `is_monday` = 1 if day_of_week == 0
- `is_friday` = 1 if day_of_week == 4

**Note:** These use the timestamp of bar N (the bar we're predicting) since we know when we're making the prediction.

## Data Leakage Checklist

✅ **Volume features:** Only use historical rolling windows
✅ **Price context:** Only use session data up to current bar
❌ **Swing features:** NEED TO FIX - currently looks forward
✅ **Returns:** Only use historical price differences
✅ **Volatility:** Only use historical rolling calculations
✅ **Microstructure:** Only use current bar data
✅ **Time features:** Only use current timestamp

## Critical Fix Needed: Swing Point Detection

The swing point logic needs to be completely rewritten to avoid future data leakage.