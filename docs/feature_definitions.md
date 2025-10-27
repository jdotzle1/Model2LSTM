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

## Feature Categories (43 Total Features)

### 1. Volume Features (4 features)
**Purpose:** Detect volume exhaustion patterns and breakout quality in ES futures

**Core Strategy:** Identify when high volume attempts fail (exhaustion) vs succeed (momentum)

#### Features:

- `volume_ratio_30s` = volume[N-1] / rolling_mean(volume[N-30:N-1])
  - **Purpose:** Current volume magnitude vs recent baseline
  - **Interpretation:** >2.0 = high volume, <1.5 = low volume
  - **Uses:** Bars [N-30 to N-1] for baseline calculation

- `volume_slope_30s` = slope(rolling_mean(volume, 5)[N-30:N-1])
  - **Purpose:** Medium-term volume trend (setup phase detection)
  - **Calculation:** Linear slope of 5-bar volume MA over 30-bar window
  - **Interpretation:** Positive = building volume, Negative = declining interest

- `volume_slope_5s` = slope(volume[N-5:N-1])
  - **Purpose:** Short-term volume trend (exhaustion detection)
  - **Calculation:** Linear slope of raw volume over last 5 bars
  - **Interpretation:** Negative after high volume = exhaustion signal

- `volume_exhaustion` = volume_ratio_30s × volume_slope_5s
  - **Purpose:** Combined exhaustion/momentum signal
  - **Interpretation:** Large negative = fade signal, Large positive = momentum signal

#### Trading Logic & Scenarios:

**🔴 Exhaustion Pattern (Fade Signal):**
```
Scenario: Price at resistance, attempting breakout
volume_ratio_30s = 2.8     # High volume (280% of average)
volume_slope_30s = +25     # Volume was building (good setup)
volume_slope_5s = -45      # Volume collapsing NOW
volume_exhaustion = -126   # Strong fade signal

→ INTERPRETATION: High volume breakout attempt is failing
→ TRADE SIGNAL: Fade the breakout, expect reversal
```

**🟢 Momentum Pattern (Follow Signal):**
```
Scenario: Price breaking key support level
volume_ratio_30s = 3.2     # Very high volume
volume_slope_30s = +15     # Volume building over time
volume_slope_5s = +30      # Volume accelerating
volume_exhaustion = +96    # Strong momentum signal

→ INTERPRETATION: Volume supporting the breakout
→ TRADE SIGNAL: Follow the breakout direction
```

**⚪ Weak Attempt (Wait Signal):**
```
Scenario: Price at key level but unconvincing
volume_ratio_30s = 1.3     # Low volume
volume_slope_30s = -5      # Volume declining
volume_slope_5s = +10      # Slight recent uptick
volume_exhaustion = +13    # Weak signal

→ INTERPRETATION: Insufficient volume for meaningful move
→ TRADE SIGNAL: Wait for better setup
```

#### Key Insights:
- **ES exhaustions happen in 1-10 seconds** - volume_slope_5s captures this
- **Setup matters** - volume_slope_30s identifies if volume was building properly
- **Combined signal is key** - volume_exhaustion integrates magnitude and trend
- **No future data** - All calculations use only historical bars through N-1

### 2. Price Context Features (5 features)
**Purpose:** Price position relative to key reference levels and VWAP behavior

**Core Strategy:** Identify price behavior relative to VWAP and session extremes

#### Features:

- `vwap` = cumulative VWAP from session start to bar N-1
  - **Calculation:** sum(price[session_start:N-1] × volume[session_start:N-1]) / sum(volume[session_start:N-1])
  - **Purpose:** Fair value reference for the session
  - **Uses:** Only completed bars up to N-1

- `distance_from_vwap_pct` = (close[N-1] - vwap[N-1]) / vwap[N-1] × 100
  - **Purpose:** Percentage distance from fair value (signed)
  - **Interpretation:** +0.5% = 0.5% above VWAP, -0.3% = 0.3% below VWAP
  - **Range:** Typically ±0.1% to ±1.0% for ES intraday

- `vwap_slope` = slope(vwap[N-30:N-1])
  - **Purpose:** VWAP trend direction and strength
  - **Calculation:** Linear slope of VWAP over last 30 bars
  - **Interpretation:** Positive = rising VWAP (bullish), Negative = falling VWAP (bearish)

- `distance_from_rth_high` = close[N-1] - max(high[session_start:N-1])
  - **Purpose:** Distance from session high (always ≤ 0)
  - **Interpretation:** -2.0 = 2 points below session high, 0 = at session high
  - **Use Case:** Resistance level proximity

- `distance_from_rth_low` = close[N-1] - min(low[session_start:N-1])
  - **Purpose:** Distance from session low (always ≥ 0)
  - **Interpretation:** +3.5 = 3.5 points above session low, 0 = at session low
  - **Use Case:** Support level proximity

#### Trading Logic & Scenarios:

**🔵 VWAP Mean Reversion Setup:**
```
Scenario: Price extended from VWAP, potential reversion
distance_from_vwap_pct = +0.8%    # Significantly above VWAP
vwap_slope = +2.5                 # VWAP rising (bullish trend)
distance_from_rth_high = -1.5     # 1.5 points below session high

→ INTERPRETATION: Price extended above rising VWAP, near resistance
→ TRADE SIGNAL: Watch for short-term reversion to VWAP
```

**🟡 VWAP Breakout Confirmation:**
```
Scenario: Price breaking above VWAP with momentum
distance_from_vwap_pct = +0.2%    # Just above VWAP
vwap_slope = +1.8                 # VWAP rising steadily
distance_from_rth_low = +4.2      # Well above session low

→ INTERPRETATION: Clean break above rising VWAP
→ TRADE SIGNAL: Momentum continuation likely
```

**🔴 Session Extreme Test:**
```
Scenario: Price testing session high
distance_from_vwap_pct = +0.6%    # Above VWAP
vwap_slope = -0.5                 # VWAP slightly declining
distance_from_rth_high = -0.25    # Very close to session high

→ INTERPRETATION: Testing resistance with declining VWAP
→ TRADE SIGNAL: High probability rejection setup
```

#### Key Insights:
- **VWAP slope matters** - Price behaves differently with rising vs falling VWAP
- **Percentage distance** - More consistent across different price levels than absolute
- **Session extremes** - RTH high/low provide key resistance/support levels
- **Complementary to recent highs/lows** - Session levels for major S/R, recent levels for rotations

### 3. Consolidation & Range Features (10 features)
**Purpose:** Identify consolidation patterns across multiple timeframes with proper retouch counting

**Core Strategy:** Compare short-term vs medium-term consolidation patterns for fade setups

#### Short-Term Consolidation Features (300s = 5 minutes):

- `short_range_high` = max(high[N-300:N-1])
  - **Purpose:** Short-term resistance level (5-minute lookback)

- `short_range_low` = min(low[N-300:N-1])
  - **Purpose:** Short-term support level (5-minute lookback)

- `short_range_size` = short_range_high - short_range_low
  - **Purpose:** Size of short-term consolidation
  - **Interpretation:** <2.0 = tight, 2-4 = normal, >4 = wide

- `position_in_short_range` = (close[N-1] - short_range_low) / short_range_size
  - **Purpose:** Position in short-term range (0 to 1)
  - **Fade zones:** <0.15 (bottom) or >0.85 (top)

#### Medium-Term Consolidation Features (900s = 15 minutes):

- `medium_range_high` = max(high[N-900:N-1])
  - **Purpose:** Medium-term resistance level (15-minute lookback)

- `medium_range_low` = min(low[N-900:N-1])
  - **Purpose:** Medium-term support level (15-minute lookback)

- `medium_range_size` = medium_range_high - medium_range_low
  - **Purpose:** Size of medium-term consolidation

- `range_compression_ratio` = short_range_size / medium_range_size
  - **Purpose:** Short-term vs medium-term range comparison
  - **Interpretation:** <0.5 = compressed (coiling), >0.8 = similar ranges

#### Retouch Counting (with 30-second cooldown):

**Retouch Logic:**
```python
def count_retouches_with_cooldown(highs, lows, threshold_high, threshold_low, cooldown_bars=30):
    """
    Count distinct retouch events with cooldown period
    
    A retouch event occurs when:
    1. Price enters the zone (high ≥ threshold_high OR low ≤ threshold_low)
    2. After entering, must exit zone for cooldown_bars before next retouch counts
    
    This prevents counting every second within the zone as a separate touch
    """
    retouches = 0
    last_retouch_bar = -cooldown_bars - 1  # Allow first retouch
    
    for i, (high, low) in enumerate(zip(highs, lows)):
        # Check if price is in retouch zone
        in_upper_zone = high >= threshold_high
        in_lower_zone = low <= threshold_low
        
        if (in_upper_zone or in_lower_zone) and (i - last_retouch_bar >= cooldown_bars):
            retouches += 1
            last_retouch_bar = i
    
    return retouches
```

**Applied Features:**
- `short_range_retouches` = retouch count in short-term range boundaries (10% zones, 30s cooldown)
- `medium_range_retouches` = retouch count in medium-term range boundaries (10% zones, 30s cooldown)

#### Trading Logic & Scenarios:

**🔴 Perfect Compressed Fade Setup:**
```
Scenario: Short-term consolidation within larger range
position_in_short_range = 0.92           # Near short-term resistance
short_range_size = 2.25                  # Tight recent consolidation
medium_range_size = 5.50                 # Wider 15-minute range
range_compression_ratio = 0.41           # Highly compressed (coiling)
short_range_retouches = 4                # 4 distinct tests (with cooldown)
medium_range_retouches = 8               # 8 tests over longer period

→ INTERPRETATION: Coiled spring - tight range within larger consolidation
→ TRADE SIGNAL: High confidence fade, expect sharp reversal
```

**🟢 Perfect Support Fade Setup:**
```
Scenario: Testing well-established support in both timeframes
position_in_short_range = 0.08           # Near short-term support
short_range_size = 3.75                  # Normal consolidation size
range_compression_ratio = 0.68           # Moderately compressed
short_range_retouches = 3                # 3 recent tests
medium_range_retouches = 7               # 7 tests over 15 minutes

→ INTERPRETATION: Well-tested support across timeframes
→ TRADE SIGNAL: High confidence long fade
```

**⚠️ Caution - Expanding Range:**
```
Scenario: Recent range expanding vs longer-term
position_in_short_range = 0.89           # Near short-term high
short_range_size = 4.50                  # Wide recent range
medium_range_size = 4.75                 # Similar medium-term range
range_compression_ratio = 0.95           # Not compressed (expanding)
short_range_retouches = 2                # Few recent tests
medium_range_retouches = 3               # Few overall tests

→ INTERPRETATION: Range expanding, breakout setup developing
→ TRADE SIGNAL: Avoid fade, watch for momentum breakout
```

**🚨 Avoid - Breakout Pattern:**
```
Scenario: Wide ranges with minimal retouches
position_in_short_range = 0.96           # At short-term high
short_range_size = 6.25                  # Very wide recent range
range_compression_ratio = 0.89           # Similar to medium-term
short_range_retouches = 1                # Minimal testing
medium_range_retouches = 2               # Minimal overall testing

→ INTERPRETATION: Wide, untested ranges = momentum setup
→ TRADE SIGNAL: Avoid fade, consider breakout play
```

#### Consolidation Quality Matrix:

| Compression Ratio | Retouches | Confidence | Strategy |
|------------------|-----------|------------|----------|
| <0.5 (Coiled) | 4+ | **Very High** | Fade with size |
| 0.5-0.7 (Compressed) | 3+ | **High** | Standard fade |
| 0.7-0.9 (Normal) | 2+ | **Medium** | Cautious fade |
| >0.9 (Expanding) | Any | **Low** | Avoid fade |

#### Retouch Cooldown Benefits:

**Without Cooldown (Bad):**
```
Price enters 10% zone at bar 100
Stays in zone for bars 100-105 (6 seconds)
Exits zone at bar 106
→ Counts as 6 retouches (WRONG!)
```

**With 30-Second Cooldown (Good):**
```
Price enters 10% zone at bar 100
Stays in zone for bars 100-105 (6 seconds)  
Exits zone at bar 106
Must wait until bar 130+ for next retouch to count
→ Counts as 1 retouch (CORRECT!)
```

#### Key Insights:
- **Dual timeframes** - Compare short vs medium-term consolidation patterns
- **Compression ratio** - Coiled springs (ratio <0.5) are best fade setups
- **Proper retouch counting** - 30-second cooldown prevents overcounting
- **Quality over quantity** - Compressed ranges with multiple retouches = high confidence
- **Breakout detection** - Wide ranges with few retouches = avoid fades

### 4. Return Features (5 features)
**Purpose:** Momentum analysis with acceleration and consistency metrics

**Core Strategy:** Identify momentum direction, acceleration, and quality for trade timing

#### Basic Momentum Features:

- `return_30s` = (close[N-1] - close[N-31]) / close[N-31]
  - **Purpose:** Short-term momentum (30-second trend)
  - **Interpretation:** +0.001 = +0.1% move, -0.002 = -0.2% move

- `return_60s` = (close[N-1] - close[N-61]) / close[N-61]
  - **Purpose:** Medium-term momentum (1-minute trend)
  - **Interpretation:** Captures slightly longer directional bias

- `return_300s` = (close[N-1] - close[N-301]) / close[N-301]
  - **Purpose:** Long-term momentum (5-minute trend)
  - **Interpretation:** Aligns with consolidation timeframe analysis

#### Advanced Momentum Features:

- `momentum_acceleration` = return_30s - return_60s
  - **Purpose:** Momentum acceleration/deceleration
  - **Calculation:** Short-term return minus medium-term return
  - **Interpretation:** 
    - **Positive** = Momentum accelerating (getting stronger)
    - **Negative** = Momentum decelerating (weakening)
    - **Near zero** = Steady momentum

- `momentum_consistency` = rolling_std(return_1s, 30) using bars [N-30:N-1]
  - **Purpose:** Momentum quality indicator
  - **Calculation:** Standard deviation of 1-second returns over 30 bars
  - **Interpretation:**
    - **Low values** = Smooth, consistent trend
    - **High values** = Choppy, erratic movement

#### Trading Logic & Scenarios:

**🟢 Perfect Momentum Setup:**
```
Scenario: Strong, accelerating, smooth momentum
return_30s = +0.0015                     # +0.15% in 30 seconds
return_60s = +0.0008                     # +0.08% in 60 seconds  
return_300s = +0.0025                    # +0.25% in 5 minutes
momentum_acceleration = +0.0007          # Accelerating (+0.07%)
momentum_consistency = 0.0002            # Very smooth trend

→ INTERPRETATION: Strong, accelerating, smooth uptrend
→ TRADE SIGNAL: High confidence momentum continuation
```

**🔴 Momentum Exhaustion Setup:**
```
Scenario: Strong momentum but decelerating and choppy
return_30s = +0.0020                     # +0.20% in 30 seconds
return_60s = +0.0035                     # +0.35% in 60 seconds
return_300s = +0.0045                    # +0.45% in 5 minutes
momentum_acceleration = -0.0015          # Decelerating (-0.15%)
momentum_consistency = 0.0008            # Choppy movement

→ INTERPRETATION: Momentum weakening and becoming erratic
→ TRADE SIGNAL: Potential reversal, consider fade
```

**⚪ Neutral/Consolidation Setup:**
```
Scenario: Low momentum with mixed signals
return_30s = +0.0002                     # +0.02% in 30 seconds
return_60s = -0.0001                     # -0.01% in 60 seconds
return_300s = +0.0005                    # +0.05% in 5 minutes
momentum_acceleration = +0.0003          # Slight acceleration
momentum_consistency = 0.0003            # Moderate choppiness

→ INTERPRETATION: Consolidating, no clear momentum
→ TRADE SIGNAL: Wait for clearer directional bias
```

**🚨 False Momentum (Avoid):**
```
Scenario: Apparent momentum but very choppy
return_30s = +0.0012                     # +0.12% in 30 seconds
return_60s = +0.0010                     # +0.10% in 60 seconds
return_300s = +0.0008                    # +0.08% in 5 minutes
momentum_acceleration = +0.0002          # Slight acceleration
momentum_consistency = 0.0012            # Very choppy

→ INTERPRETATION: Momentum present but poor quality
→ TRADE SIGNAL: Avoid - likely whipsaw conditions
```

#### Momentum Quality Matrix:

| Acceleration | Consistency | Quality | Strategy |
|-------------|-------------|---------|----------|
| Positive | Low (<0.0003) | **Excellent** | Follow momentum |
| Positive | Medium (0.0003-0.0006) | **Good** | Follow with caution |
| Negative | Low (<0.0003) | **Reversal** | Consider fade |
| Any | High (>0.0006) | **Poor** | Avoid trades |

#### Key Insights:
- **Acceleration matters** - Accelerating momentum is more reliable than decelerating
- **Consistency is crucial** - Smooth trends are more trustworthy than choppy moves
- **Timeframe alignment** - 30s/60s/300s align with our consolidation analysis
- **Quality over quantity** - Better to have fewer, higher-quality momentum signals
- **ES-specific thresholds** - Consistency values calibrated for 1-second ES data

### 5. Volatility Features (6 features)
**Purpose:** Volatility analysis for position sizing and movement expectations at range edges

**Core Strategy:** Understand current volatility regime and trend for optimal trade sizing

#### Absolute Volatility Measures:

- `atr_30s` = rolling_mean(true_range, 30 bars) using bars [N-30:N-1]
  - **Purpose:** Current short-term volatility level
  - **Interpretation:** Typical bar range in points (e.g., 1.25 = 1.25 point average range)
  - **Use Case:** Position sizing - higher ATR = wider stops needed

- `atr_300s` = rolling_mean(true_range, 300 bars) using bars [N-300:N-1]
  - **Purpose:** Longer-term volatility baseline
  - **Interpretation:** 5-minute average volatility context
  - **Use Case:** Regime identification and comparison baseline

#### Volatility Trend & Regime Features:

- `volatility_regime` = atr_30s / atr_300s
  - **Purpose:** Current volatility vs longer-term average
  - **Interpretation:** 
    - **>1.3** = High volatility regime (expand ranges/stops)
    - **0.8-1.3** = Normal volatility regime
    - **<0.8** = Low volatility regime (tighten ranges/stops)

- `volatility_acceleration` = (atr_30s - atr_60s) / atr_60s
  - **Purpose:** Volatility trend direction and speed
  - **Calculation:** Percentage change from 60s to 30s ATR
  - **Interpretation:**
    - **Positive** = Volatility increasing (expect larger moves)
    - **Negative** = Volatility decreasing (expect smaller moves)

- `volatility_breakout` = (atr_30s - atr_300s) / rolling_std(atr_30s, 300)
  - **Purpose:** Statistical volatility spike detection
  - **Calculation:** Z-score of current ATR vs 300-bar distribution
  - **Interpretation:**
    - **>2.0** = Volatility breakout (major move likely)
    - **1.0-2.0** = Elevated volatility
    - **<1.0** = Normal/quiet conditions

- `atr_percentile` = percentile_rank(atr_30s, atr_values[N-300:N-1])
  - **Purpose:** Current volatility rank within recent history
  - **Interpretation:** 
    - **>90%** = Very high volatility (top 10% of recent period)
    - **50%** = Median volatility
    - **<10%** = Very low volatility (bottom 10%)

#### Trading Logic & Scenarios:

**🔥 High Volatility Breakout Environment:**
```
Scenario: Volatility spiking, expect large moves
atr_30s = 2.8                           # High current volatility
atr_300s = 1.6                          # Normal baseline
volatility_regime = 1.75                # High vol regime (75% above normal)
volatility_acceleration = +0.25         # Vol increasing 25%
volatility_breakout = 2.4               # Statistical breakout (2.4 std devs)
atr_percentile = 95                     # Top 5% of recent volatility

→ INTERPRETATION: Major volatility expansion, large moves expected
→ TRADE SIGNAL: Widen stops to 4-5 points, expect 3-6 point moves
```

**📉 Low Volatility Compression Environment:**
```
Scenario: Volatility compressed, expect smaller moves
atr_30s = 0.9                           # Low current volatility
atr_300s = 1.4                          # Normal baseline
volatility_regime = 0.64                # Low vol regime (36% below normal)
volatility_acceleration = -0.15         # Vol decreasing 15%
volatility_breakout = -1.2              # Below normal volatility
atr_percentile = 15                     # Bottom 15% of recent volatility

→ INTERPRETATION: Compressed volatility, tight ranges expected
→ TRADE SIGNAL: Tighten stops to 1-2 points, expect 0.5-2 point moves
```

**⚡ Volatility Expansion Setup:**
```
Scenario: Volatility starting to increase from low levels
atr_30s = 1.2                           # Moderate current volatility
atr_300s = 1.5                          # Slightly higher baseline
volatility_regime = 0.80                # Still below normal
volatility_acceleration = +0.35         # Vol increasing rapidly (35%)
volatility_breakout = 0.5               # Approaching normal levels
atr_percentile = 45                     # Below median but rising

→ INTERPRETATION: Volatility awakening from compression
→ TRADE SIGNAL: Prepare for range expansion, adjust position sizes
```

**🎯 Normal Volatility Environment:**
```
Scenario: Stable, predictable volatility conditions
atr_30s = 1.6                           # Normal current volatility
atr_300s = 1.5                          # Similar baseline
volatility_regime = 1.07                # Normal regime
volatility_acceleration = +0.05         # Stable volatility
volatility_breakout = 0.2               # Normal conditions
atr_percentile = 55                     # Near median

→ INTERPRETATION: Stable volatility environment
→ TRADE SIGNAL: Standard position sizing, 2-3 point stops typical
```

#### Volatility-Based Position Sizing Guide:

| Volatility Regime | ATR Range | Stop Distance | Expected Move | Position Size |
|------------------|-----------|---------------|---------------|---------------|
| High (>1.3) | >2.0 points | 4-6 points | 3-8 points | **Reduce 30%** |
| Normal (0.8-1.3) | 1.0-2.0 points | 2-4 points | 1-4 points | **Standard** |
| Low (<0.8) | <1.0 points | 1-2 points | 0.5-2 points | **Increase 20%** |

#### Key Insights:
- **ATR for absolute levels** - Know what size moves to expect
- **Regime detection** - Adjust expectations based on current environment
- **Volatility acceleration** - Anticipate expanding/contracting ranges
- **Breakout detection** - Identify when volatility spikes significantly
- **Position sizing** - Scale trade size inversely with volatility
- **ES-specific ranges** - Calibrated for typical ES volatility patterns

### 6. Microstructure Features (6 features)
**Purpose:** Tick flow analysis and price action quality for trend identification

**Core Strategy:** Use tick flow percentages to identify underlying trend strength and direction

#### Basic Price Action:

- `bar_range` = high[N-1] - low[N-1]
  - **Purpose:** Absolute bar size for current volatility context
  - **Use Case:** Compare with ATR to identify unusually large/small bars

- `relative_bar_size` = bar_range / atr_30s
  - **Purpose:** Bar size relative to current volatility regime
  - **Interpretation:** 
    - **>1.5** = Large bar for current volatility (significant move)
    - **0.5-1.5** = Normal bar size
    - **<0.5** = Small bar (consolidation/indecision)

#### Bar Direction Flow Analysis:

- `uptick_pct_30s` = count(up_bars[N-30:N-1]) / 30 × 100
  - **Purpose:** Percentage of up bars in last 30 seconds
  - **Calculation:** up_bar = 1 if close[i] > close[i-1], else 0
  - **Interpretation:** 
    - **>60%** = Strong upward pressure (18+ of 30 bars up)
    - **40-60%** = Neutral/mixed flow (12-18 of 30 bars up)
    - **<40%** = Weak upward pressure (<12 of 30 bars up)

- `uptick_pct_60s` = count(up_bars[N-60:N-1]) / 60 × 100
  - **Purpose:** Percentage of up bars in last 60 seconds
  - **Calculation:** Same logic over 60 1-second bars
  - **Use Case:** Medium-term directional bias identification

- `bar_flow_consistency` = abs(uptick_pct_30s - uptick_pct_60s)
  - **Purpose:** Consistency between short and medium-term bar direction flow
  - **Interpretation:**
    - **Low (<10%)** = Consistent trend direction across timeframes
    - **High (>20%)** = Changing trend or choppy conditions

- `directional_strength` = abs(uptick_pct_30s - 50) × 2
  - **Purpose:** Strength of directional bar flow (0-100 scale)
  - **Calculation:** Distance from 50% (neutral) × 2
  - **Example:** 70% up bars → abs(70-50) × 2 = 40 strength score
  - **Interpretation:**
    - **>70** = Very strong directional flow (>85% or <15% up bars)
    - **30-70** = Moderate directional bias (65-85% or 15-35% up bars)
    - **<30** = Weak/neutral flow (35-65% up bars)

#### Trading Logic & Scenarios:

**🟢 Strong Uptrend Confirmation:**
```
Scenario: Consistent upward bar flow across timeframes
uptick_pct_30s = 72%                    # 22 of last 30 bars were up
uptick_pct_60s = 68%                    # 41 of last 60 bars were up
bar_flow_consistency = 4%               # Very consistent trend
directional_strength = 44              # Moderate-strong momentum
relative_bar_size = 1.3                # Normal-sized bars

→ INTERPRETATION: Consistent upward trend with good momentum
→ TRADE SIGNAL: Favor long setups, avoid short fades
```

**🔴 Trend Exhaustion Pattern:**
```
Scenario: Recent up bars but weakening vs longer timeframe
uptick_pct_30s = 65%                    # 20 of last 30 bars up
uptick_pct_60s = 78%                    # 47 of last 60 bars up (was stronger)
bar_flow_consistency = 13%              # Moderate inconsistency
directional_strength = 30              # Weakening momentum
relative_bar_size = 0.7                # Smaller bars (losing steam)

→ INTERPRETATION: Uptrend losing momentum, potential exhaustion
→ TRADE SIGNAL: Consider fade setups, avoid momentum longs
```

**⚪ Neutral/Consolidation Pattern:**
```
Scenario: Balanced tick flow, no clear direction
uptick_pct_30s = 52%                    # Slightly bullish but neutral
uptick_pct_60s = 48%                    # Slightly bearish but neutral
tick_flow_consistency = 4%              # Consistent (consistently neutral)
tick_momentum_strength = 4              # Very weak momentum
relative_bar_size = 0.6                 # Small bars (consolidation)

→ INTERPRETATION: Consolidation mode, no directional bias
→ TRADE SIGNAL: Range trading, wait for breakout confirmation
```

**🚨 Choppy/Avoid Conditions:**
```
Scenario: Inconsistent tick flow, whipsaw conditions
uptick_pct_30s = 35%                    # Bearish short-term
uptick_pct_60s = 65%                    # Bullish medium-term
tick_flow_consistency = 30%             # Very inconsistent
tick_momentum_strength = 30             # Moderate but conflicting
relative_bar_size = 1.8                 # Large bars (volatility)

→ INTERPRETATION: Conflicting signals, choppy conditions
→ TRADE SIGNAL: Avoid trades, wait for clarity
```

#### Tick Flow Quality Matrix:

| Uptick % | Consistency | Momentum Strength | Quality | Strategy |
|----------|-------------|-------------------|---------|----------|
| >65% | <10% | >50 | **Excellent Up** | Follow uptrend |
| <35% | <10% | >50 | **Excellent Down** | Follow downtrend |
| 45-55% | <10% | <30 | **Neutral** | Range trading |
| Any | >20% | Any | **Poor** | Avoid trades |

#### Key Insights:
- **Percentage-based flow** - More meaningful than raw tick counts
- **Multiple timeframes** - 30s vs 60s shows trend consistency
- **Consistency matters** - Conflicting timeframes = avoid
- **Relative bar size** - Context matters more than absolute size
- **Momentum strength** - Quantifies how strong the directional bias is
- **ES-specific thresholds** - Calibrated for 1-second ES tick data

### 7. Time Features (7 features)
**Purpose:** ES session period identification for behavioral pattern recognition

**Core Strategy:** Identify distinct trading periods with different volatility and trend characteristics

#### Session Period Features:

**Timezone Handling:**
```python
# Data Analysis Results:
# - Databento timestamps are in UTC (ts_event index)
# - RTH data runs 08:00-15:00 Central Time (09:00-16:00 Eastern Time)
# - This matches ES RTH: 9:30 AM - 4:00 PM ET = 8:30 AM - 3:00 PM CT
# - Data appears to start at 8:00 CT (30 minutes before official RTH)

import pytz
central_tz = pytz.timezone('US/Central')
ct_time = utc_timestamp.astimezone(central_tz)
ct_hour_minute = ct_time.hour + ct_time.minute/60.0
```

#### ES Session Periods (Central Time - Based on Actual Data):

- `is_eth` = 1 if (15:00 ≤ ct_time < 24:00) OR (0:00 ≤ ct_time < 7:30)
  - **Purpose:** Extended Trading Hours - overnight electronic session
  - **Coverage:** 15:00 CT to 07:30 CT next day (overnight)
  - **Characteristics:** Lower volume, algorithmic trading, news reactions
  - **Strategy:** Avoid or use very tight ranges, news-driven moves

- `is_pre_open` = 1 if 7:30 ≤ ct_time < 8:30 (07:30-08:30 CT)
  - **Purpose:** Pre-market preparation period
  - **Characteristics:** Building volume, gap setup, institutional positioning
  - **Strategy:** Watch for gap direction, prepare for RTH open

- `is_rth_open` = 1 if 8:30 ≤ ct_time < 9:15 (08:30-09:15 CT)
  - **Purpose:** RTH opening period - first 45 minutes
  - **Characteristics:** Gap reactions, high volatility, emotional trading
  - **Strategy:** Fade overreactions, watch for gap fills

- `is_morning` = 1 if 9:15 ≤ ct_time < 11:00 (09:15-11:00 CT)  
  - **Purpose:** Morning trend establishment
  - **Characteristics:** Institutional flow, sustained trend development
  - **Strategy:** Follow momentum, breakout plays work best

- `is_lunch` = 1 if 11:00 ≤ ct_time < 13:00 (11:00-13:00 CT)
  - **Purpose:** Lunch doldrums - low volume consolidation
  - **Characteristics:** Tight ranges, mean reversion, low volume
  - **Strategy:** Range trading only, fade extremes

- `is_afternoon` = 1 if 13:00 ≤ ct_time < 14:30 (13:00-14:30 CT)
  - **Purpose:** Afternoon institutional activity
  - **Characteristics:** Volume returns, trend resumption or reversal
  - **Strategy:** Major directional decisions, trend continuation/reversal

- `is_rth_close` = 1 if 14:30 ≤ ct_time < 15:00 (14:30-15:00 CT)
  - **Purpose:** RTH closing period - settlement and positioning
  - **Characteristics:** Highest volume, profit-taking, position squaring
  - **Strategy:** Fade extremes, expect mean reversion to VWAP

#### Trading Logic & Behavioral Patterns:

**� RETH - Extended Hours (15:00-07:30 CT):**
```
Characteristics:
- Overnight electronic session
- Lower volume, wider spreads
- Algorithmic and news-driven moves
- Less predictable price action

Strategy:
- Avoid trading or use very tight ranges
- News reactions can be extreme
- Wait for RTH for better liquidity
- Useful for gap analysis only
```

**🌅 Pre-Open (07:30-08:30 CT):**
```
Characteristics:
- Building volume toward RTH open
- Gap setup and direction establishment
- Institutional pre-positioning
- Anticipation of RTH opening

Strategy:
- Analyze gap size and direction
- Watch for volume buildup patterns
- Prepare for RTH opening strategy
- Don't chase pre-market moves
```

**📈 RTH Open (08:30-09:15 CT):**
```
Characteristics:
- Official RTH opening period
- Gap reactions and emotional trading
- Highest volatility of the day
- Often reverses initial direction

Strategy:
- Fade overreactions to overnight news
- Watch for gap fill opportunities
- Use wider stops due to volatility
- First 45 minutes set daily tone
```

**� Morrning Trend (09:15-11:00 CT):**
```
Characteristics:
- Prime trending period after open volatility
- Institutional flow and sustained moves
- Best breakout follow-through
- High conviction directional moves

Strategy:
- Momentum continuation trades
- Breakout plays work best
- Avoid fading strong moves
- This is "prime time" for ES trading
```

**😴 Lunch Doldrums (11:00-13:00 CT):**
```
Characteristics:
- Lowest volume period
- Tight consolidation ranges
- Mean reversion dominant
- Institutional lunch break

Strategy:
- Range trading only
- Fade range extremes
- Tighten stops (lower volatility)
- Avoid momentum plays
```

**⚡ Afternoon Session (13:00-14:45 CT):**
```
Characteristics:
- Institutional return from lunch
- Trend resumption or major reversals
- Volume picks up significantly
- Important directional decisions

Strategy:
- Watch for trend continuation
- Major reversal patterns develop
- Increased position sizing opportunity
- Key decision point for daily direction
```

**🔔 RTH Close (14:30-15:00 CT):**
```
Characteristics:
- Highest volume period (last 30 minutes)
- Profit-taking and position squaring
- Often reverses intraday extremes
- Settlement-driven activity

Strategy:
- Fade extreme moves from VWAP
- Expect mean reversion
- High volume = high opportunity
- Position squaring creates reversals
```

#### Session Period Quality Matrix:

| Period | Volatility | Trend Quality | Volume | Best Strategy |
|--------|------------|---------------|---------|---------------|
| ETH | Medium | Poor (news-driven) | **Low** | **Avoid/Watch only** |
| Pre-Open | Low-Medium | Poor (setup) | Building | **Gap analysis** |
| RTH Open | **High** | Poor (choppy) | **High** | **Fade overreactions** |
| Morning | Medium-High | **Excellent** | **High** | **Follow momentum** |
| Lunch | **Low** | Poor (range) | **Low** | **Range trading** |
| Afternoon | Medium-High | **Good** | **High** | **Trend/Reversal** |
| RTH Close | **High** | Poor (reversal) | **Highest** | **Fade extremes** |

#### Timezone Implementation Notes:

```python
# Robust timezone handling for ES data
def get_es_session_period(utc_timestamp):
    """
    Convert UTC timestamp to Central Time and determine ES session period
    Handles DST automatically
    """
    import pytz
    from datetime import datetime
    
    # Convert to Central Time (handles DST automatically)
    central_tz = pytz.timezone('US/Central')
    ct_time = utc_timestamp.astimezone(central_tz)
    
    # Get hour.minute as decimal (e.g., 8:30 = 8.5)
    ct_decimal = ct_time.hour + ct_time.minute/60.0
    
    # Determine session period
    if (15.0 <= ct_decimal < 24.0) or (0.0 <= ct_decimal < 7.5):  # ETH
        return 'eth'
    elif 7.5 <= ct_decimal < 8.5:     # 7:30-8:30
        return 'pre_open'
    elif 8.5 <= ct_decimal < 9.25:    # 8:30-9:15
        return 'rth_open'
    elif 9.25 <= ct_decimal < 11.0:   # 9:15-11:00
        return 'morning'
    elif 11.0 <= ct_decimal < 13.0:   # 11:00-13:00
        return 'lunch'
    elif 13.0 <= ct_decimal < 14.5:   # 13:00-14:30
        return 'afternoon'
    elif 14.5 <= ct_decimal < 15.0:   # 14:30-15:00
        return 'rth_close'
    else:
        return 'unknown'
```

#### Key Insights:
- **Central Time anchoring** - ES trades on Chicago time, always use CT
- **DST handling** - pytz automatically handles daylight saving transitions
- **One-hot encoding** - 7 binary features better than 1 categorical for LSTM
- **No ordinal assumption** - Periods don't have meaningful numerical order
- **Trading interpretability** - Can see which periods drive model decisions
- **Behavioral patterns** - Each period has distinct characteristics requiring separate weights

## Data Leakage Checklist

✅ **Volume features:** Only use historical rolling windows
✅ **Price context:** Only use session data up to current bar
✅ **Recent high/low features:** Only use historical lookback windows
✅ **Returns:** Only use historical price differences
✅ **Volatility:** Only use historical rolling calculations
✅ **Microstructure:** Only use current bar data
✅ **Time features:** Only use current timestamp

## ✅ All Features Now Safe From Data Leakage

All feature categories now use only historical data through bar N-1 to predict bar N.