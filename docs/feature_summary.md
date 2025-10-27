# ES Trading Model - Feature Summary

## 43 Features Across 7 Categories

### 1. Volume Features (4)
| Feature | Range | Description |
|---------|-------|-------------|
| `volume_ratio_30s` | 0.1 - 5.0+ | Current volume vs 30s average |
| `volume_slope_30s` | -100 to +100 | Medium-term volume trend (setup detection) |
| `volume_slope_5s` | -200 to +200 | Short-term volume trend (exhaustion detection) |
| `volume_exhaustion` | -1000 to +1000 | Combined signal: ratio × short slope (fade signal) |

### 2. Price Context Features (5)
| Feature | Range | Description |
|---------|-------|-------------|
| `vwap` | 4000 - 7000 | Volume-weighted average price (fair value) |
| `distance_from_vwap_pct` | -2.0% to +2.0% | Percentage distance from VWAP (signed) |
| `vwap_slope` | -10 to +10 | VWAP trend direction and strength |
| `distance_from_rth_high` | -50 to 0 | Distance from session high (≤ 0) |
| `distance_from_rth_low` | 0 to +50 | Distance from session low (≥ 0) |

### 3. Consolidation & Range Features (10)
| Feature | Range | Description |
|---------|-------|-------------|
| `short_range_high` | 4000 - 7000 | 5-minute resistance level |
| `short_range_low` | 4000 - 7000 | 5-minute support level |
| `short_range_size` | 0.5 - 20.0 | Size of 5-minute consolidation |
| `position_in_short_range` | 0.0 - 1.0 | Position in 5-minute range (fade at >0.85, <0.15) |
| `medium_range_high` | 4000 - 7000 | 15-minute resistance level |
| `medium_range_low` | 4000 - 7000 | 15-minute support level |
| `medium_range_size` | 1.0 - 50.0 | Size of 15-minute consolidation |
| `range_compression_ratio` | 0.1 - 2.0 | Short vs medium range (<0.5 = coiled spring) |
| `short_range_retouches` | 0 - 20 | Distinct tests of 5-min range boundaries (30s cooldown) |
| `medium_range_retouches` | 0 - 50 | Distinct tests of 15-min range boundaries (30s cooldown) |

### 4. Return Features (5)
| Feature | Range | Description |
|---------|-------|-------------|
| `return_30s` | -0.01 to +0.01 | 30-second momentum (% change) |
| `return_60s` | -0.02 to +0.02 | 60-second momentum (% change) |
| `return_300s` | -0.05 to +0.05 | 300-second momentum (% change) |
| `momentum_acceleration` | -0.02 to +0.02 | Momentum acceleration (30s - 60s returns) |
| `momentum_consistency` | 0.0001 - 0.002 | Movement quality (low = smooth, high = choppy) |

### 5. Volatility Features (6)
| Feature | Range | Description |
|---------|-------|-------------|
| `atr_30s` | 0.5 - 5.0 | Current short-term volatility (points) |
| `atr_300s` | 0.8 - 3.0 | Longer-term volatility baseline (points) |
| `volatility_regime` | 0.3 - 3.0 | Current vs baseline volatility (>1.3 = high vol) |
| `volatility_acceleration` | -0.5 to +0.5 | Volatility trend (% change) |
| `volatility_breakout` | -3.0 to +5.0 | Statistical volatility spike (Z-score) |
| `atr_percentile` | 0 - 100 | Current volatility rank in recent history |

### 6. Microstructure Features (6)
| Feature | Range | Description |
|---------|-------|-------------|
| `bar_range` | 0.0 - 10.0 | Last bar's high-low range (points) |
| `relative_bar_size` | 0.1 - 3.0 | Bar size vs current volatility (>1.5 = large) |
| `uptick_pct_30s` | 0% - 100% | % of up bars in last 30 seconds |
| `uptick_pct_60s` | 0% - 100% | % of up bars in last 60 seconds |
| `bar_flow_consistency` | 0% - 50% | Consistency between 30s/60s flow (<10% = consistent) |
| `directional_strength` | 0 - 100 | Strength of directional bias (>70 = very strong) |

### 7. Time Features (7)
| Feature | Range | Description |
|---------|-------|-------------|
| `is_eth` | 0 or 1 | Extended hours (15:00-07:30 CT) - avoid trading |
| `is_pre_open` | 0 or 1 | Pre-market (07:30-08:30 CT) - gap analysis |
| `is_rth_open` | 0 or 1 | RTH open (08:30-09:15 CT) - fade overreactions |
| `is_morning` | 0 or 1 | Morning trend (09:15-11:00 CT) - follow momentum |
| `is_lunch` | 0 or 1 | Lunch period (11:00-13:00 CT) - range trading |
| `is_afternoon` | 0 or 1 | Afternoon (13:00-14:30 CT) - trend/reversal |
| `is_rth_close` | 0 or 1 | RTH close (14:30-15:00 CT) - fade extremes |

## Key Trading Applications

**Volume Exhaustion**: `volume_exhaustion` < -100 = fade signal  
**Range Fades**: `position_in_short_range` > 0.85 + `range_compression_ratio` < 0.5 + `short_range_retouches` > 3 = high confidence fade  
**Momentum**: `momentum_acceleration` > 0 + `momentum_consistency` < 0.0003 = follow trend  
**Volatility Regime**: `volatility_regime` > 1.3 = widen stops, reduce size  
**Session Timing**: Different periods favor different strategies (momentum vs fade)

**Total: 43 features** designed for ES futures fade and momentum strategies with proper data leakage prevention.