# Swing Point Detection - No Future Data Leakage

## Current Problem
The existing swing logic looks forward to confirm swing points:
```python
# BAD - Uses future data
swing_high = high[N] if high[N] == max(high[N-5:N+5])  # Looks at N+1 to N+5!
```

## ✅ Corrected Approach: Delayed Swing Confirmation

### Method 1: Fixed Delay Confirmation
```python
# Look back 5 bars and check if it was a swing point
def identify_swing_high_5s(df, current_idx):
    if current_idx < 10:  # Need enough history
        return np.nan
    
    check_idx = current_idx - 5  # Look back 5 bars
    check_high = df.iloc[check_idx]['high']
    
    # Check if check_idx was highest in 10-bar window centered on it
    start_idx = max(0, check_idx - 5)
    end_idx = min(len(df), check_idx + 6)  # +6 because we're now 5 bars later
    
    window_highs = df.iloc[start_idx:end_idx]['high']
    
    if check_high == window_highs.max():
        return check_high
    else:
        return np.nan
```

### Method 2: Historical Pattern Recognition
```python
# Identify swing based on historical pattern only
def identify_swing_high_historical(df, current_idx, lookback=5):
    if current_idx < lookback * 2:
        return np.nan
    
    # Look at a completed pattern in the past
    center_idx = current_idx - lookback
    
    # Check if center_idx was highest in its neighborhood (all historical)
    start_idx = center_idx - lookback
    end_idx = center_idx + lookback + 1
    
    if df.iloc[center_idx]['high'] == df.iloc[start_idx:end_idx]['high'].max():
        return df.iloc[center_idx]['high']
    else:
        return np.nan
```

## Recommended Solution: Method 1 (Fixed Delay)

**Advantages:**
- No future data leakage
- Consistent 5-bar delay for all swing points
- Simple to implement and understand
- Realistic - in live trading, you'd need time to confirm swing points anyway

**Trade-off:**
- Swing points are identified 5 bars after they occur
- This is actually realistic - you can't know it's a swing point until price moves away

## Implementation Plan

1. **Rewrite swing detection functions** to use fixed delay method
2. **Test on sample data** to ensure no future data leakage
3. **Verify swing points make sense** visually
4. **Proceed with other features** that don't have leakage issues

## Alternative: Skip Swing Features Initially

If swing point detection is complex, we could:
1. **Start with 45 features** (skip the 10 swing features)
2. **Implement and test other feature categories first**
3. **Add swing features later** once we have the main pipeline working

What's your preference? Fix swing detection now or skip it initially?