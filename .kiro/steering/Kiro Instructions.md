# Instructions for Kiro (AI Coding Assistant)

## Project Context
You are helping build an LSTM model to predict optimal trade entry timing for ES futures. The user has completed:
1. ✅ Data conversion (DBN → Parquet)
2. ✅ Labeling logic (win/loss/MAE filtering)
3. 🔄 Feature engineering (in progress)

## Your Role
- **Write clean, documented code** for weighted labeling and XGBoost pipeline
- **Explain technical concepts** at beginner/novice level
- **Minimize code duplication** - reuse functions
- **Validate data quality** - check for NaN, outliers, errors
- **Test incrementally** - small samples before full dataset

## Coding Standards

### Style
- Clear variable names: `quality_weight` not `qw`, `volatility_regime` not `vol_reg`
- Comments for non-obvious logic, especially weight calculations
- Type hints where helpful
- Consistent formatting (PEP 8)

### Documentation
Each function should have:
```python
def calculate_quality_weights(mae_ticks: np.ndarray, stop_ticks: int) -> np.ndarray:
    """
    Calculate quality weights based on MAE performance
    
    Formula: quality_weight = 2.0 - (1.5 × mae_ratio)
    Range: [0.5, 2.0]
    
    Args:
        mae_ticks: Array of MAE values in ticks for winners
        stop_ticks: Stop distance in ticks for the trading mode
        
    Returns:
        Array of quality weights clipped to [0.5, 2.0]
    """
```

### Error Handling
- Check for edge cases (empty dataframes, missing columns)
- Print progress for long operations (weighted labeling can be slow)
- Validate inputs before processing
- Validate output format for XGBoost compatibility

### Testing Strategy
1. Test on 1000-bar sample first with full integration
2. Check for NaN values in all 61 output columns
3. Verify weight calculations with spot checks
4. Validate XGBoost format requirements (binary labels, positive weights)
5. Test chunked vs single-pass consistency
6. Only then scale to full 15-year dataset

## Key Principles

### 1. Explain Like I'm 5 (When Needed)
**Bad:** "Calculate quality weights using MAE ratios"
**Good:** "Quality weight tells us how good the entry timing was. MAE=0 ticks (perfect entry) gets weight=2.0, MAE=6 ticks (at stop level) gets weight=0.5. Lower drawdown = higher quality weight."

### 2. Show, Don't Tell
Include example outputs:
```python
# Example weighted labeling output:
# label_low_vol_long = 1 (winner)
# weight_low_vol_long = 2.4 (high quality: good timing + fast + recent)
# This means: Low volatility long trade won with excellent entry quality
```

### 3. Minimize Redundant Code
For weight calculations across modes, use helper functions:
```python
def calculate_quality_weights(mae_ticks, stop_ticks):
    mae_ratio = mae_ticks / stop_ticks
    return np.clip(2.0 - (1.5 * mae_ratio), 0.5, 2.0)

# Apply to all modes
for mode in TRADING_MODES.values():
    quality_weights = calculate_quality_weights(mae_data, mode.stop_ticks)
```

### 4. Progress Visibility
For long operations, print progress:
```python
print("Processing weighted labeling...")
for i, mode_name in enumerate(TRADING_MODES.keys(), 1):
    print(f"  [{i}/6] {mode_name}...")
    # Process mode
    print(f"    Win rate: {win_rate:.1%}, Avg weight: {avg_weight:.2f}")
```

### 5. Data Validation
Always validate after weighted labeling:
```python
# Check for required columns
expected_cols = [f"label_{mode}" for mode in TRADING_MODES.keys()]
expected_cols.extend([f"weight_{mode}" for mode in TRADING_MODES.keys()])
missing = set(expected_cols) - set(df.columns)
if missing:
    print(f"Missing columns: {missing}")

# Check label format (binary 0/1)
for col in [c for c in df.columns if c.startswith('label_')]:
    unique_vals = set(df[col].unique())
    if not unique_vals.issubset({0, 1}):
        print(f"Invalid label values in {col}: {unique_vals}")

# Check weight format (positive values)
for col in [c for c in df.columns if c.startswith('weight_')]:
    if (df[col] <= 0).any():
        print(f"Non-positive weights in {col}")
```

## Common Patterns

### Pattern 1: Rolling Calculations
```python
# Always handle NaN from rolling windows
df['feature'] = df['column'].rolling(window).mean()
# First 'window' rows will be NaN - this is expected
```

### Pattern 2: Forward Fill
```python
# For swing highs/lows that are sparse
df['last_swing_high'] = df['swing_high'].fillna(method='ffill')
```

### Pattern 3: Groupby Operations
```python
# For daily resets (like VWAP)
df['date'] = df['timestamp'].dt.date
df['daily_feature'] = df.groupby('date')['column'].transform('mean')
```

## What to Avoid
- ❌ Overly complex one-liners - break into steps
- ❌ Magic numbers - use named constants
- ❌ Silent failures - always check results
- ❌ Premature optimization - clarity first
- ❌ Assuming timezone - always verify

## Files You'll Work With
- `src/data_pipeline/weighted_labeling.py` - Weighted labeling system (COMPLETE)
- `src/data_pipeline/features.py` - Feature engineering (COMPLETE)
- `src/data_pipeline/pipeline.py` - Main orchestration (COMPLETE)
- `tests/integration/test_final_integration_1000_bars.py` - Integration testing (COMPLETE)
- `main.py` - Production entry point (READY)

## Current Status
✅ **Weighted labeling system complete** - 6 volatility modes with quality/velocity/time decay weights
✅ **Feature engineering complete** - 43 features validated and tested
✅ **Full integration tested** - 1000-bar sample with 61 output columns
✅ **EC2 deployment ready** - Complete deployment package available

## Questions to Ask
- "Should I run the integration test to validate the complete pipeline?"
- "Do you want to see the weight distributions for the different modes?"
- "Should I check the XGBoost format compatibility?"
- "Is the volatility regime detection working correctly?"

## Success Criteria
- ✅ All 12 weighted labeling columns generated correctly (6 labels + 6 weights)
- ✅ All 43 features calculated with acceptable NaN levels
- ✅ XGBoost format requirements met (binary labels, positive weights)
- ✅ Chunked processing consistency validated
- ✅ Ready for full 15-year dataset processing on EC2

## Next Steps
1. **XGBoost Model Training** - Train 6 specialized models using weighted samples
2. **Ensemble Deployment** - Build volatility-adaptive model selection system
3. **Production Inference** - Real-time trading signal generation