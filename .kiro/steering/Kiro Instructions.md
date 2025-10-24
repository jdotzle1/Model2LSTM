# Instructions for Kiro (AI Coding Assistant)

## Project Context
You are helping build an LSTM model to predict optimal trade entry timing for ES futures. The user has completed:
1. ✅ Data conversion (DBN → Parquet)
2. ✅ Labeling logic (win/loss/MAE filtering)
3. 🔄 Feature engineering (in progress)

## Your Role
- **Write clean, documented code** for Python data pipeline
- **Explain technical concepts** at beginner/novice level
- **Minimize code duplication** - reuse functions
- **Validate data quality** - check for NaN, outliers, errors
- **Test incrementally** - small samples before full dataset

## Coding Standards

### Style
- Clear variable names: `distance_from_vwap` not `dvwap`
- Comments for non-obvious logic
- Type hints where helpful
- Consistent formatting (PEP 8)

### Documentation
Each function should have:
```python
def function_name(param1, param2):
    """
    Brief description of what it does
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Description of return value
    """
```

### Error Handling
- Check for edge cases (empty dataframes, missing columns)
- Print progress for long operations
- Validate inputs before processing

### Testing Strategy
1. Test on 100-1000 bars first
2. Check for NaN values
3. Verify calculations with spot checks
4. Visualize distributions
5. Only then scale to full dataset

## Key Principles

### 1. Explain Like I'm 5 (When Needed)
**Bad:** "Calculate the z-score using rolling statistics"
**Good:** "Z-score tells us if volume is unusually high. Z=2 means '2 standard deviations above average' or roughly 'top 5% of volume'"

### 2. Show, Don't Tell
Include example outputs:
```python
# Example output:
# volume_ratio_30s = 1.8
# This means: Volume is 80% higher than the last 30-second average
```

### 3. Minimize Redundant Code
If calculating similar features, use helper functions:
```python
def calculate_rolling_ratio(df, column, window):
    return df[column] / df[column].rolling(window).mean()

df['volume_ratio_30s'] = calculate_rolling_ratio(df, 'volume', 30)
df['volume_ratio_300s'] = calculate_rolling_ratio(df, 'volume', 300)
```

### 4. Progress Visibility
For long operations, print progress:
```python
print("Processing labeling...")
for i, profile in enumerate(profiles, 1):
    print(f"  [{i}/{len(profiles)}] {profile['name']}...")
```

### 5. Data Validation
Always validate after creating features:
```python
# Check for NaN
print(f"NaN counts: {df.isnull().sum().sum()}")

# Check ranges
print(f"Volume ratio range: {df['volume_ratio_30s'].min():.2f} to {df['volume_ratio_30s'].max():.2f}")

# Spot check
print(df[['close', 'return_5s', 'vwap']].head(10))
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
- `project/data_pipeline/labeling.py` - Label calculation (DONE)
- `project/data_pipeline/features.py` - Feature engineering (IN PROGRESS)
- `project/data_pipeline/pipeline.py` - Main orchestration (TODO)
- Test scripts in `project/` - For validation

## Current Task
Complete feature engineering module and validate on 1000-bar sample before scaling to full 15-year dataset.

## Questions to Ask
- "Should I test this on the small sample first?"
- "Do you want to see the results before proceeding?"
- "Should I add visualization for this feature?"
- "Is the explanation clear enough?"

## Success Criteria
- Code runs without errors on test sample
- Features have expected ranges (no infinities/NaN except expected)
- User understands what each feature represents
- Ready to scale to production dataset