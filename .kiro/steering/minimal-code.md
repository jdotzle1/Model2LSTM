# Minimal Code Implementation Guidelines

## Core Principle: Write the ABSOLUTE MINIMUM code needed

You have a tendency to over-engineer and write verbose implementations. This project requires **minimal viable code** that gets the job done efficiently.

## Implementation Rules

### ✅ DO Write:
- **Single-purpose functions** that do one thing well
- **Direct calculations** without unnecessary abstractions
- **Simple loops** over complex vectorized operations when clearer
- **Minimal error handling** - basic validation only
- **Essential comments** - only for non-obvious logic
- **Straightforward variable names** - no over-descriptive names

### ❌ DON'T Write:
- **Classes when functions suffice** - avoid OOP unless absolutely necessary
- **Abstract base classes or interfaces** - this isn't enterprise software
- **Extensive error handling** - basic validation is enough
- **Comprehensive logging** - simple print statements for progress
- **Configuration systems** - hardcode reasonable defaults
- **Helper utilities** - write inline code instead
- **Docstrings for obvious functions** - save time and lines
- **Type hints everywhere** - only where truly helpful

## Examples

### ❌ Over-Engineered (DON'T DO THIS):
```python
class VolumeFeatureCalculator:
    def __init__(self, config: VolumeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_volume_ratio(self, volume_series: pd.Series, window: int = 30) -> pd.Series:
        """
        Calculate volume ratio using rolling mean with comprehensive error handling
        
        Args:
            volume_series: Time series of volume data
            window: Rolling window size for mean calculation
            
        Returns:
            Series of volume ratios
            
        Raises:
            ValueError: If volume_series is empty or contains negative values
        """
        if volume_series.empty:
            raise ValueError("Volume series cannot be empty")
        if (volume_series < 0).any():
            raise ValueError("Volume cannot be negative")
            
        self.logger.info(f"Calculating volume ratio with window={window}")
        rolling_mean = volume_series.rolling(window=window, min_periods=1).mean()
        return volume_series / rolling_mean
```

### ✅ Minimal Viable (DO THIS):
```python
def volume_ratio_30s(volume):
    """Volume vs 30-bar rolling mean"""
    return volume / volume.rolling(30).mean()
```

## Weighted Labeling Specifics

### Weight Calculation - Core functions:
```python
def _calculate_quality_weights(self, mae_ticks):
    mae_ratio = mae_ticks / self.mode.stop_ticks
    return np.clip(2.0 - (1.5 * mae_ratio), 0.5, 2.0)

def _calculate_velocity_weights(self, seconds_to_target):
    velocity_weights = 2.0 - (1.5 * (seconds_to_target - 300) / 600)
    return np.clip(velocity_weights, 0.5, 2.0)

def _calculate_time_decay(self, timestamps):
    months_ago = timestamps.apply(lambda x: self._months_between(x, timestamps.max()))
    return np.exp(-0.05 * months_ago)
```

### Feature Engineering - Core functions:
```python
def volume_ratio_30s(volume):
    return volume / volume.rolling(30).mean()

def volatility_regime(atr_30s, atr_300s):
    return atr_30s / atr_300s

def position_in_short_range(close, short_range_high, short_range_low):
    return (close - short_range_low) / (short_range_high - short_range_low)
```

### Key Points:
- **Direct formulas** - no unnecessary abstractions
- **Vectorized operations** - numpy/pandas built-ins
- **Clear variable names** - self-documenting code
- **Minimal validation** - basic checks only

## Testing Approach
- **Integration test** - test complete pipeline on 1000-bar sample
- **Real data validation** - use actual market data patterns
- **XGBoost format validation** - ensure output compatibility
- **Performance testing** - validate on large datasets

## File Structure
```
project/data_pipeline/
├── weighted_labeling.py    # 6 modes, 3 weight components
├── features.py            # 43 features in 7 categories
└── pipeline.py           # Integration orchestration

tests/
├── test_weighted_labeling_comprehensive.py  # Weight calculation tests
├── test_features_comprehensive.py          # Feature validation tests
└── test_final_integration_1000_bars.py     # Complete pipeline test
```

## Success Metrics
- **Weighted labeling: Binary labels + positive weights** for all 6 modes
- **Feature engineering: 43 features** with acceptable NaN levels
- **Integration: 61 total columns** (6 original + 12 labeling + 43 features)
- **XGBoost ready: Proper format** for 6 specialized model training
- **Performance: Chunked processing** for memory efficiency
- **Prioritize correctness over line count** - get the weight formulas right

## Remember
This is a **data processing pipeline**, not enterprise software. Write code like you're doing a quick analysis in a Jupyter notebook, then clean it up just enough to be reusable.

**MINIMAL. VIABLE. CODE.**