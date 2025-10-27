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

## Feature Engineering Specifics

### Volume Features - 4 functions:
```python
def volume_ratio_30s(volume):
    return volume / volume.rolling(30).mean()

def volume_slope_30s(volume):
    vol_ma = volume.rolling(5).mean()
    return vol_ma.rolling(30).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

def volume_slope_5s(volume):
    return volume.rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

def volume_exhaustion(volume):
    ratio = volume_ratio_30s(volume)
    slope = volume_slope_5s(volume)
    return ratio * slope
```

### Key Points:
- **4 functions, ~10 lines total**
- **No classes, no config, no extensive validation**
- **Direct calculations using pandas built-ins**
- **Minimal comments - function names are self-documenting**

## Testing Approach
- **Single test file** with basic input/output validation
- **No mocking, no fixtures** - use real small datasets
- **Test core logic only** - don't test pandas built-ins

## File Structure
```
project/data_pipeline/
├── features.py          # All 43 features in ~200 lines
└── __init__.py         # Empty

tests/
└── test_features.py    # Basic validation tests
```

## Success Metrics
- **Feature engineering module: <400 lines total** (guideline, not hard limit)
- **Each feature: 1-3 lines of actual calculation**
- **No abstraction layers** - direct pandas operations
- **Readable by junior developers** - no complex patterns
- **Prioritize correctness over line count** - get the requirements right first

## Remember
This is a **data processing pipeline**, not enterprise software. Write code like you're doing a quick analysis in a Jupyter notebook, then clean it up just enough to be reusable.

**MINIMAL. VIABLE. CODE.**