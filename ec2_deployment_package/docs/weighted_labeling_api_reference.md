# Weighted Labeling System - API Reference

## Overview

This document provides comprehensive API reference for the weighted labeling system, including all classes, functions, configuration options, and data structures.

## Core Classes

### WeightedLabelingEngine

Main processing engine for the weighted labeling system.

```python
class WeightedLabelingEngine:
    def __init__(self, config: LabelingConfig = None)
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame
```

#### Constructor

```python
WeightedLabelingEngine(config: LabelingConfig = None)
```

**Parameters:**
- `config` (LabelingConfig, optional): Configuration object. Uses default configuration if None.

**Example:**
```python
from project.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig

# Default configuration
engine = WeightedLabelingEngine()

# Custom configuration
config = LabelingConfig(chunk_size=50_000, memory_limit_gb=4.0)
engine = WeightedLabelingEngine(config)
```

#### process_dataframe()

```python
process_dataframe(df: pd.DataFrame) -> pd.DataFrame
```

Process DataFrame through complete weighted labeling pipeline.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame with OHLCV data

**Returns:**
- `pd.DataFrame`: DataFrame with original columns plus 12 new columns (6 labels + 6 weights)

**Raises:**
- `ValidationError`: If input validation fails
- `ProcessingError`: If processing fails  
- `PerformanceError`: If performance targets are not met

**Required Input Columns:**
- `timestamp` (datetime64): Timestamp for each bar
- `open` (float): Open price
- `high` (float): High price
- `low` (float): Low price
- `close` (float): Close price
- `volume` (int/float): Trading volume

**Generated Output Columns:**
- `label_low_vol_long` (int): Binary labels for low volatility long trades
- `label_normal_vol_long` (int): Binary labels for normal volatility long trades
- `label_high_vol_long` (int): Binary labels for high volatility long trades
- `label_low_vol_short` (int): Binary labels for low volatility short trades
- `label_normal_vol_short` (int): Binary labels for normal volatility short trades
- `label_high_vol_short` (int): Binary labels for high volatility short trades
- `weight_low_vol_long` (float): Training weights for low volatility long trades
- `weight_normal_vol_long` (float): Training weights for normal volatility long trades
- `weight_high_vol_long` (float): Training weights for high volatility long trades
- `weight_low_vol_short` (float): Training weights for low volatility short trades
- `weight_normal_vol_short` (float): Training weights for normal volatility short trades
- `weight_high_vol_short` (float): Training weights for high volatility short trades

**Example:**
```python
import pandas as pd
from project.data_pipeline.weighted_labeling import WeightedLabelingEngine

# Load your data
df = pd.read_parquet('es_futures_data.parquet')

# Process
engine = WeightedLabelingEngine()
df_labeled = engine.process_dataframe(df)

print(f"Added {len(df_labeled.columns) - len(df.columns)} new columns")
```

### LabelingConfig

Configuration dataclass for the weighted labeling system.

```python
@dataclass
class LabelingConfig:
    chunk_size: int = 100_000
    timeout_seconds: int = 900
    decay_rate: float = 0.05
    tick_size: float = 0.25
    performance_target_rows_per_minute: int = 167_000
    memory_limit_gb: float = 8.0
    enable_parallel_processing: bool = True
    enable_progress_tracking: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    progress_update_interval: int = 10_000
```

#### Parameters

**Processing Configuration:**
- `chunk_size` (int): Number of rows to process per chunk for large datasets. Default: 100,000
- `timeout_seconds` (int): Maximum lookforward time in seconds. Default: 900 (15 minutes)

**Performance Configuration:**
- `performance_target_rows_per_minute` (int): Target processing speed. Default: 167,000
- `memory_limit_gb` (float): Memory usage limit in GB. Default: 8.0

**Feature Flags:**
- `enable_parallel_processing` (bool): Enable parallel mode processing. Default: True
- `enable_progress_tracking` (bool): Show progress updates. Default: True
- `enable_performance_monitoring` (bool): Track performance metrics. Default: True
- `enable_memory_optimization` (bool): Use memory optimizations. Default: True

**Progress Configuration:**
- `progress_update_interval` (int): Progress update frequency in rows. Default: 10,000

**Weight Calculation Parameters:**
- `decay_rate` (float): Monthly decay rate for time weights. Default: 0.05
- `tick_size` (float): ES tick size in points. Default: 0.25

**Example:**
```python
from project.data_pipeline.weighted_labeling import LabelingConfig

# Custom configuration
config = LabelingConfig(
    chunk_size=50_000,
    memory_limit_gb=4.0,
    enable_progress_tracking=True,
    timeout_seconds=600  # 10 minutes
)
```

### TradingMode

Configuration for individual trading modes.

```python
@dataclass
class TradingMode:
    name: str
    direction: str  # 'long' or 'short'
    stop_ticks: int
    target_ticks: int
    
    @property
    def label_column(self) -> str
    
    @property
    def weight_column(self) -> str
```

#### Properties

- `label_column`: Returns column name for labels (e.g., "label_low_vol_long")
- `weight_column`: Returns column name for weights (e.g., "weight_low_vol_long")

#### Predefined Trading Modes

```python
TRADING_MODES = {
    'low_vol_long': TradingMode('low_vol_long', 'long', 6, 12),
    'normal_vol_long': TradingMode('normal_vol_long', 'long', 8, 16),
    'high_vol_long': TradingMode('high_vol_long', 'long', 10, 20),
    'low_vol_short': TradingMode('low_vol_short', 'short', 6, 12),
    'normal_vol_short': TradingMode('normal_vol_short', 'short', 8, 16),
    'high_vol_short': TradingMode('high_vol_short', 'short', 10, 20),
}
```

### InputDataFrame

Validates and wraps input DataFrame.

```python
class InputDataFrame:
    def __init__(self, df: pd.DataFrame)
    def validate(self) -> None
    
    @property
    def size(self) -> int
    
    @property
    def date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]
    
    @property
    def columns(self) -> List[str]
```

#### Validation Rules

- **Required Columns**: timestamp, open, high, low, close, volume
- **Data Types**: timestamp must be datetime, OHLCV must be numeric
- **RTH Only**: Data must be within Regular Trading Hours (07:30-15:00 CT)
- **Data Quality**: No NaN values, positive prices, non-negative volume
- **OHLC Relationships**: High ≥ max(Open, Close), Low ≤ min(Open, Close)

### OutputDataFrame

Validates and wraps output DataFrame with statistics.

```python
class OutputDataFrame:
    def __init__(self, df: pd.DataFrame, original_columns: List[str])
    def validate(self) -> None
    def get_statistics(self) -> Dict[str, Dict[str, float]]
    def validate_quality_assurance(self) -> Dict[str, bool]
```

#### get_statistics()

Returns comprehensive statistics for each trading mode.

**Returns:**
```python
{
    'mode_name': {
        'win_rate': float,           # Win rate (0.0 to 1.0)
        'total_winners': int,        # Number of winning trades
        'total_samples': int,        # Total number of samples
        'avg_weight': float,         # Average weight
        'weight_std': float,         # Weight standard deviation
        'min_weight': float,         # Minimum weight
        'max_weight': float,         # Maximum weight
        'validation_passed': bool    # Whether validation passed
    }
}
```

#### validate_quality_assurance()

Performs comprehensive quality checks.

**Returns:**
```python
{
    'mode_name_labels_binary': bool,      # Labels are 0 or 1
    'mode_name_weights_positive': bool,   # Weights are positive
    'mode_name_win_rate_reasonable': bool, # Win rate in 5-50% range
    'all_validations_passed': bool        # Overall validation status
}
```

## Convenience Functions

### process_weighted_labeling()

```python
def process_weighted_labeling(df: pd.DataFrame, 
                            config: LabelingConfig = None) -> pd.DataFrame
```

Convenience function for processing DataFrame with weighted labeling.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame with OHLCV data
- `config` (LabelingConfig, optional): Configuration, uses defaults if None

**Returns:**
- `pd.DataFrame`: DataFrame with original columns plus 12 new columns

**Example:**
```python
from project.data_pipeline.weighted_labeling import process_weighted_labeling

# Simple usage
df_labeled = process_weighted_labeling(df)

# With custom configuration
config = LabelingConfig(chunk_size=50_000)
df_labeled = process_weighted_labeling(df, config)
```

## Validation Utilities

### run_comprehensive_validation()

```python
def run_comprehensive_validation(df: pd.DataFrame, 
                               df_original: Optional[pd.DataFrame] = None,
                               print_reports: bool = True) -> Dict[str, Dict]
```

Run comprehensive validation suite on weighted labeling results.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with weighted labeling results
- `df_original` (pd.DataFrame, optional): Original labeling results for comparison
- `print_reports` (bool): Whether to print detailed reports

**Returns:**
```python
{
    'label_distributions': Dict,      # Label distribution analysis
    'weight_distributions': Dict,     # Weight distribution analysis  
    'data_quality': Dict,            # Data quality checks
    'original_comparison': Dict,      # Comparison with original system
    'overall_validation': {
        'passed': bool,              # Overall validation status
        'timestamp': str,            # Validation timestamp
        'dataset_size': int,         # Number of rows validated
        'summary': {
            'label_validation_passed': bool,
            'weight_validation_passed': bool,
            'data_quality_passed': bool,
            'xgboost_ready': bool
        }
    }
}
```

**Example:**
```python
from project.data_pipeline.validation_utils import run_comprehensive_validation

# Run validation
results = run_comprehensive_validation(df_labeled)

# Check if validation passed
if results['overall_validation']['passed']:
    print("✅ Ready for XGBoost training")
else:
    print("❌ Issues found - check validation report")
```

### Individual Validators

#### LabelDistributionValidator

```python
class LabelDistributionValidator:
    def validate_label_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]
    def print_label_distribution_report(self, results: Dict[str, Dict[str, float]]) -> None
```

#### WeightDistributionValidator

```python
class WeightDistributionValidator:
    def validate_weight_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]
    def print_weight_distribution_report(self, results: Dict[str, Dict[str, float]]) -> None
```

#### DataQualityChecker

```python
class DataQualityChecker:
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Dict]
    def print_data_quality_report(self, results: Dict[str, Dict]) -> None
```

## Performance Monitoring

### PerformanceMonitor

```python
class PerformanceMonitor:
    def __init__(self, target_rows_per_minute: int = 167_000, 
                 memory_limit_gb: float = 8.0)
    def start_monitoring(self, total_rows: int) -> None
    def update_progress(self, rows_processed: int, stage: str = None) -> None
    def finish_monitoring(self) -> PerformanceMetrics
    def get_current_memory_gb(self) -> float
    def validate_performance_target(self, total_rows: int = 10_000_000) -> Dict[str, bool]
    def print_performance_report(self) -> None
```

### PerformanceMetrics

```python
@dataclass
class PerformanceMetrics:
    start_time: float
    end_time: Optional[float] = None
    rows_processed: int = 0
    memory_usage_mb: List[float] = field(default_factory=list)
    processing_stages: Dict[str, float] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> float
    
    @property
    def rows_per_minute(self) -> float
    
    @property
    def peak_memory_mb(self) -> float
    
    @property
    def peak_memory_gb(self) -> float
```

### performance_context()

```python
@contextmanager
def performance_context(monitor: PerformanceMonitor, total_rows: int):
```

Context manager for automatic performance monitoring.

**Example:**
```python
from project.data_pipeline.performance_monitor import PerformanceMonitor, performance_context

monitor = PerformanceMonitor()

with performance_context(monitor, len(df)) as perf:
    df_labeled = process_weighted_labeling(df)
    # Performance report printed automatically
```

## Exception Classes

### ValidationError

```python
class ValidationError(Exception):
    """Raised when input validation fails"""
```

**Common Causes:**
- Missing required columns
- Invalid data types
- Non-RTH data
- Invalid OHLC relationships
- NaN values in required columns

### ProcessingError

```python
class ProcessingError(Exception):
    """Raised when processing fails"""
```

**Common Causes:**
- Memory allocation failures
- Calculation errors
- Data corruption during processing

### PerformanceError

```python
class PerformanceError(Exception):
    """Raised when performance targets are not met"""
```

**Common Causes:**
- Processing speed below target
- Memory usage above limit
- Projected time exceeds 60 minutes for 10M rows

## Constants

### System Constants

```python
TICK_SIZE = 0.25              # ES tick size in points
TIMEOUT_SECONDS = 900         # Default timeout (15 minutes)
DECAY_RATE = 0.05            # Default monthly decay rate
```

### Trading Mode Constants

```python
TRADING_MODES = {
    'low_vol_long': TradingMode('low_vol_long', 'long', 6, 12),
    'normal_vol_long': TradingMode('normal_vol_long', 'long', 8, 16),
    'high_vol_long': TradingMode('high_vol_long', 'long', 10, 20),
    'low_vol_short': TradingMode('low_vol_short', 'short', 6, 12),
    'normal_vol_short': TradingMode('normal_vol_short', 'short', 8, 16),
    'high_vol_short': TradingMode('high_vol_short', 'short', 10, 20),
}
```

## Usage Patterns

### Basic Usage Pattern

```python
# 1. Import
from project.data_pipeline.weighted_labeling import process_weighted_labeling

# 2. Load data
df = pd.read_parquet('your_data.parquet')

# 3. Process
df_labeled = process_weighted_labeling(df)

# 4. Validate
from project.data_pipeline.validation_utils import run_comprehensive_validation
validation_results = run_comprehensive_validation(df_labeled)

# 5. Use for XGBoost training
if validation_results['overall_validation']['passed']:
    # Prepare for training...
```

### Advanced Usage Pattern

```python
# 1. Custom configuration
config = LabelingConfig(
    chunk_size=50_000,
    memory_limit_gb=4.0,
    enable_performance_monitoring=True
)

# 2. Create engine
engine = WeightedLabelingEngine(config)

# 3. Process with monitoring
df_labeled = engine.process_dataframe(df)

# 4. Check performance
if engine.performance_monitor:
    metrics = engine.performance_monitor.metrics
    print(f"Speed: {metrics.rows_per_minute:,.0f} rows/min")
    print(f"Memory: {metrics.peak_memory_gb:.2f} GB")

# 5. Detailed validation
from project.data_pipeline.validation_utils import (
    LabelDistributionValidator,
    WeightDistributionValidator,
    DataQualityChecker
)

label_validator = LabelDistributionValidator()
label_results = label_validator.validate_label_distributions(df_labeled)
label_validator.print_label_distribution_report(label_results)
```

### Error Handling Pattern

```python
from project.data_pipeline.weighted_labeling import (
    WeightedLabelingEngine,
    ValidationError,
    ProcessingError,
    PerformanceError
)

try:
    engine = WeightedLabelingEngine()
    df_labeled = engine.process_dataframe(df)
    
except ValidationError as e:
    print(f"Input validation failed: {e}")
    # Fix input data issues
    
except ProcessingError as e:
    print(f"Processing failed: {e}")
    # Check system resources, data quality
    
except PerformanceError as e:
    print(f"Performance target not met: {e}")
    # Optimize configuration or upgrade hardware
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # General error handling
```

This API reference provides complete documentation for all public interfaces in the weighted labeling system.