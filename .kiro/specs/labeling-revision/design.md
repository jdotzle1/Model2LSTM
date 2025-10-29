# Design Document - Revised Labeling System for XGBoost Models

## Overview

This document outlines the technical design for implementing a weighted binary labeling system for ES futures trading data. The system will generate 12 new columns (6 label + 6 weight columns) for training 6 specialized XGBoost models based on volatility regimes.

## Architecture

### High-Level Architecture

```
Input DataFrame (OHLCV + Features)
           ↓
    RTH Data Validation
           ↓
    Parallel Mode Processing
    ┌─────────────────────────┐
    │  6 Trading Modes        │
    │  - low_vol_long         │
    │  - normal_vol_long      │
    │  - high_vol_long        │
    │  - low_vol_short        │
    │  - normal_vol_short     │
    │  - high_vol_short       │
    └─────────────────────────┘
           ↓
    Label & Weight Calculation
           ↓
    Output DataFrame (Original + 12 new columns)
```

### Core Components

1. **ModeProcessor**: Handles individual trading mode calculations
2. **LabelCalculator**: Determines win/loss for each bar
3. **WeightCalculator**: Computes quality, velocity, and time decay weights
4. **ValidationEngine**: Ensures data quality and performance
5. **ProgressTracker**: Provides processing updates

## Components and Interfaces

### 1. Trading Mode Configuration

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TradingMode:
    """Configuration for a single trading mode"""
    name: str
    direction: str  # 'long' or 'short'
    stop_ticks: int
    target_ticks: int
    
    @property
    def label_column(self) -> str:
        return f"label_{self.name}"
    
    @property
    def weight_column(self) -> str:
        return f"weight_{self.name}"

# Trading mode definitions
TRADING_MODES = {
    'low_vol_long': TradingMode('low_vol_long', 'long', 6, 12),
    'normal_vol_long': TradingMode('normal_vol_long', 'long', 8, 16),
    'high_vol_long': TradingMode('high_vol_long', 'long', 10, 20),
    'low_vol_short': TradingMode('low_vol_short', 'short', 6, 12),
    'normal_vol_short': TradingMode('normal_vol_short', 'short', 8, 16),
    'high_vol_short': TradingMode('high_vol_short', 'short', 10, 20),
}

# Constants
TICK_SIZE = 0.25  # ES tick size in points
TIMEOUT_SECONDS = 900  # 15 minutes
DECAY_RATE = 0.05  # Monthly decay rate
```

### 2. Core Processing Interface

```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class LabelingEngine(ABC):
    """Abstract base class for labeling engines"""
    
    @abstractmethod
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataframe and return with new columns"""
        pass
    
    @abstractmethod
    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate input dataframe structure"""
        pass

class WeightedLabelingEngine(LabelingEngine):
    """Main implementation of weighted labeling system"""
    
    def __init__(self, chunk_size: int = 100_000):
        self.chunk_size = chunk_size
        self.modes = TRADING_MODES
        
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry point for processing"""
        self.validate_input(df)
        
        # Process in chunks for memory efficiency
        if len(df) > self.chunk_size:
            return self._process_chunked(df)
        else:
            return self._process_single(df)
```

### 3. Label Calculation Component

```python
class LabelCalculator:
    """Handles win/loss determination for trading modes"""
    
    def __init__(self, mode: TradingMode):
        self.mode = mode
        
    def calculate_labels(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate labels, MAE, and seconds to target
        
        Returns:
            labels: Binary array (0 or 1)
            mae_ticks: MAE in ticks for winners (NaN for losers)
            seconds_to_target: Time to target for winners (NaN for losers)
        """
        n_bars = len(df)
        labels = np.zeros(n_bars, dtype=int)
        mae_ticks = np.full(n_bars, np.nan)
        seconds_to_target = np.full(n_bars, np.nan)
        
        # Convert to numpy arrays for speed
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        
        for i in range(n_bars - 1):  # -1 because we need next bar for entry
            result = self._check_single_entry(
                i, opens, highs, lows, n_bars
            )
            
            labels[i] = result['label']
            if result['label'] == 1:  # Winner
                mae_ticks[i] = result['mae_ticks']
                seconds_to_target[i] = result['seconds_to_target']
        
        return labels, mae_ticks, seconds_to_target
    
    def _check_single_entry(self, entry_idx: int, opens: np.ndarray, 
                           highs: np.ndarray, lows: np.ndarray, 
                           n_bars: int) -> Dict[str, Any]:
        """Check single entry for win/loss"""
        
        # Entry price is next bar's open
        if entry_idx + 1 >= n_bars:
            return {'label': 0, 'mae_ticks': np.nan, 'seconds_to_target': np.nan}
            
        entry_price = opens[entry_idx + 1]
        
        # Calculate target and stop prices
        if self.mode.direction == 'long':
            target_price = entry_price + (self.mode.target_ticks * TICK_SIZE)
            stop_price = entry_price - (self.mode.stop_ticks * TICK_SIZE)
        else:  # short
            target_price = entry_price - (self.mode.target_ticks * TICK_SIZE)
            stop_price = entry_price + (self.mode.stop_ticks * TICK_SIZE)
        
        # Look forward from entry bar
        start_idx = entry_idx + 1
        end_idx = min(start_idx + TIMEOUT_SECONDS, n_bars)
        
        worst_adverse = 0.0
        
        for j in range(start_idx, end_idx):
            if self.mode.direction == 'long':
                target_hit = highs[j] >= target_price
                stop_hit = lows[j] <= stop_price
                adverse_move = entry_price - lows[j]
            else:  # short
                target_hit = lows[j] <= target_price
                stop_hit = highs[j] >= stop_price
                adverse_move = highs[j] - entry_price
            
            worst_adverse = max(worst_adverse, adverse_move)
            
            # Check for hits (target first wins in case of same bar)
            if target_hit and stop_hit:
                # Conservative: assume stop hit first
                return {'label': 0, 'mae_ticks': np.nan, 'seconds_to_target': np.nan}
            elif target_hit:
                mae_ticks = worst_adverse / TICK_SIZE
                seconds = j - start_idx
                return {'label': 1, 'mae_ticks': mae_ticks, 'seconds_to_target': seconds}
            elif stop_hit:
                return {'label': 0, 'mae_ticks': np.nan, 'seconds_to_target': np.nan}
        
        # Timeout
        return {'label': 0, 'mae_ticks': np.nan, 'seconds_to_target': np.nan}
```

### 4. Weight Calculation Component

```python
class WeightCalculator:
    """Handles weight calculations for training examples"""
    
    def __init__(self, mode: TradingMode):
        self.mode = mode
        
    def calculate_weights(self, labels: np.ndarray, mae_ticks: np.ndarray, 
                         seconds_to_target: np.ndarray, 
                         timestamps: pd.Series) -> np.ndarray:
        """Calculate final weights for all examples"""
        
        n_samples = len(labels)
        weights = np.ones(n_samples)
        
        # Calculate time decay for all samples
        time_decay = self._calculate_time_decay(timestamps)
        
        # Calculate quality and velocity weights for winners
        winner_mask = labels == 1
        
        if winner_mask.any():
            quality_weights = self._calculate_quality_weights(
                mae_ticks[winner_mask]
            )
            velocity_weights = self._calculate_velocity_weights(
                seconds_to_target[winner_mask]
            )
            
            # Combine weights for winners
            weights[winner_mask] = quality_weights * velocity_weights * time_decay[winner_mask]
        
        # Losers get only time decay
        loser_mask = labels == 0
        weights[loser_mask] = time_decay[loser_mask]
        
        return weights
    
    def _calculate_quality_weights(self, mae_ticks: np.ndarray) -> np.ndarray:
        """Calculate quality weights based on MAE"""
        mae_ratio = mae_ticks / self.mode.stop_ticks
        quality_weights = 2.0 - (1.5 * mae_ratio)
        return np.clip(quality_weights, 0.5, 2.0)
    
    def _calculate_velocity_weights(self, seconds_to_target: np.ndarray) -> np.ndarray:
        """Calculate velocity weights based on speed to target"""
        velocity_weights = 2.0 - (1.5 * (seconds_to_target - 300) / 600)
        return np.clip(velocity_weights, 0.5, 2.0)
    
    def _calculate_time_decay(self, timestamps: pd.Series) -> np.ndarray:
        """Calculate time decay weights"""
        most_recent_date = timestamps.max()
        
        # Calculate months ago for each timestamp
        months_ago = timestamps.apply(
            lambda x: self._months_between(x, most_recent_date)
        ).values
        
        return np.exp(-DECAY_RATE * months_ago)
    
    def _months_between(self, date1: pd.Timestamp, date2: pd.Timestamp) -> int:
        """Calculate months between two dates"""
        return (date2.year - date1.year) * 12 + (date2.month - date1.month)
```

## Data Models

### Input Data Model

```python
from typing import List, Optional
import pandas as pd

class InputDataFrame:
    """Validates and wraps input DataFrame"""
    
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validate()
    
    def validate(self) -> None:
        """Validate input DataFrame structure"""
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            raise ValueError("timestamp column must be datetime type")
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValueError(f"{col} column must be numeric type")
    
    @property
    def size(self) -> int:
        return len(self.df)
    
    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.df['timestamp'].min(), self.df['timestamp'].max()
```

### Output Data Model

```python
class OutputDataFrame:
    """Validates and wraps output DataFrame"""
    
    def __init__(self, df: pd.DataFrame, original_columns: List[str]):
        self.df = df
        self.original_columns = original_columns
        self.validate()
    
    def validate(self) -> None:
        """Validate output DataFrame structure"""
        # Check that all original columns are preserved
        missing_original = set(self.original_columns) - set(self.df.columns)
        if missing_original:
            raise ValueError(f"Missing original columns: {missing_original}")
        
        # Check that all expected new columns are present
        expected_new_columns = []
        for mode in TRADING_MODES.values():
            expected_new_columns.extend([mode.label_column, mode.weight_column])
        
        missing_new = set(expected_new_columns) - set(self.df.columns)
        if missing_new:
            raise ValueError(f"Missing new columns: {missing_new}")
        
        # Validate label columns (must be 0 or 1)
        for mode in TRADING_MODES.values():
            label_col = mode.label_column
            if not self.df[label_col].isin([0, 1]).all():
                raise ValueError(f"{label_col} must contain only 0 or 1")
        
        # Validate weight columns (must be positive)
        for mode in TRADING_MODES.values():
            weight_col = mode.weight_column
            if not (self.df[weight_col] > 0).all():
                raise ValueError(f"{weight_col} must contain only positive values")
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each mode"""
        stats = {}
        
        for mode in TRADING_MODES.values():
            label_col = mode.label_column
            weight_col = mode.weight_column
            
            win_rate = self.df[label_col].mean()
            avg_weight = self.df[weight_col].mean()
            weight_std = self.df[weight_col].std()
            
            stats[mode.name] = {
                'win_rate': win_rate,
                'avg_weight': avg_weight,
                'weight_std': weight_std,
                'total_samples': len(self.df)
            }
        
        return stats
```

## Error Handling

### Exception Hierarchy

```python
class LabelingError(Exception):
    """Base exception for labeling system"""
    pass

class ValidationError(LabelingError):
    """Raised when input validation fails"""
    pass

class ProcessingError(LabelingError):
    """Raised when processing fails"""
    pass

class PerformanceError(LabelingError):
    """Raised when performance targets are not met"""
    pass
```

### Error Handling Strategy

1. **Input Validation**: Fail fast on invalid input data
2. **Processing Errors**: Log and continue with remaining modes
3. **Performance Monitoring**: Track processing speed and memory usage
4. **Graceful Degradation**: Return partial results if some modes fail

## Testing Strategy

### Unit Tests

```python
class TestLabelCalculator:
    """Test label calculation logic"""
    
    def test_long_winner(self):
        """Test long trade that hits target"""
        pass
    
    def test_long_loser(self):
        """Test long trade that hits stop"""
        pass
    
    def test_timeout(self):
        """Test trade that times out"""
        pass
    
    def test_mae_calculation(self):
        """Test MAE calculation accuracy"""
        pass

class TestWeightCalculator:
    """Test weight calculation logic"""
    
    def test_quality_weights(self):
        """Test quality weight calculation"""
        pass
    
    def test_velocity_weights(self):
        """Test velocity weight calculation"""
        pass
    
    def test_time_decay(self):
        """Test time decay calculation"""
        pass
    
    def test_month_calculation(self):
        """Test month calculation across year boundaries"""
        pass
```

### Integration Tests

```python
class TestEndToEnd:
    """Test complete labeling pipeline"""
    
    def test_small_dataset(self):
        """Test on 1000-bar sample"""
        pass
    
    def test_performance_target(self):
        """Test 10M rows within 60 minutes"""
        pass
    
    def test_memory_usage(self):
        """Test memory efficiency"""
        pass
    
    def test_chunked_processing(self):
        """Test chunked vs single processing consistency"""
        pass
```

## Performance Optimizations

### Memory Efficiency

1. **Chunked Processing**: Process large datasets in chunks
2. **Numpy Arrays**: Use numpy for numerical computations
3. **Vectorization**: Minimize Python loops where possible
4. **Memory Monitoring**: Track memory usage during processing

### Speed Optimizations

1. **Parallel Processing**: Process multiple modes simultaneously
2. **Compiled Functions**: Use numba for hot paths if needed
3. **Efficient Data Structures**: Use appropriate data types
4. **Progress Tracking**: Provide user feedback on long operations

### Scalability Considerations

1. **Horizontal Scaling**: Support distributed processing if needed
2. **Incremental Processing**: Support processing new data only
3. **Caching**: Cache intermediate results where appropriate
4. **Resource Management**: Monitor CPU and memory usage

## Deployment Considerations

### Integration Points

1. **Input**: Integrates with existing feature engineering pipeline on EC2
2. **Output**: Produces data ready for XGBoost training on same EC2 instance
3. **Monitoring**: Provides processing statistics and validation
4. **Error Handling**: Graceful failure modes with detailed logging
5. **Deployment**: Simple EC2 deployment, no complex orchestration needed

### Configuration Management

```python
@dataclass
class LabelingConfig:
    """Configuration for labeling system"""
    chunk_size: int = 100_000
    timeout_seconds: int = 900
    decay_rate: float = 0.05
    tick_size: float = 0.25
    performance_target_rows_per_minute: int = 167_000  # 10M in 60 min
    memory_limit_gb: float = 8.0
    enable_parallel_processing: bool = True
    enable_progress_tracking: bool = True
```

This design provides a robust, scalable, and maintainable implementation of the weighted labeling system that meets all requirements while being optimized for performance and reliability.