"""
Weighted Labeling System for XGBoost Models

This module implements a revised labeling system that generates 12 columns 
(6 labels + 6 weights) for training 6 specialized XGBoost models based on 
volatility regimes.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import time
import psutil
import gc


@dataclass
class TradingMode:
    """Configuration for a single trading mode"""
    name: str
    direction: str  # 'long' or 'short'
    stop_ticks: int
    target_ticks: int
    
    @property
    def label_column(self) -> str:
        """Column name for labels"""
        return f"label_{self.name}"
    
    @property
    def weight_column(self) -> str:
        """Column name for weights"""
        return f"weight_{self.name}"


# Trading mode definitions based on volatility regimes
TRADING_MODES = {
    'low_vol_long': TradingMode('low_vol_long', 'long', 6, 12),
    'normal_vol_long': TradingMode('normal_vol_long', 'long', 8, 16),
    'high_vol_long': TradingMode('high_vol_long', 'long', 10, 20),
    'low_vol_short': TradingMode('low_vol_short', 'short', 6, 12),
    'normal_vol_short': TradingMode('normal_vol_short', 'short', 8, 16),
    'high_vol_short': TradingMode('high_vol_short', 'short', 10, 20),
}

# System constants
TICK_SIZE = 0.25  # ES tick size in points
TIMEOUT_SECONDS = 900  # 15 minutes maximum timeout
DECAY_RATE = 0.05  # Monthly decay rate for time decay calculation


# Custom exception classes
class ValidationError(Exception):
    """Raised when input validation fails"""
    pass


class ProcessingError(Exception):
    """Raised when processing fails"""
    pass


class PerformanceError(Exception):
    """Raised when performance targets are not met"""
    pass


class InputDataFrame:
    """Validates and wraps input DataFrame for weighted labeling system"""
    
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    RTH_START = time(7, 30)  # 07:30 CT
    RTH_END = time(15, 0)    # 15:00 CT
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with validation
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Raises:
            ValidationError: If validation fails
        """
        self.df = df.copy()
        self.validate()
    
    def validate(self) -> None:
        """
        Validate input DataFrame structure and content
        
        Raises:
            ValidationError: If any validation check fails
        """
        # Check for empty DataFrame
        if len(self.df) == 0:
            raise ValidationError("DataFrame is empty")
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")
        
        # Validate data types
        self._validate_data_types()
        
        # Validate RTH-only data
        self._validate_rth_data()
        
        # Check for basic data quality
        self._validate_data_quality()
    
    def _validate_data_types(self) -> None:
        """Validate column data types"""
        # Check timestamp column
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            raise ValidationError("timestamp column must be datetime type")
        
        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValidationError(f"{col} column must be numeric type")
    
    def _validate_rth_data(self) -> None:
        """Validate that data is RTH-only (07:30-15:00 CT)"""
        import pytz
        
        # Convert timestamp to Central Time first, then extract time
        timestamps = pd.to_datetime(self.df['timestamp'])
        
        # Handle timezone conversion properly
        if timestamps.dt.tz is None:
            # Assume UTC if no timezone
            timestamps = timestamps.dt.tz_localize(pytz.UTC)
        
        # Convert to Central Time
        central_times = timestamps.dt.tz_convert(pytz.timezone('US/Central'))
        times = central_times.dt.time
        
        # Check if all times are within RTH
        rth_mask = (times >= self.RTH_START) & (times <= self.RTH_END)
        non_rth_count = (~rth_mask).sum()
        
        if non_rth_count > 0:
            raise ValidationError(
                f"Found {non_rth_count} bars outside RTH (07:30-15:00 CT). "
                "Only RTH data is supported."
            )
    
    def _validate_data_quality(self) -> None:
        """Validate basic data quality"""
        # Check for NaN values in required columns
        for col in self.REQUIRED_COLUMNS:
            nan_count = self.df[col].isnull().sum()
            if nan_count > 0:
                raise ValidationError(f"Column '{col}' contains {nan_count} NaN values")
        
        # Check for negative volume
        if (self.df['volume'] < 0).any():
            raise ValidationError("Volume cannot be negative")
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (self.df[col] <= 0).any():
                raise ValidationError(f"Column '{col}' contains zero or negative prices")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (self.df['high'] < self.df['low']) |
            (self.df['high'] < self.df['open']) |
            (self.df['high'] < self.df['close']) |
            (self.df['low'] > self.df['open']) |
            (self.df['low'] > self.df['close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            raise ValidationError(f"Found {invalid_count} bars with invalid OHLC relationships")
    
    @property
    def size(self) -> int:
        """Number of rows in DataFrame"""
        return len(self.df)
    
    @property
    def date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Date range of the data"""
        return self.df['timestamp'].min(), self.df['timestamp'].max()
    
    @property
    def columns(self) -> List[str]:
        """Column names in DataFrame"""
        return list(self.df.columns)


class OutputDataFrame:
    """Validates and wraps output DataFrame with labels and weights"""
    
    def __init__(self, df: pd.DataFrame, original_columns: List[str]):
        """
        Initialize with validation
        
        Args:
            df: Output DataFrame with original + new columns
            original_columns: List of original column names
            
        Raises:
            ValidationError: If validation fails
        """
        self.df = df
        self.original_columns = original_columns
        self.validate()
    
    def validate(self) -> None:
        """
        Validate output DataFrame structure and content
        
        Raises:
            ValidationError: If any validation check fails
        """
        # Check that all original columns are preserved
        missing_original = set(self.original_columns) - set(self.df.columns)
        if missing_original:
            raise ValidationError(f"Missing original columns: {missing_original}")
        
        # Check that all expected new columns are present
        expected_new_columns = []
        for mode in TRADING_MODES.values():
            expected_new_columns.extend([mode.label_column, mode.weight_column])
        
        missing_new = set(expected_new_columns) - set(self.df.columns)
        if missing_new:
            raise ValidationError(f"Missing new columns: {missing_new}")
        
        # Validate label columns (must be 0 or 1)
        for mode in TRADING_MODES.values():
            label_col = mode.label_column
            if not self.df[label_col].isin([0, 1]).all():
                raise ValidationError(f"{label_col} must contain only 0 or 1")
        
        # Validate weight columns (must be positive)
        for mode in TRADING_MODES.values():
            weight_col = mode.weight_column
            if not (self.df[weight_col] > 0).all():
                raise ValidationError(f"{weight_col} must contain only positive values")
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive statistics for each trading mode
        
        Returns:
            Dictionary with statistics per mode including win rates, weight distributions,
            and validation checks as specified in requirement 10.7
        """
        stats = {}
        
        for mode in TRADING_MODES.values():
            label_col = mode.label_column
            weight_col = mode.weight_column
            
            # Basic statistics
            win_rate = self.df[label_col].mean()
            total_winners = self.df[label_col].sum()
            total_samples = len(self.df)
            
            # Weight statistics
            avg_weight = self.df[weight_col].mean()
            weight_std = self.df[weight_col].std()
            min_weight = self.df[weight_col].min()
            max_weight = self.df[weight_col].max()
            
            # Validation checks
            has_nan_labels = self.df[label_col].isnull().any()
            has_nan_weights = self.df[weight_col].isnull().any()
            has_infinite_weights = (~np.isfinite(self.df[weight_col])).any()
            
            # Win rate validation (5-50% range as per requirement 10.5)
            win_rate_reasonable = 0.05 <= win_rate <= 0.50
            
            stats[mode.name] = {
                # Basic metrics
                'win_rate': win_rate,
                'total_winners': int(total_winners),
                'total_samples': total_samples,
                
                # Weight distribution
                'avg_weight': avg_weight,
                'weight_std': weight_std,
                'min_weight': min_weight,
                'max_weight': max_weight,
                
                # Quality checks
                'has_nan_labels': has_nan_labels,
                'has_nan_weights': has_nan_weights,
                'has_infinite_weights': has_infinite_weights,
                'win_rate_reasonable': win_rate_reasonable,
                
                # Validation status
                'validation_passed': (
                    not has_nan_labels and 
                    not has_nan_weights and 
                    not has_infinite_weights and 
                    win_rate_reasonable
                )
            }
        
        return stats
    
    def validate_quality_assurance(self) -> Dict[str, bool]:
        """
        Perform comprehensive quality assurance checks as per requirement 10
        
        Returns:
            Dictionary with validation results for each check
        """
        checks = {}
        
        # Check for NaN or infinite values in output columns (requirement 10.6)
        for mode in TRADING_MODES.values():
            label_col = mode.label_column
            weight_col = mode.weight_column
            
            checks[f'{mode.name}_labels_no_nan'] = not self.df[label_col].isnull().any()
            checks[f'{mode.name}_weights_no_nan'] = not self.df[weight_col].isnull().any()
            checks[f'{mode.name}_weights_no_infinite'] = np.isfinite(self.df[weight_col]).all()
            
            # Check label values are exactly 0 or 1 (requirement 10.1)
            checks[f'{mode.name}_labels_binary'] = self.df[label_col].isin([0, 1]).all()
            
            # Check weight values are positive (requirement 10.2)
            checks[f'{mode.name}_weights_positive'] = (self.df[weight_col] > 0).all()
            
            # Check winning percentages are reasonable (requirement 10.5)
            win_rate = self.df[label_col].mean()
            checks[f'{mode.name}_win_rate_reasonable'] = 0.05 <= win_rate <= 0.50
        
        # Overall validation
        checks['all_validations_passed'] = all(checks.values())
        
        return checks
    
    @property
    def size(self) -> int:
        """Number of rows in DataFrame"""
        return len(self.df)
    
    @property
    def columns(self) -> List[str]:
        """Column names in DataFrame"""
        return list(self.df.columns)


class LabelCalculator:
    """Handles win/loss determination and MAE tracking for individual trading modes"""
    
    def __init__(self, mode: TradingMode, enable_vectorization: bool = True, 
                 roll_detection_threshold: float = 20.0):
        """
        Initialize calculator for a specific trading mode
        
        Args:
            mode: Trading mode configuration
            enable_vectorization: Whether to use vectorized calculations
            roll_detection_threshold: Minimum price change (in points) to detect contract rolls
        
        Args:
            mode: TradingMode configuration
            enable_vectorization: Whether to use numpy vectorization for optimization
        """
        self.mode = mode
        self.enable_vectorization = enable_vectorization
        self.roll_detection_threshold = roll_detection_threshold
    
    def _detect_contract_rolls(self, df: pd.DataFrame) -> np.ndarray:
        """
        Detect contract roll events based on large price jumps
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Boolean array indicating bars affected by contract rolls
        """
        if len(df) < 2:
            return np.zeros(len(df), dtype=bool)
        
        # Calculate price changes between consecutive bars
        price_changes = df['close'].diff().abs()
        
        # Identify roll events (large sudden price jumps)
        roll_events = price_changes > self.roll_detection_threshold
        
        # Mark bars around roll events as affected
        # This includes the bar with the roll and the next few bars
        affected_bars = np.zeros(len(df), dtype=bool)
        
        for i in range(len(roll_events)):
            if roll_events.iloc[i]:
                # Mark current bar and next 5 bars as affected by roll
                start_idx = max(0, i - 1)  # Include previous bar
                end_idx = min(len(df), i + 6)  # Include next 5 bars
                affected_bars[start_idx:end_idx] = True
        
        return affected_bars

    def calculate_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate labels, MAE, and seconds to target for all bars with optional vectorization
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (labels, mae_ticks, seconds_to_target)
            - labels: Binary array (0 or 1) 
            - mae_ticks: MAE in ticks for winners (NaN for losers)
            - seconds_to_target: Time to target for winners (NaN for losers)
        """
        n_bars = len(df)
        labels = np.zeros(n_bars, dtype=int)
        mae_ticks = np.full(n_bars, np.nan)
        seconds_to_target = np.full(n_bars, np.nan)
        
        # Detect contract roll events
        roll_affected_bars = self._detect_contract_rolls(df)
        
        # Convert to numpy arrays for speed
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df['timestamp'].values
        
        if self.enable_vectorization and n_bars > 1000:
            # Use optimized vectorized approach for larger datasets
            return self._calculate_labels_vectorized(df)
        else:
            # Use standard loop-based approach for smaller datasets or when vectorization disabled
            # Process each bar (except last one since we need next bar for entry)
            for i in range(n_bars - 1):
                # Skip bars affected by contract rolls
                if roll_affected_bars[i]:
                    labels[i] = 0  # Mark as non-winner (excluded from training)
                    continue
                
                result = self._check_single_entry(
                    i, opens, highs, lows, timestamps, n_bars, roll_affected_bars
                )
                
                labels[i] = result['label']
                if result['label'] == 1:  # Winner
                    mae_ticks[i] = result['mae_ticks']
                    seconds_to_target[i] = result['seconds_to_target']
            
            return labels, mae_ticks, seconds_to_target
    
    def _calculate_labels_vectorized(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized calculation of labels using numpy operations for better performance
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (labels, mae_ticks, seconds_to_target)
        """
        n_bars = len(df)
        labels = np.zeros(n_bars, dtype=int)
        mae_ticks = np.full(n_bars, np.nan)
        seconds_to_target = np.full(n_bars, np.nan)
        
        # Detect contract roll events
        roll_affected_bars = self._detect_contract_rolls(df)
        
        # Convert to numpy arrays
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df['timestamp'].values
        
        # Pre-calculate entry prices (next bar's open)
        entry_prices = opens[1:]  # Skip first bar since it can't be an entry
        
        # Calculate target and stop prices for all potential entries
        if self.mode.direction == 'long':
            target_prices = entry_prices + (self.mode.target_ticks * TICK_SIZE)
            stop_prices = entry_prices - (self.mode.stop_ticks * TICK_SIZE)
        else:  # short
            target_prices = entry_prices - (self.mode.target_ticks * TICK_SIZE)
            stop_prices = entry_prices + (self.mode.stop_ticks * TICK_SIZE)
        
        # Process entries in batches for memory efficiency
        batch_size = min(10000, len(entry_prices))
        
        for batch_start in range(0, len(entry_prices), batch_size):
            batch_end = min(batch_start + batch_size, len(entry_prices))
            
            # Process this batch of entries
            for i in range(batch_start, batch_end):
                # Skip bars affected by contract rolls
                if roll_affected_bars[i]:
                    labels[i] = 0  # Mark as non-winner (excluded from training)
                    continue
                
                # Still need to check each entry individually due to lookforward logic
                result = self._check_single_entry(
                    i, opens, highs, lows, timestamps, n_bars, roll_affected_bars
                )
                
                labels[i] = result['label']
                if result['label'] == 1:  # Winner
                    mae_ticks[i] = result['mae_ticks']
                    seconds_to_target[i] = result['seconds_to_target']
        
        return labels, mae_ticks, seconds_to_target
    
    def _check_single_entry(self, entry_idx: int, opens: np.ndarray, 
                           highs: np.ndarray, lows: np.ndarray,
                           timestamps: np.ndarray, n_bars: int, 
                           roll_affected_bars: np.ndarray) -> Dict[str, float]:
        """
        Check single entry for win/loss determination with MAE and timing tracking
        
        Args:
            entry_idx: Index of the signal bar
            opens: Array of open prices
            highs: Array of high prices  
            lows: Array of low prices
            timestamps: Array of timestamps
            n_bars: Total number of bars
            
        Returns:
            Dictionary with label, mae_ticks, and seconds_to_target
        """
        # Entry price is next bar's open (entry_idx + 1)
        if entry_idx + 1 >= n_bars:
            return {'label': 0, 'mae_ticks': np.nan, 'seconds_to_target': np.nan}
        
        entry_price = opens[entry_idx + 1]
        entry_time = timestamps[entry_idx + 1]
        
        # Calculate target and stop prices based on direction
        target_price, stop_price = self._calculate_target_stop_prices(entry_price)
        
        # Look forward from entry bar up to timeout
        start_idx = entry_idx + 1
        end_idx = min(start_idx + TIMEOUT_SECONDS, n_bars)
        
        # Track worst adverse excursion
        worst_adverse = 0.0
        
        # Check each bar for target/stop hits
        for j in range(start_idx, end_idx):
            current_time = timestamps[j]
            
            # Skip if this bar is affected by a contract roll
            if roll_affected_bars[j]:
                # If we encounter a roll event, treat as timeout (no win/loss)
                return {'label': 0, 'mae_ticks': np.nan, 'seconds_to_target': np.nan}
            
            # Check for target and stop hits with tick precision
            target_hit, stop_hit = self._check_price_hits(
                highs[j], lows[j], target_price, stop_price
            )
            
            # Update MAE tracking
            adverse_move = self._calculate_adverse_move(
                entry_price, highs[j], lows[j]
            )
            worst_adverse = max(worst_adverse, adverse_move)
            
            # Determine outcome (target first wins in case of same bar)
            if target_hit and stop_hit:
                # Conservative: assume stop hit first if both hit same bar
                return {'label': 0, 'mae_ticks': np.nan, 'seconds_to_target': np.nan}
            elif target_hit:
                # Winner: calculate MAE and timing
                mae_ticks = worst_adverse / TICK_SIZE
                # Convert numpy timedelta to seconds
                time_diff = current_time - entry_time
                if hasattr(time_diff, 'total_seconds'):
                    seconds = time_diff.total_seconds()
                else:
                    # Handle numpy timedelta64
                    seconds = float(time_diff / np.timedelta64(1, 's'))
                return {
                    'label': 1, 
                    'mae_ticks': mae_ticks, 
                    'seconds_to_target': seconds
                }
            elif stop_hit:
                # Loser: stop hit first
                return {'label': 0, 'mae_ticks': np.nan, 'seconds_to_target': np.nan}
        
        # Timeout: neither target nor stop hit within time limit
        return {'label': 0, 'mae_ticks': np.nan, 'seconds_to_target': np.nan}
    
    def _calculate_target_stop_prices(self, entry_price: float) -> Tuple[float, float]:
        """
        Calculate target and stop prices based on trading mode direction
        
        Args:
            entry_price: Entry price (next bar's open)
            
        Returns:
            Tuple of (target_price, stop_price)
        """
        if self.mode.direction == 'long':
            target_price = entry_price + (self.mode.target_ticks * TICK_SIZE)
            stop_price = entry_price - (self.mode.stop_ticks * TICK_SIZE)
        else:  # short
            target_price = entry_price - (self.mode.target_ticks * TICK_SIZE)
            stop_price = entry_price + (self.mode.stop_ticks * TICK_SIZE)
        
        return target_price, stop_price
    
    def _check_price_hits(self, bar_high: float, bar_low: float, 
                         target_price: float, stop_price: float) -> Tuple[bool, bool]:
        """
        Check if target or stop prices were hit with tick precision
        
        Args:
            bar_high: High price of current bar
            bar_low: Low price of current bar
            target_price: Target price level
            stop_price: Stop price level
            
        Returns:
            Tuple of (target_hit, stop_hit)
        """
        if self.mode.direction == 'long':
            target_hit = bar_high >= target_price
            stop_hit = bar_low <= stop_price
        else:  # short
            target_hit = bar_low <= target_price
            stop_hit = bar_high >= stop_price
        
        return target_hit, stop_hit
    
    def _calculate_adverse_move(self, entry_price: float, 
                              bar_high: float, bar_low: float) -> float:
        """
        Calculate adverse price movement for MAE tracking
        
        Args:
            entry_price: Entry price
            bar_high: High price of current bar
            bar_low: Low price of current bar
            
        Returns:
            Adverse move in points (always positive)
        """
        if self.mode.direction == 'long':
            # For long trades, adverse move is how far price went below entry
            adverse_move = max(0.0, entry_price - bar_low)
        else:  # short
            # For short trades, adverse move is how far price went above entry
            adverse_move = max(0.0, bar_high - entry_price)
        
        return adverse_move


class WeightCalculator:
    """Handles weight calculations for training examples based on quality, velocity, and time decay"""
    
    def __init__(self, mode: TradingMode, enable_vectorization: bool = True):
        """
        Initialize calculator for a specific trading mode
        
        Args:
            mode: TradingMode configuration
            enable_vectorization: Whether to use numpy vectorization for optimization
        """
        self.mode = mode
        self.enable_vectorization = enable_vectorization
    
    def calculate_weights(self, labels: np.ndarray, mae_ticks: np.ndarray, 
                         seconds_to_target: np.ndarray, 
                         timestamps: pd.Series) -> np.ndarray:
        """
        Calculate final weights for all examples combining quality, velocity, and time decay
        
        Args:
            labels: Binary array (0 or 1) indicating wins/losses
            mae_ticks: MAE in ticks for winners (NaN for losers)
            seconds_to_target: Time to target for winners (NaN for losers)
            timestamps: Timestamp series for time decay calculation
            
        Returns:
            Array of final weights for each example
        """
        if self.enable_vectorization:
            return self._calculate_weights_vectorized(labels, mae_ticks, seconds_to_target, timestamps)
        else:
            return self._calculate_weights_standard(labels, mae_ticks, seconds_to_target, timestamps)
    
    def _calculate_weights_standard(self, labels: np.ndarray, mae_ticks: np.ndarray, 
                                  seconds_to_target: np.ndarray, 
                                  timestamps: pd.Series) -> np.ndarray:
        """Standard weight calculation method"""
        n_samples = len(labels)
        weights = np.ones(n_samples)
        
        # Calculate time decay for all samples
        time_decay = self._calculate_time_decay(timestamps)
        
        # Calculate quality and velocity weights for winners only
        winner_mask = labels == 1
        
        if winner_mask.any():
            # Get MAE and timing data for winners
            winner_mae = mae_ticks[winner_mask]
            winner_timing = seconds_to_target[winner_mask]
            
            # Calculate component weights
            quality_weights = self._calculate_quality_weights(winner_mae)
            velocity_weights = self._calculate_velocity_weights(winner_timing)
            
            # Combine all weights for winners: quality × velocity × time_decay
            weights[winner_mask] = (
                quality_weights * velocity_weights * time_decay[winner_mask]
            )
        
        # Losers get only time decay (quality=1.0, velocity=1.0)
        loser_mask = labels == 0
        weights[loser_mask] = time_decay[loser_mask]
        
        return weights
    
    def _calculate_weights_vectorized(self, labels: np.ndarray, mae_ticks: np.ndarray, 
                                    seconds_to_target: np.ndarray, 
                                    timestamps: pd.Series) -> np.ndarray:
        """
        Vectorized weight calculation for better performance
        
        Uses numpy operations to calculate all weights simultaneously
        """
        from .performance_monitor import OptimizedCalculations
        
        n_samples = len(labels)
        
        # Calculate time decay for all samples (vectorized)
        most_recent_date = timestamps.max()
        months_ago = timestamps.apply(
            lambda x: self._months_between(x, most_recent_date)
        ).values
        time_decay = np.exp(-DECAY_RATE * months_ago)
        
        # Initialize weights with time decay (applies to all samples)
        weights = time_decay.copy()
        
        # Calculate quality and velocity weights for winners only
        winner_mask = labels == 1
        
        if winner_mask.any():
            # Get data for winners only
            winner_mae = mae_ticks[winner_mask]
            winner_timing = seconds_to_target[winner_mask]
            winner_months = months_ago[winner_mask]
            
            # Use vectorized calculations
            quality_weights, velocity_weights, winner_time_decay = OptimizedCalculations.vectorized_weight_calculations(
                winner_mae, winner_timing, self.mode.stop_ticks, winner_months, DECAY_RATE
            )
            
            # Combine all weights for winners: quality × velocity × time_decay
            weights[winner_mask] = quality_weights * velocity_weights * winner_time_decay
        
        return weights
    
    def _calculate_quality_weights(self, mae_ticks: np.ndarray) -> np.ndarray:
        """
        Calculate quality weights based on MAE performance
        
        Lower MAE = higher quality weight
        Formula: quality_weight = 2.0 - (1.5 × mae_ratio)
        Range: [0.5, 2.0]
        
        Args:
            mae_ticks: Array of MAE values in ticks for winners
            
        Returns:
            Array of quality weights
        """
        # Calculate MAE ratio (MAE / stop distance)
        mae_ratio = mae_ticks / self.mode.stop_ticks
        
        # Apply quality weight formula
        quality_weights = 2.0 - (1.5 * mae_ratio)
        
        # Clip to valid range [0.5, 2.0]
        return np.clip(quality_weights, 0.5, 2.0)
    
    def _calculate_velocity_weights(self, seconds_to_target: np.ndarray) -> np.ndarray:
        """
        Calculate velocity weights based on speed to target
        
        Faster to target = higher velocity weight
        Formula: velocity_weight = 2.0 - (1.5 × (seconds_to_target - 300) / 600)
        Optimal time: 300 seconds (5 minutes)
        Range: [0.5, 2.0]
        
        Args:
            seconds_to_target: Array of time to target in seconds for winners
            
        Returns:
            Array of velocity weights
        """
        # Apply velocity weight formula
        # 300 seconds = optimal time (5 minutes)
        # 600 seconds = normalization factor (10 minutes range)
        velocity_weights = 2.0 - (1.5 * (seconds_to_target - 300) / 600)
        
        # Clip to valid range [0.5, 2.0]
        return np.clip(velocity_weights, 0.5, 2.0)
    
    def _calculate_time_decay(self, timestamps: pd.Series) -> np.ndarray:
        """
        Calculate time decay weights for all samples
        
        More recent data gets higher weights
        Formula: time_decay = exp(-0.05 × months_ago)
        
        Args:
            timestamps: Series of timestamps
            
        Returns:
            Array of time decay weights
        """
        # Find the most recent date in the entire dataset
        most_recent_date = timestamps.max()
        
        # Calculate months ago for each timestamp
        months_ago = timestamps.apply(
            lambda x: self._months_between(x, most_recent_date)
        ).values
        
        # Apply exponential decay
        return np.exp(-DECAY_RATE * months_ago)
    
    def _months_between(self, date1: pd.Timestamp, date2: pd.Timestamp) -> int:
        """
        Calculate months between two dates with proper year boundary handling
        
        Formula: months_ago = (most_recent_year - row_year) × 12 + (most_recent_month - row_month)
        
        Args:
            date1: Earlier date
            date2: Later date (most recent)
            
        Returns:
            Number of months between dates
            
        Examples:
            January 2023 to December 2024: (2024-2023)×12 + (12-1) = 23 months
            March 2024 to March 2024: (2024-2024)×12 + (3-3) = 0 months
            December 2023 to January 2024: (2024-2023)×12 + (1-12) = 1 month
        """
        return (date2.year - date1.year) * 12 + (date2.month - date1.month)


@dataclass
class LabelingConfig:
    """Configuration for the weighted labeling system"""
    chunk_size: int = 100_000
    timeout_seconds: int = TIMEOUT_SECONDS
    decay_rate: float = DECAY_RATE
    tick_size: float = TICK_SIZE
    performance_target_rows_per_minute: int = 167_000  # 10M in 60 min
    memory_limit_gb: float = 8.0
    enable_parallel_processing: bool = True
    enable_progress_tracking: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    progress_update_interval: int = 10_000  # Update every 10K rows


class WeightedLabelingEngine:
    """Main processing engine for the weighted labeling system"""
    
    def __init__(self, config: LabelingConfig = None):
        """
        Initialize the weighted labeling engine
        
        Args:
            config: Configuration object, uses defaults if None
        """
        self.config = config or LabelingConfig()
        self.label_calculators = {}
        self.weight_calculators = {}
        self.performance_monitor = None
        
        # Initialize calculators for all trading modes with vectorization enabled
        enable_vectorization = self.config.enable_memory_optimization
        for mode_name, mode in TRADING_MODES.items():
            self.label_calculators[mode_name] = LabelCalculator(mode, enable_vectorization)
            self.weight_calculators[mode_name] = WeightCalculator(mode, enable_vectorization)
        
        # Initialize performance monitoring if enabled
        if self.config.enable_performance_monitoring:
            from .performance_monitor import PerformanceMonitor
            self.performance_monitor = PerformanceMonitor(
                target_rows_per_minute=self.config.performance_target_rows_per_minute,
                memory_limit_gb=self.config.memory_limit_gb
            )
    
    def process_dataframe(self, df: pd.DataFrame, validate_performance: bool = True) -> pd.DataFrame:
        """
        Process DataFrame through complete weighted labeling pipeline with performance monitoring
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with original columns plus 12 new columns (6 labels + 6 weights)
            
        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
            PerformanceError: If performance targets are not met
        """
        # Validate input
        input_data = InputDataFrame(df)
        
        # Start performance monitoring
        if self.performance_monitor:
            self.performance_monitor.start_monitoring(input_data.size)
        
        if self.config.enable_progress_tracking:
            print(f"Processing {input_data.size:,} rows with weighted labeling system...")
            print(f"Date range: {input_data.date_range[0]} to {input_data.date_range[1]}")
        
        try:
            # Memory optimization: ensure clean start
            if self.config.enable_memory_optimization:
                gc.collect()
            
            # Update progress for validation stage
            if self.performance_monitor:
                self.performance_monitor.update_progress(0, "input_validation")
            
            # Determine processing strategy based on size
            if input_data.size <= self.config.chunk_size:
                # Single-pass processing for smaller datasets
                result_df = self._process_single_chunk(input_data.df)
            else:
                # Chunked processing for larger datasets
                result_df = self._process_chunked(input_data.df)
            
            # Update progress for output validation stage
            if self.performance_monitor:
                self.performance_monitor.update_progress(input_data.size, "output_validation")
            
            # Validate output
            output_data = OutputDataFrame(result_df, input_data.columns)
            
            # Final memory cleanup
            if self.config.enable_memory_optimization:
                gc.collect()
            
            # Finish performance monitoring and validate targets
            if self.performance_monitor:
                metrics = self.performance_monitor.finish_monitoring()
                
                if self.config.enable_progress_tracking:
                    # Show comprehensive statistics (requirement 10.7)
                    stats = output_data.get_statistics()
                    print("\nMode Statistics:")
                    for mode_name, mode_stats in stats.items():
                        print(f"  {mode_name}: {mode_stats['win_rate']:.1%} win rate "
                              f"({mode_stats['total_winners']} winners), "
                              f"avg weight: {mode_stats['avg_weight']:.3f}")
                    
                    # Show quality assurance results
                    qa_results = output_data.validate_quality_assurance()
                    failed_checks = [check for check, passed in qa_results.items() if not passed]
                    
                    if qa_results['all_validations_passed']:
                        print("\n✓ All quality assurance checks passed")
                    else:
                        print(f"\n⚠ Quality assurance issues found:")
                        for failed_check in failed_checks:
                            if failed_check != 'all_validations_passed':
                                print(f"  ❌ {failed_check}")
                                
                        # Don't raise error for QA issues, just warn
                        print("  Note: Processing completed but data quality should be reviewed")
                
                # Validate performance requirements (optional for testing)
                if validate_performance:
                    from .performance_monitor import validate_performance_requirements
                    validate_performance_requirements(self.performance_monitor)
            
            return result_df
            
        except Exception as e:
            # Ensure monitoring is properly finished even on error
            if self.performance_monitor:
                self.performance_monitor.finish_monitoring()
            raise
    
    def _process_single_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process DataFrame in single pass (for smaller datasets)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with labels and weights added
        """
        result_df = df.copy()
        
        # Process all trading modes
        if self.config.enable_parallel_processing:
            # Process modes in parallel (simulated with sequential for now)
            for mode_name in TRADING_MODES.keys():
                self._process_single_mode(result_df, mode_name)
        else:
            # Sequential processing
            for mode_name in TRADING_MODES.keys():
                self._process_single_mode(result_df, mode_name)
        
        return result_df
    
    def _process_chunked(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process DataFrame in chunks with performance monitoring and memory optimization
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with labels and weights added
        """
        n_rows = len(df)
        chunk_size = self.config.chunk_size
        n_chunks = (n_rows + chunk_size - 1) // chunk_size  # Ceiling division
        
        if self.config.enable_progress_tracking:
            print(f"Processing in {n_chunks} chunks of {chunk_size:,} rows each...")
        
        # Initialize result DataFrame
        result_df = df.copy()
        
        # Add empty columns for all modes
        for mode in TRADING_MODES.values():
            result_df[mode.label_column] = 0
            result_df[mode.weight_column] = 1.0
        
        # Process each chunk
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_rows)
            
            # Update performance monitoring
            if self.performance_monitor:
                self.performance_monitor.update_progress(
                    end_idx, f"chunk_{chunk_idx + 1}_of_{n_chunks}"
                )
            
            if self.config.enable_progress_tracking and chunk_idx % 10 == 0:
                current_memory = self.performance_monitor.get_current_memory_gb() if self.performance_monitor else 0
                print(f"  Processing chunk {chunk_idx + 1}/{n_chunks} "
                      f"(rows {start_idx:,}-{end_idx:,}) - Memory: {current_memory:.2f} GB")
            
            # Extract chunk with some overlap for lookforward calculations
            # Need extra rows for the 15-minute timeout lookforward
            extended_end = min(end_idx + TIMEOUT_SECONDS, n_rows)
            chunk_df = df.iloc[start_idx:extended_end].copy()
            
            # Use optimized batch processing if available
            if self.config.enable_memory_optimization and hasattr(self, '_use_optimized_processing'):
                results = self._process_chunk_optimized(chunk_df)
                
                # Update results
                actual_chunk_size = end_idx - start_idx
                for mode_name, mode_results in results.items():
                    mode = TRADING_MODES[mode_name]
                    labels = mode_results['labels'][:actual_chunk_size]
                    
                    # Calculate weights for this chunk
                    weights = self.weight_calculators[mode_name].calculate_weights(
                        labels, mode_results['mae_ticks'][:actual_chunk_size], 
                        mode_results['seconds_to_target'][:actual_chunk_size], 
                        chunk_df['timestamp'].iloc[:actual_chunk_size]
                    )
                    
                    result_df.iloc[start_idx:end_idx, result_df.columns.get_loc(mode.label_column)] = labels
                    result_df.iloc[start_idx:end_idx, result_df.columns.get_loc(mode.weight_column)] = weights
            else:
                # Standard processing
                for mode_name in TRADING_MODES.keys():
                    labels, mae_ticks, seconds_to_target = self.label_calculators[mode_name].calculate_labels(chunk_df)
                    weights = self.weight_calculators[mode_name].calculate_weights(
                        labels, mae_ticks, seconds_to_target, chunk_df['timestamp']
                    )
                    
                    # Only update the actual chunk range (not the extended part)
                    actual_chunk_size = end_idx - start_idx
                    mode = TRADING_MODES[mode_name]
                    result_df.iloc[start_idx:end_idx, result_df.columns.get_loc(mode.label_column)] = labels[:actual_chunk_size]
                    result_df.iloc[start_idx:end_idx, result_df.columns.get_loc(mode.weight_column)] = weights[:actual_chunk_size]
            
            # Memory cleanup after each chunk
            if self.config.enable_memory_optimization and chunk_idx % 10 == 0:
                del chunk_df
                gc.collect()
        
        return result_df
    
    def _process_chunk_optimized(self, chunk_df: pd.DataFrame) -> Dict:
        """
        Process chunk using optimized vectorized calculations
        
        Args:
            chunk_df: DataFrame chunk to process
            
        Returns:
            Dictionary with results for each mode
        """
        from .performance_monitor import OptimizedCalculations
        
        # Prepare mode configurations for batch processing
        mode_configs = {}
        for mode_name, mode in TRADING_MODES.items():
            mode_configs[mode_name] = {
                'direction': mode.direction,
                'stop_ticks': mode.stop_ticks,
                'target_ticks': mode.target_ticks
            }
        
        # Use optimized batch processing
        return OptimizedCalculations.batch_process_entries(
            chunk_df, mode_configs, self.config.timeout_seconds
        )
    
    def _process_single_mode(self, df: pd.DataFrame, mode_name: str) -> None:
        """
        Process a single trading mode and add results to DataFrame
        
        Args:
            df: DataFrame to modify in-place
            mode_name: Name of trading mode to process
        """
        mode = TRADING_MODES[mode_name]
        
        # Calculate labels and tracking data
        labels, mae_ticks, seconds_to_target = self.label_calculators[mode_name].calculate_labels(df)
        
        # Calculate weights
        weights = self.weight_calculators[mode_name].calculate_weights(
            labels, mae_ticks, seconds_to_target, df['timestamp']
        )
        
        # Add columns to DataFrame
        df[mode.label_column] = labels
        df[mode.weight_column] = weights
        
        if self.config.enable_progress_tracking:
            win_rate = labels.mean()
            avg_weight = weights.mean()
            print(f"  {mode_name}: {win_rate:.1%} win rate, avg weight: {avg_weight:.3f}")


def process_weighted_labeling(df: pd.DataFrame, config: LabelingConfig = None) -> pd.DataFrame:
    """
    Convenience function for processing DataFrame with weighted labeling
    
    Args:
        df: Input DataFrame with OHLCV data
        config: Optional configuration, uses defaults if None
        
    Returns:
        DataFrame with original columns plus 12 new columns (6 labels + 6 weights)
    """
    engine = WeightedLabelingEngine(config)
    return engine.process_dataframe(df)