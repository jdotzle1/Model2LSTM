#!/usr/bin/env python3
"""
Session-Aware Utilities for RTH Data Processing

Handles rolling calculations and lookforward logic that respect trading session boundaries.
Prevents contamination between trading days when overnight data has been filtered out.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


def session_aware_rolling(df: pd.DataFrame, column: str, window: int, 
                         operation: str = 'mean', session_col: str = 'session_number') -> pd.Series:
    """
    Perform rolling calculations that respect session boundaries
    
    Args:
        df: DataFrame with session information
        column: Column to calculate rolling statistic on
        window: Rolling window size
        operation: 'mean', 'std', 'min', 'max', 'sum'
        session_col: Column containing session identifiers
        
    Returns:
        Series with session-aware rolling calculation
    """
    if session_col not in df.columns:
        # Fallback to regular rolling if no session info
        print(f"Warning: {session_col} not found, using regular rolling calculation")
        return getattr(df[column].rolling(window), operation)()
    
    result = pd.Series(index=df.index, dtype=float)
    
    # Process each session separately
    for session_id in df[session_col].unique():
        session_mask = df[session_col] == session_id
        session_data = df.loc[session_mask, column]
        
        # Apply rolling calculation within session
        session_result = getattr(session_data.rolling(window), operation)()
        result.loc[session_mask] = session_result
    
    return result


def session_aware_lookforward(df: pd.DataFrame, entry_idx: int, max_seconds: int,
                              session_col: str = 'session_number') -> int:
    """
    Calculate safe lookforward range that doesn't cross session boundaries
    
    Args:
        df: DataFrame with session information
        entry_idx: Index of entry bar
        max_seconds: Maximum lookforward in seconds (bars)
        session_col: Column containing session identifiers
        
    Returns:
        Actual lookforward range (may be less than max_seconds)
    """
    if session_col not in df.columns:
        # Fallback to regular lookforward if no session info
        return min(max_seconds, len(df) - entry_idx - 1)
    
    entry_session = df.iloc[entry_idx][session_col]
    
    # Find end of current session
    session_mask = df[session_col] == entry_session
    session_indices = df.index[session_mask]
    
    # Get the last index of current session
    session_end_idx = session_indices[-1]
    
    # Calculate safe lookforward range
    max_lookforward_idx = entry_idx + max_seconds
    safe_end_idx = min(max_lookforward_idx, session_end_idx)
    
    return safe_end_idx - entry_idx


def validate_session_boundaries(df: pd.DataFrame, session_col: str = 'session_number') -> dict:
    """
    Validate that session boundaries are properly identified
    
    Returns:
        Dictionary with validation results
    """
    if session_col not in df.columns:
        return {'error': f'{session_col} column not found'}
    
    results = {
        'total_sessions': df[session_col].nunique(),
        'total_bars': len(df),
        'avg_bars_per_session': len(df) / df[session_col].nunique(),
        'session_gaps_detected': 0
    }
    
    # Check for time gaps between sessions
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()
        
        # Gaps longer than 5 minutes (300 seconds) indicate session boundaries
        large_gaps = time_diffs > pd.Timedelta(minutes=5)
        results['session_gaps_detected'] = large_gaps.sum()
    
    return results


def add_session_features(df: pd.DataFrame, session_col: str = 'session_number') -> pd.DataFrame:
    """
    Add session-related features that help with modeling
    
    Args:
        df: DataFrame with session information
        session_col: Column containing session identifiers
        
    Returns:
        DataFrame with additional session features
    """
    if session_col not in df.columns:
        print(f"Warning: {session_col} not found, skipping session features")
        return df
    
    df = df.copy()
    
    # Time within session (minutes since session start)
    df['minutes_since_session_start'] = df.groupby(session_col).cumcount() / 60.0
    
    # Time until session end (minutes until session end)
    session_lengths = df.groupby(session_col).size()
    df['session_length'] = df[session_col].map(session_lengths)
    df['minutes_until_session_end'] = (df['session_length'] - df.groupby(session_col).cumcount()) / 60.0
    
    # Session progress (0.0 = start, 1.0 = end)
    df['session_progress'] = df.groupby(session_col).cumcount() / (df['session_length'] - 1)
    
    # Clean up temporary column
    df.drop('session_length', axis=1, inplace=True)
    
    return df


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with session boundaries
    dates = pd.date_range('2024-01-01 07:30:00', periods=1000, freq='1s')
    
    # Create artificial session breaks (simulate day boundaries)
    session_breaks = [0, 450, 900]  # Sessions of ~7.5 hours each
    session_numbers = []
    current_session = 1
    
    for i in range(len(dates)):
        if i in session_breaks[1:]:
            current_session += 1
        session_numbers.append(current_session)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': 4750 + np.random.randn(1000).cumsum() * 0.5,
        'volume': np.random.randint(100, 2000, 1000),
        'session_number': session_numbers
    })
    
    # Test session-aware rolling
    df['regular_ma'] = df['close'].rolling(30).mean()
    df['session_ma'] = session_aware_rolling(df, 'close', 30, 'mean')
    
    # Test lookforward
    test_idx = 400
    regular_lookforward = min(900, len(df) - test_idx - 1)
    session_lookforward = session_aware_lookforward(df, test_idx, 900)
    
    print(f"Regular lookforward: {regular_lookforward} bars")
    print(f"Session-aware lookforward: {session_lookforward} bars")
    
    # Validate sessions
    validation = validate_session_boundaries(df)
    print(f"Session validation: {validation}")