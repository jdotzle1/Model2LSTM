import pandas as pd
import numpy as np

# ============================================
# MAIN FEATURE CREATION FUNCTION
# ============================================

def create_all_features(df):
    """
    Create all 55 features for Model 2
    
    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        
    Returns:
        df: DataFrame with added feature columns
    """
    
    print("Creating features...")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("  Creating volume features...")
    df = create_volume_features(df)
    
    print("  Creating price context features...")
    df = create_price_context_features(df)
    
    print("  Creating swing features...")
    df = create_swing_features(df)
    
    print("  Creating return features...")
    df = create_return_features(df)
    
    print("  Creating volatility features...")
    df = create_volatility_features(df)
    
    print("  Creating microstructure features...")
    df = create_microstructure_features(df)
    
    print("  Creating time features...")
    df = create_time_features(df)
    
    print(f"✓ Created {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} feature columns")
    
    return df


# ============================================
# VOLUME FEATURES
# ============================================

def create_volume_features(df):
    """Volume ratios, percentiles, z-scores"""
    
    # Volume ratios (relative to recent average)
    df['volume_ratio_30s'] = df['volume'] / df['volume'].rolling(30).mean()
    df['volume_ratio_300s'] = df['volume'] / df['volume'].rolling(300).mean()
    
    # Volume percentile rank (intraday)
    df['volume_pct_rank'] = df.groupby(df['timestamp'].dt.date)['volume'].rank(pct=True)
    
    # Volume momentum
    df['volume_change_5s'] = df['volume'].pct_change(5)
    
    # Volume z-score
    rolling_mean = df['volume'].rolling(300).mean()
    rolling_std = df['volume'].rolling(300).std()
    df['volume_zscore_300s'] = (df['volume'] - rolling_mean) / rolling_std
    
    return df


# ============================================
# PRICE CONTEXT FEATURES
# ============================================

def create_price_context_features(df):
    """VWAP, RTH levels, distances"""
    
    # VWAP (reset daily)
    df['date'] = df['timestamp'].dt.date
    df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    df['cum_price_volume'] = df.groupby('date').apply(
        lambda x: (x['close'] * x['volume']).cumsum()
    ).reset_index(level=0, drop=True)
    df['vwap'] = df['cum_price_volume'] / df['cum_volume']
    
    # Distance from VWAP
    df['distance_from_vwap'] = df['close'] - df['vwap']
    df['distance_from_vwap_pct'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    
    # RTH high/low (session high/low so far)
    df['rth_high'] = df.groupby('date')['high'].cummax()
    df['rth_low'] = df.groupby('date')['low'].cummin()
    
    # Distance from RTH levels
    df['distance_from_rth_high'] = df['close'] - df['rth_high']
    df['distance_from_rth_low'] = df['close'] - df['rth_low']
    
    # Position in RTH range (0 = at low, 1 = at high)
    df['rth_range'] = df['rth_high'] - df['rth_low']
    df['position_in_rth_range'] = (df['close'] - df['rth_low']) / df['rth_range']
    
    # Cleanup temp columns
    df.drop(['date', 'cum_volume', 'cum_price_volume'], axis=1, inplace=True)
    
    return df


# ============================================
# SWING HIGH/LOW FEATURES
# ============================================

def create_swing_features(df):
    """Swing highs/lows and distances"""
    
    # 5-second swing highs/lows
    df = identify_swing_points(df, lookback=5, suffix='5s')
    
    # 60-second swing highs/lows
    df = identify_swing_points(df, lookback=60, suffix='60s')
    
    return df


def identify_swing_points(df, lookback=5, suffix='5s'):
    """Identify swing highs and lows"""
    
    swing_highs = []
    swing_lows = []
    
    for i in range(len(df)):
        # Check if we have enough data on both sides
        if i < lookback or i >= len(df) - lookback:
            swing_highs.append(np.nan)
            swing_lows.append(np.nan)
            continue
        
        # Get window
        window_high = df['high'].iloc[i-lookback:i+lookback+1]
        window_low = df['low'].iloc[i-lookback:i+lookback+1]
        
        # Check if current bar is swing high
        if df['high'].iloc[i] == window_high.max():
            swing_highs.append(df['high'].iloc[i])
        else:
            swing_highs.append(np.nan)
        
        # Check if current bar is swing low
        if df['low'].iloc[i] == window_low.min():
            swing_lows.append(df['low'].iloc[i])
        else:
            swing_lows.append(np.nan)
    
    df[f'swing_high_{suffix}'] = swing_highs
    df[f'swing_low_{suffix}'] = swing_lows
    
    # Forward fill to get last swing high/low
    df[f'last_swing_high_{suffix}'] = df[f'swing_high_{suffix}'].fillna(method='ffill')
    df[f'last_swing_low_{suffix}'] = df[f'swing_low_{suffix}'].fillna(method='ffill')
    
    # Distance from swing levels
    df[f'distance_from_swing_high_{suffix}'] = df['close'] - df[f'last_swing_high_{suffix}']
    df[f'distance_from_swing_low_{suffix}'] = df['close'] - df[f'last_swing_low_{suffix}']
    
    # Position in swing range
    swing_range = df[f'last_swing_high_{suffix}'] - df[f'last_swing_low_{suffix}']
    df[f'position_in_swing_range_{suffix}'] = (df['close'] - df[f'last_swing_low_{suffix}']) / swing_range
    
    # Bars since last swing
    df[f'bars_since_swing_high_{suffix}'] = calculate_bars_since_event(df[f'swing_high_{suffix}'])
    
    return df


def calculate_bars_since_event(series):
    """Calculate bars since last non-NaN value"""
    bars_since = []
    counter = 0
    
    for val in series:
        if pd.notna(val):
            counter = 0
        else:
            counter += 1
        bars_since.append(counter)
    
    return bars_since


# ============================================
# RETURN FEATURES
# ============================================

def create_return_features(df):
    """Returns at multiple timeframes"""
    
    df['return_1s'] = df['close'].pct_change(1)
    df['return_5s'] = df['close'].pct_change(5)
    df['return_10s'] = df['close'].pct_change(10)
    df['return_30s'] = df['close'].pct_change(30)
    df['return_60s'] = df['close'].pct_change(60)
    df['return_300s'] = df['close'].pct_change(300)
    
    return df


# ============================================
# VOLATILITY FEATURES
# ============================================

def create_volatility_features(df):
    """ATR and realized volatility"""
    
    # Calculate ATR
    df = calculate_atr(df)
    
    # Realized volatility (std of returns)
    df['realized_vol_30s'] = df['return_1s'].rolling(30).std()
    df['realized_vol_60s'] = df['return_1s'].rolling(60).std()
    df['realized_vol_300s'] = df['return_1s'].rolling(300).std()
    
    # Volatility ratio
    df['vol_ratio_30_300'] = df['realized_vol_30s'] / df['realized_vol_300s']
    
    # ATR as percentage of price
    df['atr_pct'] = df['atr_30s'] / df['close'] * 100
    
    return df


def calculate_atr(df):
    """Calculate Average True Range"""
    
    df['prev_close'] = df['close'].shift(1)
    
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    df['atr_30s'] = df['true_range'].rolling(30).mean()
    df['atr_60s'] = df['true_range'].rolling(60).mean()
    df['atr_300s'] = df['true_range'].rolling(300).mean()
    
    # Cleanup temp columns
    df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
    
    return df


# ============================================
# MICROSTRUCTURE FEATURES
# ============================================

def create_microstructure_features(df):
    """Bar characteristics and tick direction"""
    
    # Bar range
    df['bar_range'] = df['high'] - df['low']
    df['bar_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    
    # Bar body
    df['bar_body'] = abs(df['close'] - df['open'])
    df['body_ratio'] = df['bar_body'] / df['bar_range']
    
    # Tick direction
    df['tick_direction'] = np.sign(df['close'].diff())
    
    # Consecutive up/down ticks
    df['consecutive_up'] = calculate_consecutive_values(df['tick_direction'], 1)
    df['consecutive_down'] = calculate_consecutive_values(df['tick_direction'], -1)
    
    # Net ticks over window
    df['net_ticks_60s'] = df['tick_direction'].rolling(60).sum()
    
    return df


def calculate_consecutive_values(series, target_value):
    """Count consecutive occurrences of target_value"""
    consecutive = []
    counter = 0
    
    for val in series:
        if val == target_value:
            counter += 1
        else:
            counter = 0
        consecutive.append(counter)
    
    return consecutive


# ============================================
# TIME FEATURES
# ============================================

def create_time_features(df):
    """Time-based features"""
    
    # Basic time components
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Seconds since RTH open (9:30 AM ET = 14:30 UTC)
    # Assuming timestamp is in UTC
    df['seconds_since_open'] = (
        (df['hour'] - 14) * 3600 + 
        (df['minute'] - 30) * 60 + 
        df['timestamp'].dt.second
    )
    
    # Seconds until RTH close (4:00 PM ET = 21:00 UTC)
    df['seconds_until_close'] = (
        (21 - df['hour']) * 3600 - 
        df['minute'] * 60 - 
        df['timestamp'].dt.second
    )
    
    # Session period flags
    df['is_opening'] = ((df['hour'] == 14) & (df['minute'] >= 30)) | ((df['hour'] == 15) & (df['minute'] < 0))
    df['is_lunch'] = ((df['hour'] >= 16) & (df['hour'] < 19))
    df['is_close'] = (df['hour'] == 20)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    return df