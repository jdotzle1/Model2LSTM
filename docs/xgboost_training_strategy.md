# XGBoost Training Strategy for ES Trading Models

## Overview
Train 6 specialized XGBoost models (one per volatility-based trading mode) to predict optimal entry timing for ES futures trades using weighted binary classification.

## Why XGBoost Over LSTM?

### ✅ **XGBoost Advantages for Trading**
1. **Tabular Data Excellence**: XGBoost is specifically designed for structured/tabular data
2. **Feature Interpretability**: Clear feature importance rankings for each model
3. **Faster Training**: Minutes vs hours for LSTM on same dataset
4. **Less Data Required**: Works well with smaller datasets, no sequence requirements
5. **Robust to Overfitting**: Built-in regularization and early stopping
6. **Production Ready**: Faster inference, smaller model files
7. **Hyperparameter Stability**: Less sensitive to hyperparameter tuning

### ❌ **LSTM Disadvantages for This Use Case**
1. **Sequence Dependency**: Requires careful sequence construction and padding
2. **Data Hungry**: Needs much larger datasets for good performance
3. **Black Box**: Harder to interpret which features drive decisions
4. **Training Complexity**: GPU requirements, longer training times
5. **Overfitting Risk**: Easy to overfit on financial time series
6. **Deployment Overhead**: Larger models, more complex serving infrastructure

## Model Architecture

### 6 Independent XGBoost Models (Volatility-Based)

```python
models = {
    'low_vol_long': {
        'label_column': 'label_low_vol_long',
        'weight_column': 'weight_low_vol_long',
        'target_ticks': 12,
        'stop_ticks': 6,
        'direction': 'long'
    },
    'normal_vol_long': {
        'label_column': 'label_normal_vol_long',
        'weight_column': 'weight_normal_vol_long', 
        'target_ticks': 16,
        'stop_ticks': 8,
        'direction': 'long'
    },
    'high_vol_long': {
        'label_column': 'label_high_vol_long',
        'weight_column': 'weight_high_vol_long',
        'target_ticks': 20, 
        'stop_ticks': 10,
        'direction': 'long'
    },
    'low_vol_short': {
        'label_column': 'label_low_vol_short',
        'weight_column': 'weight_low_vol_short',
        'target_ticks': 12,
        'stop_ticks': 6, 
        'direction': 'short'
    },
    'normal_vol_short': {
        'label_column': 'label_normal_vol_short',
        'weight_column': 'weight_normal_vol_short',
        'target_ticks': 16,
        'stop_ticks': 8,
        'direction': 'short'
    },
    'high_vol_short': {
        'label_column': 'label_high_vol_short',
        'weight_column': 'weight_high_vol_short',
        'target_ticks': 20,
        'stop_ticks': 10,
        'direction': 'short'
    }
}
```

## Training Data Preparation

### Weighted Binary Classification
Use the new weighted labeling system directly:

```python
def prepare_weighted_xgboost_data(df, label_column, weight_column):
    """
    Prepare data for weighted XGBoost training
    
    New labeling system:
    1 = Winner (target hit first)
    0 = Loser (stop hit first or timeout)
    
    Weights incorporate:
    - Quality weight (based on MAE)
    - Velocity weight (based on speed to target)  
    - Time decay (recent data weighted higher)
    """
    # Labels are already binary (0 or 1), no conversion needed
    df_clean = df.copy()
    
    # Validate labels and weights
    assert df_clean[label_column].isin([0, 1]).all(), "Labels must be 0 or 1"
    assert (df_clean[weight_column] > 0).all(), "Weights must be positive"
    
    return df_clean[label_column], df_clean[weight_column]
```

### Weight System Benefits
The new weighting system provides significant advantages:

1. **Quality Weighting**: Better entries (lower MAE) get higher importance
2. **Velocity Weighting**: Faster-moving trades prioritized  
3. **Time Decay**: Recent market conditions weighted higher
4. **Balanced Learning**: Winners and losers both contribute meaningfully

### Feature Selection
Use all 43 engineered features:
- Volume features (4)
- Price context features (5) 
- Consolidation features (10)
- Return features (5)
- Volatility features (6)
- Microstructure features (6)
- Time features (7)

### Train/Validation/Test Split

```python
# Chronological split (no random shuffling for time series)
def create_time_splits(df, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    """
    Create chronological splits for time series data
    """
    n = len(df)
    
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end] 
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

# Example split for 15 years of data:
# Train: 2010-2020 (10.5 years)
# Validation: 2021-2022 (2.25 years) 
# Test: 2023-2025 (2.25 years)
```

## XGBoost Hyperparameters

### Starting Configuration
```python
xgb_params = {
    # Core parameters
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',  # Fast histogram-based algorithm
    
    # Model complexity
    'max_depth': 6,         # Start conservative
    'min_child_weight': 1,
    'subsample': 0.8,       # Row sampling
    'colsample_bytree': 0.8, # Feature sampling
    
    # Learning rate
    'eta': 0.1,             # Learning rate
    'n_estimators': 1000,   # Will use early stopping
    
    # Regularization  
    'reg_alpha': 0.1,       # L1 regularization
    'reg_lambda': 1.0,      # L2 regularization
    
    # Other
    'random_state': 42,
    'n_jobs': -1,           # Use all CPU cores
    'verbosity': 1
}
```

### Hyperparameter Tuning Strategy
1. **Start Simple**: Use default parameters first
2. **Early Stopping**: Monitor validation AUC, stop when no improvement
3. **Grid Search**: Tune max_depth, eta, regularization
4. **Profile-Specific**: Each model may need different parameters

## Training Process

### 1. Data Loading and Preparation
```python
# Load processed dataset
df = pd.read_parquet('processed_es_dataset.parquet')

# Feature columns (43 features)
feature_cols = [col for col in df.columns if col not in 
               ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + 
               [col for col in df.columns if '_label' in col or '_outcome' in col]]

print(f"Training with {len(feature_cols)} features")
```

### 2. Train Each Model with Sample Weights
```python
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

trained_models = {}

for model_name, config in models.items():
    print(f"\nTraining {model_name}...")
    
    # Prepare weighted data for this mode
    label_col = config['label_column']
    weight_col = config['weight_column']
    
    # Split data chronologically
    train_df, val_df, test_df = create_time_splits(df)
    
    # Prepare features, labels, and weights
    X_train = train_df[feature_cols]
    y_train, sample_weights_train = prepare_weighted_xgboost_data(
        train_df, label_col, weight_col
    )
    
    X_val = val_df[feature_cols] 
    y_val, sample_weights_val = prepare_weighted_xgboost_data(
        val_df, label_col, weight_col
    )
    
    # Train XGBoost model with sample weights
    model = xgb.XGBClassifier(**xgb_params)
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,  # Key addition: use sample weights
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[sample_weights_val],  # Weighted validation too
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluate
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred, sample_weight=sample_weights_val)
    
    print(f"Weighted Validation AUC: {val_auc:.4f}")
    
    # Report weight statistics
    win_rate = y_train.mean()
    avg_winner_weight = sample_weights_train[y_train == 1].mean()
    avg_loser_weight = sample_weights_train[y_train == 0].mean()
    
    print(f"Win rate: {win_rate:.3f}")
    print(f"Avg winner weight: {avg_winner_weight:.3f}")
    print(f"Avg loser weight: {avg_loser_weight:.3f}")
    
    # Store model
    trained_models[model_name] = {
        'model': model,
        'config': config,
        'val_auc': val_auc,
        'win_rate': win_rate,
        'feature_importance': model.feature_importances_
    }
```

### 3. Model Evaluation
```python
# Evaluate each model on test set
for model_name, model_info in trained_models.items():
    model = model_info['model']
    
    # Test predictions
    test_pred = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    
    print(f"{model_name} Test AUC: {test_auc:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Top 10 features for {model_name}:")
    print(importance_df.head(10))
```

## Expected Performance Metrics

### Target Metrics
- **Weighted AUC**: >0.55 (anything above 0.5 is profitable)
- **Win Rate**: 15-40% (varies by volatility mode)
- **Precision**: >0.03 (3% of high-confidence signals should win)
- **Recall**: >0.70 (catch 70% of winning entries)

### Expected Win Rates by Mode
Based on volatility and risk parameters:
- **Low Vol (6/12 ticks)**: 35-40% win rate (tighter stops, easier targets)
- **Normal Vol (8/16 ticks)**: 25-30% win rate (balanced risk/reward)
- **High Vol (10/20 ticks)**: 15-25% win rate (wider stops, harder targets)

### Weighted Training Benefits
- **Quality Focus**: High-quality entries (low MAE) get 2x weight
- **Speed Preference**: Fast-moving setups get 2x weight
- **Recency Bias**: Recent data automatically weighted higher
- **Balanced Learning**: Both winners and losers contribute meaningfully

## Deployment Strategy

### Model Serving
```python
# Real-time inference ensemble
def predict_optimal_entries(features_dict):
    """
    Get predictions from all 6 volatility-based models
    Returns dict of probabilities for each mode
    """
    predictions = {}
    
    for model_name, model_info in trained_models.items():
        model = model_info['model']
        prob = model.predict_proba([features_dict])[0, 1]
        predictions[model_name] = prob
    
    return predictions

# Example usage with volatility-based selection
current_features = extract_features(current_bar)
current_volatility = get_current_volatility_regime(current_features)
entry_probs = predict_optimal_entries(current_features)

# Select appropriate models based on current volatility
if current_volatility == 'low':
    relevant_models = ['low_vol_long', 'low_vol_short']
elif current_volatility == 'high':
    relevant_models = ['high_vol_long', 'high_vol_short']
else:
    relevant_models = ['normal_vol_long', 'normal_vol_short']

# Trade if relevant model shows high confidence
for model_name in relevant_models:
    prob = entry_probs[model_name]
    if prob > 0.7:  # 70% confidence threshold
        direction = 'LONG' if 'long' in model_name else 'SHORT'
        print(f"High confidence {direction} signal: {model_name} ({prob:.3f})")
```

### Model Updates
- Retrain monthly with new data on EC2
- A/B test new models against existing ones
- Monitor performance degradation over time
- Simple redeployment: just update model files on EC2

## Advantages of This Weighted XGBoost Approach

1. **Volatility-Aware Models**: Each model specialized for specific market conditions
2. **Quality-Weighted Learning**: Better entries get higher training importance
3. **Velocity-Optimized**: Fast-moving setups prioritized in training
4. **Time-Aware**: Recent market conditions automatically weighted higher
5. **Fast Training**: Complete training in minutes vs hours for LSTM
6. **Interpretable**: Clear feature importance per volatility regime
7. **Robust**: Less prone to overfitting than neural networks
8. **Production Ready**: Fast inference, easy EC2 deployment
9. **Incremental Updates**: Can retrain individual models
10. **Resource Efficient**: No GPU requirements, simple EC2 setup

## Key Improvements Over Previous Approach

### **Better Label Quality**
- **Old**: 3-class system with arbitrary MAE filtering
- **New**: Binary classification with sophisticated weighting

### **Volatility Awareness**  
- **Old**: Generic small/medium/large sizing
- **New**: Volatility-based modes that adapt to market conditions

### **Weighted Training**
- **Old**: All examples weighted equally
- **New**: Quality + velocity + recency weighting

### **Simplified Deployment**
- **Old**: Complex optimal vs suboptimal distinction
- **New**: Simple binary win/loss with confidence scores

This weighted XGBoost approach should provide significantly better performance and easier maintenance compared to both LSTM models and the previous unweighted labeling system.