# XGBoost Training Strategy for ES Trading Models

## Overview
Train 6 specialized XGBoost models (one per trading profile) to predict optimal entry timing for ES futures trades.

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

### 6 Independent XGBoost Models

```python
models = {
    'long_2to1_small': {
        'target_column': 'long_2to1_small_label',
        'target_ticks': 12,
        'stop_ticks': 6,
        'direction': 'long'
    },
    'long_2to1_medium': {
        'target_column': 'long_2to1_medium_label', 
        'target_ticks': 16,
        'stop_ticks': 8,
        'direction': 'long'
    },
    'long_2to1_large': {
        'target_column': 'long_2to1_large_label',
        'target_ticks': 20, 
        'stop_ticks': 10,
        'direction': 'long'
    },
    'short_2to1_small': {
        'target_column': 'short_2to1_small_label',
        'target_ticks': 12,
        'stop_ticks': 6, 
        'direction': 'short'
    },
    'short_2to1_medium': {
        'target_column': 'short_2to1_medium_label',
        'target_ticks': 16,
        'stop_ticks': 8,
        'direction': 'short'
    },
    'short_2to1_large': {
        'target_column': 'short_2to1_large_label',
        'target_ticks': 20,
        'stop_ticks': 10,
        'direction': 'short'
    }
}
```

## Training Data Preparation

### Label Encoding for XGBoost
Convert 3-class labels to binary classification:

```python
def prepare_xgboost_labels(df, label_column):
    """
    Convert labels for XGBoost binary classification
    
    Original labels:
    +1 = Optimal entry (keep as 1)
     0 = Suboptimal entry (convert to 0) 
    -1 = Loss (convert to 0)
    NaN = Timeout (exclude from training)
    
    XGBoost labels:
    1 = Optimal entry (trade this)
    0 = Not optimal (don't trade)
    """
    # Remove timeout bars (NaN labels)
    mask = df[label_column].notna()
    df_clean = df[mask].copy()
    
    # Convert to binary: 1 stays 1, everything else becomes 0
    df_clean['xgb_label'] = (df_clean[label_column] == 1).astype(int)
    
    return df_clean
```

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

### 2. Train Each Model
```python
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

trained_models = {}

for model_name, config in models.items():
    print(f"\nTraining {model_name}...")
    
    # Prepare data for this profile
    label_col = config['target_column']
    df_model = prepare_xgboost_labels(df, label_col)
    
    # Split data chronologically
    train_df, val_df, test_df = create_time_splits(df_model)
    
    # Prepare features and labels
    X_train = train_df[feature_cols]
    y_train = train_df['xgb_label']
    X_val = val_df[feature_cols] 
    y_val = val_df['xgb_label']
    
    # Train XGBoost model
    model = xgb.XGBClassifier(**xgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluate
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    
    print(f"Validation AUC: {val_auc:.4f}")
    
    # Store model
    trained_models[model_name] = {
        'model': model,
        'config': config,
        'val_auc': val_auc,
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
- **AUC**: >0.55 (anything above 0.5 is profitable)
- **Precision**: >0.02 (2% of signals should be optimal)
- **Recall**: >0.80 (catch 80% of optimal entries)

### Class Imbalance Handling
Optimal entries are rare (~1-2% of bars):
- Use `scale_pos_weight` parameter in XGBoost
- Focus on AUC and precision rather than accuracy
- Consider threshold tuning for deployment

## Deployment Strategy

### Model Serving
```python
# Real-time inference ensemble
def predict_optimal_entries(features_dict):
    """
    Get predictions from all 6 models
    Returns dict of probabilities for each profile
    """
    predictions = {}
    
    for model_name, model_info in trained_models.items():
        model = model_info['model']
        prob = model.predict_proba([features_dict])[0, 1]
        predictions[model_name] = prob
    
    return predictions

# Example usage
current_features = extract_features(current_bar)
entry_probs = predict_optimal_entries(current_features)

# Trade if any model shows high confidence
for profile, prob in entry_probs.items():
    if prob > 0.7:  # 70% confidence threshold
        print(f"High confidence signal: {profile} ({prob:.3f})")
```

### Model Updates
- Retrain monthly with new data
- A/B test new models against existing ones
- Monitor performance degradation over time

## Advantages of This Approach

1. **Specialized Models**: Each profile optimized independently
2. **Fast Training**: Complete training in minutes vs hours
3. **Interpretable**: Clear feature importance per profile
4. **Robust**: Less prone to overfitting than neural networks
5. **Production Ready**: Fast inference, easy deployment
6. **Incremental Updates**: Can retrain individual models
7. **Resource Efficient**: No GPU requirements

This XGBoost approach should provide better performance and easier maintenance compared to a single LSTM model for this tabular trading data use case.