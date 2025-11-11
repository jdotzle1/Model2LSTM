#!/usr/bin/env python3
"""
Train 6 XGBoost Models for ES Trading

Trains one model per volatility mode:
- low_vol_long, normal_vol_long, high_vol_long
- low_vol_short, normal_vol_short, high_vol_short

Each model uses:
- Binary labels (0/1)
- Sample weights (quality × velocity × time_decay)
- 43 engineered features

Usage:
    python scripts/train_xgboost_models.py --data-dir data/xgboost
    python scripts/train_xgboost_models.py --data-dir data/xgboost --output-dir models
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.weighted_labeling import TRADING_MODES


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature column names (exclude labels, weights, metadata)"""
    exclude_prefixes = ('label_', 'weight_', 'timestamp', 'session', 
                       'symbol', 'rtype', 'publisher', 'instrument')
    
    feature_cols = [c for c in df.columns if not c.startswith(exclude_prefixes)]
    return feature_cols


def train_single_model(mode_name: str, mode, train_df: pd.DataFrame, 
                      val_df: pd.DataFrame, feature_cols: list, 
                      output_dir: Path, params: dict):
    """Train a single XGBoost model for one volatility mode"""
    
    print("\n" + "=" * 80)
    print(f"TRAINING: {mode_name}")
    print("=" * 80)
    
    label_col = mode.label_column
    weight_col = mode.weight_column
    
    # Prepare training data
    print("Preparing training data...")
    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    w_train = train_df[weight_col]
    
    # Handle NaN/inf in features (XGBoost can handle, but let's be explicit)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Winners: {y_train.sum():,} ({y_train.mean():.1%})")
    print(f"  Features: {len(feature_cols)}")
    print(f"  NaN values: {X_train.isna().sum().sum():,} ({X_train.isna().sum().sum()/(len(X_train)*len(feature_cols))*100:.2f}%)")
    
    # Prepare validation data
    print("Preparing validation data...")
    X_val = val_df[feature_cols]
    y_val = val_df[label_col]
    w_val = val_df[weight_col]
    
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Winners: {y_val.sum():,} ({y_val.mean():.1%})")
    
    # Create DMatrix with weights
    print("Creating DMatrix...")
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, 
                        feature_names=feature_cols, enable_categorical=False)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val,
                      feature_names=feature_cols, enable_categorical=False)
    
    # Train model
    print(f"\nTraining XGBoost model...")
    print(f"Parameters: {params}")
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=10,
        evals_result=evals_result
    )
    
    # Get best iteration
    best_iteration = model.best_iteration
    best_score = model.best_score
    
    print(f"\n✓ Training complete")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Best validation AUC: {best_score:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred = model.predict(dval)
    
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    
    auc = roc_auc_score(y_val, y_pred, sample_weight=w_val)
    acc = accuracy_score(y_val, (y_pred > 0.5).astype(int), sample_weight=w_val)
    precision = precision_score(y_val, (y_pred > 0.5).astype(int), sample_weight=w_val, zero_division=0)
    recall = recall_score(y_val, (y_pred > 0.5).astype(int), sample_weight=w_val, zero_division=0)
    
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Save model
    model_path = output_dir / f"model_{mode_name}.json"
    model.save_model(str(model_path))
    print(f"\n✓ Model saved: {model_path}")
    
    # Save feature importance
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame([
        {'feature': k, 'gain': v} 
        for k, v in importance.items()
    ]).sort_values('gain', ascending=False)
    
    importance_path = output_dir / f"feature_importance_{mode_name}.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")
    
    # Save training metrics
    metrics = {
        'mode': mode_name,
        'training_date': datetime.now().isoformat(),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'train_win_rate': float(y_train.mean()),
        'val_win_rate': float(y_val.mean()),
        'features': len(feature_cols),
        'best_iteration': int(best_iteration),
        'validation_metrics': {
            'auc': float(auc),
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall)
        },
        'training_history': {
            'train_auc': [float(x) for x in evals_result['train']['auc']],
            'val_auc': [float(x) for x in evals_result['val']['auc']]
        }
    }
    
    metrics_path = output_dir / f"metrics_{mode_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train 6 XGBoost models for ES trading"
    )
    
    parser.add_argument(
        "--data-dir", default="data/xgboost",
        help="Directory with train_data.parquet and val_data.parquet (default: data/xgboost)"
    )
    parser.add_argument(
        "--output-dir", default="models",
        help="Output directory for trained models (default: models)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help="XGBoost max_depth parameter (default: 6)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1,
        help="XGBoost learning_rate parameter (default: 0.1)"
    )
    parser.add_argument(
        "--subsample", type=float, default=0.8,
        help="XGBoost subsample parameter (default: 0.8)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to use: cpu or cuda (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = data_dir / 'train_data.parquet'
    val_file = data_dir / 'val_data.parquet'
    
    # Check files exist
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        print(f"Run: python scripts/prepare_train_val_split.py")
        sys.exit(1)
    
    if not val_file.exists():
        print(f"❌ Validation file not found: {val_file}")
        print(f"Run: python scripts/prepare_train_val_split.py")
        sys.exit(1)
    
    print("=" * 80)
    print("XGBOOST MODEL TRAINING")
    print("=" * 80)
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_parquet(train_file)
    print(f"✓ Loaded {len(train_df):,} training samples")
    
    print("Loading validation data...")
    val_df = pd.read_parquet(val_file)
    print(f"✓ Loaded {len(val_df):,} validation samples")
    
    # Get feature columns
    feature_cols = get_feature_columns(train_df)
    print(f"\n✓ Using {len(feature_cols)} features")
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'tree_method': 'hist',
        'device': args.device
    }
    
    # Train all 6 models
    all_metrics = {}
    
    for mode_name, mode in TRADING_MODES.items():
        try:
            metrics = train_single_model(
                mode_name, mode, train_df, val_df, 
                feature_cols, output_dir, params
            )
            all_metrics[mode_name] = metrics
        except Exception as e:
            print(f"\n❌ Error training {mode_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    for mode_name, metrics in all_metrics.items():
        val_metrics = metrics['validation_metrics']
        print(f"\n{mode_name}:")
        print(f"  AUC: {val_metrics['auc']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Win rate (val): {metrics['val_win_rate']:.1%}")
    
    print("\n" + "=" * 80)
    print(f"✅ Training complete! {len(all_metrics)}/6 models trained")
    print(f"\nModels saved in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
