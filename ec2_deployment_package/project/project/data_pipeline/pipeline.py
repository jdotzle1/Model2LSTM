"""
Unified Data Pipeline for ES Trading Model

This module provides a complete pipeline that integrates:
1. Data conversion (DBN to Parquet)
2. Weighted labeling system (6 volatility-based modes)
3. Feature engineering (43 features)
4. XGBoost model training (6 models)

Main Functions:
    process_complete_pipeline() - End-to-end processing
    process_labeling_and_features() - Labeling + features only
    process_training_only() - Training only (assumes labeled+featured data)
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime

# Import pipeline components
from .weighted_labeling import process_weighted_labeling, LabelingConfig
from .features import create_all_features, get_expected_feature_names


class PipelineConfig:
    """Configuration for the complete pipeline"""
    
    def __init__(self,
                 chunk_size: int = 500_000,
                 enable_performance_monitoring: bool = True,
                 enable_progress_tracking: bool = True,
                 enable_memory_optimization: bool = True,
                 output_dir: str = None):
        """
        Initialize pipeline configuration
        
        Args:
            chunk_size: Rows per chunk for memory-efficient processing
            enable_performance_monitoring: Track processing speed and memory
            enable_progress_tracking: Print progress updates
            enable_memory_optimization: Enable memory optimization features
            output_dir: Directory for saving intermediate and final results
        """
        self.chunk_size = chunk_size
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_progress_tracking = enable_progress_tracking
        self.enable_memory_optimization = enable_memory_optimization
        self.output_dir = output_dir or os.getcwd()
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def process_complete_pipeline(input_path: str, 
                            output_path: str = None,
                            config: PipelineConfig = None) -> pd.DataFrame:
    """
    Complete end-to-end pipeline: labeling + features + training
    
    Args:
        input_path: Path to input Parquet file with OHLCV data
        output_path: Path to save final dataset (optional)
        config: Pipeline configuration (optional)
        
    Returns:
        DataFrame with labels, weights, features, and trained models metadata
    """
    config = config or PipelineConfig()
    
    if config.enable_progress_tracking:
        print("ES Trading Model - Complete Pipeline")
        print("=" * 50)
        print(f"Input: {input_path}")
        print(f"Output: {output_path or 'in-memory only'}")
        print(f"Chunk size: {config.chunk_size:,} rows")
    
    start_time = time.time()
    
    try:
        # Step 1: Load data
        if config.enable_progress_tracking:
            print(f"\n=== STEP 1: LOADING DATA ===")
        
        df = pd.read_parquet(input_path)
        
        if config.enable_progress_tracking:
            print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
            if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        # Step 2: Apply weighted labeling
        df_labeled = process_labeling_and_features(df, config)
        
        # Step 3: Train models (optional - can be done separately)
        if config.enable_progress_tracking:
            print(f"\n=== PIPELINE COMPLETE ===")
            total_time = time.time() - start_time
            print(f"Total processing time: {total_time/60:.1f} minutes")
            print(f"Final dataset: {len(df_labeled):,} rows × {len(df_labeled.columns)} columns")
        
        # Save if output path provided
        if output_path:
            if config.enable_progress_tracking:
                print(f"Saving to: {output_path}")
            df_labeled.to_parquet(output_path, compression='snappy')
        
        return df_labeled
        
    except Exception as e:
        if config.enable_progress_tracking:
            print(f"\n❌ Pipeline failed: {str(e)}")
        raise


def process_labeling_and_features(df: pd.DataFrame, 
                                config: PipelineConfig = None) -> pd.DataFrame:
    """
    Apply weighted labeling and feature engineering
    
    Args:
        df: Input DataFrame with OHLCV data
        config: Pipeline configuration (optional)
        
    Returns:
        DataFrame with labels, weights, and features added
    """
    config = config or PipelineConfig()
    
    if config.enable_progress_tracking:
        print(f"\n=== LABELING AND FEATURE ENGINEERING ===")
    
    # Step 1: Apply weighted labeling
    if config.enable_progress_tracking:
        print(f"\nStep 1: Weighted Labeling (6 volatility-based modes)")
    
    # Configure labeling system
    labeling_config = LabelingConfig(
        chunk_size=config.chunk_size,
        enable_performance_monitoring=config.enable_performance_monitoring,
        enable_progress_tracking=config.enable_progress_tracking,
        enable_memory_optimization=config.enable_memory_optimization
    )
    
    df_labeled = process_weighted_labeling(df, labeling_config)
    
    # Validate labeling results
    expected_label_columns = []
    for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                 'low_vol_short', 'normal_vol_short', 'high_vol_short']:
        expected_label_columns.extend([f'label_{mode}', f'weight_{mode}'])
    
    missing_label_cols = set(expected_label_columns) - set(df_labeled.columns)
    if missing_label_cols:
        raise ValueError(f"Missing expected label/weight columns: {missing_label_cols}")
    
    if config.enable_progress_tracking:
        print(f"  ✓ Added 12 labeling columns (6 labels + 6 weights)")
    
    # Step 2: Apply feature engineering
    if config.enable_progress_tracking:
        print(f"\nStep 2: Feature Engineering (43 features)")
    
    df_featured = create_all_features(df_labeled)
    
    # Validate feature engineering results
    expected_features = get_expected_feature_names()
    missing_features = set(expected_features) - set(df_featured.columns)
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")
    
    if config.enable_progress_tracking:
        print(f"  ✓ Added {len(expected_features)} feature columns")
        print(f"  Final dataset: {len(df_featured):,} rows × {len(df_featured.columns)} columns")
    
    return df_featured


def train_xgboost_models(df: pd.DataFrame, 
                        config: PipelineConfig = None,
                        save_models: bool = True) -> Dict:
    """
    Train 6 XGBoost models using weighted binary classification
    
    Args:
        df: DataFrame with labels, weights, and features
        config: Pipeline configuration (optional)
        save_models: Whether to save trained models to disk
        
    Returns:
        Dictionary containing trained models and metadata
    """
    config = config or PipelineConfig()
    
    if config.enable_progress_tracking:
        print(f"\n=== XGBOOST MODEL TRAINING ===")
    
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    import joblib
    import json
    
    # Identify feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    exclude_cols.extend([col for col in df.columns if col.startswith(('label_', 'weight_'))])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if config.enable_progress_tracking:
        print(f"  Training with {len(feature_cols)} features")
        print(f"  Dataset size: {len(df):,} rows")
    
    models = {}
    trading_modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                    'low_vol_short', 'normal_vol_short', 'high_vol_short']
    
    # Chronological split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    
    for i, mode in enumerate(trading_modes, 1):
        if config.enable_progress_tracking:
            print(f"  [{i}/6] Training {mode} model...")
        
        # Prepare data for this mode
        X = df[feature_cols]
        y = df[f'label_{mode}']
        sample_weights = df[f'weight_{mode}']
        
        # Chronological split
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        w_train, w_test = sample_weights.iloc[:split_idx], sample_weights.iloc[split_idx:]
        
        # Remove NaN values if any
        valid_mask = ~(y_train.isna() | w_train.isna())
        if not valid_mask.all():
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            w_train = w_train[valid_mask]
        
        # Check data quality
        win_rate = y_train.mean()
        total_samples = len(y_train)
        
        if config.enable_progress_tracking:
            print(f"    Data: {total_samples:,} samples, win rate: {win_rate:.1%}")
        
        # XGBoost parameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eta': 0.1,
            'n_estimators': 1000,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Train model
        model = xgb.XGBClassifier(**xgb_params)
        
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            eval_sample_weight=[w_test],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate
        test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred, sample_weight=w_test)
        
        # Weight statistics
        avg_winner_weight = w_train[y_train == 1].mean()
        avg_loser_weight = w_train[y_train == 0].mean()
        
        if config.enable_progress_tracking:
            print(f"    Weighted Test AUC: {test_auc:.4f}")
            print(f"    Avg Winner Weight: {avg_winner_weight:.3f}")
            print(f"    Avg Loser Weight: {avg_loser_weight:.3f}")
        
        # Store model with metadata
        model_info = {
            'model': model,
            'test_auc': test_auc,
            'win_rate': win_rate,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
            'training_samples': total_samples,
            'avg_winner_weight': avg_winner_weight,
            'avg_loser_weight': avg_loser_weight,
            'xgb_params': xgb_params
        }
        
        models[mode] = model_info
        
        # Save model if requested
        if save_models:
            model_path = os.path.join(config.output_dir, f'{mode}_model.pkl')
            joblib.dump(model, model_path)
            
            # Save feature importance
            importance_path = os.path.join(config.output_dir, f'{mode}_feature_importance.json')
            with open(importance_path, 'w') as f:
                json.dump(model_info['feature_importance'], f, indent=2)
    
    if config.enable_progress_tracking:
        print(f"✓ Trained {len(models)} XGBoost models")
        print(f"\n  Model Performance Summary:")
        for mode, model_info in models.items():
            print(f"    {mode}: AUC={model_info['test_auc']:.4f}, WinRate={model_info['win_rate']:.3f}")
    
    return models


def validate_pipeline_output(df: pd.DataFrame) -> Dict:
    """
    Validate that pipeline output contains all expected columns and data quality
    
    Args:
        df: Pipeline output DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check required OHLCV columns
    required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
    missing_ohlcv = set(required_ohlcv) - set(df.columns)
    if missing_ohlcv:
        validation_results['errors'].append(f"Missing OHLCV columns: {missing_ohlcv}")
        validation_results['valid'] = False
    
    # Check label and weight columns
    expected_label_weight_cols = []
    for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                 'low_vol_short', 'normal_vol_short', 'high_vol_short']:
        expected_label_weight_cols.extend([f'label_{mode}', f'weight_{mode}'])
    
    missing_labels = set(expected_label_weight_cols) - set(df.columns)
    if missing_labels:
        validation_results['errors'].append(f"Missing label/weight columns: {missing_labels}")
        validation_results['valid'] = False
    
    # Check feature columns
    expected_features = get_expected_feature_names()
    missing_features = set(expected_features) - set(df.columns)
    if missing_features:
        validation_results['errors'].append(f"Missing feature columns: {missing_features}")
        validation_results['valid'] = False
    
    # Validate data quality
    if validation_results['valid']:
        # Check label values (should be 0 or 1)
        for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                     'low_vol_short', 'normal_vol_short', 'high_vol_short']:
            label_col = f'label_{mode}'
            if not df[label_col].isin([0, 1]).all():
                validation_results['errors'].append(f"{label_col} contains non-binary values")
                validation_results['valid'] = False
            
            weight_col = f'weight_{mode}'
            if not (df[weight_col] > 0).all():
                validation_results['errors'].append(f"{weight_col} contains non-positive values")
                validation_results['valid'] = False
        
        # Calculate statistics
        validation_results['statistics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': {
                'start': str(df.index.min()) if hasattr(df.index, 'min') else 'unknown',
                'end': str(df.index.max()) if hasattr(df.index, 'max') else 'unknown'
            },
            'win_rates': {}
        }
        
        for mode in ['low_vol_long', 'normal_vol_long', 'high_vol_long',
                     'low_vol_short', 'normal_vol_short', 'high_vol_short']:
            label_col = f'label_{mode}'
            win_rate = df[label_col].mean()
            validation_results['statistics']['win_rates'][mode] = win_rate
            
            # Warn about unusual win rates
            if win_rate < 0.05 or win_rate > 0.6:
                validation_results['warnings'].append(
                    f"{mode} has unusual win rate: {win_rate:.1%}"
                )
    
    return validation_results


def create_pipeline_summary(df: pd.DataFrame, models: Dict = None) -> Dict:
    """
    Create a comprehensive summary of pipeline results
    
    Args:
        df: Final pipeline DataFrame
        models: Trained models dictionary (optional)
        
    Returns:
        Dictionary with pipeline summary
    """
    summary = {
        'pipeline_version': '2.0_weighted_labeling',
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'date_range': {
                'start': str(df.index.min()) if hasattr(df.index, 'min') else 'unknown',
                'end': str(df.index.max()) if hasattr(df.index, 'max') else 'unknown'
            }
        },
        'labeling_info': {
            'system': 'weighted_volatility_based',
            'modes': 6,
            'columns_added': 12
        },
        'feature_info': {
            'system': 'engineered_features',
            'features_added': 43,
            'categories': [
                'volume_features_4',
                'price_context_features_5', 
                'consolidation_features_10',
                'return_features_5',
                'volatility_features_6',
                'microstructure_features_6',
                'time_features_7'
            ]
        }
    }
    
    # Add model information if available
    if models:
        summary['model_info'] = {
            'algorithm': 'xgboost',
            'models_trained': len(models),
            'performance': {}
        }
        
        for mode, model_info in models.items():
            summary['model_info']['performance'][mode] = {
                'test_auc': model_info['test_auc'],
                'win_rate': model_info['win_rate'],
                'training_samples': model_info['training_samples']
            }
    
    return summary


# Convenience functions for common use cases
def quick_labeling_and_features(input_path: str, output_path: str = None) -> pd.DataFrame:
    """Quick processing: labeling + features only"""
    return process_complete_pipeline(input_path, output_path)


def quick_training_only(input_path: str, model_output_dir: str = None) -> Dict:
    """Quick training: assumes data already has labels and features"""
    df = pd.read_parquet(input_path)
    config = PipelineConfig(output_dir=model_output_dir)
    return train_xgboost_models(df, config)