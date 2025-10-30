# Technology Stack & Deployment

## Core Technologies

- **Python 3.x**: Primary programming language
- **pandas**: Data manipulation and analysis  
- **numpy**: Numerical computing
- **XGBoost**: Machine learning framework for 6 specialized models
- **databento**: Market data provider API and file format handling
- **Parquet**: Columnar storage format for efficient data processing

## Development Environment

- **Local Testing**: 1000-bar sample on laptop (Windows)
- **Production Processing**: EC2 deployment for full 15-year dataset
- **Data Storage**: S3 for raw data, processed datasets with weighted labels

## Common Commands

### Local Development & Testing
```bash
# Data conversion
python project/convert_dbn.py

# PRODUCTION: Weighted labeling full dataset
python ec2_weighted_labeling_pipeline.py

# Pipeline testing (small samples)
python test_final_integration_1000_bars.py       # Complete integration test
python tests/test_weighted_labeling_comprehensive.py  # Weighted labeling tests
python tests/test_features_comprehensive.py      # Feature engineering tests

# Validation (ALWAYS run before production)
python run_comprehensive_validation.py           # Complete validation suite
python validate_data_quality.py                  # Data quality checks
python validate_performance.py                   # Performance validation

# Utility scripts
python project/scripts/prepare_data.py
python project/scripts/validate_features.py
python project/scripts/visualize_features.py
python project/scripts/check_dataset_size.py
```

### Production Deployment (EC2)
```bash
# Extract deployment package
tar -xzf ec2_deployment_package_[timestamp].tar.gz

# Set up environment
export S3_BUCKET=your-bucket-name
chmod +x setup_ec2_environment.sh
./setup_ec2_environment.sh

# Validate setup
./validate_ec2_setup.sh

# Run weighted labeling pipeline
python ec2_weighted_labeling_pipeline.py --bucket $S3_BUCKET

# Monitor progress
./monitor_pipeline.sh
python check_progress.py
```

## Performance Optimizations

- **Parquet format**: Fast I/O and compression for large datasets
- **Vectorized operations**: pandas/numpy for performance
- **Chunked processing**: Memory-efficient processing for large datasets
- **Weighted labeling**: Optimized algorithms for quality/velocity/time decay calculations
- **Progress tracking**: Comprehensive monitoring for long-running operations

## Key Dependencies

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from project.data_pipeline.weighted_labeling import process_weighted_labeling
from project.data_pipeline.features import create_all_features
```

## Weighted Labeling System

### Core Components
- **6 Volatility Modes**: Low/Normal/High vol for Long/Short
- **Binary Labels**: 0 (loss) or 1 (win) for each mode
- **Triple Weighting**: Quality × Velocity × Time Decay
- **12 Output Columns**: 6 labels + 6 weights for XGBoost

### Weight Calculations
```python
# Quality weight (MAE-based)
quality_weight = 2.0 - (1.5 × mae_ratio)  # [0.5, 2.0]

# Velocity weight (speed-based)  
velocity_weight = 2.0 - (1.5 × (seconds_to_target - 300) / 600)  # [0.5, 2.0]

# Time decay weight (recency-based)
time_decay = exp(-0.05 × months_ago)

# Final weight
final_weight = quality_weight × velocity_weight × time_decay
```

## Deployment Strategy

1. **Local Development**: Test and validate on 1000-bar sample
2. **EC2 Scaling**: Deploy weighted labeling pipeline for full 15-year dataset
3. **Model Training**: 6 XGBoost models with weighted samples
4. **Ensemble Deployment**: Volatility-adaptive model selection
5. **Production**: Real-time inference with volatility regime detection

## Model Training Pipeline
```bash
# After weighted labeling is complete
python train_xgboost_models.py --input processed_dataset.parquet
python validate_model_performance.py --models model_output/
python deploy_ensemble.py --models model_output/ --config deployment_config.json
```