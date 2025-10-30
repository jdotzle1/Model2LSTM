# ES Trading Model - Weighted Labeling System

A machine learning pipeline for E-mini S&P 500 futures trading that generates weighted labels for training 6 specialized XGBoost models based on volatility regimes.

## Overview

This system processes 1-second ES futures bars through a complete pipeline:

1. **Weighted Labeling**: Generates 12 columns (6 labels + 6 weights) for 6 volatility-based trading modes
2. **Feature Engineering**: Creates 43 engineered features across 7 categories
3. **Validation**: Ensures XGBoost-ready output format

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd es-trading-model

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process data through complete pipeline
python main.py --input raw_data.parquet --output processed_data.parquet

# With validation
python main.py --input raw_data.parquet --output processed_data.parquet --validate

# Custom chunk size for large datasets
python main.py --input raw_data.parquet --output processed_data.parquet --chunk-size 5000
```

## Architecture

### 6 Volatility-Based Trading Modes

- **Low Vol Long/Short**: 6-tick stop, 12-tick target (2:1 R/R)
- **Normal Vol Long/Short**: 8-tick stop, 16-tick target (2:1 R/R)  
- **High Vol Long/Short**: 10-tick stop, 20-tick target (2:1 R/R)

### Triple Weighting System

Each winning trade gets weighted by three components:

1. **Quality Weight**: Based on MAE (Maximum Adverse Excursion)
   - Formula: `2.0 - (1.5 × mae_ratio)` [0.5, 2.0]
   - Lower drawdown = higher weight

2. **Velocity Weight**: Based on speed to target
   - Formula: `2.0 - (1.5 × (seconds_to_target - 300) / 600)` [0.5, 2.0]
   - Optimal time: 5 minutes

3. **Time Decay Weight**: Based on data recency
   - Formula: `exp(-0.05 × months_ago)`
   - Recent data gets higher weight

### Output Format

61 total columns:
- 6 original OHLCV columns
- 12 weighted labeling columns (6 labels + 6 weights)
- 43 engineered features

## Directory Structure

```
├── main.py                     # Main production entry point
├── requirements.txt            # Dependencies
├── src/                        # Production source code
│   ├── data_pipeline/         # Core pipeline modules
│   ├── config/                # Configuration files
│   └── convert_dbn.py         # Data conversion utilities
├── tests/                     # All test files
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── validation/            # Validation scripts
├── scripts/                   # Utility scripts
│   ├── analysis/              # Analysis scripts
│   └── utilities/             # Utility scripts
├── deployment/                # Deployment files
│   └── ec2/                   # EC2 deployment package
└── docs/                      # Documentation and reports
    └── reports/               # Generated reports
```

## Testing

### Run All Tests

```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests  
python -m pytest tests/integration/ -v

# Complete integration test (1000-bar sample)
python tests/integration/test_final_integration_1000_bars.py

# Comprehensive validation
python tests/validation/run_comprehensive_validation.py processed_data.parquet
```

### Key Test Files

- `tests/unit/test_weighted_labeling_comprehensive.py` - Weight calculation tests
- `tests/unit/test_features_comprehensive.py` - Feature engineering tests
- `tests/integration/test_final_integration_1000_bars.py` - Complete pipeline test

## Model Training

After processing data, train 6 XGBoost models:

```python
import xgboost as xgb
import pandas as pd

# Load processed data
df = pd.read_parquet('processed_data.parquet')

# Train model for low volatility long trades
X = df[feature_columns]  # 43 features
y = df['label_low_vol_long']  # Binary labels
weights = df['weight_low_vol_long']  # Sample weights

model = xgb.XGBClassifier()
model.fit(X, y, sample_weight=weights)
```

## Performance

- **Processing Rate**: ~1,000-1,200 rows/second
- **Memory Usage**: +56% increase (efficient for large datasets)
- **Chunked Processing**: Memory-efficient for datasets >1M rows
- **Validation**: Comprehensive quality assurance built-in

## Key Features

### Weighted Labeling
- 6 specialized volatility modes
- Binary classification (0/1) with sample weights
- Quality, velocity, and time decay weighting
- XGBoost-ready output format

### Feature Engineering
- 43 features across 7 categories
- Volume, price context, consolidation, returns, volatility, microstructure, time
- Optimized for ES futures characteristics
- Handles missing data appropriately

### Validation
- Comprehensive data quality checks
- XGBoost format validation
- Performance monitoring
- Chunked processing consistency

## Deployment

### EC2 Deployment

```bash
# Extract deployment package
cd deployment/ec2/
tar -xzf ec2_deployment_package_*.tar.gz

# Set up environment
export S3_BUCKET=your-bucket-name
./setup_ec2_environment.sh

# Run pipeline
python ec2_weighted_labeling_pipeline.py --bucket $S3_BUCKET
```

## Contributing

1. Follow the existing code structure in `src/`
2. Add tests for new features in appropriate `tests/` subdirectories
3. Update documentation for significant changes
4. Run validation suite before submitting changes

## License

[Add your license information here]