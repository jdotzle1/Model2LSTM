# Technology Stack & Deployment

## Core Technologies

- **Python 3.x**: Primary programming language
- **pandas**: Data manipulation and analysis  
- **numpy**: Numerical computing
- **databento**: Market data provider API and file format handling
- **Parquet**: Columnar storage format for efficient data processing

## Development Environment

- **Local Testing**: 1-month sample on laptop (Windows)
- **Production Processing**: EC2 deployment for full 15-year dataset
- **Data Storage**: S3 for raw data, processed datasets

## Common Commands

### Local Development & Testing
```bash
# Data conversion
python project/convert_dbn.py

# Pipeline testing (small samples)
python test_labeling.py          # Test labeling logic
python "test features.py"        # Test feature engineering  
python view_results.py           # Interactive data inspection

# Utility scripts
python project/scripts/prepare_data.py
python project/scripts/validate_features.py
python project/scripts/visualize_features.py
```

### Production Deployment (EC2)
```bash
# Scale to full dataset processing
# (Commands will be added when deploying to EC2)
```

## Performance Optimizations

- **Parquet format**: Fast I/O and compression for large datasets
- **Vectorized operations**: pandas/numpy for performance
- **Memory management**: Use `.copy()` to avoid SettingWithCopyWarning
- **Chunking**: Process large datasets in manageable pieces
- **Progress tracking**: Print status for long-running operations

## Key Dependencies

```python
import databento as db
import pandas as pd
import numpy as np
```

## Deployment Strategy

1. **Local Development**: Test and validate on 1-month sample
2. **EC2 Scaling**: Deploy pipeline for full 15-year dataset processing
3. **Model Training**: LSTM training on processed features
4. **Production**: Real-time inference deployment