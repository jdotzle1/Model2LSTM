# Project Structure

## Directory Organization

```
project/
├── config/                 # Configuration files
│   └── config.py          # Project constants and settings
├── data/                  # Data storage (organized by processing stage)
│   ├── raw/              # Original DBN.ZST files converted to Parquet
│   ├── processed/        # Fully processed datasets with labels and features
│   └── test/             # Small test datasets for development
├── data_pipeline/        # Core pipeline modules
│   ├── __init__.py
│   ├── labeling.py       # Trade outcome labeling logic
│   ├── features.py       # Feature engineering functions
│   └── pipeline.py       # Main pipeline orchestration
└── scripts/              # Utility and processing scripts
    ├── prepare_data.py
    ├── validate_features.py
    └── visualize_features.py
```

## Root Level Files

- `convert_dbn.py`: Converts Databento files to Parquet format
- `test_labeling.py`: Tests labeling pipeline on sample data
- `test features.py`: Tests feature engineering pipeline
- `view_results.py`: Interactive data inspection tool

## Data Flow

1. **Raw Data**: DBN.ZST files → `project/data/raw/` (Parquet)
2. **Labeling**: Raw Parquet → Labeled data with trade outcomes
3. **Features**: Labeled data → Full feature set (55+ columns)
4. **Output**: `project/data/processed/` or `project/data/test/`

## Module Responsibilities

### `data_pipeline/labeling.py`
- Defines 6 trading profiles with risk/reward parameters
- Implements target/stop checking logic
- Calculates Maximum Adverse Excursion (MAE)
- Applies MAE filtering for optimal trade selection

### `data_pipeline/features.py`
- Volume features (ratios, percentiles, z-scores)
- Price context (VWAP, RTH levels, distances)
- Swing high/low identification
- Return calculations at multiple timeframes
- Volatility measures (ATR, realized vol)
- Microstructure features (tick direction, bar characteristics)
- Time-based features (session periods, time since open/close)

## Naming Conventions

- **Files**: snake_case (e.g., `test_labeling.py`)
- **Functions**: snake_case (e.g., `calculate_labels_for_all_profiles`)
- **Variables**: snake_case (e.g., `profile_name`, `lookforward_seconds`)
- **Constants**: UPPER_CASE (e.g., `TICK_SIZE`, `PROFILES`)
- **Column names**: snake_case with descriptive suffixes (e.g., `volume_ratio_30s`, `long_2to1_small_label`)

## Import Patterns

```python
# Standard imports
import pandas as pd
import numpy as np

# Local imports
from project.data_pipeline.labeling import calculate_labels_for_all_profiles
from project.data_pipeline.features import create_all_features
```