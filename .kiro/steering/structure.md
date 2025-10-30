# Project Structure

## Directory Organization

```
├── main.py                     # Main production entry point
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── src/                        # Production source code
│   ├── data_pipeline/         # Core pipeline modules
│   │   ├── weighted_labeling.py  # Weighted labeling system (6 volatility modes)
│   │   ├── features.py       # Feature engineering functions (43 features)
│   │   ├── pipeline.py       # Main pipeline orchestration
│   │   └── validation_utils.py  # Validation and quality assurance
│   ├── config/                # Configuration files
│   │   └── config.py          # Project constants and settings
│   └── convert_dbn.py         # Converts Databento files to Parquet format
├── tests/                     # All test files organized by type
│   ├── unit/                  # Unit tests
│   │   ├── test_weighted_labeling_comprehensive.py  # Weight calculation tests
│   │   └── test_features_comprehensive.py          # Feature engineering tests
│   ├── integration/           # Integration tests
│   │   ├── test_final_integration_1000_bars.py     # Complete pipeline test
│   │   ├── test_performance_monitoring.py          # Performance tests
│   │   └── test_*.py          # Other integration tests
│   └── validation/            # Validation scripts
│       ├── run_comprehensive_validation.py         # Complete validation suite
│       ├── validate_*.py      # Specific validation scripts
│       └── test_*.py          # Validation test scripts
├── scripts/                   # Utility and analysis scripts
│   ├── analysis/              # Analysis scripts
│   │   ├── analyze_winner_count.py
│   │   ├── compare_labeling_systems.py
│   │   └── performance_validation_summary.py
│   └── utilities/             # Utility scripts
│       ├── integrate_features.py
│       ├── quick_performance_test.py
│       └── update_imports.py
├── deployment/                # Deployment files
│   └── ec2/                   # EC2 deployment package
│       ├── prepare_ec2_deployment.py
│       ├── deploy_ec2_weighted_pipeline.sh
│       └── ec2_deployment_package_*.tar.gz
├── docs/                      # Documentation and reports
│   └── reports/               # Generated reports and analysis
├── archive/                   # Deprecated files and old implementations
└── .kiro/steering/           # AI assistant guidance documents
```

## Production Files

- `main.py`: **MAIN ENTRY POINT** - Production pipeline with CLI interface
- `src/data_pipeline/`: **CORE MODULES** - Weighted labeling, features, validation
- `requirements.txt`: **DEPENDENCIES** - All required Python packages

## Data Flow

1. **Raw Data**: DBN.ZST files → `project/data/raw/` (Parquet)
2. **Weighted Labeling**: Raw Parquet → 12 columns (6 labels + 6 weights) for 6 volatility modes
3. **Features**: Labeled data → 43 engineered features
4. **Output**: `project/data/processed/` with 61 total columns (6 original + 12 labeling + 43 features)

## Module Responsibilities

### `data_pipeline/weighted_labeling.py` (NEW)
- Defines 6 volatility-based trading modes (Low/Normal/High vol × Long/Short)
- Implements binary labeling (0=loss, 1=win) for each mode
- Calculates three-component weights:
  - **Quality weights**: Based on MAE (Maximum Adverse Excursion)
  - **Velocity weights**: Based on speed to target
  - **Time decay weights**: Based on data recency
- Generates 12 columns for XGBoost training

### `data_pipeline/features.py`
- Volume features (4): ratios, slopes, exhaustion patterns
- Price context features (5): VWAP, distances, slopes
- Consolidation features (10): range identification, retouch counting
- Return features (5): momentum at multiple timeframes
- Volatility features (6): ATR, regime detection, breakouts
- Microstructure features (6): bar characteristics, tick flow
- Time features (7): session period identification

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