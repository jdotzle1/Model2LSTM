# Clean Directory Structure Guide

## Overview
The project has been organized following software engineering best practices with clear separation of concerns.

## Directory Structure

```
├── project/                    # Core project modules (PRODUCTION CODE)
│   ├── data_pipeline/         # Labeling and feature engineering
│   ├── scripts/               # Utility scripts
│   ├── data/                  # Data storage (raw/processed/test)
│   └── config/                # Configuration files
├── tests/                     # Testing and validation (DEVELOPMENT)
│   ├── validation/            # Algorithm validation scripts
│   └── debug/                 # Debugging utilities (archived)
├── docs/                      # Documentation and analysis
├── archive/                   # Deprecated files and old implementations
├── .kiro/steering/           # AI assistant guidance documents
├── simple_optimized_labeling.py  # MAIN PRODUCTION: Optimized algorithm
├── label_full_dataset.py        # MAIN PRODUCTION: Full dataset script
└── README.md                     # Project overview and quick start
```

## Key Principles

### 1. Production vs Development Separation
- **Root level**: Only production-ready files
- **tests/**: All testing, validation, and debugging code
- **archive/**: Deprecated code kept for reference

### 2. Clear Naming Convention
- Production files have descriptive names without debug/test prefixes
- Test files clearly indicate their purpose
- Debug files are archived but accessible

### 3. Import Path Management
- All test files properly handle Python path resolution
- Imports work from any directory level
- No relative import issues

## File Categories

### Production Files (Root Level)
- `ec2_weighted_labeling_pipeline.py` - **MAIN**: Weighted labeling pipeline for EC2
- `test_final_integration_1000_bars.py` - **MAIN TEST**: Complete integration testing
- `run_comprehensive_validation.py` - **VALIDATION**: Complete validation suite

### Core Project (`project/`)
- Weighted labeling system (6 volatility modes)
- Feature engineering (43 features)
- Pipeline orchestration and validation
- Data storage directories

### Testing (`tests/`)
- `validation/` - Chunked processing and integration tests
- `test_weighted_labeling_comprehensive.py` - Weight calculation validation
- `test_features_comprehensive.py` - Feature engineering validation
- `debug/` - Development debugging utilities (archived)

### EC2 Deployment (`ec2_deployment_package/`)
- Complete deployment archive
- Configuration files and setup scripts
- Monitoring and validation tools

### Documentation (`docs/`)
- Technical documentation
- Performance validation reports
- Analysis reports

### Archive (`archive/`)
- Deprecated implementations (original labeling system)
- One-time analysis scripts
- Historical reference code

## Usage Guidelines

### For Development
1. Always validate changes: `python tests/integration/test_final_integration_1000_bars.py`
2. Test on 1000-bar sample first with full integration
3. Use comprehensive validation suite: `python tests/validation/run_comprehensive_validation.py`
4. Run unit tests: `python -m pytest tests/unit/ -v`

### For Production
1. Use main entry point: `python main.py --input data.parquet --output processed.parquet`
2. Deploy using EC2 deployment package in `deployment/ec2/`
3. Monitor with provided monitoring tools
4. Validate XGBoost format requirements with `--validate` flag

### For Model Training
1. Use weighted labeling output (12 columns: 6 labels + 6 weights)
2. Train 6 separate XGBoost models with corresponding label/weight pairs
3. Use 43 engineered features as input for all models
4. Deploy ensemble with volatility regime detection

### For Analysis
1. Use scripts in `scripts/analysis/` for data analysis
2. Use scripts in `scripts/utilities/` for utility functions
3. Generate reports in `docs/reports/`

## Benefits of This Structure

1. **Clear separation**: Production vs development code
2. **Easy navigation**: Logical grouping of related files
3. **Maintainable**: Easy to find and update specific functionality
4. **Professional**: Follows industry best practices
5. **Scalable**: Structure supports future growth

## Import Best Practices

When creating new test files, use this pattern:
```python
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# Now import project modules
from project.data_pipeline.labeling import calculate_labels_for_all_profiles
from simple_optimized_labeling import calculate_labels_for_all_profiles_optimized
```

This ensures imports work regardless of where the script is run from.