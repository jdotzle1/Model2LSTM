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
- `simple_optimized_labeling.py` - **MAIN**: Optimized labeling algorithm
- `label_full_dataset.py` - **MAIN**: Full dataset processing script
- `README.md` - Project documentation

### Core Project (`project/`)
- Original implementations and utilities
- Data pipeline modules
- Configuration and scripts
- Data storage directories

### Testing (`tests/`)
- `validation/` - Algorithm correctness validation
- `debug/` - Development debugging utilities (archived)

### Documentation (`docs/`)
- Technical documentation
- Feature definitions
- Analysis reports

### Archive (`archive/`)
- Deprecated implementations
- One-time analysis scripts
- Historical reference code

## Usage Guidelines

### For Development
1. Always validate changes: `python tests/validation/validate_optimization.py`
2. Test on small samples first
3. Use production files for actual processing

### For Production
1. Use root-level scripts: `simple_optimized_labeling.py`, `label_full_dataset.py`
2. Validate before scaling to full dataset
3. Monitor performance and results

### For Debugging
1. Check `tests/debug/` for historical debugging approaches
2. Create new debug scripts in `tests/debug/` if needed
3. Archive debug scripts after issues are resolved

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