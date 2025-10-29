# ES Trading Model - XGBoost Prediction System

## Overview
Machine learning system to predict optimal entry timing for E-mini S&P 500 futures trades using 6 specialized XGBoost models (one per trading profile).

## Project Status
- ✅ **Data Conversion**: DBN → Parquet format complete
- ✅ **Labeling Logic**: Win/loss/MAE filtering complete and optimized (300x speedup)
- 🔄 **Feature Engineering**: In progress (55 features planned)
- ⏳ **Model Training**: Pending (6 XGBoost models, one per trading profile)

## Quick Start

### Production Scripts
```bash
# Label full dataset (optimized version)
python label_full_dataset.py

# Test labeling on sample
python project/scripts/test_labeling.py

# View results interactively  
python project/scripts/view_results.py
```

### Validation & Testing
```bash
# Validate optimization correctness
python tests/validation/validate_optimization.py

# Quick validation on small sample
python tests/validation/quick_validation.py
```

## Directory Structure

```
├── project/                    # Core project modules
│   ├── data_pipeline/         # Labeling and feature engineering
│   ├── scripts/               # Utility scripts
│   ├── data/                  # Data storage (raw/processed/test)
│   └── config/                # Configuration files
├── tests/                     # Testing and validation
│   ├── validation/            # Algorithm validation scripts
│   └── debug/                 # Debugging utilities
├── docs/                      # Documentation
├── archive/                   # Deprecated/old files
├── .kiro/steering/           # AI assistant guidance
├── simple_optimized_labeling.py  # Optimized labeling algorithm
└── label_full_dataset.py        # Production labeling script
```

## Key Files

### Production Code
- `simple_optimized_labeling.py` - Optimized labeling algorithm (300x faster)
- `label_full_dataset.py` - Script to process full 15-year dataset
- `project/data_pipeline/labeling.py` - Original labeling implementation
- `project/data_pipeline/features.py` - Feature engineering (in progress)

### Testing & Validation
- `tests/validation/validate_optimization.py` - Validates optimized vs original
- `tests/validation/test_labeling.py` - Basic labeling tests
- `tests/debug/` - Debugging utilities used during optimization

### Documentation
- `docs/` - Technical documentation and analysis
- `.kiro/steering/` - AI assistant guidance documents

## Performance

### Labeling Performance (1000 bars)
- **Original**: 341 seconds
- **Optimized**: 1.6 seconds  
- **Speedup**: 207x faster

### Full Dataset Estimates (947K bars)
- **Original**: ~28 hours
- **Optimized**: ~17 minutes

## Trading Strategy

- **Asset**: E-mini S&P 500 futures (ES)
- **Timeframe**: 1-second bars, RTH only (07:30-15:00 CT)
- **Risk/Reward**: All profiles are 2:1 reward-to-risk
- **Lookforward**: 15 minutes to determine win/loss
- **Models**: 6 XGBoost models (Long/Short × Small/Medium/Large position sizes)
- **Architecture**: Separate specialized model per trading profile for optimal performance

## Data Pipeline

1. **Raw Data**: Databento DBN.ZST → Parquet format (RTH-only filtering)
2. **Labeling**: Apply 6 trading profiles with MAE filtering
3. **Features**: 43 engineered features (volume, price, time, microstructure)
4. **Training**: 6 specialized XGBoost models (one per trading profile)
5. **Deployment**: Real-time inference ensemble system

## Development Workflow

1. Test changes on small samples (100-1000 bars)
2. Validate against original implementation
3. Scale to full dataset only after validation
4. Use progress tracking for long operations

## Next Steps

1. Complete feature engineering module
2. Test feature pipeline on sample data
3. Scale to full 15-year dataset processing
4. Train 6 XGBoost models on AWS SageMaker
5. Deploy real-time inference ensemble system