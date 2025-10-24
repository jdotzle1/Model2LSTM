# Documentation

## Overview
Technical documentation and analysis for the ES Trading Model project.

## Files

### Technical Documentation
- `feature_definitions.md` - Detailed explanation of all 55 engineered features
- `recommendation.md` - Analysis and recommendations from optimization work
- `updated_labeling_explanation.md` - Updated labeling logic documentation

### Analysis & Proposals  
- `swing_fix_proposal.md` - Proposal for swing high/low feature improvements

## Key Concepts

### Labeling Logic
The system uses a sophisticated labeling approach:
1. **Outcome Detection**: Win/Loss/Timeout based on target/stop hits
2. **MAE Calculation**: Maximum Adverse Excursion for winning trades
3. **Sequence Grouping**: Consecutive winners form sequences
4. **Optimal Selection**: Lowest MAE + shortest hold time within sequence

### Feature Engineering
55 features across 7 categories:
- Volume features (5) - Context and momentum
- Price context (8) - VWAP, session levels, distances
- Swing highs/lows (10) - Reversal points and distances
- Returns (6) - Multi-timeframe momentum
- Volatility (8) - ATR and realized volatility measures
- Microstructure (8) - Bar characteristics and tick direction
- Time features (10) - Session periods and time-based patterns

### Performance Optimization
Key optimizations achieved:
- Vectorized operations instead of loops
- Efficient sequence detection algorithms
- Batch processing for large datasets
- Memory-efficient data structures
- Progress tracking for long operations

## Usage

Refer to these documents when:
- Understanding feature calculations
- Implementing new features
- Debugging labeling logic
- Optimizing performance
- Planning future enhancements