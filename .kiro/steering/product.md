# ES Trading Model - Project Overview

## Purpose
Build a machine learning model (LSTM) to predict optimal entry timing for E-mini S&P 500 futures trades.

## Trading Strategy
- **Asset:** E-mini S&P 500 futures (ES)
- **Timeframe:** 1-second bars, RTH only (9:30 AM - 4:00 PM ET)
- **Risk/Reward:** All profiles are 2:1 reward-to-risk
- **Lookforward:** 15 minutes to determine win/loss

## Data
- **Training data:** 15 years (2010-2025) in S3
- **Test data:** 1 month sample on local machine
- **File format:** Parquet (converted from Databento DBN.ZST)
- **Columns:** timestamp, open, high, low, close, volume

## Model Architecture
Single LSTM model with 6 output heads (one per risk profile):
- Long 2:1 Small (target +12 ticks, stop -6 ticks)
- Long 2:1 Medium (target +16 ticks, stop -8 ticks)
- Long 2:1 Large (target +20 ticks, stop -10 ticks)
- Short 2:1 Small (target -12 ticks, stop +6 ticks)
- Short 2:1 Medium (target -16 ticks, stop +8 ticks)
- Short 2:1 Large (target -20 ticks, stop +10 ticks)

## Key Concepts

### ES Basics
- **Tick size:** 0.25 points
- **Point value:** $50 per point
- **Example:** 12 ticks = 3 points = $150 move

### Labels (What We're Predicting)
- **+1 (Optimal):** Trade won with lowest MAE in sequence
- **0 (Suboptimal):** Trade won but timing wasn't optimal
- **-1 (Loss):** Trade hit stop before target
- **NaN (Timeout):** Neither target nor stop hit in 15 minutes

### MAE (Maximum Adverse Excursion)
The worst drawdown before a winning trade hits target. Lower MAE = better entry timing.

**Example:**
- Entry: 4750.00
- Target: 4753.00 (+12 ticks)
- Path: 4750 → 4749 → 4748.75 (worst point) → 4751 → 4753 ✓
- MAE: 4750 - 4748.75 = 1.25 points = 5 ticks

## Development Workflow
1. ✅ Convert test data to Parquet
2. ✅ Build labeling logic (calculate wins/losses/MAE)
3. 🔄 Build feature engineering (55 features)
4. Test on 1000-bar sample
5. Validate results
6. Scale to full 15-year dataset on SageMaker
7. Train LSTM model
8. Evaluate and deploy

## Current Status
- Labeling module complete and tested on 1000 bars
- Feature engineering module in progress