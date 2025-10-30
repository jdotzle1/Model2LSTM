# ES Trading Model - Project Overview

## Purpose
Build 6 specialized XGBoost models to predict optimal entry timing for E-mini S&P 500 futures trades across different volatility regimes.

## Trading Strategy
- **Asset:** E-mini S&P 500 futures (ES)
- **Timeframe:** 1-second bars, RTH only (9:30 AM - 4:00 PM ET)
- **Risk/Reward:** All profiles are 2:1 reward-to-risk
- **Lookforward:** 15 minutes to determine win/loss
- **Volatility-Adaptive:** Different stop/target sizes based on market conditions

## Data
- **Training data:** 15 years (2010-2025) in S3
- **Test data:** 1 month sample on local machine
- **File format:** Parquet (converted from Databento DBN.ZST)
- **Columns:** timestamp, open, high, low, close, volume

## Model Architecture
**6 Specialized XGBoost Models** (one per volatility regime):

**Long Models:**
- **Low Vol Long:** 6-tick stop, 12-tick target
- **Normal Vol Long:** 8-tick stop, 16-tick target  
- **High Vol Long:** 10-tick stop, 20-tick target

**Short Models:**
- **Low Vol Short:** 6-tick stop, 12-tick target
- **Normal Vol Short:** 8-tick stop, 16-tick target
- **High Vol Short:** 10-tick stop, 20-tick target

## Key Concepts

### ES Basics
- **Tick size:** 0.25 points
- **Point value:** $50 per point
- **Example:** 12 ticks = 3 points = $150 move

### Weighted Labeling System
**Binary Labels (0 or 1):**
- **1 (Win):** Price hits target before stop within 15 minutes
- **0 (Loss):** Price hits stop before target, or timeout

**Three-Component Weighting:**
1. **Quality Weight:** Based on MAE (Maximum Adverse Excursion)
2. **Velocity Weight:** Based on speed to target
3. **Time Decay Weight:** Based on data recency

### MAE (Maximum Adverse Excursion)
The worst drawdown before a winning trade hits target. Lower MAE = higher quality weight.

**Example:**
- Entry: 4750.00
- Target: 4753.00 (+12 ticks)
- Path: 4750 → 4749 → 4748.75 (worst point) → 4751 → 4753 ✓
- MAE: 4750 - 4748.75 = 1.25 points = 5 ticks
- Quality Weight: `2.0 - (1.5 × 5/6) = 0.75`

### Weight Formulas
**Quality:** `2.0 - (1.5 × mae_ratio)` [0.5, 2.0]
**Velocity:** `2.0 - (1.5 × (seconds_to_target - 300) / 600)` [0.5, 2.0]
**Time Decay:** `exp(-0.05 × months_ago)`
**Final:** `quality × velocity × time_decay` (winners) or `time_decay` (losers)

## Output Format (12 Columns)
Each row gets 12 columns for XGBoost training:
- `label_low_vol_long`, `weight_low_vol_long`
- `label_normal_vol_long`, `weight_normal_vol_long`
- `label_high_vol_long`, `weight_high_vol_long`
- `label_low_vol_short`, `weight_low_vol_short`
- `label_normal_vol_short`, `weight_normal_vol_short`
- `label_high_vol_short`, `weight_high_vol_short`

## Development Workflow
1. ✅ Convert test data to Parquet
2. ✅ Build weighted labeling system (6 volatility modes)
3. ✅ Build feature engineering (43 features)
4. ✅ Test on 1000-bar sample with full integration
5. ✅ Validate results and XGBoost format compatibility
6. ✅ Scale to full dataset processing (EC2 ready)
7. 🔄 Train 6 XGBoost models with weighted samples
8. 🔄 Build ensemble deployment system
9. 🔄 Evaluate and deploy

## Current Status
- ✅ Weighted labeling system complete and tested
- ✅ Feature engineering complete (43 features)
- ✅ Full integration tested on 1000-bar sample
- ✅ EC2 deployment package ready
- 🔄 Ready for full 15-year dataset processing

## Model Training Strategy
1. **Separate Training:** Train 6 independent XGBoost models
2. **Weighted Samples:** Use corresponding weight columns for sample weighting
3. **Feature Sharing:** All models use same 43 engineered features
4. **Volatility Detection:** Deploy ensemble that selects model based on current volatility regime
5. **Binary Classification:** Each model predicts probability of winning trade