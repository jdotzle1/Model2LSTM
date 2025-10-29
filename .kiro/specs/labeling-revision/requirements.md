# Requirements Document - Revised Labeling System for XGBoost Models

## Introduction

This document specifies a revised labeling system for ES futures trading data that will be used to train 6 specialized XGBoost models. The new system introduces weighted labels and renames trading modes to reflect volatility-based position sizing rather than arbitrary size categories.

## Glossary

- **ES**: E-mini S&P 500 futures contract
- **Tick**: Minimum price movement (0.25 points for ES)
- **RTH**: Regular Trading Hours (07:30-15:00 CT)
- **MAE**: Maximum Adverse Excursion (worst drawdown before target)
- **Entry Price**: Next bar's open price after signal bar
- **Target Hit**: Price touches target level at tick precision
- **Stop Hit**: Price touches stop level at tick precision
- **Timeout**: Neither target nor stop hit within 900 seconds
- **Quality Weight**: Weight based on MAE performance
- **Velocity Weight**: Weight based on speed to target
- **Time Decay**: Weight reduction for older data

## Requirements

### Requirement 1: Trading Mode Definition

**User Story:** As a model trainer, I want 6 distinct trading modes based on volatility regimes and direction, so that I can train specialized XGBoost models for different market conditions.

#### Acceptance Criteria

1. THE System SHALL define exactly 6 trading modes with the following specifications:
   - `low_vol_long`: 6 tick stop, 12 tick target, long direction
   - `normal_vol_long`: 8 tick stop, 16 tick target, long direction  
   - `high_vol_long`: 10 tick stop, 20 tick target, long direction
   - `low_vol_short`: 6 tick stop, 12 tick target, short direction
   - `normal_vol_short`: 8 tick stop, 16 tick target, short direction
   - `high_vol_short`: 10 tick stop, 20 tick target, short direction

2. THE System SHALL use ES tick size of 0.25 points for all calculations

3. THE System SHALL maintain 2:1 reward-to-risk ratio for all modes

4. THE System SHALL use 900 seconds (15 minutes) as maximum timeout period

5. THE System SHALL process only RTH data (07:30-15:00 CT)

### Requirement 2: Label Generation

**User Story:** As a model trainer, I want binary labels (0 or 1) for each trading mode, so that I can train XGBoost models for binary classification.

#### Acceptance Criteria

1. WHEN processing each bar, THE System SHALL generate labels for all 6 trading modes simultaneously

2. FOR long modes, THE System SHALL:
   - Set entry price as next bar's open price
   - Calculate stop price as entry - (stop_ticks × 0.25)
   - Calculate target price as entry + (target_ticks × 0.25)
   - Look forward up to 900 seconds from entry bar

3. FOR short modes, THE System SHALL:
   - Set entry price as next bar's open price  
   - Calculate stop price as entry + (stop_ticks × 0.25)
   - Calculate target price as entry - (target_ticks × 0.25)
   - Look forward up to 900 seconds from entry bar

4. THE System SHALL assign label = 1 WHEN target is hit before stop

5. THE System SHALL assign label = 0 WHEN stop is hit before target OR timeout occurs

6. THE System SHALL check price hits at exact tick precision (0.25 increments)

7. THE System SHALL create 12 new columns: `label_[mode_name]` and `weight_[mode_name]` for each mode

### Requirement 3: MAE and Timing Tracking

**User Story:** As a model trainer, I want to track Maximum Adverse Excursion and timing for winning trades, so that I can calculate quality-based weights.

#### Acceptance Criteria

1. FOR winning trades (label = 1) in long modes, THE System SHALL track:
   - `mae_ticks`: Maximum adverse excursion as lowest low between entry and target, converted to ticks below entry
   - `seconds_to_target`: Time in seconds from entry to target hit

2. FOR winning trades (label = 1) in short modes, THE System SHALL track:
   - `mae_ticks`: Maximum adverse excursion as highest high between entry and target, converted to ticks above entry  
   - `seconds_to_target`: Time in seconds from entry to target hit

3. THE System SHALL convert MAE to tick precision for consistent measurement

4. THE System SHALL record exact timing to the second for velocity calculations

### Requirement 4: Quality Weight Calculation

**User Story:** As a model trainer, I want quality weights based on MAE performance, so that better-timed entries receive higher importance during training.

#### Acceptance Criteria

1. FOR winning trades, THE System SHALL calculate quality weight using formula:
   - `mae_ratio = mae_ticks / stop_ticks`
   - `quality_weight = 2.0 - (1.5 × mae_ratio)`
   - `quality_weight = clip(quality_weight, 0.5, 2.0)`

2. THE System SHALL assign higher quality weights to trades with lower MAE

3. THE System SHALL ensure quality weight range is [0.5, 2.0]

4. FOR losing trades, THE System SHALL set quality_weight = 1.0

### Requirement 5: Velocity Weight Calculation

**User Story:** As a model trainer, I want velocity weights based on speed to target, so that faster-moving trades receive higher importance.

#### Acceptance Criteria

1. FOR winning trades, THE System SHALL calculate velocity weight using formula:
   - `velocity_weight = 2.0 - (1.5 × (seconds_to_target - 300) / 600)`
   - `velocity_weight = clip(velocity_weight, 0.5, 2.0)`

2. THE System SHALL assign higher velocity weights to trades reaching target faster

3. THE System SHALL use 300 seconds as optimal target time (5 minutes)

4. THE System SHALL ensure velocity weight range is [0.5, 2.0]

5. FOR losing trades, THE System SHALL set velocity_weight = 1.0

### Requirement 6: Time Decay Calculation

**User Story:** As a model trainer, I want time decay weights for older data, so that recent market conditions have higher influence on model training.

#### Acceptance Criteria

1. THE System SHALL calculate time decay using formula:
   - `most_recent_date = maximum date in entire dataset`
   - `row_date = date of current row`
   - `months_ago = total calendar months between row_date and most_recent_date`
   - `time_decay = exp(-0.05 × months_ago)`

2. THE System SHALL apply time decay to both winning and losing trades

3. THE System SHALL calculate months_ago accounting for year boundaries using the formula:
   - `months_ago = (most_recent_year - row_year) × 12 + (most_recent_month - row_month)`
   - Example: January 2023 (2023-01) to December 2024 (2024-12) = (2024-2023)×12 + (12-1) = 23 months

4. THE System SHALL use monthly granularity for time decay calculation

5. THE System SHALL ensure more recent data receives higher weights

### Requirement 7: Final Weight Combination

**User Story:** As a model trainer, I want combined weights that reflect quality, velocity, and recency, so that XGBoost models prioritize the most relevant training examples.

#### Acceptance Criteria

1. FOR winning trades, THE System SHALL calculate final weight as:
   - `weight = quality_weight × velocity_weight × time_decay`

2. FOR losing trades, THE System SHALL calculate final weight as:
   - `weight = 1.0 × 1.0 × time_decay`

3. THE System SHALL ensure all weights are positive values

4. THE System SHALL store weights as float values in `weight_[mode_name]` columns

### Requirement 8: Data Output Format

**User Story:** As a model trainer, I want consistently formatted output data, so that I can reliably train XGBoost models.

#### Acceptance Criteria

1. THE System SHALL add exactly 12 new columns to the input DataFrame

2. THE System SHALL use naming convention:
   - Labels: `label_low_vol_long`, `label_normal_vol_long`, etc.
   - Weights: `weight_low_vol_long`, `weight_normal_vol_long`, etc.

3. THE System SHALL ensure label columns contain only integer values (0 or 1)

4. THE System SHALL ensure weight columns contain only positive float values

5. THE System SHALL preserve all original DataFrame columns and structure

### Requirement 9: Performance and Memory Efficiency

**User Story:** As a data processor, I want efficient labeling that can handle large datasets, so that I can process 15 years of ES data within reasonable time and memory constraints.

#### Acceptance Criteria

1. THE System SHALL process datasets up to 10 million rows within 60 minutes

2. THE System SHALL use vectorized operations where possible to minimize processing time

3. THE System SHALL provide progress updates every 10,000 rows processed

4. THE System SHALL handle memory efficiently for datasets up to 5GB

5. THE System SHALL support chunked processing for larger datasets

### Requirement 10: Validation and Quality Assurance

**User Story:** As a data scientist, I want validation checks on the labeling output, so that I can ensure data quality before model training.

#### Acceptance Criteria

1. THE System SHALL validate that all label values are exactly 0 or 1

2. THE System SHALL validate that all weight values are positive

3. THE System SHALL report label distribution statistics for each mode

4. THE System SHALL report weight distribution statistics for each mode

5. THE System SHALL validate that winning percentages are reasonable (5-50% range)

6. THE System SHALL check for any NaN or infinite values in output columns

7. THE System SHALL provide summary statistics including:
   - Total bars processed
   - Win rate per mode
   - Average MAE for winners per mode
   - Average time to target per mode
   - Weight distribution percentiles per mode