# Feature Engineering Implementation Plan

- [ ] 1. Set up core feature engineering module structure
  - Create `project/data_pipeline/features.py` with main processing function
  - Implement basic input validation and progress tracking
  - Add utility functions for slope calculation and retouch counting
  - _Requirements: 1.3, 1.4, 11.1, 11.2, 11.4_

- [ ] 2. Implement Volume Features (4 features)
  - Code `volume_ratio_30s` as current volume divided by 30-bar rolling mean
  - Code `volume_slope_30s` as linear slope of 5-bar volume MA over 30-bar window  
  - Code `volume_slope_5s` as linear slope of raw volume over 5-bar window
  - Code `volume_exhaustion` as volume_ratio_30s multiplied by volume_slope_5s
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 3. Implement Price Context Features (5 features)
  - Code session VWAP calculation using only completed bars from session start
  - Code `distance_from_vwap_pct` as signed percentage distance from VWAP
  - Code `vwap_slope` as linear slope of VWAP over last 30 bars
  - Code `distance_from_rth_high` and `distance_from_rth_low` using session boundaries
  - Implement Central Time conversion with DST handling for session identification
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 4. Implement Consolidation Range Features (10 features)
  - Code short-term range features using 300-bar (5-minute) lookback windows
  - Code medium-term range features using 900-bar (15-minute) lookback windows
  - Code `range_compression_ratio` as short_range_size divided by medium_range_size
  - Code `position_in_short_range` for fade zone identification
  - Implement retouch counting with 30-second cooldown and 10% zone definitions
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 5. Implement Return and Momentum Features (5 features)
  - Code return calculations at 30s, 60s, and 300s timeframes using historical price differences
  - Code `momentum_acceleration` as return_30s minus return_60s
  - Code `momentum_consistency` as rolling standard deviation of 1-second returns over 30 bars
  - Handle edge cases where price differences could result in division by zero
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 6. Implement Volatility Features (6 features)
  - Code ATR calculations using true range over 30-bar and 300-bar windows
  - Code `volatility_regime` as atr_30s divided by atr_300s
  - Code `volatility_acceleration` as percentage change from 60s to 30s ATR
  - Code `volatility_breakout` as Z-score of current ATR vs 300-bar distribution
  - Code `atr_percentile` as rank of current ATR within recent 300-bar history
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 7. Implement Microstructure Features (6 features)
  - Code `bar_range` as high minus low for previous completed bar
  - Code `relative_bar_size` as bar_range divided by atr_30s
  - Code uptick percentage calculations by counting bars where close > previous close
  - Code `bar_flow_consistency` as absolute difference between 30s and 60s uptick percentages
  - Code `directional_strength` as distance from 50% uptick rate multiplied by 2
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 8. Implement Time Features (7 features)
  - Code UTC to Central Time conversion with automatic DST handling using pytz
  - Code session period identification for all 7 periods: ETH, Pre-Open, RTH Open, Morning, Lunch, Afternoon, RTH Close
  - Create binary features for each session period using one-hot encoding
  - Ensure only one session period is active at any given time
  - Handle timezone edge cases and DST transitions correctly
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 9. Integrate with existing labeled dataset
  - Code main function to accept existing 39-column labeled dataset as input
  - Add all 43 feature columns to existing dataset structure while preserving labels
  - Ensure feature column names match exactly the names defined in feature definitions
  - Validate that feature values fall within expected ranges as documented
  - Save enhanced dataset in Parquet format with 82 total columns
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.7_

- [ ] 10. Implement memory-efficient processing for large datasets
  - Code chunked processing logic to handle datasets from 947K to 88M bars
  - Handle rolling window overlaps between chunks to maintain calculation accuracy
  - Implement progress tracking with simple print statements for long-running operations
  - Ensure identical results regardless of chunk size used
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.7_

- [ ] 11. Add data validation and error handling
  - Code input validation to check for required OHLCV columns and non-empty dataframe
  - Handle insufficient historical data by setting features to NaN with minimal validation
  - Implement fail-fast behavior for critical errors rather than complex recovery logic
  - Add basic progress logging using print statements
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 12. Create comprehensive testing suite
  - Write unit tests for each feature category with simple input/output validation
  - Write integration test that validates end-to-end processing on sample data
  - Write data leakage tests that verify no future data is used in calculations
  - Write simple performance validation on laptop processing time
  - Implement all testing in single test file under 200 lines of code
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

- [ ] 13. Validate and optimize performance
  - Test processing time on 947K bar dataset to ensure under 10 minutes on laptop
  - Validate feature calculation accuracy against known expected ranges
  - Ensure memory usage stays reasonable during processing
  - Generate summary statistics for all 43 features to validate distributions
  - _Requirements: 1.1, 1.5, 10.5, 10.6_

- [ ] 14. Prepare for SageMaker deployment
  - Ensure code works with standard pandas/numpy environment without additional dependencies
  - Test chunked processing on larger datasets to simulate SageMaker conditions
  - Validate that all features are calculated identically across different environments
  - Document any SageMaker-specific considerations for deployment
  - _Requirements: 1.2, 13.1, 13.2, 13.4_