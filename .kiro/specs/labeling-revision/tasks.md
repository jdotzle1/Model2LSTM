# Implementation Plan - Revised Labeling System

Convert the weighted labeling system design into a series of implementation tasks for building the new labeling system that generates 12 columns (6 labels + 6 weights) for training 6 XGBoost models.

## Implementation Tasks

- [x] 1. Create core data structures and configuration




  - Define TradingMode dataclass with name, direction, stop_ticks, target_ticks
  - Create TRADING_MODES dictionary with all 6 volatility-based modes
  - Define constants (TICK_SIZE, TIMEOUT_SECONDS, DECAY_RATE)
  - Create LabelingConfig dataclass for system configuration
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement input validation and data models




  - Create InputDataFrame class with validation for required columns
  - Validate data types (datetime for timestamp, numeric for OHLCV)
  - Add RTH-only data verification (07:30-15:00 CT)
  - Create custom exception classes (ValidationError, ProcessingError, PerformanceError)
  - _Requirements: 8.1, 8.2, 8.5, 10.6_

- [x] 3. Build label calculation engine





  - Create LabelCalculator class for individual trading modes
  - Implement _check_single_entry method for win/loss determination
  - Add target/stop price calculation logic for long and short modes
  - Implement forward-looking price checking with tick precision
  - Track MAE (Maximum Adverse Excursion) for winning trades
  - Track seconds_to_target timing for winning trades
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4_

- [x] 4. Build weight calculation engine





  - Create WeightCalculator class for computing example weights
  - Implement quality weight calculation based on MAE ratio
  - Implement velocity weight calculation based on speed to target
  - Implement time decay calculation with proper month handling across year boundaries
  - Combine quality × velocity × time_decay for winners
  - Apply time_decay only for losers
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4_

- [x] 5. Create main processing engine


  - Create WeightedLabelingEngine class as main entry point
  - Implement process_dataframe method for full pipeline
  - Add chunked processing support for large datasets (>100K rows)
  - Implement parallel processing for multiple trading modes
  - Add progress tracking with updates every 10,000 rows
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 6. Implement output validation and statistics



  - Create OutputDataFrame class for result validation
  - Validate all 12 new columns are present with correct naming
  - Ensure label columns contain only 0 or 1 values
  - Ensure weight columns contain only positive values
  - Generate statistics (win rates, weight distributions) per mode
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 10.1, 10.2, 10.3, 10.4, 10.5, 10.7_

- [x] 7. Add performance monitoring and optimization





  - Implement processing speed tracking (target: 167K rows/minute)
  - Add memory usage monitoring (target: <8GB)
  - Optimize with numpy vectorization for numerical computations
  - Add performance validation against 60-minute target for 10M rows
  - _Requirements: 9.1, 9.2, 9.4_

- [x] 8. Create comprehensive test suite








  - Write unit tests for LabelCalculator (long winners, long losers, timeouts)
  - Write unit tests for WeightCalculator (quality, velocity, time decay)
  - Write unit tests for month calculation across year boundaries
  - Write integration tests for end-to-end pipeline
  - Write performance tests for 10M row processing target
  - Write memory usage tests for large dataset handling
  - _Requirements: All requirements validation_

- [x] 9. Add validation and quality assurance utilities









  - Create validation script to check label distributions per mode
  - Create validation script to check weight distributions per mode
  - Add data quality checks for NaN/infinite values
  - Create comparison utility vs original labeling system
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [x] 10. Integration with EC2 pipeline









  - Update existing labeling module imports and interfaces
  - Create EC2 complete pipeline script combining conversion + labeling + features + training
  - Update feature engineering pipeline to handle new column names
  - Ensure smooth integration on single EC2 instance
  - _Requirements: 8.5_

- [x] 11. Create usage documentation and examples






  - Write usage examples for processing sample datasets
  - Document configuration options and performance tuning
  - Create troubleshooting guide for common issues
  - Add performance benchmarking examples
  - _Requirements: All requirements implementation guidance_

- [x] 12. Final integration and EC2 deployment preparation




  - Test complete pipeline on 1000-bar sample dataset
  - Validate output format matches XGBoost training requirements
  - Test chunked processing consistency vs single-pass processing
  - Create EC2 deployment script for complete pipeline
  - Validate all 12 columns are correctly generated and formatted
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 9.1_