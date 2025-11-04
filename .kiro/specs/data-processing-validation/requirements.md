# Data Processing Validation and Monthly S3 Processing

## Introduction

Fix and validate the existing data processing pipeline for XGBoost model training data. The system has substantial existing code including weighted labeling (src/data_pipeline/weighted_labeling.py), feature engineering (src/data_pipeline/features.py), monthly processing (process_monthly_chunks_fixed.py), validation scripts, and data quality fixes. The goal is to resolve current issues with labeling and feature engineering validation, implement robust rollover handling, and enable reliable processing of 15 years of monthly S3 data with comprehensive statistics logging.

## Glossary

- **WeightedLabelingEngine**: Existing class in src/data_pipeline/weighted_labeling.py that generates 6 labels + 6 weights
- **Feature Engineering System**: Existing create_all_features function that generates 43 features across 7 categories  
- **Monthly Processing Pipeline**: Existing process_monthly_chunks_fixed.py with built-in data cleaning
- **Rollover Detection**: Contract roll identification using 20-point price gap threshold (already implemented)
- **Statistics Logging**: Monthly validation metrics and quality checks (needs enhancement)
- **S3 Processing**: Existing boto3-based system for downloading/uploading monthly chunks
- **Data Quality Fixes**: Existing fix_data_quality_issues.py with price validation and cleaning
- **Validation Scripts**: Existing test_30day_pipeline.py and validate_full_dataset_logic.py
- **RTH Filtering**: Existing timezone-aware filtering to 07:30-15:00 Central Time
- **Desktop Sample**: 1-month ES data on desktop for testing (es_30day_rth.parquet)

## Requirements

### Requirement 1: Fix and Validate Existing Pipeline

**User Story:** As a quantitative trader, I want to fix the current issues in my existing labeling and feature engineering pipeline and validate it works correctly with rollover handling, so that I can trust the data quality before processing the full 15-year dataset.

#### Acceptance Criteria

1. THE System SHALL fix any issues in the existing WeightedLabelingEngine (src/data_pipeline/weighted_labeling.py)
2. THE System SHALL fix any issues in the existing feature engineering system (src/data_pipeline/features.py)
3. THE System SHALL enhance the existing rollover detection in LabelCalculator._detect_contract_rolls() to properly handle 20-point price gaps
4. THE System SHALL validate the existing test_30day_pipeline.py works end-to-end on desktop sample
5. THE System SHALL validate the existing validate_full_dataset_logic.py produces consistent results
6. THE System SHALL ensure the existing data quality fixes in fix_data_quality_issues.py handle all edge cases
7. THE System SHALL run complete validation in under 10 minutes using existing test scripts

### Requirement 2: Fix and Enhance Monthly S3 Processing

**User Story:** As a data engineer, I want to fix issues in the existing monthly processing pipeline and enhance it with better error handling and statistics logging, so that I can reliably process 15 years of ES data from S3.

#### Acceptance Criteria

1. THE System SHALL fix any issues in the existing process_monthly_chunks_fixed.py pipeline
2. THE System SHALL enhance the existing S3 file discovery to handle different path structures in "es-1-second-data" bucket
3. THE System SHALL improve the existing monthly file processing from July 2010 through October 2025
4. THE System SHALL enhance the existing file naming pattern handling: "glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1s.dbn.zst"
5. THE System SHALL improve the existing restart capability by better tracking processed months
6. THE System SHALL enhance the existing cleanup logic in cleanup_monthly_files()
7. THE System SHALL fix any memory leaks or performance issues in the existing monthly processing loop

### Requirement 3: Enhance Statistics Logging and Validation

**User Story:** As a model trainer, I want enhanced statistics logging for each processed month building on the existing OutputDataFrame.get_statistics(), so that I can identify months that need reprocessing and validate data quality across the entire dataset.

#### Acceptance Criteria

1. THE System SHALL enhance the existing OutputDataFrame.get_statistics() to include:
   - Total bars processed and date range
   - Win rates for each of the 6 trading modes (already implemented)
   - Average weights and weight distributions (already implemented)
   - Feature quality metrics (NaN percentages, outlier counts)
   - Rollover events detected and bars excluded
   - Processing time and memory usage
2. THE System SHALL enhance the existing upload_monthly_results() to save statistics as JSON metadata
3. THE System SHALL build on existing validation logic to identify months requiring reprocessing
4. THE System SHALL use existing win rate validation (5-50% range already implemented)
5. THE System SHALL add feature distribution consistency checks to existing validation
6. THE System SHALL enhance existing rollover detection statistics and flagging

### Requirement 4: Fix and Enhance Rollover Detection

**User Story:** As a trading system developer, I want to fix and enhance the existing rollover detection in LabelCalculator._detect_contract_rolls(), so that my model training data contains only valid trading opportunities.

#### Acceptance Criteria

1. THE System SHALL fix any issues in the existing _detect_contract_rolls() method that uses 20-point threshold
2. THE System SHALL enhance the existing logic that marks rollover bar and following 5 bars as affected
3. THE System SHALL ensure the existing label=0 assignment for rollover-affected bars works correctly
4. THE System SHALL enhance rollover statistics tracking in the existing system
5. THE System SHALL improve validation of the existing rollover detection logic
6. THE System SHALL ensure the existing system handles multiple rollover events per month
7. THE System SHALL integrate rollover statistics into the enhanced monthly logging system

### Requirement 5: Enhance Existing Data Quality Validation

**User Story:** As a data scientist, I want to enhance the existing data quality validation systems (InputDataFrame, OutputDataFrame, fix_data_quality_issues.py), so that I can identify and fix issues before they affect model training.

#### Acceptance Criteria

1. THE System SHALL enhance the existing clean_price_data() function to handle all edge cases:
   - Fix any remaining issues with OHLCV validation
   - Improve the existing OHLC relationship checks
   - Enhance the existing volume validation
   - Add better price range validation for ES futures
2. THE System SHALL enhance the existing RTH filtering validation:
   - Improve the existing timezone handling in Central Time conversion
   - Fix any DST transition issues in existing code
   - Validate the existing 30-40% data reduction expectations
3. THE System SHALL enhance the existing OutputDataFrame.validate_quality_assurance():
   - Fix any issues with binary label validation (0 or 1)
   - Improve existing positive weight validation
   - Enhance existing win rate range validation (5-50%)
   - Fix any NaN or infinite value detection issues
4. THE System SHALL enhance feature validation in the existing system:
   - Validate all 43 expected features are generated correctly
   - Add feature value range validation
   - Improve NaN percentage validation (< 35% threshold)
   - Add outlier detection for feature values

### Requirement 6: Fix Memory and Performance Issues

**User Story:** As a system operator, I want to fix existing memory and performance issues in the WeightedLabelingEngine and monthly processing pipeline, so that I can process large datasets reliably.

#### Acceptance Criteria

1. THE System SHALL fix memory issues in the existing WeightedLabelingEngine to stay under 8GB peak usage
2. THE System SHALL optimize the existing monthly processing to complete each month in under 30 minutes
3. THE System SHALL fix any memory leaks in the existing garbage collection between stages
4. THE System SHALL enhance the existing PerformanceMonitor to better track memory usage
5. THE System SHALL fix progress reporting in the existing system (currently every 10K rows)
6. THE System SHALL improve the existing chunked processing in WeightedLabelingEngine._process_chunked()
7. THE System SHALL optimize the existing processing order to minimize memory fragmentation

### Requirement 7: Enhance Error Handling and Recovery

**User Story:** As a production operator, I want to enhance the existing error handling in process_monthly_chunks_fixed.py and related scripts, so that processing failures don't require restarting the entire 15-year processing job.

#### Acceptance Criteria

1. THE System SHALL enhance the existing try/catch blocks in process_single_month() to continue processing remaining months
2. THE System SHALL improve the existing log_progress() function to capture detailed error information
3. THE System SHALL add retry logic to the existing S3 download/upload functions for transient errors
4. THE System SHALL enhance the existing S3 file validation in download_monthly_file()
5. THE System SHALL improve handling of corrupted files in the existing process_monthly_data()
6. THE System SHALL enhance error messages in the existing exception handling
7. THE System SHALL improve the existing processing log in /tmp/monthly_processing.log

### Requirement 8: Fix Output Data Format and Storage

**User Story:** As a model trainer, I want to fix any issues in the existing output format and enhance the S3 storage system, so that I can efficiently load and combine months for XGBoost training.

#### Acceptance Criteria

1. THE System SHALL fix any schema consistency issues in the existing Parquet output
2. THE System SHALL validate the existing column structure (original + 12 labeling + 43 features)
3. THE System SHALL ensure the existing column naming is consistent across all processed months
4. THE System SHALL optimize the existing Parquet compression in the output files
5. THE System SHALL enhance the existing S3 metadata in upload_monthly_results()
6. THE System SHALL improve the existing file naming: "monthly_YYYY-MM_timestamp.parquet"
7. THE System SHALL add output file integrity validation before the existing S3 upload

### Requirement 9: Enhance Progress Tracking and Monitoring

**User Story:** As a project manager, I want to enhance the existing progress tracking in the monthly processing pipeline, so that I can estimate completion times and identify bottlenecks.

#### Acceptance Criteria

1. THE System SHALL enhance the existing progress tracking in main() to better track 180+ months
2. THE System SHALL improve the existing time estimation logic based on completed months
3. THE System SHALL enhance the existing log_progress() to better capture start/end times
4. THE System SHALL improve the existing average processing time calculation
5. THE System SHALL add stage timing to identify bottlenecks in the existing pipeline
6. THE System SHALL enhance the existing summary statistics display
7. THE System SHALL improve the existing final processing report generation

### Requirement 10: Fix Integration Issues with Existing Codebase

**User Story:** As a developer, I want to fix any integration issues between the monthly processing system and existing weighted labeling/feature engineering code, so that the system works reliably end-to-end.

#### Acceptance Criteria

1. THE System SHALL fix any import or compatibility issues with WeightedLabelingEngine
2. THE System SHALL fix any issues with create_all_features function integration
3. THE System SHALL fix any issues with existing data quality validation function integration
4. THE System SHALL ensure the existing configuration parameters work correctly
5. THE System SHALL fix any functionality regressions while preserving existing capabilities
6. THE System SHALL ensure the existing desktop validation logic works with monthly processing
7. THE System SHALL fix any inconsistencies between desktop and S3 processing results