# Data Processing Validation and Monthly S3 Processing - Design Document

## Overview

This design document outlines the fixes and enhancements needed for the existing data processing pipeline to enable reliable validation and monthly S3 processing of 15 years of ES futures data. The system builds on substantial existing code including WeightedLabelingEngine, feature engineering, monthly processing scripts, and validation frameworks.

## Architecture

### Current System Architecture

```mermaid
graph TB
    subgraph "Desktop Validation"
        A[es_30day_rth.parquet] --> B[test_30day_pipeline.py]
        B --> C[WeightedLabelingEngine]
        C --> D[create_all_features]
        D --> E[Validation Results]
    end
    
    subgraph "Monthly S3 Processing"
        F[S3 Monthly DBN Files] --> G[process_monthly_chunks_fixed.py]
        G --> H[download_monthly_file]
        H --> I[process_monthly_data]
        I --> J[clean_price_data]
        J --> K[WeightedLabelingEngine]
        K --> L[create_all_features]
        L --> M[upload_monthly_results]
    end
    
    subgraph "Data Quality & Validation"
        N[fix_data_quality_issues.py] --> O[InputDataFrame validation]
        O --> P[OutputDataFrame validation]
        P --> Q[Statistics logging]
    end
```

### Enhanced Architecture (Target State)

```mermaid
graph TB
    subgraph "Fixed Desktop Validation"
        A[es_30day_rth.parquet] --> B[Enhanced test_30day_pipeline.py]
        B --> C[Fixed WeightedLabelingEngine]
        C --> D[Enhanced rollover detection]
        D --> E[Fixed create_all_features]
        E --> F[Comprehensive validation report]
    end
    
    subgraph "Enhanced Monthly S3 Processing"
        G[S3 Monthly DBN Files] --> H[Enhanced process_monthly_chunks_fixed.py]
        H --> I[Improved download with retry]
        I --> J[Enhanced process_monthly_data]
        J --> K[Enhanced clean_price_data]
        K --> L[Fixed WeightedLabelingEngine]
        L --> M[Fixed create_all_features]
        M --> N[Enhanced statistics collection]
        N --> O[Improved upload with metadata]
    end
    
    subgraph "Enhanced Data Quality & Monitoring"
        P[Enhanced fix_data_quality_issues.py] --> Q[Improved InputDataFrame validation]
        Q --> R[Enhanced OutputDataFrame validation]
        R --> S[Comprehensive statistics logging]
        S --> T[Monthly quality reports]
        T --> U[Reprocessing recommendations]
    end
```

## Components and Interfaces

### 1. Enhanced WeightedLabelingEngine (src/data_pipeline/weighted_labeling.py)

**Current Issues to Fix:**
- Memory leaks in chunked processing
- Rollover detection edge cases
- Performance bottlenecks in large datasets

**Enhancements:**

```python
class EnhancedWeightedLabelingEngine(WeightedLabelingEngine):
    """Enhanced version with fixes and improvements"""
    
    def __init__(self, config: LabelingConfig = None):
        super().__init__(config)
        self.rollover_stats = RolloverStatistics()
        self.memory_monitor = MemoryMonitor()
    
    def process_dataframe(self, df: pd.DataFrame, validate_performance: bool = True) -> pd.DataFrame:
        """Enhanced processing with better memory management and rollover tracking"""
        # Fix memory issues
        self.memory_monitor.start_monitoring()
        
        # Enhanced rollover detection
        rollover_events = self._detect_and_track_rollovers(df)
        
        # Process with improved chunking
        result = self._process_with_enhanced_chunking(df, rollover_events)
        
        # Collect comprehensive statistics
        stats = self._collect_processing_statistics(result, rollover_events)
        
        return result, stats
```

**Key Fixes:**
1. **Memory Management**: Fix memory leaks in `_process_chunked()` method
2. **Rollover Detection**: Enhance `_detect_contract_rolls()` with better edge case handling
3. **Performance**: Optimize vectorized calculations in `_calculate_weights_vectorized()`
4. **Statistics**: Add comprehensive rollover and processing statistics collection

### 2. Enhanced Feature Engineering (src/data_pipeline/features.py)

**Current Issues to Fix:**
- NaN handling in rolling calculations
- Feature value validation
- Memory usage optimization

**Enhancements:**

```python
class EnhancedFeatureEngine:
    """Enhanced feature engineering with better validation and error handling"""
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature creation with comprehensive validation"""
        
        # Pre-validation
        self._validate_input_data(df)
        
        # Create features with enhanced error handling
        feature_results = {}
        
        for category in FEATURE_CATEGORIES:
            try:
                features = self._create_feature_category(df, category)
                feature_results[category] = features
                self._validate_feature_category(features, category)
            except Exception as e:
                self._handle_feature_error(category, e)
        
        # Combine and validate final result
        result_df = self._combine_features(df, feature_results)
        self._validate_final_features(result_df)
        
        return result_df
```

**Key Fixes:**
1. **NaN Handling**: Improve rolling calculation edge cases
2. **Validation**: Add feature value range validation
3. **Error Handling**: Better error recovery for individual feature categories
4. **Memory**: Optimize memory usage in feature calculations

### 3. Enhanced Monthly Processing Pipeline

**Current Issues to Fix:**
- S3 path discovery reliability
- Error recovery and retry logic
- Statistics collection and logging

**Enhanced Architecture:**

```python
class EnhancedMonthlyProcessor:
    """Enhanced monthly processing with better error handling and statistics"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.statistics_collector = MonthlyStatisticsCollector()
        self.error_handler = EnhancedErrorHandler()
        self.progress_tracker = ProgressTracker()
    
    def process_all_months(self):
        """Process all months with enhanced error handling and statistics"""
        
        monthly_files = self._discover_monthly_files()
        to_process = self._filter_unprocessed_months(monthly_files)
        
        for month_info in to_process:
            try:
                result = self._process_single_month_enhanced(month_info)
                self.statistics_collector.record_success(month_info, result)
            except Exception as e:
                self.error_handler.handle_month_failure(month_info, e)
                continue
        
        # Generate final report
        self._generate_processing_report()
```

### 4. Enhanced Data Quality System

**Current Issues to Fix:**
- Edge cases in price validation
- Timezone handling improvements
- Better outlier detection

**Enhanced Data Quality Pipeline:**

```python
class EnhancedDataQualitySystem:
    """Enhanced data quality with comprehensive validation"""
    
    def __init__(self):
        self.price_validator = EnhancedPriceValidator()
        self.timezone_handler = EnhancedTimezoneHandler()
        self.outlier_detector = OutlierDetector()
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Enhanced cleaning with comprehensive reporting"""
        
        # Enhanced price cleaning
        df_clean, price_issues = self.price_validator.clean_prices(df)
        
        # Enhanced timezone validation
        timezone_issues = self.timezone_handler.validate_rth_data(df_clean)
        
        # Outlier detection
        outlier_issues = self.outlier_detector.detect_outliers(df_clean)
        
        # Generate comprehensive report
        quality_report = DataQualityReport(price_issues, timezone_issues, outlier_issues)
        
        return df_clean, quality_report
```

## Data Models

### 1. Enhanced Statistics Model

```python
@dataclass
class MonthlyProcessingStatistics:
    """Comprehensive statistics for monthly processing"""
    
    # Basic info
    month_str: str
    processing_date: datetime
    processing_time_minutes: float
    
    # Data volume
    raw_bars: int
    rth_bars: int
    final_bars: int
    
    # Data quality
    price_issues_fixed: int
    timezone_issues: int
    outliers_detected: int
    
    # Rollover events
    rollover_events: List[RolloverEvent]
    bars_excluded_rollover: int
    
    # Labeling results
    mode_statistics: Dict[str, ModeStatistics]
    
    # Feature engineering
    feature_statistics: Dict[str, FeatureStatistics]
    
    # Quality flags
    requires_reprocessing: bool
    quality_score: float
    
    def to_json(self) -> str:
        """Serialize to JSON for S3 storage"""
        return json.dumps(asdict(self), default=str, indent=2)

@dataclass
class RolloverEvent:
    """Details of a contract rollover event"""
    timestamp: datetime
    price_gap: float
    bars_affected: int
    gap_direction: str  # 'up' or 'down'

@dataclass
class ModeStatistics:
    """Statistics for a single trading mode"""
    win_rate: float
    total_winners: int
    total_samples: int
    avg_weight: float
    weight_distribution: Dict[str, float]  # percentiles
    quality_flags: List[str]

@dataclass
class FeatureStatistics:
    """Statistics for feature engineering"""
    features_generated: int
    nan_percentages: Dict[str, float]
    outlier_counts: Dict[str, int]
    value_ranges: Dict[str, Tuple[float, float]]
    quality_score: float
```

### 2. Enhanced Configuration Model

```python
@dataclass
class EnhancedLabelingConfig(LabelingConfig):
    """Enhanced configuration with additional parameters"""
    
    # Rollover detection
    rollover_threshold_points: float = 20.0
    rollover_affected_bars: int = 5
    
    # Memory management
    max_memory_gb: float = 8.0
    gc_frequency: int = 10  # chunks between garbage collection
    
    # Statistics collection
    collect_detailed_stats: bool = True
    save_rollover_details: bool = True
    
    # Error handling
    max_retries: int = 3
    retry_delay_seconds: int = 30
    
    # Quality thresholds
    min_win_rate: float = 0.05
    max_win_rate: float = 0.50
    max_nan_percentage: float = 0.35
    
    # S3 configuration
    s3_bucket: str = "es-1-second-data"
    s3_retry_attempts: int = 3
    s3_timeout_seconds: int = 300
```

## Error Handling

### 1. Enhanced Error Recovery System

```python
class EnhancedErrorHandler:
    """Comprehensive error handling with recovery strategies"""
    
    def __init__(self, config: EnhancedLabelingConfig):
        self.config = config
        self.retry_strategies = {
            'S3Error': self._handle_s3_error,
            'MemoryError': self._handle_memory_error,
            'DataQualityError': self._handle_data_quality_error,
            'ProcessingError': self._handle_processing_error
        }
    
    def handle_month_failure(self, month_info: dict, error: Exception) -> bool:
        """Handle month processing failure with appropriate recovery strategy"""
        
        error_type = type(error).__name__
        
        if error_type in self.retry_strategies:
            return self.retry_strategies[error_type](month_info, error)
        else:
            return self._handle_unknown_error(month_info, error)
    
    def _handle_s3_error(self, month_info: dict, error: Exception) -> bool:
        """Handle S3-related errors with exponential backoff retry"""
        
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(self.config.retry_delay_seconds * (2 ** attempt))
                # Retry the S3 operation
                return self._retry_s3_operation(month_info)
            except Exception as retry_error:
                if attempt == self.config.max_retries - 1:
                    self._log_permanent_failure(month_info, retry_error)
                    return False
        
        return False
```

### 2. Memory Management Enhancements

```python
class MemoryMonitor:
    """Enhanced memory monitoring and management"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.memory_history = []
    
    def check_memory_usage(self) -> float:
        """Check current memory usage and trigger cleanup if needed"""
        
        current_memory = psutil.virtual_memory().used / (1024**3)
        self.memory_history.append(current_memory)
        
        if current_memory > self.max_memory_gb * 0.8:  # 80% threshold
            self._trigger_memory_cleanup()
        
        return current_memory
    
    def _trigger_memory_cleanup(self):
        """Trigger aggressive memory cleanup"""
        gc.collect()
        
        # Force cleanup of large objects
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
```

## Testing Strategy

### 1. Enhanced Desktop Validation

**Test Scenarios:**
1. **End-to-End Pipeline Test**: Validate complete pipeline on 30-day sample
2. **Rollover Detection Test**: Test rollover detection with synthetic rollover events
3. **Memory Stress Test**: Process large chunks to validate memory management
4. **Error Recovery Test**: Test error handling with corrupted data

**Enhanced Test Implementation:**

```python
class EnhancedPipelineValidator:
    """Comprehensive pipeline validation"""
    
    def run_comprehensive_validation(self) -> ValidationReport:
        """Run all validation tests and generate comprehensive report"""
        
        results = {}
        
        # Test 1: End-to-end pipeline
        results['end_to_end'] = self._test_end_to_end_pipeline()
        
        # Test 2: Rollover detection
        results['rollover_detection'] = self._test_rollover_detection()
        
        # Test 3: Memory management
        results['memory_management'] = self._test_memory_management()
        
        # Test 4: Error recovery
        results['error_recovery'] = self._test_error_recovery()
        
        # Test 5: Data quality
        results['data_quality'] = self._test_data_quality_fixes()
        
        return ValidationReport(results)
```

### 2. Monthly Processing Validation

**Validation Strategy:**
1. **Single Month Test**: Test complete monthly processing on one month
2. **Multi-Month Test**: Test processing multiple months with error injection
3. **S3 Integration Test**: Test S3 upload/download with retry logic
4. **Statistics Validation**: Validate statistics collection and reporting

## Performance Optimizations

### 1. Memory Optimization Strategy

```python
class MemoryOptimizedProcessor:
    """Memory-optimized processing strategies"""
    
    def process_large_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process large dataset with memory optimization"""
        
        # Strategy 1: Chunked processing with overlap
        chunk_size = self._calculate_optimal_chunk_size(df)
        
        # Strategy 2: Feature-wise processing
        features_by_memory_usage = self._sort_features_by_memory_usage()
        
        # Strategy 3: Incremental garbage collection
        for i, chunk in enumerate(self._get_chunks(df, chunk_size)):
            result_chunk = self._process_chunk_optimized(chunk)
            
            if i % 10 == 0:  # GC every 10 chunks
                gc.collect()
        
        return self._combine_results(result_chunks)
```

### 2. S3 Optimization Strategy

```python
class OptimizedS3Handler:
    """Optimized S3 operations with retry and compression"""
    
    def upload_with_optimization(self, file_path: str, s3_key: str) -> bool:
        """Upload with compression and retry logic"""
        
        # Compress file before upload
        compressed_file = self._compress_parquet_file(file_path)
        
        # Upload with retry and progress tracking
        return self._upload_with_retry(compressed_file, s3_key)
    
    def _compress_parquet_file(self, file_path: str) -> str:
        """Optimize Parquet compression for S3 storage"""
        
        df = pd.read_parquet(file_path)
        
        # Optimize data types
        df_optimized = self._optimize_data_types(df)
        
        # Save with optimal compression
        compressed_path = file_path.replace('.parquet', '_compressed.parquet')
        df_optimized.to_parquet(
            compressed_path,
            compression='snappy',
            index=False,
            engine='pyarrow'
        )
        
        return compressed_path
```

## Deployment Strategy

### 1. Phased Deployment Approach

**Phase 1: Desktop Validation Fixes**
- Fix WeightedLabelingEngine issues
- Enhance rollover detection
- Fix feature engineering issues
- Validate on 30-day sample

**Phase 2: Monthly Processing Enhancements**
- Fix S3 integration issues
- Enhance error handling
- Add comprehensive statistics
- Test on single month

**Phase 3: Full Pipeline Deployment**
- Deploy enhanced monthly processing
- Process all 180+ months
- Monitor and optimize performance
- Generate final dataset

### 2. Monitoring and Alerting

```python
class ProcessingMonitor:
    """Monitor processing progress and alert on issues"""
    
    def __init__(self):
        self.alerts = []
        self.metrics = {}
    
    def monitor_monthly_processing(self, month_info: dict, stats: MonthlyProcessingStatistics):
        """Monitor monthly processing and generate alerts"""
        
        # Check processing time
        if stats.processing_time_minutes > 45:  # Alert if > 45 minutes
            self.alerts.append(f"Slow processing for {month_info['month_str']}: {stats.processing_time_minutes:.1f} min")
        
        # Check data quality
        if stats.quality_score < 0.8:  # Alert if quality score < 80%
            self.alerts.append(f"Low quality score for {month_info['month_str']}: {stats.quality_score:.2f}")
        
        # Check win rates
        for mode, mode_stats in stats.mode_statistics.items():
            if not (0.05 <= mode_stats.win_rate <= 0.50):
                self.alerts.append(f"Unusual win rate for {mode} in {month_info['month_str']}: {mode_stats.win_rate:.1%}")
```

This design builds on your existing architecture while addressing the specific issues and enhancements needed for reliable data processing validation and monthly S3 processing. The focus is on fixing existing code rather than rebuilding from scratch, while adding the necessary enhancements for production-scale processing.