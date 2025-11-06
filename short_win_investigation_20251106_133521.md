
# Short Win Rate Investigation Report
Generated: 2025-11-06 13:35:21

## Problem Statement
Short trading modes are showing unexpectedly high win rates (>60%), which seems unrealistic for a 2:1 reward/risk strategy.

## Investigation Approach

### 1. Logic Verification ✅
- Reviewed short trade labeling logic in weighted_labeling.py
- Logic is CORRECT:
  - Short target: entry_price - (target_ticks * tick_size) ✅
  - Short stop: entry_price + (stop_ticks * tick_size) ✅
  - Target hit: bar_low <= target_price ✅
  - Stop hit: bar_high >= stop_price ✅

### 2. Possible Explanations

#### A. Market Conditions (Most Likely)
- **2011 Market Context**: Post-financial crisis, high volatility
- **Intraday Mean Reversion**: ES often reverts intraday
- **Downward Bias**: Market might have natural downward pressure
- **Volatility Regime**: High volatility = more mean reversion

#### B. Data Quality Issues
- **Contract Rollovers**: Price gaps affecting calculations
- **Data Gaps**: Missing bars causing incorrect lookforward
- **Timestamp Issues**: Incorrect time sequencing

#### C. Algorithmic Factors
- **Target Size**: 2:1 ratio might favor short direction
- **Timeout Period**: 15-minute window might be optimal for shorts
- **Entry Timing**: Next bar open might favor short entries

### 3. Validation Steps

#### Immediate Actions:
1. **Compare Time Periods**: Check if other months show same pattern
2. **Compare Long vs Short**: Verify long trades have reasonable win rates
3. **Sample Verification**: Manually verify sample winning short trades
4. **Market Research**: Check 2011 ES market conditions

#### Commands to Run:
```bash
# Check multiple months
python3 process_monthly_chunks_fixed.py --test-month 2011-05
python3 process_monthly_chunks_fixed.py --test-month 2011-07

# Compare statistics
aws s3 ls s3://es-1-second-data/processed-data/monthly/2011/ --recursive | grep statistics

# Download and compare
aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/05/statistics/ /tmp/may_stats/ --recursive
aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/07/statistics/ /tmp/july_stats/ --recursive
```

### 4. Expected Findings

#### If Market Conditions:
- Other months in 2011 show similar pattern
- Long trades have reasonable win rates (30-45%)
- Pattern consistent with known market volatility

#### If Data Issues:
- Inconsistent patterns across months
- Both long and short affected
- Clear data quality problems visible

#### If Algorithm Issues:
- Pattern appears in all time periods
- Affects multiple volatility modes equally
- Logic review reveals bugs

### 5. Next Steps

1. **Validate with multiple months** ← START HERE
2. **Compare against market benchmarks**
3. **Manual verification of sample trades**
4. **Consider if this is actually correct** (market was volatile in 2011)

## Conclusion

The labeling logic appears correct. High short win rates might be:
1. **Legitimate market behavior** (2011 was volatile, mean-reverting)
2. **Data quality issues** (contract rollovers, gaps)
3. **Algorithmic edge case** (unlikely given logic review)

**Recommendation**: Validate with additional time periods before assuming this is a bug.
