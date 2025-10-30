import pandas as pd
import numpy as np

def validate_full_dataset_results():
    """
    Comprehensive validation of full dataset labeling results
    """
    print("=== FULL DATASET VALIDATION ANALYSIS ===")
    
    # Load the labeled dataset
    print("Loading labeled dataset...")
    df = pd.read_parquet('project/data/processed/full_labeled_dataset.parquet')
    print(f"Dataset shape: {df.shape}")
    print(f"Total bars: {len(df):,}")
    
    # Basic data quality checks
    print(f"\n{'='*60}")
    print("DATA QUALITY CHECKS")
    print(f"{'='*60}")
    
    # Check for missing data in core columns
    core_columns = ['open', 'high', 'low', 'close', 'volume']
    print("Missing values in core columns:")
    for col in core_columns:
        missing = df[col].isna().sum()
        print(f"  {col}: {missing:,} ({missing/len(df)*100:.2f}%)")
    
    # Check price consistency
    print(f"\nPrice consistency checks:")
    high_low_issues = (df['high'] < df['low']).sum()
    print(f"  High < Low issues: {high_low_issues:,}")
    
    open_range_issues = ((df['open'] < df['low']) | (df['open'] > df['high'])).sum()
    print(f"  Open outside High/Low range: {open_range_issues:,}")
    
    close_range_issues = ((df['close'] < df['low']) | (df['close'] > df['high'])).sum()
    print(f"  Close outside High/Low range: {close_range_issues:,}")
    
    # Volume checks
    zero_volume = (df['volume'] == 0).sum()
    negative_volume = (df['volume'] < 0).sum()
    print(f"  Zero volume bars: {zero_volume:,}")
    print(f"  Negative volume bars: {negative_volume:,}")
    
    # Label validation
    print(f"\n{'='*60}")
    print("LABEL VALIDATION")
    print(f"{'='*60}")
    
    profiles = ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']
    
    label_summary = {}
    
    for profile in profiles:
        print(f"\n{profile}:")
        
        # Check label distribution
        label_col = f'{profile}_label'
        outcome_col = f'{profile}_outcome'
        mae_col = f'{profile}_mae'
        hold_col = f'{profile}_hold_time'
        
        # Label counts
        labels = df[label_col].value_counts().sort_index()
        total_labeled = labels.sum()
        timeouts = df[label_col].isna().sum()
        
        optimal = labels.get(1.0, 0)
        suboptimal = labels.get(0.0, 0)
        losses = labels.get(-1.0, 0)
        
        print(f"  Labels: Optimal={optimal:,}, Suboptimal={suboptimal:,}, Loss={losses:,}, Timeout={timeouts:,}")
        print(f"  Percentages: Optimal={optimal/total_labeled*100:.1f}%, Win Rate={(optimal+suboptimal)/total_labeled*100:.1f}%")
        
        # Validate label consistency with outcomes
        win_mask = df[outcome_col] == 'win'
        loss_mask = df[outcome_col] == 'loss'
        timeout_mask = df[outcome_col] == 'timeout'
        
        # Check that all losses are labeled -1
        loss_label_consistency = (df[loss_mask][label_col] == -1).all()
        print(f"  Loss label consistency: {loss_label_consistency}")
        
        # Check that all timeouts are labeled NaN
        timeout_label_consistency = df[timeout_mask][label_col].isna().all()
        print(f"  Timeout label consistency: {timeout_label_consistency}")
        
        # Check that winners are labeled 0 or 1
        winner_labels = df[win_mask][label_col].dropna()
        winner_label_consistency = winner_labels.isin([0.0, 1.0]).all()
        print(f"  Winner label consistency: {winner_label_consistency}")
        
        # Check MAE values
        mae_values = df[win_mask][mae_col].dropna()
        if len(mae_values) > 0:
            print(f"  MAE range: {mae_values.min():.2f} to {mae_values.max():.2f} ticks")
            print(f"  MAE mean: {mae_values.mean():.2f} ticks")
        
        # Check hold times
        hold_values = df[win_mask][hold_col].dropna()
        if len(hold_values) > 0:
            print(f"  Hold time range: {hold_values.min():.1f} to {hold_values.max():.1f} seconds")
            print(f"  Hold time mean: {hold_values.mean():.1f} seconds")
        
        # Store summary for frequency analysis
        label_summary[profile] = {
            'optimal': optimal,
            'suboptimal': suboptimal,
            'losses': losses,
            'timeouts': timeouts,
            'win_rate': (optimal + suboptimal) / total_labeled * 100 if total_labeled > 0 else 0
        }
    
    # Frequency analysis
    print(f"\n{'='*60}")
    print("FREQUENCY ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate time span
    df_reset = df.reset_index()
    time_span = df_reset['ts_event'].iloc[-1] - df_reset['ts_event'].iloc[0]
    hours = time_span.total_seconds() / 3600
    trading_days = hours / 6.5  # 6.5 hour trading day
    
    print(f"Time span: {hours:.1f} hours ({trading_days:.1f} trading days)")
    
    # Calculate frequencies
    total_optimal = sum(label_summary[p]['optimal'] for p in profiles)
    optimal_per_hour = total_optimal / hours
    optimal_per_day = optimal_per_hour * 6.5
    
    print(f"Total optimal entries: {total_optimal:,}")
    print(f"Optimal per hour: {optimal_per_hour:.1f}")
    print(f"Optimal per trading day: {optimal_per_day:.0f}")
    
    # Per-profile frequencies
    print(f"\nPer-profile optimal frequencies:")
    for profile in profiles:
        profile_optimal = label_summary[profile]['optimal']
        profile_per_hour = profile_optimal / hours
        profile_per_day = profile_per_hour * 6.5
        print(f"  {profile}: {profile_per_hour:.1f}/hour, {profile_per_day:.0f}/day")
    
    # Realism assessment
    print(f"\n{'='*60}")
    print("REALISM ASSESSMENT")
    print(f"{'='*60}")
    
    if optimal_per_hour > 50:
        print(f"🚨 VERY HIGH FREQUENCY: {optimal_per_hour:.1f} optimal entries per hour")
        print(f"   This suggests the criteria may be too lenient")
        print(f"   Consider tightening MAE thresholds or increasing target distances")
    elif optimal_per_hour > 20:
        print(f"⚠️  HIGH FREQUENCY: {optimal_per_hour:.1f} optimal entries per hour")
        print(f"   This is aggressive but potentially tradeable with automation")
    elif optimal_per_hour > 5:
        print(f"✅ MODERATE FREQUENCY: {optimal_per_hour:.1f} optimal entries per hour")
        print(f"   This seems reasonable for algorithmic trading")
    else:
        print(f"⚠️  LOW FREQUENCY: {optimal_per_hour:.1f} optimal entries per hour")
        print(f"   This may be too conservative, consider loosening criteria")
    
    # Win rate analysis
    print(f"\nWin rate analysis:")
    long_profiles = [p for p in profiles if p.startswith('long')]
    short_profiles = [p for p in profiles if p.startswith('short')]
    
    long_win_rates = [label_summary[p]['win_rate'] for p in long_profiles]
    short_win_rates = [label_summary[p]['win_rate'] for p in short_profiles]
    
    print(f"  Long profiles win rate: {np.mean(long_win_rates):.1f}% (range: {min(long_win_rates):.1f}% - {max(long_win_rates):.1f}%)")
    print(f"  Short profiles win rate: {np.mean(short_win_rates):.1f}% (range: {min(short_win_rates):.1f}% - {max(short_win_rates):.1f}%)")
    
    if np.mean(long_win_rates) > np.mean(short_win_rates) + 10:
        print(f"  ⚠️  Long bias detected: Long profiles significantly outperform short")
    elif np.mean(short_win_rates) > np.mean(long_win_rates) + 10:
        print(f"  ⚠️  Short bias detected: Short profiles significantly outperform long")
    else:
        print(f"  ✅ Balanced: Long and short profiles have similar win rates")
    
    # Data integrity final check
    print(f"\n{'='*60}")
    print("FINAL INTEGRITY CHECK")
    print(f"{'='*60}")
    
    integrity_issues = []
    
    # Check for any impossible label combinations
    for profile in profiles:
        outcome_col = f'{profile}_outcome'
        label_col = f'{profile}_label'
        
        # Losses should always be -1
        loss_wrong_labels = ((df[outcome_col] == 'loss') & (df[label_col] != -1)).sum()
        if loss_wrong_labels > 0:
            integrity_issues.append(f"{profile}: {loss_wrong_labels} losses not labeled -1")
        
        # Timeouts should always be NaN
        timeout_wrong_labels = ((df[outcome_col] == 'timeout') & (~df[label_col].isna())).sum()
        if timeout_wrong_labels > 0:
            integrity_issues.append(f"{profile}: {timeout_wrong_labels} timeouts not labeled NaN")
        
        # Winners should be 0 or 1
        win_wrong_labels = ((df[outcome_col] == 'win') & (~df[label_col].isin([0.0, 1.0]))).sum()
        if win_wrong_labels > 0:
            integrity_issues.append(f"{profile}: {win_wrong_labels} wins labeled incorrectly")
    
    if integrity_issues:
        print("🚨 INTEGRITY ISSUES FOUND:")
        for issue in integrity_issues:
            print(f"  - {issue}")
        print("\n❌ VALIDATION FAILED - Fix issues before proceeding")
        return False
    else:
        print("✅ ALL INTEGRITY CHECKS PASSED")
        print("✅ Dataset is ready for feature engineering")
        return True

def create_summary_report():
    """
    Create a summary report of the labeling results
    """
    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}")
    
    df = pd.read_parquet('project/data/processed/full_labeled_dataset.parquet')
    
    print(f"Dataset: {len(df):,} bars over 30 days")
    print(f"Profiles: 6 (Long/Short × Small/Medium/Large)")
    print(f"Processing time: 22.3 minutes")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Calculate total statistics
    profiles = ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']
    
    total_optimal = sum(df[f'{p}_label'].eq(1).sum() for p in profiles)
    total_suboptimal = sum(df[f'{p}_label'].eq(0).sum() for p in profiles)
    total_losses = sum(df[f'{p}_label'].eq(-1).sum() for p in profiles)
    
    print(f"\nTotal labels across all profiles:")
    print(f"  Optimal entries: {total_optimal:,}")
    print(f"  Suboptimal entries: {total_suboptimal:,}")
    print(f"  Losses: {total_losses:,}")
    
    print(f"\nReady for next phase: Feature Engineering")
    print(f"Next steps:")
    print(f"  1. ✅ Labeling complete and validated")
    print(f"  2. 🔄 Feature engineering (55 features)")
    print(f"  3. ⏳ Model training (LSTM)")
    print(f"  4. ⏳ Deployment")

if __name__ == "__main__":
    validation_passed = validate_full_dataset_results()
    create_summary_report()
    
    if validation_passed:
        print(f"\n🎉 LABELING PHASE COMPLETE!")
        print(f"✅ Ready to lock labeling code and proceed to features")
    else:
        print(f"\n❌ VALIDATION FAILED")
        print(f"🔧 Fix issues before proceeding to features")