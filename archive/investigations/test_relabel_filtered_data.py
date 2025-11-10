"""
Re-label the contract-filtered data to verify win rates are now reasonable
"""
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_pipeline.weighted_labeling import WeightedLabelingEngine
from src.data_pipeline.features import create_all_features

def relabel_filtered_data():
    """Re-label contract-filtered data"""
    
    print("=" * 80)
    print("RE-LABELING CONTRACT-FILTERED DATA - July 2010")
    print("=" * 80)
    
    # Load contract-filtered data
    filtered_path = r"C:\Users\jdotzler\Desktop\monthly_2010-07_contract_filtered.parquet"
    print(f"\n📖 Loading filtered data: {filtered_path}")
    
    df = pd.read_parquet(filtered_path)
    print(f"   Loaded: {len(df):,} rows")
    
    # Apply weighted labeling
    print(f"\n🏷️  Applying weighted labeling...")
    try:
        engine = WeightedLabelingEngine()
        df_labeled = engine.process_dataframe(df, validate_performance=False)
        print(f"   ✅ Labeling complete: {len(df_labeled.columns)} columns")
    except Exception as e:
        print(f"   ❌ Labeling failed: {e}")
        return None
    
    # Check win rates
    print(f"\n📊 WIN RATES AFTER CONTRACT FILTERING:")
    label_cols = [c for c in df_labeled.columns if c.startswith('label_')]
    
    for col in label_cols:
        win_rate = df_labeled[col].mean()
        wins = (df_labeled[col] == 1).sum()
        losses = (df_labeled[col] == 0).sum()
        print(f"   {col:<25}: {win_rate:>6.1%} ({wins:>6,} wins / {losses:>6,} losses)")
    
    # Check if win rates are reasonable
    print(f"\n🔍 WIN RATE ANALYSIS:")
    long_win_rate = df_labeled['label_low_vol_long'].mean()
    short_win_rate = df_labeled['label_low_vol_short'].mean()
    
    if 0.15 <= long_win_rate <= 0.35 and 0.15 <= short_win_rate <= 0.35:
        print(f"   ✅ WIN RATES ARE REASONABLE!")
        print(f"      Long: {long_win_rate:.1%} (expected 15-30%)")
        print(f"      Short: {short_win_rate:.1%} (expected 15-30%)")
        print(f"      Contract filtering FIXED the issue!")
    elif long_win_rate < 0.10 or short_win_rate > 0.70:
        print(f"   ⚠️  WIN RATES STILL PROBLEMATIC")
        print(f"      Long: {long_win_rate:.1%}")
        print(f"      Short: {short_win_rate:.1%}")
        print(f"      May need additional filtering or investigation")
    else:
        print(f"   ⚙️  WIN RATES IMPROVED BUT NOT IDEAL")
        print(f"      Long: {long_win_rate:.1%}")
        print(f"      Short: {short_win_rate:.1%}")
    
    # Apply feature engineering
    print(f"\n🔧 Applying feature engineering...")
    try:
        df_final = create_all_features(df_labeled)
        print(f"   ✅ Features complete: {len(df_final.columns)} columns")
    except Exception as e:
        print(f"   ❌ Feature engineering failed: {e}")
        return df_labeled
    
    # Save final processed data
    output_path = r"C:\Users\jdotzler\Desktop\monthly_2010-07_FIXED.parquet"
    df_final.to_parquet(output_path, index=False)
    print(f"\n💾 Saved final processed data to: {output_path}")
    
    print(f"\n" + "=" * 80)
    print("RE-LABELING COMPLETE")
    print("=" * 80)
    
    return df_final

if __name__ == "__main__":
    df_final = relabel_filtered_data()
