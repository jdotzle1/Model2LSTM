"""
Test contract filtering on July 2010 processed data
"""
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_pipeline.contract_filtering import detect_and_filter_contracts, validate_contract_filtering

def test_contract_filtering():
    """Test contract filtering on July 2010 data"""
    
    print("=" * 80)
    print("TESTING CONTRACT FILTERING - July 2010")
    print("=" * 80)
    
    # Load processed data
    parquet_path = r"C:\Users\jdotzler\Desktop\monthly_2010-07_20251107_152756.parquet"
    print(f"\n📖 Loading: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"   Loaded: {len(df):,} rows")
    
    # Extract only the columns needed for filtering
    df_for_filtering = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    print(f"\n🔍 BEFORE FILTERING:")
    print(f"   Total rows: {len(df_for_filtering):,}")
    
    # Check for price gaps
    df_sorted = df_for_filtering.sort_values('timestamp')
    gaps_before = abs(df_sorted['close'].diff()) > 5.0
    print(f"   Large price gaps (>5 pts): {gaps_before.sum()}")
    
    # Apply contract filtering
    print(f"\n🔄 APPLYING CONTRACT FILTERING...")
    df_filtered, stats = detect_and_filter_contracts(df_for_filtering)
    
    # Validate
    print(f"\n✅ AFTER FILTERING:")
    print(f"   Total rows: {len(df_filtered):,}")
    
    validation = validate_contract_filtering(df_for_filtering, df_filtered)
    
    # Show detailed stats
    print(f"\n📊 DETAILED STATISTICS:")
    print(f"   Rows removed: {stats['removed_rows']:,} ({stats['removal_percentage']:.1f}%)")
    print(f"   Days with filtering: {stats['days_with_contract_filtering']}")
    print(f"   Gaps removed: {validation['gaps_removed']} ({validation['gap_reduction_pct']:.1f}%)")
    
    # Show days with significant filtering
    print(f"\n📅 DAYS WITH SIGNIFICANT FILTERING (>5%):")
    significant_days = [(date, info) for date, info in stats['daily_contract_info'].items() 
                       if info['removal_pct'] > 5.0]
    
    if significant_days:
        for date, info in sorted(significant_days):
            print(f"   {date}: {info['removed_bars']:,} bars removed ({info['removal_pct']:.1f}%) - Volume: {info['dominant_volume']:,}")
    else:
        print(f"   No days with >5% filtering")
    
    # Save filtered data for re-labeling
    output_path = r"C:\Users\jdotzler\Desktop\monthly_2010-07_contract_filtered.parquet"
    df_filtered.to_parquet(output_path, index=False)
    print(f"\n💾 Saved filtered data to: {output_path}")
    
    print(f"\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Re-run labeling and feature engineering on filtered data")
    print(f"   2. Check if win rates are now reasonable (15-30%)")
    print(f"   3. If successful, deploy to EC2 pipeline")
    
    return df_filtered, stats, validation

if __name__ == "__main__":
    df_filtered, stats, validation = test_contract_filtering()
