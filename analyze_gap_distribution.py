"""
Analyze gap distribution by hour to verify gaps are during lull periods
"""
import pandas as pd
import pytz

print("=" * 80)
print("GAP DISTRIBUTION ANALYSIS - OCTOBER 2025")
print("=" * 80)

# Load processed data
df = pd.read_parquet("oct2025_processed_FINAL.parquet")

print(f"\nTotal rows: {len(df):,}")
print(f"Rows with volume=0 (filled gaps): {(df['volume'] == 0).sum():,}")
print(f"Rows with volume>0 (actual data): {(df['volume'] > 0).sum():,}")
print(f"Gap percentage: {(df['volume'] == 0).sum() / len(df) * 100:.1f}%")

# Convert to Central Time
central_tz = pytz.timezone('US/Central')
df['timestamp_ct'] = df['timestamp'].dt.tz_convert(central_tz)
df['hour_ct'] = df['timestamp_ct'].dt.hour
df['is_gap'] = df['volume'] == 0

# Analyze by hour
print("\n" + "=" * 80)
print("GAP DISTRIBUTION BY HOUR (Central Time)")
print("=" * 80)
print(f"\n{'Hour':<10} {'Total Bars':<12} {'Gaps':<12} {'Gap %':<10} {'Session'}")
print("-" * 80)

for hour in range(7, 15):  # 07:00 to 14:59
    hour_data = df[df['hour_ct'] == hour]
    total_bars = len(hour_data)
    gaps = (hour_data['volume'] == 0).sum()
    gap_pct = (gaps / total_bars * 100) if total_bars > 0 else 0
    
    # Classify session period
    if hour == 7:
        session = "Pre-Open"
    elif hour in [8, 9]:
        session = "Open"
    elif hour in [10, 11]:
        session = "Morning"
    elif hour in [12, 13]:
        session = "Lunch 🍽️"
    elif hour == 14:
        session = "Afternoon"
    else:
        session = ""
    
    print(f"{hour:02d}:00-{hour:02d}:59  {total_bars:>10,}  {gaps:>10,}  {gap_pct:>8.1f}%  {session}")

# More detailed lunch analysis
print("\n" + "=" * 80)
print("DETAILED LUNCH PERIOD ANALYSIS (12:00-13:59 CT)")
print("=" * 80)

lunch_data = df[df['hour_ct'].isin([12, 13])]
print(f"\nLunch period total bars: {len(lunch_data):,}")
print(f"Lunch period gaps: {(lunch_data['volume'] == 0).sum():,}")
print(f"Lunch period gap %: {(lunch_data['volume'] == 0).sum() / len(lunch_data) * 100:.1f}%")

# Compare to most active period (open)
open_data = df[df['hour_ct'].isin([8, 9])]
print(f"\nOpen period total bars: {len(open_data):,}")
print(f"Open period gaps: {(open_data['volume'] == 0).sum():,}")
print(f"Open period gap %: {(open_data['volume'] == 0).sum() / len(open_data) * 100:.1f}%")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

total_gaps = (df['volume'] == 0).sum()
lunch_gaps = (lunch_data['volume'] == 0).sum()
lunch_gap_pct = (lunch_gaps / total_gaps * 100) if total_gaps > 0 else 0

print(f"\nTotal gaps filled: {total_gaps:,}")
print(f"Gaps during lunch (12:00-13:59 CT): {lunch_gaps:,} ({lunch_gap_pct:.1f}% of all gaps)")
print(f"\nLunch hours represent: {len(lunch_data) / len(df) * 100:.1f}% of trading time")
print(f"But contain: {lunch_gap_pct:.1f}% of all gaps")

if lunch_gap_pct > (len(lunch_data) / len(df) * 100):
    print(f"\n✅ CONFIRMED: Gaps are disproportionately during lunch lull")
    print(f"   Lunch is {lunch_gap_pct / (len(lunch_data) / len(df) * 100):.1f}x more likely to have gaps")
else:
    print(f"\n⚠️  Gaps are NOT concentrated during lunch")

# Show gap distribution across all hours as percentages
print("\n" + "=" * 80)
print("GAP DISTRIBUTION ACROSS HOURS")
print("=" * 80)

gap_by_hour = df[df['is_gap']].groupby('hour_ct').size()
total_gaps_check = gap_by_hour.sum()

print(f"\n{'Hour':<10} {'Gaps':<12} {'% of Total Gaps'}")
print("-" * 50)
for hour in range(7, 15):
    gaps = gap_by_hour.get(hour, 0)
    pct = (gaps / total_gaps_check * 100) if total_gaps_check > 0 else 0
    print(f"{hour:02d}:00-{hour:02d}:59  {gaps:>10,}  {pct:>15.1f}%")

print("\n" + "=" * 80)
