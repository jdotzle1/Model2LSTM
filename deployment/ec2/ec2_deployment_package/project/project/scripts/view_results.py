import pandas as pd

# Configure pandas to show all columns and more rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 15)

# Load the labeled results
df = pd.read_parquet('project/data/test/test_labeled_1000.parquet')

print("Dataset shape:", df.shape)
print("\nAll columns:")
for i, col in enumerate(df.columns):
    print(f"{i:2d}: {col}")

# Show first few rows with all columns
print("\n" + "="*120)
print("FIRST 5 ROWS - ALL COLUMNS")
print("="*120)
print(df.head())

# Show just the core OHLCV data first
print("\n" + "="*80)
print("CORE OHLCV DATA (first 10 rows)")
print("="*80)
core_cols = ['open', 'high', 'low', 'close', 'volume']
print(df[core_cols].head(10))

# Show label distributions
print("\n" + "="*80)
print("LABEL DISTRIBUTIONS")
print("="*80)
for profile in ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']:
    print(f"\n{profile}:")
    print(df[f'{profile}_label'].value_counts().sort_index())

# Show detailed view of optimal trades
print("\n" + "="*120)
print("DETAILED OPTIMAL TRADES - LONG_2TO1_SMALL")
print("="*120)
optimal_mask = df['long_2to1_small_label'] == 1
if optimal_mask.any():
    # Show all relevant columns for optimal trades
    relevant_cols = ['open', 'high', 'low', 'close', 'volume',
                    'long_2to1_small_outcome', 'long_2to1_small_mae', 
                    'long_2to1_small_sequence_id', 'long_2to1_small_label']
    optimal_df = df[optimal_mask][relevant_cols]
    print(f"Found {len(optimal_df)} optimal trades:")
    print(optimal_df)
else:
    print("No optimal trades found for long_2to1_small")

# Show a sample of each label type
print("\n" + "="*120)
print("SAMPLE OF EACH LABEL TYPE - LONG_2TO1_SMALL")
print("="*120)
for label_val, label_name in [(-1, "LOSS"), (0, "SUBOPTIMAL WIN"), (1, "OPTIMAL WIN")]:
    mask = df['long_2to1_small_label'] == label_val
    if mask.any():
        print(f"\n{label_name} (label={label_val}) - showing first 3:")
        sample_cols = ['close', 'long_2to1_small_outcome', 'long_2to1_small_mae', 'long_2to1_small_label']
        print(df[mask][sample_cols].head(3))
    else:
        print(f"\nNo {label_name} trades found")

# Show basic stats
print("\n" + "="*80)
print("BASIC STATISTICS")
print("="*80)
print(df.describe())
# Interactive browsing function
def browse_data():
    """Interactive function to browse different sections of the data"""
    print("\n" + "="*120)
    print("INTERACTIVE DATA BROWSER")
    print("="*120)
    print("Available commands:")
    print("1. Type 'profile_X' to see all columns for a specific profile (e.g., 'long_2to1_small')")
    print("2. Type 'rows_X_Y' to see rows X through Y (e.g., 'rows_100_110')")
    print("3. Type 'winners_X' to see all winners for profile X")
    print("4. Type 'optimal_X' to see all optimal trades for profile X")
    print("5. Type 'quit' to exit")
    
    profiles = ['long_2to1_small', 'long_2to1_medium', 'long_2to1_large',
                'short_2to1_small', 'short_2to1_medium', 'short_2to1_large']
    
    while True:
        try:
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 'quit':
                break
            elif cmd.startswith('profile_'):
                profile = cmd.replace('profile_', '')
                if profile in profiles:
                    profile_cols = [col for col in df.columns if profile in col]
                    base_cols = ['open', 'high', 'low', 'close', 'volume']
                    all_cols = base_cols + profile_cols
                    print(f"\n{profile.upper()} - All related columns (first 10 rows):")
                    print(df[all_cols].head(10))
                else:
                    print(f"Profile '{profile}' not found. Available: {profiles}")
            
            elif cmd.startswith('rows_'):
                parts = cmd.replace('rows_', '').split('_')
                if len(parts) == 2:
                    start, end = int(parts[0]), int(parts[1])
                    print(f"\nRows {start} to {end}:")
                    print(df.iloc[start:end+1])
                else:
                    print("Format: rows_START_END (e.g., rows_100_110)")
            
            elif cmd.startswith('winners_'):
                profile = cmd.replace('winners_', '')
                if profile in profiles:
                    winners_mask = df[f'{profile}_outcome'] == 'win'
                    if winners_mask.any():
                        winner_cols = ['close', f'{profile}_outcome', f'{profile}_mae', 
                                     f'{profile}_sequence_id', f'{profile}_label']
                        print(f"\nAll winners for {profile} ({winners_mask.sum()} total):")
                        print(df[winners_mask][winner_cols])
                    else:
                        print(f"No winners found for {profile}")
                else:
                    print(f"Profile '{profile}' not found. Available: {profiles}")
            
            elif cmd.startswith('optimal_'):
                profile = cmd.replace('optimal_', '')
                if profile in profiles:
                    optimal_mask = df[f'{profile}_label'] == 1
                    if optimal_mask.any():
                        optimal_cols = ['open', 'high', 'low', 'close', 'volume',
                                      f'{profile}_outcome', f'{profile}_mae', 
                                      f'{profile}_sequence_id', f'{profile}_label']
                        print(f"\nAll optimal trades for {profile} ({optimal_mask.sum()} total):")
                        print(df[optimal_mask][optimal_cols])
                    else:
                        print(f"No optimal trades found for {profile}")
                else:
                    print(f"Profile '{profile}' not found. Available: {profiles}")
            
            else:
                print("Unknown command. Try 'profile_long_2to1_small', 'rows_0_10', 'winners_long_2to1_small', etc.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

# Uncomment the line below to start interactive browsing
# browse_data()