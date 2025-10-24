import pandas as pd

def check_duplicate_timestamps():
    """
    Check for duplicate timestamps in the dataset
    """
    print("=== CHECKING FOR DUPLICATE TIMESTAMPS ===")
    
    # Load data
    df_full = pd.read_parquet('project/data/raw/test_sample.parquet')
    df_test = df_full.head(1000).copy()
    
    print(f"Total rows: {len(df_test)}")
    print(f"Unique timestamps: {df_test.index.nunique()}")
    print(f"Duplicate timestamps: {len(df_test) - df_test.index.nunique()}")
    
    # Find duplicate timestamps
    duplicate_mask = df_test.index.duplicated(keep=False)
    duplicates = df_test[duplicate_mask]
    
    if len(duplicates) > 0:
        print(f"\nFound {len(duplicates)} rows with duplicate timestamps:")
        
        # Group by timestamp to see duplicates
        for timestamp, group in duplicates.groupby(duplicates.index):
            print(f"\nTimestamp {timestamp} appears {len(group)} times:")
            print(f"  Row indices: {group.index.get_indexer(group.index).tolist()}")
            print(f"  DataFrame positions: {[df_test.index.get_loc(timestamp) if not isinstance(df_test.index.get_loc(timestamp), slice) else 'multiple' for _ in range(len(group))]}")
            
            # Show the actual data
            print(f"  Data:")
            for i, (idx, row) in enumerate(group.iterrows()):
                pos = df_test.index.get_loc(idx)
                if isinstance(pos, slice):
                    pos = f"slice({pos.start}:{pos.stop})"
                elif hasattr(pos, '__len__'):
                    pos = f"array({pos})"
                print(f"    Position {pos}: open={row['open']}, high={row['high']}, low={row['low']}, close={row['close']}, volume={row['volume']}")
    
    # Check specific problematic timestamps
    problem_timestamps = [
        pd.Timestamp('2025-09-22 00:47:51+00:00', tz='UTC'),
        pd.Timestamp('2025-09-22 00:48:34+00:00', tz='UTC')
    ]
    
    print(f"\n=== CHECKING SPECIFIC PROBLEMATIC TIMESTAMPS ===")
    for ts in problem_timestamps:
        if ts in df_test.index:
            matches = df_test.loc[ts]
            if isinstance(matches, pd.Series):
                print(f"\nTimestamp {ts}: 1 row")
            else:
                print(f"\nTimestamp {ts}: {len(matches)} rows")
                print(matches[['open', 'high', 'low', 'close', 'volume']])
                
                # Show what happens when we set a label
                print(f"  If we set label=1 for this timestamp, it affects rows:")
                positions = df_test.index.get_loc(ts)
                if isinstance(positions, slice):
                    affected_rows = list(range(positions.start, positions.stop))
                elif hasattr(positions, '__len__'):
                    affected_rows = positions.tolist()
                else:
                    affected_rows = [positions]
                print(f"    {affected_rows}")

if __name__ == "__main__":
    check_duplicate_timestamps()