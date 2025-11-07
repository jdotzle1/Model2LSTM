import pandas as pd

df = pd.read_parquet(r'C:\Users\jdotzler\Desktop\test_single_month_20251103_170901.parquet')
print(f'Rows: {len(df):,}')
print(f'Columns: {len(df.columns)}')
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
