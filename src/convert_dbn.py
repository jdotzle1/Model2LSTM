import databento as db
import pandas as pd

# Read the DBN.ZST file directly
print("Reading DBN.ZST file...")
dbn_path = r"C:\Users\jdotzler\Desktop\Databento\second\ES\glbx-mdp3-20250922-20251021.ohlcv-1s.dbn.zst"
store = db.DBNStore.from_file(dbn_path)
df = store.to_df()

print(f"Loaded {len(df):,} bars")
print(f"Columns: {df.columns.tolist()}")

# Save as Parquet
output_path = r"data\raw\test_sample.parquet"
df.to_parquet(output_path)
print(f"Saved to {output_path}")