"""
Process October 2025 DBN file using the pipeline's conversion
"""
import sys
import os
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the pipeline's DBN conversion
try:
    import databento as db
    print("✅ databento module available")
    
    dbn_path = r"C:\Users\jdotzler\Downloads\glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst"
    output_path = r"C:\Users\jdotzler\Desktop\monthly_2025-10_raw.parquet"
    
    print(f"\n📖 Reading DBN file...")
    print(f"   File: {dbn_path}")
    
    store = db.DBNStore.from_file(dbn_path)
    metadata = store.metadata
    
    print(f"\n📊 Metadata:")
    print(f"   Schema: {metadata.schema}")
    print(f"   Dataset: {metadata.dataset}")
    print(f"   Symbols: {metadata.symbols}")
    
    print(f"\n🔄 Converting to DataFrame...")
    df = store.to_df()
    
    print(f"\n📋 DataFrame Info:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Create timestamp if needed
    if 'timestamp' not in df.columns:
        print(f"\n⏱️  Creating timestamps...")
        start_ns = metadata.start
        end_ns = metadata.end
        timestamps = pd.date_range(
            start=pd.to_datetime(start_ns, unit='ns', utc=True),
            end=pd.to_datetime(end_ns, unit='ns', utc=True),
            periods=len(df)
        )
        df['timestamp'] = timestamps
    
    # Reset index if timestamp is in index
    if df.index.name == 'timestamp':
        df = df.reset_index()
    
    print(f"\n💾 Saving to Parquet...")
    df.to_parquet(output_path, index=False)
    print(f"   Saved: {output_path}")
    print(f"   Size: {len(df):,} rows")
    
    # Quick analysis
    print(f"\n📊 Quick Analysis:")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    if 'open' in df.columns and 'close' in df.columns:
        print(f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    print(f"\n✅ Conversion complete!")
    
except ImportError as e:
    print(f"❌ databento module not available: {e}")
    print(f"\nTo install databento, run:")
    print(f"   pip install databento")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
