"""
Convert October 2025 DBN file to Parquet for analysis
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Check if we can import the conversion function
try:
    from src.convert_dbn import convert_dbn_file
    print("✅ Using src/convert_dbn.py")
    HAS_CONVERTER = True
except ImportError:
    print("❌ Cannot import convert_dbn")
    HAS_CONVERTER = False

if not HAS_CONVERTER:
    print("\nTrying direct databento import...")
    try:
        import databento as db
        print("✅ databento available")
        
        dbn_path = r"C:\Users\jdotzler\Downloads\glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst"
        output_path = r"C:\Users\jdotzler\Desktop\monthly_2025-10_raw.parquet"
        
        print(f"\n📖 Reading DBN file: {dbn_path}")
        store = db.DBNStore.from_file(dbn_path)
        df = store.to_df()
        
        print(f"   Loaded: {len(df):,} rows")
        print(f"   Columns: {list(df.columns)}")
        
        # Save to parquet
        df.to_parquet(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
        
    except ImportError:
        print("❌ databento not available")
        print("\nPlease install databento or use the pipeline conversion")
        sys.exit(1)
else:
    dbn_path = r"C:\Users\jdotzler\Downloads\glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst"
    output_path = r"C:\Users\jdotzler\Desktop\monthly_2025-10_raw.parquet"
    
    print(f"\n📖 Converting: {dbn_path}")
    df = convert_dbn_file(dbn_path, rth_only=False)  # Don't filter RTH yet
    
    print(f"\n💾 Saving to: {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"   Saved: {len(df):,} rows")
