"""
Manually decompress and inspect DBN file without databento library
"""
import zstandard as zstd
import struct
from pathlib import Path

def decompress_dbn_file(dbn_zst_path, output_path=None):
    """Decompress .dbn.zst file to .dbn"""
    
    print(f"📖 Reading compressed file: {dbn_zst_path}")
    
    # Read compressed file
    with open(dbn_zst_path, 'rb') as f:
        compressed_data = f.read()
    
    print(f"   Compressed size: {len(compressed_data):,} bytes ({len(compressed_data)/1024/1024:.2f} MB)")
    
    # Decompress
    print(f"\n🔓 Decompressing...")
    dctx = zstd.ZstdDecompressor()
    decompressed_data = dctx.decompress(compressed_data)
    
    print(f"   Decompressed size: {len(decompressed_data):,} bytes ({len(decompressed_data)/1024/1024:.2f} MB)")
    print(f"   Compression ratio: {len(compressed_data)/len(decompressed_data):.2%}")
    
    # Save decompressed file if output path provided
    if output_path:
        print(f"\n💾 Saving decompressed file: {output_path}")
        with open(output_path, 'wb') as f:
            f.write(decompressed_data)
        print(f"   Saved!")
    
    # Try to read DBN header
    print(f"\n📋 DBN File Header:")
    print(f"   First 100 bytes (hex): {decompressed_data[:100].hex()}")
    print(f"   First 100 bytes (ascii, errors ignored): {decompressed_data[:100].decode('ascii', errors='ignore')}")
    
    return decompressed_data

if __name__ == "__main__":
    try:
        import zstandard
        print("✅ zstandard module available")
    except ImportError:
        print("❌ zstandard module not available")
        print("Installing zstandard...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "zstandard"])
        import zstandard
        print("✅ zstandard installed")
    
    # Decompress October 2025 file
    dbn_zst_path = r"C:\Users\jdotzler\Downloads\glbx-mdp3-20251001-20251031.ohlcv-1s.dbn.zst"
    output_path = r"C:\Users\jdotzler\Desktop\monthly_2025-10_raw.dbn"
    
    try:
        data = decompress_dbn_file(dbn_zst_path, output_path)
        print(f"\n✅ Decompression complete!")
        print(f"\nNote: The .dbn file still needs databento library to parse properly.")
        print(f"      But we can see the file is valid and decompresses successfully.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
