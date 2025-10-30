#!/usr/bin/env python3
"""
Test only the chunked processing validation
"""

import pandas as pd
import os
import sys

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_pipeline.features import validate_chunked_processing

def main():
    # Load dataset
    dataset_path = 'project/data/processed/full_labeled_dataset.parquet'
    df = pd.read_parquet(dataset_path)
    
    # Test on smaller subset
    subset_size = 20000
    df_subset = df.head(subset_size).copy()
    
    print(f"Testing chunked processing on {len(df_subset):,} rows...")
    
    # Test chunked processing validation
    chunk_size = 5000
    validation_passed = validate_chunked_processing(df_subset, chunk_size)
    
    if validation_passed:
        print("✅ PASS: Chunked processing validation successful")
    else:
        print("❌ FAIL: Chunked processing validation failed")

if __name__ == "__main__":
    main()