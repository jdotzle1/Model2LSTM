"""
Monthly Batch Processor for ES Trading Model

Handles processing of 15 years of ES data in monthly chunks from S3.
Uses modular components for S3 operations, pipeline processing, and monitoring.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import calendar
import time
import gc
from typing import Dict, List, Optional

from .s3_operations import EnhancedS3Operations
from .corrected_contract_filtering import process_complete_pipeline
from .pipeline import process_labeling_and_features, PipelineConfig


class MonthlyProcessor:
    """Process ES data in monthly chunks from S3"""
    
    def __init__(self, bucket_name: str = "es-1-second-data"):
        self.bucket_name = bucket_name
        self.s3_ops = EnhancedS3Operations(bucket_name)
        self.temp_dir = Path("/tmp/monthly_processing")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_monthly_file_list(self, start_year=2010, start_month=7, 
                                   end_year=2025, end_month=10) -> List[Dict]:
        """Generate list of monthly files to process"""
        monthly_files = []
        
        for year in range(start_year, end_year + 1):
            first_month = start_month if year == start_year else 1
            last_month = end_month if year == end_year else 12
            
            for month in range(first_month, last_month + 1):
                first_day = 1
                last_day = calendar.monthrange(year, month)[1]
                
                start_date = f"{year:04d}{month:02d}{first_day:02d}"
                end_date = f"{year:04d}{month:02d}{last_day:02d}"
                
                month_str = f"{year:04d}-{month:02d}"
                filename = f"glbx-mdp3-{start_date}-{end_date}.ohlcv-1s.dbn.zst"
                
                monthly_files.append({
                    'year': year,
                    'month': month,
                    'month_str': month_str,
                    'filename': filename,
                    's3_key': f"raw-data/databento/{filename}",
                    'local_file': str(self.temp_dir / month_str / "input.dbn.zst"),
                    'output_file': str(self.temp_dir / month_str / "processed.parquet")
                })
        
        return monthly_files
    
    def check_existing_processed(self, monthly_files: List[Dict]) -> List[Dict]:
        """Check which files are already processed in S3"""
        print("🔍 Checking for existing processed files in S3...")
        
        try:
            # List objects in processed-data prefix
            paginator = self.s3_ops.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix='processed-data/monthly/'
            )
            
            existing = set()
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Extract month from filename (e.g., monthly_2025-10_timestamp.parquet)
                        filename = obj['Key'].split('/')[-1]
                        if 'monthly_' in filename:
                            parts = filename.split('_')
                            if len(parts) >= 2:
                                month_str = parts[1]  # e.g., "2025-10"
                                existing.add(month_str)
            
            to_process = [f for f in monthly_files if f['month_str'] not in existing]
            
            print(f"✅ Already processed: {len(existing)} months")
            print(f"🔄 Need to process: {len(to_process)} months")
            
            return to_process
            
        except Exception as e:
            print(f"⚠️  Could not check existing files: {e}")
            print(f"   Proceeding with all {len(monthly_files)} files")
            return monthly_files
    
    def process_single_month(self, file_info: Dict) -> Optional[str]:
        """Process a single month of data"""
        month_str = file_info['month_str']
        print(f"\n{'='*80}")
        print(f"PROCESSING {month_str}")
        print(f"{'='*80}")
        
        try:
            # Stage 1: Download
            print(f"📥 Stage 1: Downloading...", flush=True)
            if not self.s3_ops.download_monthly_file_optimized(file_info):
                print(f"❌ Download failed for {month_str}", flush=True)
                return None
            
            # Stage 2: Process (contract filtering + gap filling)
            print(f"🔧 Stage 2: Processing...", flush=True)
            local_file = Path(file_info['local_file'])
            
            import databento as db
            store = db.DBNStore.from_file(str(local_file))
            df_raw = store.to_df()
            
            if df_raw.index.name == 'ts_event':
                df_raw = df_raw.reset_index()
                df_raw = df_raw.rename(columns={'ts_event': 'timestamp'})
            
            df_processed, stats = process_complete_pipeline(df_raw)
            
            # Stage 3: Add features and labels
            print(f"🏷️  Stage 3: Labeling and features...", flush=True)
            config = PipelineConfig(chunk_size=500_000)
            df_final = process_labeling_and_features(df_processed, config)
            
            # Calculate labeling statistics for reporting
            print(f"\n📊 Calculating final statistics...", flush=True)
            labeling_stats = self._calculate_labeling_stats(df_final)
            stats['labeling'] = labeling_stats
            
            # Print summary
            print(f"\n{'='*80}", flush=True)
            print(f"PROCESSING SUMMARY - {month_str}", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Final dataset: {len(df_final):,} rows × {len(df_final.columns)} columns", flush=True)
            print(f"\nWin Rates:", flush=True)
            for mode, mode_stats in labeling_stats['modes'].items():
                print(f"  {mode}: {mode_stats['win_rate']:.1%} ({mode_stats['winners']:,} winners)", flush=True)
            print(f"{'='*80}\n", flush=True)
            
            # Stage 4: Save
            print(f"💾 Stage 4: Saving...", flush=True)
            output_file = Path(file_info['output_file'])
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_final.to_parquet(output_file)
            
            # Stage 5: Upload
            print(f"📤 Stage 5: Uploading...", flush=True)
            if not self.s3_ops.upload_monthly_results_optimized(file_info, str(output_file), stats):
                print(f"⚠️  Upload failed for {month_str}", flush=True)
            
            # Cleanup
            print(f"🧹 Stage 6: Cleanup...", flush=True)
            self._cleanup_month(file_info)
            gc.collect()
            
            print(f"✅ {month_str} completed successfully", flush=True)
            return str(output_file)
            
        except Exception as e:
            print(f"❌ {month_str} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            self._cleanup_month(file_info)
            return None
    
    def _calculate_labeling_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics from labeled data"""
        from .weighted_labeling import TRADING_MODES
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'modes': {}
        }
        
        for mode_name, mode in TRADING_MODES.items():
            label_col = mode.label_column
            weight_col = mode.weight_column
            
            if label_col in df.columns and weight_col in df.columns:
                winners = df[label_col] == 1
                losers = df[label_col] == 0
                
                mode_stats = {
                    'win_rate': df[label_col].mean(),
                    'winners': int(winners.sum()),
                    'losers': int(losers.sum()),
                    'avg_weight': float(df[weight_col].mean()),
                    'avg_winner_weight': float(df.loc[winners, weight_col].mean()) if winners.any() else 0.0,
                    'avg_loser_weight': float(df.loc[losers, weight_col].mean()) if losers.any() else 0.0
                }
                stats['modes'][mode_name] = mode_stats
        
        return stats
    
    def _cleanup_month(self, file_info: Dict):
        """Clean up temporary files for a month"""
        month_dir = Path(file_info['local_file']).parent
        if month_dir.exists():
            import shutil
            shutil.rmtree(month_dir, ignore_errors=True)
    
    def process_all_months(self, monthly_files: List[Dict]) -> Dict:
        """Process all months and return summary"""
        start_time = time.time()
        results = {
            'total': len(monthly_files),
            'successful': 0,
            'failed': 0,
            'failed_months': []
        }
        
        for i, file_info in enumerate(monthly_files, 1):
            month_str = file_info['month_str']
            print(f"\n[{i}/{len(monthly_files)}] Processing {month_str}...")
            
            month_start = time.time()
            result = self.process_single_month(file_info)
            month_duration = time.time() - month_start
            
            if result:
                results['successful'] += 1
                print(f"✅ {month_str} completed in {month_duration/60:.1f} minutes", flush=True)
            else:
                results['failed'] += 1
                results['failed_months'].append(month_str)
                print(f"❌ {month_str} failed after {month_duration/60:.1f} minutes", flush=True)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(monthly_files) - i) * avg_time
            print(f"📊 Progress: {i}/{len(monthly_files)} - ETA: {remaining/3600:.1f}h", flush=True)
        
        total_time = time.time() - start_time
        results['total_time_hours'] = total_time / 3600
        
        return results
