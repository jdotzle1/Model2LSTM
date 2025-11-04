#!/usr/bin/env python3
"""
Standalone Final Processing Report Generator

This script generates comprehensive final processing reports for the monthly data processing pipeline.
It can be run independently to analyze processing results and generate recommendations.

Usage:
    python generate_final_processing_report.py [--output-dir /path/to/processing/logs]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.final_processing_report import generate_final_processing_report


def main():
    """Main function for standalone report generation"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive final processing report for monthly data processing pipeline"
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/monthly_processing',
        help='Directory containing processing logs and statistics (default: /tmp/monthly_processing)'
    )
    
    parser.add_argument(
        '--report-file',
        type=str,
        help='Custom filename for the report (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show preview of report content in console'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    print("📊 Final Processing Report Generator")
    print("=" * 50)
    
    # Check if output directory exists
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Error: Output directory does not exist: {output_dir}")
        print(f"   Please ensure the processing pipeline has been run and logs are available.")
        sys.exit(1)
    
    if args.verbose:
        print(f"📁 Processing logs directory: {output_dir}")
        
        # List available files
        log_files = list(output_dir.glob("*.log"))
        stats_files = list(output_dir.glob("statistics/*.json")) if (output_dir / "statistics").exists() else []
        
        print(f"   Found {len(log_files)} log files")
        print(f"   Found {len(stats_files)} statistics files")
    
    # Generate the report
    try:
        print("🔄 Generating final processing report...")
        
        from src.data_pipeline.final_processing_report import FinalProcessingReportGenerator
        
        generator = FinalProcessingReportGenerator(str(output_dir))
        
        # Load processing results
        if not generator.load_processing_results():
            print("⚠️  Warning: Could not load complete processing results")
            print("   Report will be generated with available data")
        
        # Generate report
        report_content = generator.generate_comprehensive_final_report()
        
        # Save report
        report_path = generator.save_report_to_file(report_content, args.report_file)
        
        if report_path:
            print(f"✅ Final processing report generated successfully!")
            print(f"📄 Report saved to: {report_path}")
            
            # Show report statistics
            lines = report_content.split('\n')
            print(f"📊 Report contains {len(lines)} lines")
            
            # Extract key metrics from report
            if generator.processing_summary:
                summary = generator.processing_summary
                print(f"📈 Processing Summary:")
                print(f"   • Success Rate: {summary.success_rate:.1f}%")
                print(f"   • Total Months: {summary.total_months_attempted}")
                print(f"   • Processing Time: {summary.total_processing_time_hours:.1f} hours")
                print(f"   • Average Time/Month: {summary.avg_processing_time_minutes:.1f} minutes")
            
            # Show reprocessing recommendations count
            recommendations = generator._generate_reprocessing_recommendations()
            if recommendations:
                high_priority = len([r for r in recommendations if r.priority == 'high'])
                medium_priority = len([r for r in recommendations if r.priority == 'medium'])
                low_priority = len([r for r in recommendations if r.priority == 'low'])
                
                print(f"🔄 Reprocessing Recommendations:")
                print(f"   • High Priority: {high_priority} months")
                print(f"   • Medium Priority: {medium_priority} months")
                print(f"   • Low Priority: {low_priority} months")
            else:
                print("✅ No reprocessing required!")
            
            # Show preview if requested
            if args.preview:
                print("\n" + "=" * 80)
                print("📋 REPORT PREVIEW (First 30 lines)")
                print("=" * 80)
                
                for i, line in enumerate(lines[:30]):
                    print(line)
                
                if len(lines) > 30:
                    print(f"\n... and {len(lines) - 30} more lines")
                    print(f"📄 Full report available at: {report_path}")
                
                print("=" * 80)
        
        else:
            print("❌ Failed to save final processing report")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error generating final processing report: {e}")
        if args.verbose:
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()