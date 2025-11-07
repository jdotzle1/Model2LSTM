#!/usr/bin/env python3
"""
Trace Actual Execution

Since the fix is confirmed in place but results are still wrong,
we need to trace what's actually happening during execution.
"""

def generate_trace_commands():
    """Generate commands to trace the actual execution"""
    
    print("🔍 TRACE ACTUAL EXECUTION")
    print("=" * 60)
    print("The fix is confirmed in place, but results are still wrong.")
    print("We need to trace what's happening during actual processing.")
    print()
    
    print("Run these commands on EC2 (as ssm-user):")
    print()
    
    commands = [
        "# Navigate to correct directory",
        "cd /home/ssm-user/Model2LSTM",
        "",
        "# Test the WeightedLabelingEngine directly with trace",
        "python3 -c \"",
        "import sys",
        "sys.path.insert(0, '.')",
        "from src.data_pipeline.weighted_labeling import WeightedLabelingEngine, LabelingConfig, TRADING_MODES",
        "import pandas as pd",
        "from datetime import datetime",
        "import numpy as np",
        "",
        "# Create test data that should show clear results",
        "print('🧪 TESTING WITH CLEAR SCENARIO')",
        "data = {",
        "    'timestamp': [datetime(2011, 6, 10, 9, 30, i) for i in range(10)],",
        "    'open': [1300.00 + i*0.5 for i in range(10)],  # Steadily rising",
        "    'high': [1300.50 + i*0.5 for i in range(10)],",
        "    'low': [1299.50 + i*0.5 for i in range(10)],",
        "    'close': [1300.00 + i*0.5 for i in range(10)],  # Rising 0.5 per bar",
        "    'volume': [100 + i*10 for i in range(10)]",
        "}",
        "",
        "df = pd.DataFrame(data)",
        "print('Test data (rising prices):')  ",
        "for i in range(5):",
        "    print(f'  Bar {i}: Close {df.iloc[i][\\\"close\\\"]:.2f}')",
        "",
        "# Test with engine",
        "config = LabelingConfig()",
        "engine = WeightedLabelingEngine(config)",
        "",
        "print('\\\\n🔍 PROCESSING WITH ENGINE...')  ",
        "result = engine.process_dataframe(df)",
        "",
        "# Check results",
        "short_cols = [col for col in result.columns if 'label_' in col and 'short' in col]",
        "long_cols = [col for col in result.columns if 'label_' in col and 'long' in col]",
        "",
        "print('\\\\n📊 RESULTS:')  ",
        "print('Short trades (should lose in rising market):')  ",
        "for col in short_cols:",
        "    rate = result[col].mean()",
        "    wins = result[col].sum()",
        "    total = len(result)",
        "    print(f'  {col}: {rate:.1%} ({wins}/{total})')",
        "",
        "print('Long trades (should also lose - targets too far):')  ",
        "for col in long_cols:",
        "    rate = result[col].mean()",
        "    wins = result[col].sum()",
        "    total = len(result)",
        "    print(f'  {col}: {rate:.1%} ({wins}/{total})')",
        "",
        "# Check if results make sense",
        "avg_short = sum(result[col].mean() for col in short_cols) / len(short_cols)",
        "avg_long = sum(result[col].mean() for col in long_cols) / len(long_cols)",
        "",
        "print(f'\\\\nAverage short win rate: {avg_short:.1%}')  ",
        "print(f'Average long win rate: {avg_long:.1%}')  ",
        "",
        "if avg_short > 0.5:",
        "    print('❌ BUG: Shorts winning in rising market!')  ",
        "elif avg_short == 0.0:",
        "    print('✅ CORRECT: Shorts losing in rising market')  ",
        "else:",
        "    print(f'⚠️  UNEXPECTED: Short rate is {avg_short:.1%}')  ",
        "\"",
    ]
    
    for cmd in commands:
        print(cmd)

def generate_manual_calculation():
    """Generate commands to manually calculate what the results should be"""
    
    print(f"\n📊 MANUAL CALCULATION TEST")
    print("=" * 60)
    print("Let's manually calculate what the win rates should be for June 2011:")
    print()
    
    commands = [
        "# Download a small sample of June 2011 data",
        "cd /home/ssm-user/Model2LSTM",
        "",
        "# Get the processed data",
        "aws s3 cp s3://es-1-second-data/processed-data/monthly/2011/06/ /tmp/june_debug/ --recursive --region us-east-1",
        "",
        "# Analyze the actual data",
        "python3 -c \"",
        "import pandas as pd",
        "import numpy as np",
        "from glob import glob",
        "",
        "# Find the parquet file",
        "files = glob('/tmp/june_debug/*.parquet')",
        "if files:",
        "    df = pd.read_parquet(files[0])",
        "    print(f'Loaded {len(df):,} rows')  ",
        "    ",
        "    # Check the actual label values",
        "    label_cols = [col for col in df.columns if col.startswith('label_')]",
        "    print('\\\\nActual label statistics:')  ",
        "    for col in label_cols:",
        "        unique_vals = sorted(df[col].unique())",
        "        win_rate = df[col].mean()",
        "        print(f'  {col}: {win_rate:.1%} (values: {unique_vals})')  ",
        "    ",
        "    # Sample some data to see what's happening",
        "    print('\\\\nSample data (first 10 rows):')  ",
        "    sample_cols = ['timestamp', 'open', 'high', 'low', 'close'] + label_cols[:2]",
        "    print(df[sample_cols].head(10))",
        "    ",
        "    # Check for any obvious patterns",
        "    print('\\\\nPrice movement analysis:')  ",
        "    df['price_change'] = df['close'].diff()",
        "    up_moves = (df['price_change'] > 0).sum()",
        "    down_moves = (df['price_change'] < 0).sum()",
        "    print(f'Up moves: {up_moves:,}, Down moves: {down_moves:,}')  ",
        "    print(f'Up/Down ratio: {up_moves/down_moves:.2f}' if down_moves > 0 else 'No down moves')  ",
        "else:",
        "    print('No parquet files found')  ",
        "\"",
        "",
        "# Clean up",
        "rm -rf /tmp/june_debug/",
    ]
    
    for cmd in commands:
        print(cmd)

def generate_nuclear_option():
    """Generate the nuclear option - completely bypass existing code"""
    
    print(f"\n🚀 NUCLEAR OPTION")
    print("=" * 60)
    print("If nothing else works, create a completely new labeling script:")
    print()
    
    script = '''
# Create a fresh labeling script that bypasses all existing code
cat > /tmp/fresh_labeling_test.py << 'EOF'
import pandas as pd
import numpy as np
from datetime import datetime

def simple_short_labeling(df):
    """Simple short labeling - no fancy classes or caching"""
    
    labels = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - 1):  # Can't enter on last bar
        # Entry price is next bar's open
        entry_price = df.iloc[i + 1]['open']
        
        # Short trade: target goes down, stop goes up
        target_price = entry_price - (16 * 0.25)  # 16 ticks down
        stop_price = entry_price + (8 * 0.25)     # 8 ticks up
        
        # Check the entry bar and subsequent bars
        for j in range(i + 1, min(i + 901, len(df))):  # 900 second timeout
            bar = df.iloc[j]
            
            target_hit = bar['low'] <= target_price
            stop_hit = bar['high'] >= stop_price
            
            if target_hit and not stop_hit:
                labels[i] = 1  # Win
                break
            elif stop_hit:
                labels[i] = 0  # Loss
                break
        # If neither hit, labels[i] remains 0 (loss)
    
    return labels

# Test with rising price data
data = {
    'timestamp': [datetime(2011, 6, 10, 9, 30, i) for i in range(10)],
    'open': [1300.00 + i*0.5 for i in range(10)],
    'high': [1300.50 + i*0.5 for i in range(10)],
    'low': [1299.50 + i*0.5 for i in range(10)],
    'close': [1300.00 + i*0.5 for i in range(10)],
    'volume': [100] * 10
}

df = pd.DataFrame(data)
labels = simple_short_labeling(df)

print("Fresh labeling results (rising market):")
print(f"Labels: {labels}")
print(f"Win rate: {labels.mean():.1%}")

if labels.mean() == 0.0:
    print("✅ CORRECT: Fresh code shows 0% short wins in rising market")
else:
    print(f"❌ STILL WRONG: Fresh code shows {labels.mean():.1%} short wins")
EOF

python3 /tmp/fresh_labeling_test.py
'''
    
    print(script)

def main():
    """Main execution"""
    
    print("🚨 THE MYSTERY DEEPENS")
    print("The fix is confirmed in place, but results are still 66%.")
    print("We need to trace what's actually happening during execution.")
    print()
    
    generate_trace_commands()
    generate_manual_calculation()
    generate_nuclear_option()
    
    print(f"\n🎯 PRIORITY ORDER:")
    print("1. Run the trace test first - this will show if the engine is working")
    print("2. If engine works but production doesn't, check the manual calculation")
    print("3. If all else fails, use the nuclear option to bypass everything")
    print()
    print("This is the most puzzling bug I've ever encountered! 🤯")


if __name__ == "__main__":
    main()