#!/usr/bin/env python3
"""
Analyze the winner count issue to understand why we're getting 1624 winners from 1000 bars
"""

import pandas as pd
from project.data_pipeline.weighted_labeling import process_weighted_labeling

def analyze_winner_count():
    # Load and process Oct 14 data
    df = pd.read_parquet('project/data/test/oct14_high_activity_1000.parquet')
    result_df = process_weighted_labeling(df)

    print('=== UNDERSTANDING THE WINNER COUNT ===')
    print('Total bars:', len(result_df))

    # Check how many bars are winners in each mode
    modes = ['low_vol_long', 'normal_vol_long', 'high_vol_long', 'low_vol_short', 'normal_vol_short', 'high_vol_short']

    print('\nWinners per mode:')
    for mode in modes:
        label_col = f'label_{mode}'
        winners = result_df[label_col].sum()
        print(f'  {mode}: {winners} bars are winners ({winners/1000:.1%})')

    # Check how many bars are winners in ANY mode
    any_winner = result_df[[f'label_{mode}' for mode in modes]].max(axis=1)
    total_bars_with_wins = any_winner.sum()
    print(f'\nBars that are winners in at least one mode: {total_bars_with_wins} ({total_bars_with_wins/1000:.1%})')

    # Check how many bars are winners in multiple modes
    winner_counts_per_bar = result_df[[f'label_{mode}' for mode in modes]].sum(axis=1)
    print(f'\nDistribution of winner modes per bar:')
    for i in range(7):  # 0 to 6 modes
        count = (winner_counts_per_bar == i).sum()
        if count > 0:
            print(f'  {i} modes: {count} bars ({count/1000:.1%})')

    # Show some examples of bars that win in multiple modes
    multi_winners = result_df[winner_counts_per_bar > 1]
    if len(multi_winners) > 0:
        print(f'\nExample bars winning in multiple modes (first 5):')
        for idx in multi_winners.head(5).index:
            winning_modes = []
            for mode in modes:
                if result_df.loc[idx, f'label_{mode}'] == 1:
                    winning_modes.append(mode)
            print(f'  Bar {idx}: {len(winning_modes)} modes - {winning_modes}')

    print(f'\nSUMMARY:')
    total_winner_instances = result_df[[f'label_{mode}' for mode in modes]].sum().sum()
    print(f'• Total winner instances: {total_winner_instances}')
    print(f'• Unique bars with wins: {total_bars_with_wins}')
    if total_bars_with_wins > 0:
        print(f'• Average modes per winning bar: {total_winner_instances / total_bars_with_wins:.1f}')
    
    # This explains the confusion
    print(f'\n🔍 EXPLANATION:')
    print(f'• Each bar can be a winner in multiple trading modes simultaneously')
    print(f'• We have {total_winner_instances} total "winner instances" across all modes')
    print(f'• But only {total_bars_with_wins} unique bars actually contain winning trades')
    print(f'• The high count comes from bars winning in multiple modes at once')

if __name__ == "__main__":
    analyze_winner_count()