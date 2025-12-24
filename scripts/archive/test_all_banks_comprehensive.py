"""Comprehensive quintile analysis for all 5 significant banks"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.loader import merge_all_data
from analysis.lead_lag import lead_lag_analysis_full
from config.config import get_sector_config, FORWARD_RETURN_HORIZONS
from utils.constants import *

# 5 banks that were identified as significant
TICKERS = ['ACB', 'OCB', 'VPB', 'SHB', 'SSB']

# Get banking config
config_banking = get_sector_config('banking')

print("="*80)
print("COMPREHENSIVE QUINTILE ANALYSIS - ALL 5 SIGNIFICANT BANKS")
print("="*80)
print(f"Tickers: {', '.join(TICKERS)}")
print(f"Horizons: {FORWARD_RETURN_HORIZONS}")
print("="*80)

# Load all data
print("\nLoading data...")
data = merge_all_data(config_banking, tickers=TICKERS)

# Summary results
summary_results = []

for ticker in TICKERS:
    if ticker not in data:
        print(f"\nWARNING: {ticker} not in loaded data, skipping...")
        continue

    print(f"\n{'='*80}")
    print(f"ANALYZING {ticker}")
    print(f"{'='*80}")

    # Run analysis
    results = lead_lag_analysis_full(data[ticker])

    # Extract results for each horizon
    for horizon in FORWARD_RETURN_HORIZONS:
        hkey = f'T+{horizon}'
        if hkey in results:
            r = results[hkey]
            summary_results.append({
                'Ticker': ticker,
                'Horizon': hkey,
                'Q5_Mean': r['q5_mean'],
                'Q1_Mean': r['q1_mean'],
                'Spread': r['spread'],
                'T-Stat': r['t_stat'],
                'P-Value': r['p_value'],
                'Significant': 'YES' if r['significant'] else 'NO',
                'Positive_Spread': 'YES' if r['spread'] > 0 else 'NO'
            })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_results)

print("\n" + "="*80)
print("COMPREHENSIVE SUMMARY - ALL STOCKS, ALL HORIZONS")
print("="*80)

# Display full summary
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)

print("\nFULL RESULTS:")
print(summary_df.to_string(index=False))

# Count significant results per ticker
print("\n" + "="*80)
print("SIGNIFICANCE COUNT BY TICKER")
print("="*80)

for ticker in TICKERS:
    ticker_data = summary_df[summary_df['Ticker'] == ticker]
    sig_count = (ticker_data['Significant'] == 'YES').sum()
    pos_spread_count = (ticker_data['Positive_Spread'] == 'YES').sum()
    both_count = ((ticker_data['Significant'] == 'YES') & (ticker_data['Positive_Spread'] == 'YES')).sum()

    print(f"\n{ticker}:")
    print(f"  Significant (p<=0.05): {sig_count}/6")
    print(f"  Positive Spread: {pos_spread_count}/6")
    print(f"  Both (sig + pos): {both_count}/6")

    # Show which horizons meet criteria
    if both_count > 0:
        sig_horizons = ticker_data[(ticker_data['Significant'] == 'YES') & (ticker_data['Positive_Spread'] == 'YES')]['Horizon'].tolist()
        print(f"  Significant horizons: {', '.join(sig_horizons)}")

# Filter to show only significant results with positive spread
print("\n" + "="*80)
print("FILTERED: ONLY SIGNIFICANT WITH POSITIVE SPREAD")
print("="*80)

filtered_df = summary_df[(summary_df['Significant'] == 'YES') & (summary_df['Positive_Spread'] == 'YES')]

if len(filtered_df) > 0:
    print(f"\n{len(filtered_df)} significant results found:")
    print(filtered_df[['Ticker', 'Horizon', 'Spread', 'T-Stat', 'P-Value']].to_string(index=False))
else:
    print("\nNO SIGNIFICANT RESULTS WITH POSITIVE SPREAD FOUND!")
    print("\nThis suggests:")
    print("1. The data may have changed since the original analysis")
    print("2. The analysis parameters may be different")
    print("3. The zero-filtering or data preparation may differ")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
