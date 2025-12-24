"""
Final comprehensive analysis of all 17 banks with 6-year data + prices
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from data.loader import load_foreign_trading
from analysis.lead_lag import lead_lag_analysis_full
from config import config_banking

HORIZONS = [1, 3, 5, 10, 20, 30]

print("=" * 80)
print("FINAL ANALYSIS: 17 BANKS - 6 YEAR DATA WITH PRICES")
print("Period: Matched to each stock's foreign trading data")
print("=" * 80)
print()

# Load data
print("Loading data...")
data_dict = load_foreign_trading(config_banking)
print(f"Loaded {len(data_dict)} banks\n")

# Results storage
all_results = []

for ticker in sorted(data_dict.keys()):
    df = data_dict[ticker]

    # Get date column (might be 'date' or 'Ngay')
    date_col = 'date' if 'date' in df.columns else 'Ngay'

    print(f"{'=' * 70}")
    print(f"{ticker}: {len(df)} rows, {df[date_col].min().strftime('%d/%m/%Y')} - {df[date_col].max().strftime('%d/%m/%Y')}")
    print('=' * 70)

    # Run analysis
    result = lead_lag_analysis_full(df, horizons=HORIZONS)

    if 'error' in result:
        print(f"ERROR: {result['error']}\n")
        continue

    # Process each horizon
    for horizon in HORIZONS:
        horizon_key = f'T+{horizon}'
        horizon_result = result.get(horizon_key, {})

        if not horizon_result:
            continue

        p_val = horizon_result.get('p_value')
        spread = horizon_result.get('spread')

        if p_val is None or spread is None:
            continue

        # Classify significance
        if p_val <= 0.05 and spread > 0:
            if p_val < 0.01:
                sig, marker = "STRONG", "***"
            elif p_val < 0.03:
                sig, marker = "MODERATE", "**"
            else:
                sig, marker = "MARGINAL", "*"
        elif p_val <= 0.10 and spread > 0:
            sig, marker = "NEAR", "~"
        else:
            sig, marker = "NO", ""

        if p_val <= 0.10 and spread > 0:
            print(f"  T+{horizon:2d}: p={p_val:.4f}, spread={spread*100:5.2f}% {marker} {sig}")
            all_results.append({
                'ticker': ticker,
                'horizon': horizon,
                'p_value': p_val,
                'spread': spread,
                'significance': sig
            })

    print()

# Summary
print("\n" + "=" * 80)
print("SUMMARY - STATISTICAL SIGNIFICANCE")
print("=" * 80)

df_results = pd.DataFrame(all_results)

if len(df_results) == 0:
    print("\nNo significant results found!")
else:
    sig_only = df_results[df_results['significance'].isin(['STRONG', 'MODERATE', 'MARGINAL'])]

    if len(sig_only) > 0:
        print("\nBANKS WITH SIGNIFICANCE (p <= 0.05, spread > 0):")
        print("-" * 80)

        ticker_counts = sig_only.groupby('ticker').size().sort_values(ascending=False)

        for ticker in ticker_counts.index:
            group = sig_only[sig_only['ticker'] == ticker]
            count = len(group)
            horizons = [f"T+{h}" for h in sorted(group['horizon'])]
            min_p = group['p_value'].min()
            max_spread = group['spread'].max()

            strong = len(group[group['significance'] == 'STRONG'])
            moderate = len(group[group['significance'] == 'MODERATE'])
            marginal = len(group[group['significance'] == 'MARGINAL'])

            breakdown = []
            if strong > 0:
                breakdown.append(f"{strong} STRONG")
            if moderate > 0:
                breakdown.append(f"{moderate} MODERATE")
            if marginal > 0:
                breakdown.append(f"{marginal} MARGINAL")

            print(f"\n{ticker}: {count}/6 horizons")
            print(f"  Horizons: {', '.join(horizons)}")
            print(f"  Min p: {min_p:.4f}, Max spread: {max_spread*100:.2f}%")
            print(f"  Breakdown: {', '.join(breakdown)}")

    near_only = df_results[df_results['significance'] == 'NEAR']
    if len(near_only) > 0:
        print("\n\nNEAR SIGNIFICANCE (0.05 < p <= 0.10):")
        print("-" * 80)
        for ticker, group in near_only.groupby('ticker'):
            horizons = [f"T+{h}" for h in sorted(group['horizon'])]
            print(f"  {ticker}: {', '.join(horizons)}")

print("\n" + "=" * 80)
print("LEGEND: *** STRONG (p<0.01) | ** MODERATE (p<0.03) | * MARGINAL (p<0.05) | ~ NEAR (p<0.10)")
print("=" * 80)

# Top recommendations
if len(sig_only) > 0:
    print("\n" + "=" * 80)
    print("TOP RECOMMENDATIONS FOR Q1 FOREIGN LEAD/LAG")
    print("=" * 80)

    for i, ticker in enumerate(ticker_counts.index[:5], 1):
        group = sig_only[sig_only['ticker'] == ticker]
        count = len(group)
        min_p = group['p_value'].min()
        max_spread = group['spread'].max()
        print(f"{i}. {ticker}: {count}/6 horizons (p_min={min_p:.4f}, spread_max={max_spread*100:.2f}%)")

    print("=" * 80)
