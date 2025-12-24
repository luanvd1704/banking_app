"""Test SHB quintile analysis to debug inf/-inf issue"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.loader import merge_all_data
from analysis.lead_lag import lead_lag_analysis_full
from config.config import get_sector_config
from utils.constants import *

# Get banking config
config_banking = get_sector_config('banking')

print("="*80)
print("TESTING SHB QUINTILE ANALYSIS")
print("="*80)

# Load SHB data
print("\nLoading SHB data...")
data = merge_all_data(config_banking, tickers=['SHB'])

if 'SHB' not in data:
    print("ERROR: SHB data not loaded")
    sys.exit(1)

df = data['SHB']

print(f"\nLoaded {len(df)} rows")
print(f"Date range: {df[DATE].min()} to {df[DATE].max()}")

# Check foreign_net_buy_val column
print(f"\nForeign net buy value stats:")
print(f"  Total values: {len(df[FOREIGN_NET_BUY_VAL])}")
print(f"  Non-null: {df[FOREIGN_NET_BUY_VAL].notna().sum()}")
print(f"  Zeros: {(df[FOREIGN_NET_BUY_VAL] == 0).sum()}")
print(f"  Non-zero: {((df[FOREIGN_NET_BUY_VAL] != 0) & df[FOREIGN_NET_BUY_VAL].notna()).sum()}")

# Check close price
print(f"\nClose price stats:")
print(f"  Non-null: {df[CLOSE].notna().sum()}")
print(f"  Null: {df[CLOSE].isna().sum()}")

# Run analysis
print("\n" + "="*80)
print("RUNNING ANALYSIS...")
print("="*80)

results = lead_lag_analysis_full(df)

# Check T+1 results (should be significant)
print("\nT+1 Results:")
if 'T+1' in results:
    r = results['T+1']
    print(f"  Q5 Mean: {r['q5_mean']}")
    print(f"  Q1 Mean: {r['q1_mean']}")
    print(f"  Spread: {r['spread']}")
    print(f"  T-stat: {r['t_stat']}")
    print(f"  P-value: {r['p_value']}")
    print(f"  Significant: {r['significant']}")

    # Check quintile stats
    print(f"\n  Quintile Statistics:")
    print(r['quintile_stats'])
else:
    print("  ERROR: T+1 not in results")

# Check T+3 results (should be significant)
print("\nT+3 Results:")
if 'T+3' in results:
    r = results['T+3']
    print(f"  Q5 Mean: {r['q5_mean']}")
    print(f"  Q1 Mean: {r['q1_mean']}")
    print(f"  Spread: {r['spread']}")
    print(f"  T-stat: {r['t_stat']}")
    print(f"  P-value: {r['p_value']}")
    print(f"  Significant: {r['significant']}")

    # Check quintile stats
    print(f"\n  Quintile Statistics:")
    print(r['quintile_stats'])
else:
    print("  ERROR: T+3 not in results")

# Debug: Check if data has quintiles assigned
print("\n" + "="*80)
print("DEBUGGING QUINTILE ASSIGNMENT")
print("="*80)

from analysis.lead_lag import create_quintiles_by_foreign_trading, prepare_lead_lag_data

# Prepare data
df_prepared = prepare_lead_lag_data(df, [1, 3])

# Create quintiles
df_quintiles = create_quintiles_by_foreign_trading(df_prepared)

print(f"\nQuintile distribution:")
if QUINTILE in df_quintiles.columns:
    print(df_quintiles[QUINTILE].value_counts())

    # Check each quintile's forward returns
    print(f"\nForward returns by quintile (T+1):")
    if 'fwd_excess_return_1d' in df_quintiles.columns:
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_data = df_quintiles[df_quintiles[QUINTILE] == q]['fwd_excess_return_1d']
            print(f"  {q}: count={len(q_data)}, mean={q_data.mean():.6f}, std={q_data.std():.6f}")
    else:
        print("  ERROR: fwd_excess_return_1d not in columns")
        print(f"  Available columns: {list(df_quintiles.columns)}")
else:
    print("ERROR: Quintile column not created")
    print(f"Available columns: {list(df_quintiles.columns)}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
