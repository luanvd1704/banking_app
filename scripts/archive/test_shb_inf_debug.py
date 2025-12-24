"""Debug the inf values in Q1 quintile for SHB"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.loader import merge_all_data
from analysis.lead_lag import prepare_lead_lag_data, create_quintiles_by_foreign_trading
from config.config import get_sector_config
from utils.constants import *

# Get banking config
config_banking = get_sector_config('banking')

print("="*80)
print("DEBUGGING INF VALUES IN Q1 QUINTILE")
print("="*80)

# Load SHB data
data = merge_all_data(config_banking, tickers=['SHB'])
df = data['SHB']

# Prepare data
df_prepared = prepare_lead_lag_data(df, [1, 3])

# Create quintiles
df_quintiles = create_quintiles_by_foreign_trading(df_prepared)

# Check for inf values in forward returns
print("\nChecking for inf values in forward returns...")
print(f"Inf values in fwd_excess_return_1d: {np.isinf(df_quintiles['fwd_excess_return_1d']).sum()}")
print(f"NaN values in fwd_excess_return_1d: {df_quintiles['fwd_excess_return_1d'].isna().sum()}")

# Find rows with inf values
inf_rows = df_quintiles[np.isinf(df_quintiles['fwd_excess_return_1d'])]
print(f"\nFound {len(inf_rows)} rows with inf forward returns")

if len(inf_rows) > 0:
    print("\nFirst 10 inf rows:")
    cols_to_show = [DATE, CLOSE, FOREIGN_NET_BUY_VAL, QUINTILE, 'fwd_return_1d', 'fwd_excess_return_1d']
    print(inf_rows[cols_to_show].head(10))

    # Check the quintile distribution of inf values
    print("\nQuintile distribution of inf values:")
    print(inf_rows[QUINTILE].value_counts())

# Check for zero or near-zero prices
print("\n" + "="*80)
print("Checking for problematic close prices...")
print("="*80)

zero_price_rows = df_quintiles[df_quintiles[CLOSE] == 0]
print(f"Rows with zero close price: {len(zero_price_rows)}")

if len(zero_price_rows) > 0:
    print("\nRows with zero prices:")
    print(zero_price_rows[[DATE, CLOSE, FOREIGN_NET_BUY_VAL]].head(10))

# Check for very small prices that might cause issues
small_price_rows = df_quintiles[(df_quintiles[CLOSE] > 0) & (df_quintiles[CLOSE] < 0.01)]
print(f"\nRows with very small close price (< 0.01): {len(small_price_rows)}")

if len(small_price_rows) > 0:
    print("\nRows with very small prices:")
    print(small_price_rows[[DATE, CLOSE, FOREIGN_NET_BUY_VAL]].head(10))

# Check Q1 data specifically
print("\n" + "="*80)
print("Checking Q1 quintile data...")
print("="*80)

q1_data = df_quintiles[df_quintiles[QUINTILE] == 'Q1']
print(f"Q1 total rows: {len(q1_data)}")
print(f"Q1 inf returns: {np.isinf(q1_data['fwd_excess_return_1d']).sum()}")
print(f"Q1 non-inf returns: {(~np.isinf(q1_data['fwd_excess_return_1d'])).sum()}")

# Check if we're including zeros in Q1
q1_zeros = q1_data[q1_data[FOREIGN_NET_BUY_VAL] == 0]
print(f"\nQ1 rows with zero foreign net buy: {len(q1_zeros)}")
print(f"Total zero foreign net buy rows: {(df_quintiles[FOREIGN_NET_BUY_VAL] == 0).sum()}")

# Show quintile distribution including zeros
print("\n" + "="*80)
print("QUINTILE DISTRIBUTION OF ZERO FOREIGN NET BUY VALUES")
print("="*80)

zero_foreign_rows = df_quintiles[df_quintiles[FOREIGN_NET_BUY_VAL] == 0]
print(f"Total rows with zero foreign net buy: {len(zero_foreign_rows)}")

if len(zero_foreign_rows) > 0:
    print("\nQuintile distribution of zero values:")
    print(zero_foreign_rows[QUINTILE].value_counts())

# Show foreign_net_buy_val distribution by quintile
print("\n" + "="*80)
print("FOREIGN NET BUY VALUE RANGE BY QUINTILE")
print("="*80)

for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    q_data = df_quintiles[df_quintiles[QUINTILE] == q][FOREIGN_NET_BUY_VAL]
    print(f"\n{q}:")
    print(f"  Count: {len(q_data)}")
    print(f"  Min: {q_data.min():.2f}")
    print(f"  Max: {q_data.max():.2f}")
    print(f"  Mean: {q_data.mean():.2f}")
    print(f"  Zeros: {(q_data == 0).sum()}")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
