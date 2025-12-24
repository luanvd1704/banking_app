"""Verify SHB data to understand why analysis shows no significance"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Direct load from Excel to check raw data
print("="*80)
print("VERIFYING SHB RAW DATA")
print("="*80)

# Load foreign trading data directly
foreign_file = "data/bank_foreign_trading.xlsx"
df = pd.read_excel(foreign_file, sheet_name='SHB')

print(f"\nRaw data loaded: {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Check date range
if 'Ngay' in df.columns:
    df['Ngay'] = pd.to_datetime(df['Ngay'], format='%d/%m/%Y')
    print(f"Date range: {df['Ngay'].min()} to {df['Ngay'].max()}")

# Check foreign trading values
if 'GTDGRong' in df.columns:
    print(f"\nForeign Net Buy Value (GTDGRong) stats:")
    print(f"  Total: {len(df)}")
    print(f"  Non-null: {df['GTDGRong'].notna().sum()}")
    print(f"  Zeros: {(df['GTDGRong'] == 0).sum()}")
    print(f"  Non-zero: {((df['GTDGRong'] != 0) & df['GTDGRong'].notna()).sum()}")
    print(f"  Mean: {df['GTDGRong'].mean():.2f}")
    print(f"  Std: {df['GTDGRong'].std():.2f}")

# Check close prices
if 'Close' in df.columns:
    print(f"\nClose price stats:")
    print(f"  Non-null: {df['Close'].notna().sum()}")
    print(f"  Zeros: {(df['Close'] == 0).sum()}")
    print(f"  Mean: {df['Close'].mean():.2f}")
    print(f"  Std: {df['Close'].std():.2f}")

    # Show zero price dates
    if (df['Close'] == 0).sum() > 0:
        print(f"\nDates with zero close prices:")
        zero_dates = df[df['Close'] == 0]['Ngay']
        for date in zero_dates:
            print(f"  {date}")

# Check recent data
print(f"\n" + "="*80)
print("RECENT DATA (last 10 rows)")
print("="*80)
cols_to_show = ['Ngay', 'GTDGRong', 'Close']
print(df[cols_to_show].tail(10))

# Check for any data issues
print(f"\n" + "="*80)
print("DATA QUALITY CHECK")
print("="*80)

# Check if we have both fields needed for analysis
if 'GTDGRong' in df.columns and 'Close' in df.columns:
    # Find rows with valid data for both
    valid_mask = (df['GTDGRong'].notna()) & (df['Close'].notna()) & (df['Close'] > 0)
    print(f"Rows with valid foreign trading and close price: {valid_mask.sum()}")

    # Check non-zero foreign trading with valid prices
    non_zero_valid = (df['GTDGRong'] != 0) & valid_mask
    print(f"Rows with non-zero foreign trading and valid price: {non_zero_valid.sum()}")

print("\n" + "="*80)
