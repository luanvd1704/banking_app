"""
Fetch 6-year price data silently (suppress vnstock output)
"""
import sys
import os
import io
import contextlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datetime import datetime
import time

# Suppress all output during vnstock import
@contextlib.contextmanager
def suppress_output():
    """Suppress stdout and stderr"""
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr

# Import vnstock silently
with suppress_output():
    from vnstock3 import Vnstock

FOREIGN_FILE = 'data/bank_foreign_trading.xlsx'

# All 17 tickers in 3 batches
BATCH_1 = ['VCB', 'TCB', 'MBB', 'ACB', 'VPB', 'BID']
BATCH_2 = ['CTG', 'STB', 'HDB', 'TPB', 'VIB', 'SSB']
BATCH_3 = ['SHB', 'MSB', 'LPB', 'OCB', 'EIB']

print("=" * 80)
print("FETCHING 6-YEAR PRICE DATA FOR 17 BANKS")
print("Period: 23/12/2019 to 22/12/2025")
print("=" * 80)

# Load existing foreign trading data
print("\nLoading foreign trading data...")
excel_file = pd.ExcelFile(FOREIGN_FILE)
foreign_data = {}
for ticker in excel_file.sheet_names:
    df = pd.read_excel(FOREIGN_FILE, sheet_name=ticker)
    df['Ngay'] = pd.to_datetime(df['Ngay'])
    foreign_data[ticker] = df

print(f"Loaded {len(foreign_data)} stocks")

def fetch_prices_batch(batch_name, tickers):
    """Fetch prices for one batch"""
    print(f"\n{'=' * 60}")
    print(f"{batch_name}: {len(tickers)} stocks")
    print('=' * 60)

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}...", end=" ")

        try:
            # Get date range from foreign data
            foreign_df = foreign_data.get(ticker)
            if foreign_df is None:
                print("SKIP (no foreign data)")
                continue

            start_date = foreign_df['Ngay'].min()
            end_date = foreign_df['Ngay'].max()

            # Fetch prices with vnstock (suppress output)
            with suppress_output():
                stock = Vnstock().stock(symbol=ticker, source='VCI')
                price_df = stock.quote.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1D'
                )

            if price_df is not None and not price_df.empty:
                # Process price data
                if 'time' in price_df.columns:
                    price_df['Ngay'] = pd.to_datetime(price_df['time'])
                else:
                    price_df['Ngay'] = pd.to_datetime(price_df.index)

                if 'close' in price_df.columns:
                    price_df['Close'] = price_df['close']

                price_df = price_df[['Ngay', 'Close']].copy()

                # Merge with foreign data
                merged_df = pd.merge(foreign_df, price_df, on='Ngay', how='left')
                foreign_data[ticker] = merged_df

                close_count = merged_df['Close'].notna().sum()
                print(f"OK ({len(price_df)} rows, {close_count} merged)")
            else:
                print("FAIL (no data)")

        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")

        # Delay
        if i < len(tickers):
            time.sleep(15)

# Process all batches
fetch_prices_batch("BATCH 1/3", BATCH_1)
print("\nWaiting 120 seconds...")
time.sleep(120)

fetch_prices_batch("BATCH 2/3", BATCH_2)
print("\nWaiting 120 seconds...")
time.sleep(120)

fetch_prices_batch("BATCH 3/3", BATCH_3)

# Save updated data
print("\n" + "=" * 80)
print("SAVING UPDATED DATA")
print("=" * 80)

with pd.ExcelWriter(FOREIGN_FILE, engine='openpyxl') as writer:
    for ticker in sorted(foreign_data.keys()):
        df = foreign_data[ticker]
        df.to_excel(writer, sheet_name=ticker, index=False)

        has_close = 'Close' in df.columns
        close_count = df['Close'].notna().sum() if has_close else 0
        total = len(df)
        pct = (close_count / total * 100) if total > 0 else 0

        print(f"  {ticker}: {total} rows, {close_count} with Close ({pct:.1f}%)")

print(f"\nOK Saved to {FOREIGN_FILE}")
print("=" * 80)
