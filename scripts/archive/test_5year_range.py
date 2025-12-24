"""Test CaféF API with 5-year date range"""
import sys
import os
import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Stock-analyst'))

from fetch_cafef_trade_data import fetch_cafef_foreign_trades

# Same date range as the main script
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5*365)
start_date_str = start_date.strftime("%d/%m/%Y")
end_date_str = end_date.strftime("%d/%m/%Y")

print(f"Testing with 5-year date range:")
print(f"Start: {start_date_str}")
print(f"End: {end_date_str}")
print("=" * 60)

ticker = "VCB"
print(f"\nTesting {ticker}...")

try:
    df = fetch_cafef_foreign_trades(ticker, start_date_str, end_date_str)
    print(f"Result: {len(df)} records")
    if not df.empty:
        print(f"Date range in data: {df['Ngay'].min()} to {df['Ngay'].max()}")
        print(f"\nFirst record:")
        print(df.head(1))
    else:
        print("DataFrame is EMPTY!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
