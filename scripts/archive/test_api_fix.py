"""Test the fixed CaféF API parsing"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Stock-analyst'))

from fetch_cafef_trade_data import fetch_cafef_foreign_trades

print("Testing fixed CaféF API parsing...")
print("=" * 60)

# Test with VCB for last 5 days
ticker = "VCB"
start_date = "18/12/2025"
end_date = "22/12/2025"

print(f"Ticker: {ticker}")
print(f"Date range: {start_date} to {end_date}")
print()

try:
    df = fetch_cafef_foreign_trades(ticker, start_date, end_date)

    if df.empty:
        print("FAILED: No data returned")
    else:
        print(f"SUCCESS: {len(df)} records fetched")
        print()
        print("First 3 records:")
        print(df.head(3)[['Ngay', 'KLGDRong', 'GTDGRong', 'ThayDoi']])
        print()
        print("Columns:", list(df.columns))

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
