"""
Quick verification script to check if new file has exactly 8 metrics
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import config_banking

# Load data
file_path = config_banking.FINANCIAL_FILE
print(f"Loading: {file_path}\n")

# Read first sheet
xl = pd.ExcelFile(file_path)
first_sheet = xl.sheet_names[0]
df = pd.read_excel(file_path, sheet_name=first_sheet)

print(f"Sheet: {first_sheet}")
print(f"Columns: {list(df.columns)}\n")

# Expected metrics
expected_metrics = ['roa', 'net_profit_yoy', 'loan_growth', 'operating_income_yoy',
                    'cir', 'equity_assets', 'fee_ratio', 'ocf_net_profit']

# Check which metrics exist
found_metrics = [m for m in expected_metrics if m in df.columns]
missing_metrics = [m for m in expected_metrics if m not in df.columns]

# Check removed metrics
removed_metrics = ['nim', 'credit_cost', 'ldr']
still_present = [m for m in removed_metrics if m in df.columns]

print(f"Expected 8 metrics: {expected_metrics}")
print(f"\nFound {len(found_metrics)} metrics: {found_metrics}")

if missing_metrics:
    print(f"\nMISSING metrics: {missing_metrics}")

if still_present:
    print(f"\nWARNING: Removed metrics still present: {still_present}")
else:
    print(f"\n✓ Removed metrics (nim, credit_cost, ldr) successfully removed!")

print(f"\n{'='*60}")
if len(found_metrics) == 8 and not still_present:
    print("✓ VERIFICATION PASSED: Exactly 8 metrics, no removed metrics")
else:
    print("✗ VERIFICATION FAILED")
