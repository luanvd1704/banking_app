"""Test CaféF API directly"""
import requests
import json

# Test với VCB
symbol = "VCB"
url = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/GDKhoiNgoai.ashx"

params = {
    "Symbol": symbol,
    "PageIndex": 1,
    "PageSize": 10,
    "StartDate": "18/12/2025",
    "EndDate": "22/12/2025"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://cafef.vn/",
    "Accept": "application/json, text/plain, */*"
}

print("Testing CafeF API...")
print(f"URL: {url}")
print(f"Symbol: {symbol}")
print(f"Date range: {params['StartDate']} -> {params['EndDate']}")
print()

try:
    response = requests.get(url, params=params, headers=headers, timeout=30)
    print(f"Status code: {response.status_code}")
    print()

    if response.status_code == 200:
        data = response.json()
        print("JSON Response:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print()

        # Check structure
        if "Data" in data:
            if "Data" in data["Data"]:
                if "Data" in data["Data"]["Data"]:
                    records = data["Data"]["Data"]["Data"]
                    print(f"Number of records: {len(records)}")
                    if len(records) > 0:
                        print("First record:")
                        print(json.dumps(records[0], indent=2, ensure_ascii=False))
                    else:
                        print("NO RECORDS RETURNED!")
                else:
                    print("Missing Data['Data']['Data']")
            else:
                print("Missing Data['Data']")
        else:
            print("Missing 'Data' key")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
