# Banking Sector Analysis Platform 🏦

Nền tảng phân tích định lượng ngành ngân hàng Việt Nam

## Overview

Platform này phân tích **17 ngân hàng** hàng đầu Việt Nam với 6 câu hỏi nghiên cứu và ranking theo 8 chỉ số tài chính.

### 🏦 17 Ngân hàng được phân tích:
VCB, TCB, MBB, ACB, VPB, BID, CTG, STB, HDB, TPB, VIB, SSB, SHB, MSB, LPB, OCB, EIB

## Features

### 📋 6 Câu hỏi Nghiên cứu:

1. **Q1: Foreign Lead/Lag** 🔍
   - Khối ngoại có thể dự đoán lợi nhuận T+1/T+3/T+5/T+10 không?
   - Phân tích quintile và kiểm định thống kê
   - Tìm cửa sổ normalization tối ưu

2. **Q2: Self-Trading Signals** 💼
   - Tự doanh có sinh lợi không?
   - So sánh ADV20 vs GTGD normalization
   - Information Coefficient analysis

3. **Q3: Foreign vs Self Conflicts** ⚔️
   - Ai dẫn dắt khi có xung đột?
   - Granger causality test
   - Event window analysis

4. **Q4: Valuation Percentiles** 💰
   - PE/PB thấp → lợi nhuận cao hơn?
   - Phân tích percentile và decile
   - Zone identification (cheap/expensive)

5. **Q5: Composite Score** 🎯
   - Kết hợp tín hiệu: z(Foreign) + z(Self) - percentile(PE/PB)
   - Quintile backtest
   - CAPM alpha analysis

6. **Ranking by Financial Metrics** 🏆
   - Xếp hạng theo 8 chỉ số tài chính
   - Cross-sectional analysis
   - Quintile performance comparison

### 💰 8 Chỉ Số Tài Chính (TTM + YTD Methodology):

**Profitability:**
- ROA (Return on Assets) - TTM - Trọng số 1.0

**Growth:**
- Net Profit YoY - 9M YTD - Trọng số 1.0
- Operating Income YoY - 9M YTD - Trọng số 1.0
- Loan Growth - End-Quarter - Trọng số 0.5

**Efficiency:**
- CIR (Cost-to-Income Ratio) - TTM - Trọng số 1.0

**Capital & Liquidity:**
- Equity/Assets - End-Quarter - Trọng số 1.0

**Income Structure:**
- Fee Ratio - TTM - Trọng số 1.0

**Cashflow Quality:**
- OCF/Net Profit - TTM - Trọng số 0.25 (Cờ cảnh báo)

## Data

| Dataset | Thời gian | Tickers |
|---------|-----------|---------|
| **Foreign Trading** | 2020-12 → 2025-12 | 17 banks |
| **Self-Trading** | 2022-11 → 2025-12 | 17 banks ⚠️ |
| **Valuation** | 2019-12 → 2025-12 | 17 banks |
| **Financial Metrics** | Quarterly (8Q) | 17 banks |

⚠️ **Lưu ý**: Dữ liệu tự doanh chỉ có 3 năm → Q2, Q3, Q5 có giới hạn

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/banking-flow-analysis.git
cd banking-flow-analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run banking_app.py
```

The app will open in your browser at `http://localhost:8501`

### Updating Data

Use the scripts in the `data-collector/` folder to fetch latest data:

```bash
cd data-collector
python export_excel.py
```

This will update the following Excel files:
- `steel_foreign_trading.xlsx` - Foreign trading data
- `steel_self_trading.xlsx` - Self trading data
- `steel_valuation.xlsx` - Valuation metrics
- `vnindex_market.xlsx` - Market data

## Project Structure

```
banking-flow-analysis/
├── banking_app.py              # Main Streamlit app entry point
├── pages_banking/              # Streamlit pages for each analysis
│   ├── 1_📊_Overview.py
│   ├── 2_🔍_Q1_Foreign_LeadLag.py
│   ├── 3_💰_Q4_Valuation.py
│   ├── 4_🎯_Q5_Composite.py
│   └── 5_🏆_Ranking.py
├── data-collector/             # Data fetching and Excel export
│   ├── export_excel.py
│   ├── fetch_cafef_trade_data.py
│   ├── fetch_smoney_trade_data.py
│   └── *.xlsx                  # Data files
├── config/                     # Configuration files
├── data/                       # Processed data
├── utils/                      # Utility functions
├── scripts/                    # Helper scripts
├── calculate_8_metrics.py      # Financial metrics calculation
├── banking_metrics.csv         # Calculated metrics
├── Bank_Metrics_Formulas.txt   # Formula documentation
└── requirements.txt            # Python dependencies
```

## Methodology

- **Event study**: Phân tích sự kiện giao dịch
- **Quintile analysis**: Chia nhóm và so sánh hiệu suất
- **Statistical testing**: T-tests, p-values, confidence intervals
- **Cross-sectional ranking**: Xếp hạng theo chỉ số tài chính
- **CAPM analysis**: Risk-adjusted returns

## Deployment

This app is designed to be deployed on Streamlit Cloud:

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click
4. Auto-redeploy on every push

## Data Updates

Data is automatically updated when you run:
```bash
python data-collector/export_excel.py
```

Commit and push the updated Excel files to trigger redeployment.

## Disclaimer

⚠️ **Disclaimer**: Đây là nghiên cứu định lượng, không phải khuyến nghị đầu tư.

This is quantitative research for educational purposes only. Not investment advice.

## License

Copyright © 2025 Banking Sector Analysis

## Contact

For questions or suggestions, please open an issue on GitHub.
