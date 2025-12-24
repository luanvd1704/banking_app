"""
Banking Sector Analysis - Configuration
All configuration parameters for the banking flow analysis platform
"""
import os

# ============================================
# SECTOR INFO
# ============================================
SECTOR_NAME = "Banking"
SECTOR_CODE = "banking"

# ============================================
# TICKERS - 17 Banks
# ============================================
TICKERS = [
    "VCB",  # Vietcombank
    "TCB",  # Techcombank
    "MBB",  # MB Bank
    "ACB",  # ACB
    "VPB",  # VPBank
    "BID",  # BIDV
    "CTG",  # VietinBank
    "STB",  # Sacombank
    "HDB",  # HDBank
    "TPB",  # TPBank
    "VIB",  # VIB
    "SSB",  # Southeast Asia Bank
    "SHB",  # SHB
    "MSB",  # MSB
    "LPB",  # LienVietPostBank
    "OCB",  # OCB
    "EIB"   # Eximbank
]

# ============================================
# FILE PATHS
# ============================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data-collector')

# Data files
FOREIGN_TRADING_FILE = os.path.join(DATA_DIR, 'bank_foreign_trading.xlsx')
VALUATION_FILE = os.path.join(DATA_DIR, 'bank_valuation.xlsx')
FINANCIAL_FILE = os.path.join(DATA_DIR, 'bank_financials.xlsx')
VNINDEX_FILE = os.path.join(DATA_DIR, 'vnindex_market.xlsx')

# ============================================
# STREAMLIT CONFIG
# ============================================
PAGE_TITLE = "Banking Sector Analysis"
PAGE_ICON = "🏦"
LAYOUT = "wide"
CACHE_TTL = 3600  # 1 hour

# ============================================
# SECTOR-SPECIFIC PARAMETERS
# ============================================
HAS_FINANCIAL_METRICS = True
HAS_RANKING_PAGE = True

# ============================================
# ANALYSIS PARAMETERS
# ============================================
# Event windows for analysis
EVENT_WINDOWS = [(1, 5), (1, 10)]

# Forward return horizons (in trading days)
FORWARD_RETURN_HORIZONS = [1, 3, 5, 10, 20, 30]

# Rolling window parameters
ADV_WINDOW = 20              # Average daily volume window
ZSCORE_WINDOW = 252          # Z-score window (1 year trading days)
PERCENTILE_WINDOW = 756      # Percentile window (3 years trading days)

# Statistical parameters
SIGNIFICANCE_LEVEL = 0.05    # Alpha for hypothesis testing
MIN_SAMPLE_SIZE = 30         # Minimum sample size for t-tests

# Market regime parameters
MA_WINDOW_REGIME = 200       # Moving average for bull/bear classification

# Backtest parameters
REBALANCE_FREQ = 'M'         # Monthly rebalancing
MIN_HOLDING_PERIOD = 1       # Minimum holding period in days

# Normalization methods
NORMALIZATION_METHODS = ['ADV20', 'GTGD']

# ============================================
# VISUALIZATION PARAMETERS
# ============================================
# Color schemes
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'neutral': '#7f7f7f'
}

# Quintile colors (for Q1, Q5 analysis)
QUINTILE_COLORS = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c', '#1f77b4']

# Tercile labels
TERCILE_LABELS = ['T1', 'T2', 'T3']

# Chart defaults
CHART_HEIGHT = 500
CHART_WIDTH = 800

# ============================================
# DATA QUALITY PARAMETERS
# ============================================
MAX_MISSING_PCT = 0.5  # 50%
MIN_DATA_POINTS = 100

# ============================================
# BANKING FINANCIAL METRICS (9 metrics)
# ============================================
BANK_METRICS = {
    'roa': {
        'name': 'ROA',
        'full_name': 'Return on Assets (TTM)',
        'group': 'Profitability',
        'direction': 'higher_is_better',
        'unit': '%',
        'formula': 'Σ Net Profit (4Q) / Avg Total Assets * 100',
        'description': 'TTM return on assets - measures profitability efficiency over last 12 months',
        'weight': 1.0
    },
    'net_profit_yoy': {
        'name': 'Net Profit YoY',
        'full_name': 'Net Profit YoY Growth (9M YTD)',
        'group': 'Growth',
        'direction': 'higher_is_better',
        'unit': '%',
        'formula': '(9M This Year - 9M Last Year) / 9M Last Year * 100',
        'description': 'Year-to-date net profit growth (9 months) - more stable than quarterly comparison',
        'weight': 1.0
    },
    'loan_growth': {
        'name': 'Loan Growth',
        'full_name': 'Loan Growth YoY (End-Quarter)',
        'group': 'Growth',
        'direction': 'higher_is_better',
        'unit': '%',
        'formula': '(Loans End-Q This Year - Loans End-Q Last Year) / Loans Last Year * 100',
        'description': 'Year-over-year loan portfolio growth comparing same quarter end-points (lower weight due to noise)',
        'weight': 0.5
    },
    'operating_income_yoy': {
        'name': 'Operating Income YoY',
        'full_name': 'Operating Income Growth (9M YTD)',
        'group': 'Growth',
        'direction': 'higher_is_better',
        'unit': '%',
        'formula': '(9M This Year - 9M Last Year) / 9M Last Year * 100',
        'description': 'Year-to-date operating income growth (9 months) - more stable than quarterly',
        'weight': 1.0
    },
    'cir': {
        'name': 'CIR',
        'full_name': 'Cost-to-Income Ratio (TTM)',
        'group': 'Efficiency',
        'direction': 'lower_is_better',
        'unit': '%',
        'formula': 'Σ Operating Expense (4Q) / Σ Operating Income (4Q) * 100',
        'description': 'TTM operating efficiency - lower is better, calculated over last 12 months',
        'weight': 1.0
    },
    'equity_assets': {
        'name': 'Equity/Assets',
        'full_name': 'Equity to Assets Ratio (End-Quarter)',
        'group': 'Capital & Liquidity',
        'direction': 'optimal_range',
        'optimal_range': (7, 12),
        'unit': '%',
        'formula': 'Equity End-Q / Total Assets End-Q * 100',
        'description': 'Capital adequacy at quarter-end - optimal range 7-12%',
        'weight': 1.0
    },
    'fee_ratio': {
        'name': 'Fee Ratio',
        'full_name': 'Fee Income Ratio (TTM)',
        'group': 'Income Structure',
        'direction': 'higher_is_better',
        'unit': '%',
        'formula': 'Σ Fee Income (4Q) / Σ Operating Income (4Q) * 100',
        'description': 'TTM non-interest income diversification over last 12 months',
        'weight': 1.0
    },
    'ocf_net_profit': {
        'name': 'OCF/Net Profit',
        'full_name': 'Operating Cash Flow to Net Profit (TTM)',
        'group': 'Cashflow Quality',
        'direction': 'higher_is_better',
        'unit': 'ratio',
        'formula': 'Σ OCF (4Q) / Σ Net Profit (4Q)',
        'description': 'TTM cash generation quality - warning flag only, not main driver (very high noise in banking CFO)',
        'weight': 0.25
    },
    'ldr': {
        'name': 'LDR',
        'full_name': 'Loan-to-Deposit Ratio (End-Quarter)',
        'group': 'Capital & Liquidity',
        'direction': 'lower_is_better',
        'unit': '%',
        'formula': 'Loans End-Q / Deposits End-Q * 100',
        'description': 'Tỷ lệ cho vay trên huy động - thấp hơn = thanh khoản tốt hơn',
        'weight': 0.10
    }
}

# Metric groups for UI organization
METRIC_GROUPS = {
    'Profitability': ['roa'],
    'Growth': ['net_profit_yoy', 'loan_growth', 'operating_income_yoy'],
    'Efficiency': ['cir'],
    'Capital & Liquidity': ['equity_assets', 'ldr'],
    'Income Structure': ['fee_ratio'],
    'Cashflow Quality': ['ocf_net_profit']
}

# Default metric for ranking page
DEFAULT_RANKING_METRIC = 'roa'

# ============================================
# WARNINGS AND DISCLAIMERS
# ============================================
SECTOR_WARNING = """
⚠️ **Lưu ý về dữ liệu Banking - 9 Metrics với TTM + YTD Methodology**:
- **TTM (Trailing Twelve Months)**: ROA, CIR, Fee Ratio, OCF/Net Profit
  → Tổng 4 quý gần nhất, phản ánh performance thực tế 12 tháng
- **9M YTD (Year-to-Date)**: Net Profit YoY, Operating Income YoY
  → So sánh 9 tháng năm nay vs 9 tháng năm trước, ổn định hơn quarterly YoY
- **End-Quarter**: Loan Growth, Equity/Assets, LDR
  → Snapshot cuối quý, balance sheet metrics
- **Trọng số phân tầng**:
  - **1.0 (Bình thường)**: ROA, Net Profit YoY, Operating Income YoY, CIR, Equity/Assets, Fee Ratio
  - **0.5 (Tham khảo)**: Loan Growth (độ nhiễu cao)
  - **0.25 (Cờ cảnh báo)**: OCF/Net Profit (CFO ngân hàng rất nhiễu, chỉ dùng để cảnh báo)
  - **0.10 (Theo dõi)**: LDR (liquidity monitoring)
- Dữ liệu BCTC quarterly được forward-fill sang daily cho ranking
- Cần ít nhất 4-5 quarters dữ liệu lịch sử để tính TTM và YTD chính xác
"""

BACKTEST_DISCLAIMER = """
⚠️ **Disclaimer**: Kết quả backtest là phân tích lịch sử và không đảm bảo hiệu suất tương lai.
Không nên sử dụng làm khuyến nghị đầu tư.
"""

DATA_LIMITATION_WARNING = """
⚠️ **Giới hạn dữ liệu**: Một số thời kỳ có thể thiếu dữ liệu giao dịch.
Các giá trị bị thiếu được forward-fill cho giá, nhưng dữ liệu giao dịch giữ nguyên NaN.
"""
