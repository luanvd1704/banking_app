"""
Banking Flow Analysis - Helper Functions
All constants, data loading, calculations, and utilities in one place
Organized into clear sections for maintainability
"""
import pandas as pd
import numpy as np
import streamlit as st
import os
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from datetime import datetime, timedelta

# ============================================
# SECTION 1: CONSTANTS
# ============================================

# Column Names - Foreign Trading
FOREIGN_NET_BUY_VOL = 'foreign_net_buy_vol'
FOREIGN_NET_BUY_VAL = 'foreign_net_buy_val'
FOREIGN_BUY_VOL = 'foreign_buy_vol'
FOREIGN_SELL_VOL = 'foreign_sell_vol'

# Column Names - Self Trading
SELF_NET_BUY_VOL = 'self_net_buy_vol'
SELF_NET_BUY_VAL = 'self_net_buy_val'
SELF_BUY_VAL = 'self_buy_val'
SELF_SELL_VAL = 'self_sell_val'

# Column Names - Price and Volume
CLOSE = 'close'
VOLUME = 'volume'
HIGH = 'high'
LOW = 'low'
OPEN = 'open'

# Column Names - Valuation
PE = 'pe'
PB = 'pb'
PCFS = 'pcfs'

# Column Names - Market
VNINDEX_CLOSE = 'vnindex_close'
MARKET_RETURN = 'market_return'
STOCK_RETURN = 'return'

# Column Names - Derived
EXCESS_RETURN = 'excess_return'
MA200 = 'ma200'
BULL_MARKET = 'bull_market'

# Column Names - Normalized Signals
FOREIGN_SIGNAL_ADV20 = 'foreign_signal_adv20'
SELF_SIGNAL_ADV20 = 'self_signal_adv20'
SELF_SIGNAL_GTGD = 'self_signal_gtgd'

# Column Names - Percentiles
PE_PERCENTILE = 'pe_percentile'
PB_PERCENTILE = 'pb_percentile'
PCFS_PERCENTILE = 'pcfs_percentile'

# Column Names - Z-scores
FOREIGN_ZSCORE = 'foreign_zscore'
SELF_ZSCORE = 'self_zscore'

# Column Names - Composite
COMPOSITE_SCORE = 'composite_score'
COMPOSITE_RANK = 'composite_rank'

# Column Names - Grouping
QUINTILE = 'quintile'
TERCILE = 'tercile'
DECILE = 'decile'

# Column Names - Forward Returns
FWD_RETURN_1D = 'fwd_return_1d'
FWD_RETURN_3D = 'fwd_return_3d'
FWD_RETURN_5D = 'fwd_return_5d'
FWD_RETURN_10D = 'fwd_return_10d'

# Column Names - States
CONFLICT_STATE = 'conflict_state'

# Conflict States
BOTH_BUY = 'Both Buy'
FOREIGN_BUY_SELF_SELL = 'Foreign Buy, Self Sell'
FOREIGN_SELL_SELF_BUY = 'Foreign Sell, Self Buy'
BOTH_SELL = 'Both Sell'

# Market Regimes
BULL = 'Bull'
BEAR = 'Bear'

# Statistical Terms
T_STAT = 't_statistic'
P_VALUE = 'p_value'
MEAN = 'mean'
STD = 'std'
SHARPE = 'sharpe_ratio'
ALPHA = 'alpha'
BETA = 'beta'
IR = 'information_ratio'

# Labels
QUINTILE_LABELS = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)']
TERCILE_LABELS = ['T1 (Sell)', 'T2 (Neutral)', 'T3 (Buy)']
DECILE_LABELS = [f'D{i+1}' for i in range(10)]

# Date Columns
DATE = 'date'
TRADING_DATE = 'TradingDate'


# ============================================
# SECTION 2: DATA LOADING FUNCTIONS
# ============================================

def standardize_date_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Standardize date column to datetime format

    Args:
        df: DataFrame with date column
        date_col: Name of date column

    Returns:
        DataFrame with standardized date column
    """
    df = df.copy()

    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col])

    return df


def load_foreign_trading(foreign_file: str, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load foreign trading data from Excel

    Args:
        foreign_file: Path to foreign trading Excel file
        tickers: List of tickers to load

    Returns:
        Dictionary mapping ticker to DataFrame
    """
    if not os.path.exists(foreign_file):
        raise FileNotFoundError(f"Foreign trading file not found: {foreign_file}")

    excel_file = pd.ExcelFile(foreign_file)
    data_dict = {}

    for ticker in tickers:
        if ticker in excel_file.sheet_names:
            df = pd.read_excel(foreign_file, sheet_name=ticker)

            # Column mapping from Vietnamese to English
            column_map = {
                'Ngay': DATE,
                'ngay': DATE,
                'KLGDRong': FOREIGN_NET_BUY_VOL,
                'klgdrong': FOREIGN_NET_BUY_VOL,
                'GTDGRong': FOREIGN_NET_BUY_VAL,
                'gtdgrong': FOREIGN_NET_BUY_VAL,
                'Close': CLOSE,
                'close': CLOSE,
                'TradingDate': DATE,
                'tradingdate': DATE,
                'date': DATE,
                'Date': DATE
            }

            df = df.rename(columns=column_map)

            # Check if date column exists
            if DATE not in df.columns:
                raise ValueError(f"No date column found for {ticker}. Available columns: {list(df.columns)}")

            df = standardize_date_column(df, DATE)

            # Select only needed columns
            cols_to_keep = [DATE] + [col for col in [FOREIGN_NET_BUY_VOL, FOREIGN_NET_BUY_VAL, CLOSE] if col in df.columns]
            df = df[cols_to_keep]

            # Sort by date
            df = df.sort_values(DATE).reset_index(drop=True)

            data_dict[ticker] = df
        else:
            print(f"Warning: {ticker} sheet not found in foreign trading file")

    return data_dict


def load_valuation(val_file: str, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load valuation data (PE, PB, PCFS) from Excel

    Args:
        val_file: Path to valuation Excel file
        tickers: List of tickers to load

    Returns:
        Dictionary mapping ticker to DataFrame
    """
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Valuation file not found: {val_file}")

    excel_file = pd.ExcelFile(val_file)
    data_dict = {}

    for ticker in tickers:
        if ticker in excel_file.sheet_names:
            df = pd.read_excel(val_file, sheet_name=ticker)

            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()

            if 'date' not in df.columns:
                raise ValueError(f"No date column found for {ticker}")

            df = df.rename(columns={'date': DATE})
            df = standardize_date_column(df, DATE)

            # Select only needed columns
            cols_to_keep = [DATE]
            for col in [PE, PB, PCFS]:
                if col in df.columns:
                    cols_to_keep.append(col)

            df = df[cols_to_keep]

            # Sort by date
            df = df.sort_values(DATE).reset_index(drop=True)

            data_dict[ticker] = df
        else:
            print(f"Warning: {ticker} sheet not found in valuation file")

    return data_dict


def load_vnindex(vnindex_file: str) -> pd.DataFrame:
    """
    Load VN-Index market data from Excel

    Args:
        vnindex_file: Path to VN-Index Excel file

    Returns:
        DataFrame with VN-Index data
    """
    if not os.path.exists(vnindex_file):
        raise FileNotFoundError(f"VN-Index file not found: {vnindex_file}")

    df = pd.read_excel(vnindex_file)

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Standardize date column
    if 'tradingdate' in df.columns:
        df = df.rename(columns={'tradingdate': DATE})
    elif 'date' not in df.columns:
        raise ValueError("No date column found in VN-Index file")

    df = standardize_date_column(df, DATE)

    # Rename close column to vnindex_close
    if CLOSE in df.columns:
        df = df.rename(columns={CLOSE: VNINDEX_CLOSE})

    # Sort by date
    df = df.sort_values(DATE).reset_index(drop=True)

    return df


def merge_all_data(foreign_file: str, val_file: str, vnindex_file: str,
                   tickers: List[str], ma_window: int = 200) -> Dict[str, pd.DataFrame]:
    """
    Load and merge all data sources for each ticker

    Merge strategy:
    - Outer join on date (preserve all dates)
    - Forward fill prices (close, vnindex)
    - Leave trading data as NaN where missing

    Args:
        foreign_file: Path to foreign trading Excel file
        val_file: Path to valuation Excel file
        vnindex_file: Path to VN-Index Excel file
        tickers: List of tickers to load
        ma_window: Moving average window for bull/bear regime

    Returns:
        Dictionary mapping ticker to merged DataFrame
    """
    print("Loading data files...")
    foreign_data = load_foreign_trading(foreign_file, tickers)
    valuation_data = load_valuation(val_file, tickers)
    vnindex_data = load_vnindex(vnindex_file)

    print("Merging data...")
    merged_data = {}

    for ticker in tickers:
        print(f"Processing {ticker}...")

        # Start with foreign trading data (has price and date)
        if ticker not in foreign_data:
            print(f"Warning: No foreign trading data for {ticker}, skipping...")
            continue

        df = foreign_data[ticker].copy()

        # Merge valuation
        if ticker in valuation_data:
            df = df.merge(
                valuation_data[ticker],
                on=DATE,
                how='outer'
            )

        # Merge VN-Index
        df = df.merge(
            vnindex_data[[DATE, VNINDEX_CLOSE]],
            on=DATE,
            how='outer'
        )

        # Sort by date
        df = df.sort_values(DATE).reset_index(drop=True)

        # Forward fill prices only (NOT trading data)
        price_cols = [CLOSE, VNINDEX_CLOSE, PE, PB]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Calculate returns
        df[STOCK_RETURN] = df[CLOSE].pct_change()
        df[MARKET_RETURN] = df[VNINDEX_CLOSE].pct_change()

        # Calculate excess return
        df[EXCESS_RETURN] = df[STOCK_RETURN] - df[MARKET_RETURN]

        # Calculate MA200 for bull/bear regime
        df[MA200] = df[VNINDEX_CLOSE].rolling(window=ma_window, min_periods=1).mean()
        df[BULL_MARKET] = (df[VNINDEX_CLOSE] > df[MA200]).astype(int)

        merged_data[ticker] = df

    print(f"Data loading complete. Loaded {len(merged_data)} tickers.")

    return merged_data


def get_data_summary(merged_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Get summary statistics of loaded data

    Args:
        merged_data: Dictionary of merged DataFrames

    Returns:
        Summary DataFrame
    """
    summary_data = []

    for ticker, df in merged_data.items():
        summary = {
            'Ticker': ticker,
            'Start Date': df[DATE].min(),
            'End Date': df[DATE].max(),
            'Total Days': len(df),
            'Foreign Data Points': df[FOREIGN_NET_BUY_VAL].notna().sum(),
            'Valuation Data Points': df[PE].notna().sum() if PE in df.columns else 0,
            'Missing Price %': f"{(df[CLOSE].isna().sum() / len(df) * 100):.1f}%"
        }
        summary_data.append(summary)

    return pd.DataFrame(summary_data)


# ============================================
# SECTION 3: CALCULATION FUNCTIONS
# ============================================

def calculate_return(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate return over specified periods"""
    return prices.pct_change(periods)


def calculate_excess_return(stock_return: pd.Series, market_return: pd.Series) -> pd.Series:
    """Calculate excess return (stock return - market return)"""
    return stock_return - market_return


def create_quintiles(series: pd.Series, labels: Optional[List[str]] = None) -> pd.Series:
    """
    Create quintiles from a series

    Args:
        series: Input series
        labels: Optional labels for quintiles

    Returns:
        Quintile assignments
    """
    if labels is None:
        labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

    # Drop NaN values for binning
    series_clean = series.dropna()

    if len(series_clean) == 0:
        return pd.Series(index=series.index, dtype='category')

    try:
        result = pd.qcut(series_clean, q=5, labels=labels, duplicates='drop')
    except (ValueError, TypeError):
        # If duplicates='drop' causes label mismatch, use rank-based approach
        percentiles = series_clean.rank(pct=True)
        result = pd.cut(percentiles, bins=5, labels=labels, include_lowest=True)

    # Reindex to match original series (NaN values will be NaN in result)
    return result.reindex(series.index)


def create_terciles(series: pd.Series, labels: Optional[List[str]] = None) -> pd.Series:
    """Create terciles from a series"""
    if labels is None:
        labels = ['T1', 'T2', 'T3']

    series_clean = series.dropna()

    if len(series_clean) == 0:
        return pd.Series(index=series.index, dtype='category')

    try:
        result = pd.qcut(series_clean, q=3, labels=labels, duplicates='drop')
    except (ValueError, TypeError):
        percentiles = series_clean.rank(pct=True)
        result = pd.cut(percentiles, bins=3, labels=labels, include_lowest=True)

    return result.reindex(series.index)


def create_deciles(series: pd.Series, labels: Optional[List[str]] = None) -> pd.Series:
    """Create deciles from a series"""
    if labels is None:
        labels = [f'D{i+1}' for i in range(10)]

    series_clean = series.dropna()

    if len(series_clean) == 0:
        return pd.Series(index=series.index, dtype='category')

    try:
        result = pd.qcut(series_clean, q=10, labels=labels, duplicates='drop')
    except (ValueError, TypeError):
        percentiles = series_clean.rank(pct=True)
        result = pd.cut(percentiles, bins=10, labels=labels, include_lowest=True)

    return result.reindex(series.index)


def calculate_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Calculate rolling z-score

    Args:
        series: Input series
        window: Rolling window size

    Returns:
        Z-score series
    """
    rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
    rolling_std = series.rolling(window=window, min_periods=window//2).std()

    return (series - rolling_mean) / rolling_std


def calculate_percentile(series: pd.Series, window: int = 756) -> pd.Series:
    """
    Calculate rolling percentile rank (0-100)

    Args:
        series: Input series
        window: Rolling window size

    Returns:
        Percentile series (0-100)
    """
    def percentile_rank(x):
        if len(x) < 2:
            return np.nan
        return stats.percentileofscore(x[:-1], x.iloc[-1])

    return series.rolling(window=window, min_periods=window//2).apply(percentile_rank, raw=False)


def calculate_forward_returns(prices: pd.Series, horizons: List[int]) -> pd.DataFrame:
    """
    Calculate forward returns for multiple horizons

    Args:
        prices: Price series
        horizons: List of forward periods

    Returns:
        DataFrame with forward returns for each horizon
    """
    fwd_returns = pd.DataFrame(index=prices.index)

    for h in horizons:
        fwd_returns[f'fwd_return_{h}d'] = prices.pct_change(h).shift(-h)

    return fwd_returns


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio

    Args:
        returns: Return series
        periods_per_year: Number of periods in a year (252 for daily)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return np.nan

    mean_return = returns.mean() * periods_per_year
    std_return = returns.std() * np.sqrt(periods_per_year)

    return mean_return / std_return


def calculate_information_coefficient(signal: pd.Series, forward_return: pd.Series) -> float:
    """
    Calculate Information Coefficient (IC) - correlation between signal and forward return

    Args:
        signal: Trading signal
        forward_return: Forward return

    Returns:
        IC (correlation coefficient)
    """
    valid_data = pd.DataFrame({'signal': signal, 'return': forward_return}).dropna()

    if len(valid_data) < 2:
        return np.nan

    return valid_data['signal'].corr(valid_data['return'])


def perform_ttest(group1: pd.Series, group2: pd.Series) -> Tuple[float, float]:
    """
    Perform two-sample t-test

    Args:
        group1: First group
        group2: Second group

    Returns:
        Tuple of (t-statistic, p-value)
    """
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()

    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan, np.nan

    t_stat, p_val = stats.ttest_ind(group1_clean, group2_clean)

    return t_stat, p_val


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Calculate drawdown series"""
    cumulative_max = equity_curve.expanding().max()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return drawdown


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown"""
    drawdown = calculate_drawdown(equity_curve)
    return drawdown.min()


def winsorize_series(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """Winsorize series to handle outliers"""
    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(upper_pct)
    return series.clip(lower=lower_bound, upper=upper_bound)


# ============================================
# SECTION 4: FORMATTING FUNCTIONS
# ============================================

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage"""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


# ============================================
# SECTION 5: UI UTILITIES (Streamlit)
# ============================================

def display_sidebar_logo():
    """
    Display logo in sidebar
    Compatible with older Streamlit versions
    """
    logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.jpg')

    if os.path.exists(logo_path):
        with st.sidebar:
            st.image(logo_path, use_column_width=True)
            st.markdown("---")


# ============================================
# SECTION 6: DATE UTILITIES
# ============================================

def is_business_day(date: pd.Timestamp) -> bool:
    """Check if date is a business day (Monday-Friday)"""
    return date.weekday() < 5  # Monday=0, Friday=4


def get_business_days(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Get all business days between start and end dates"""
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    business_days = all_dates[all_dates.weekday < 5]
    return business_days


def add_business_days(date: pd.Timestamp, days: int) -> pd.Timestamp:
    """Add business days to a date"""
    if days == 0:
        return date

    direction = 1 if days > 0 else -1
    days_to_add = abs(days)
    current_date = date

    while days_to_add > 0:
        current_date += timedelta(days=direction)
        if is_business_day(current_date):
            days_to_add -= 1

    return current_date


# ============================================
# MISSING HELPER FUNCTIONS FOR PAGES
# ============================================

def valuation_summary(df: pd.DataFrame, metrics: List[str] = None) -> Dict:
    """
    Get current valuation summary with latest values, percentiles and zones
    
    Args:
        df: DataFrame with valuation data and percentiles
        metrics: List of valuation metrics (e.g., [PE, PB, PCFS])
    
    Returns:
        Dict with 'date', 'valuations', 'percentiles', 'zones'
    """
    if df.empty:
        return {}
    
    if metrics is None:
        metrics = [PE, PB, PCFS]
    
    latest = df.iloc[-1]
    
    result = {
        'date': latest.get(DATE),
        'valuations': {},
        'percentiles': {},
        'zones': {}
    }
    
    for metric in metrics:
        # Get metric value
        val = latest.get(metric)
        result['valuations'][metric] = val if not pd.isna(val) else None
        
        # Get percentile
        pct_col = f'{metric}_percentile'
        pct = latest.get(pct_col)
        result['percentiles'][metric] = pct if not pd.isna(pct) else None
        
        # Determine zone based on percentile
        if pct is not None and not pd.isna(pct):
            if pct <= 20:
                zone = 'Rẻ'
            elif pct <= 40:
                zone = 'Hợp lý thấp'
            elif pct <= 60:
                zone = 'Trung bình'
            elif pct <= 80:
                zone = 'Hợp lý cao'
            else:
                zone = 'Đắt'
            result['zones'][metric] = zone
        else:
            result['zones'][metric] = 'N/A'
    
    return result


def lead_lag_analysis_full(df: pd.DataFrame) -> Dict:
    """
    Run full lead/lag analysis for all forward return horizons
    Wrapper around analyze_foreign_lead_lag that reformats results
    
    Args:
        df: DataFrame with foreign trading and return data
    
    Returns:
        Dict with keys like 'T+1', 'T+3', 'T+5', 'T+10', 'T+20', 'T+30'
    """
    from analysis.analysis import analyze_foreign_lead_lag, prepare_lead_lag_data
    import config
    
    # Get horizons from config
    horizons = config.FORWARD_RETURN_HORIZONS if hasattr(config, 'FORWARD_RETURN_HORIZONS') else [1, 3, 5, 10, 20, 30]
    
    # Prepare data with forward returns
    df = prepare_lead_lag_data(df, horizons)
    
    # Run analysis
    raw_results = analyze_foreign_lead_lag(df, horizons)
    
    # Reformat results from '5d' format to 'T+5' format
    formatted_results = {}
    for key, value in raw_results.items():
        # Extract number from '5d' -> 5
        horizon_num = int(key.replace('d', ''))
        # Reformat to 'T+5'
        new_key = f'T+{horizon_num}'
        formatted_results[new_key] = value
    
    return formatted_results


def analyze_percentile_returns(df: pd.DataFrame, percentile_col: str, forward_horizon: int) -> Dict:
    """
    Analyze returns by percentile deciles
    
    Args:
        df: DataFrame with percentile and forward return data
        percentile_col: Name of percentile column
        forward_horizon: Forward return horizon in days
    
    Returns:
        Dictionary with analysis results
    """
    fwd_ret_col = f'fwd_return_{forward_horizon}d'
    
    if percentile_col not in df.columns or fwd_ret_col not in df.columns:
        return {'error': f'Missing required columns: {percentile_col} or {fwd_ret_col}'}
    
    # Create deciles from percentiles
    df_clean = df[[percentile_col, fwd_ret_col]].dropna()
    
    if len(df_clean) < 10:
        return {'error': 'Insufficient data for decile analysis'}
    
    # Create 10 decile bins
    df_clean['decile'] = pd.qcut(df_clean[percentile_col], q=10, labels=False, duplicates='drop') + 1
    
    # Calculate decile statistics
    decile_stats = df_clean.groupby('decile')[fwd_ret_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('count', 'count')
    ]).reset_index()
    
    # Calculate zone returns
    cheap_zone = df_clean[df_clean[percentile_col] <= 20][fwd_ret_col]
    expensive_zone = df_clean[df_clean[percentile_col] >= 80][fwd_ret_col]
    
    cheap_zone_return = cheap_zone.mean() if len(cheap_zone) > 0 else 0
    expensive_zone_return = expensive_zone.mean() if len(expensive_zone) > 0 else 0
    
    # Monotonicity test (Spearman correlation)
    from scipy import stats as sp_stats
    correlation, p_value = sp_stats.spearmanr(df_clean[percentile_col], df_clean[fwd_ret_col])
    
    # ANOVA test
    groups = [group[fwd_ret_col].values for name, group in df_clean.groupby('decile')]
    f_stat, anova_p = sp_stats.f_oneway(*groups) if len(groups) > 1 else (np.nan, np.nan)
    
    return {
        'decile_stats': decile_stats,
        'cheap_zone_return': cheap_zone_return,
        'expensive_zone_return': expensive_zone_return,
        'cheap_expensive_spread': cheap_zone_return - expensive_zone_return,
        'monotonicity': {
            'correlation': correlation,
            P_VALUE: p_value,
            'is_monotonic': p_value < 0.05 and correlation < 0  # Negative correlation expected
        },
        'anova': {
            'f_stat': f_stat,
            P_VALUE: anova_p,
            'significant': anova_p < 0.05 if not np.isnan(anova_p) else False
        }
    }


def predict_forward_return(df: pd.DataFrame, input_percentile: float, 
                          percentile_col: str, forward_horizon: int) -> Dict:
    """
    Predict forward return based on input percentile
    
    Args:
        df: DataFrame with historical data
        input_percentile: Input percentile value (0-100)
        percentile_col: Name of percentile column
        forward_horizon: Forward return horizon
    
    Returns:
        Dictionary with prediction results
    """
    fwd_ret_col = f'fwd_return_{forward_horizon}d'
    
    if percentile_col not in df.columns or fwd_ret_col not in df.columns:
        return {'error': 'Missing required columns'}
    
    # Find decile
    decile = int(input_percentile // 10) + 1
    if decile > 10:
        decile = 10
    
    # Filter data in similar percentile range (+/- 10%)
    df_clean = df[[percentile_col, fwd_ret_col]].dropna()
    similar_data = df_clean[
        (df_clean[percentile_col] >= input_percentile - 10) &
        (df_clean[percentile_col] <= input_percentile + 10)
    ][fwd_ret_col]
    
    if len(similar_data) < 5:
        return {'error': 'Insufficient historical data for this percentile range'}
    
    # Calculate statistics
    expected_return = similar_data.mean()
    std_return = similar_data.std()
    
    # 95% confidence interval
    from scipy import stats as sp_stats
    confidence_interval = sp_stats.t.interval(
        0.95, len(similar_data)-1,
        loc=expected_return,
        scale=std_return/np.sqrt(len(similar_data))
    )
    
    return {
        'expected_return': expected_return,
        'confidence_low': confidence_interval[0],
        'confidence_high': confidence_interval[1],
        'sample_size': len(similar_data),
        'current_percentile': input_percentile,
        'decile': decile
    }


def compare_valuation_metrics(df: pd.DataFrame, forward_horizon: int) -> Dict:
    """
    Compare different valuation metrics
    
    Args:
        df: DataFrame with valuation data
        forward_horizon: Forward return horizon
    
    Returns:
        Dictionary with comparison results for each metric
    """
    metrics = [PE, PB, PCFS]
    results = {}
    
    for metric in metrics:
        percentile_col = f'{metric}_percentile'
        
        if percentile_col in df.columns:
            analysis = analyze_percentile_returns(df, percentile_col, forward_horizon)
            
            if 'error' not in analysis:
                results[metric] = {
                    'monotonicity_correlation': analysis['monotonicity']['correlation'],
                    'monotonicity_pvalue': analysis['monotonicity'][P_VALUE],
                    'cheap_expensive_spread': analysis['cheap_expensive_spread'],
                    'anova_pvalue': analysis['anova'][P_VALUE]
                }
    
    return results
