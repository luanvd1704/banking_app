"""
Banking Flow Analysis - All Analysis Logic
Combines Q1 (Foreign Lead/Lag), Q4 (Valuation), Q5 (Composite), and Ranking
Organized into clear sections for each research question
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from helpers import *
from analysis.statistics import ttest_two_groups, test_quintile_spread, calculate_information_coefficient

# ============================================
# SECTION 1: SIGNAL NORMALIZATION
# ============================================

def normalize_by_adv(net_buy_value: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Normalize net buy value by Average Daily Volume (ADV)"""
    volume = net_buy_value / close
    avg_volume = volume.abs().rolling(window=window, min_periods=window//2).mean()
    normalized = volume / avg_volume
    return normalized


def calculate_foreign_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate normalized foreign trading signals"""
    df = df.copy()

    if FOREIGN_NET_BUY_VAL in df.columns and CLOSE in df.columns:
        df[FOREIGN_SIGNAL_ADV20] = normalize_by_adv(
            df[FOREIGN_NET_BUY_VAL],
            df[CLOSE],
            window=window
        )

        # Calculate z-score for composite scoring
        df[FOREIGN_ZSCORE] = calculate_zscore(
            df[FOREIGN_NET_BUY_VAL],
            window=config.ZSCORE_WINDOW
        )

    return df


def calculate_valuation_percentiles(df: pd.DataFrame, window: int = 756) -> pd.DataFrame:
    """Calculate rolling percentiles for valuation metrics"""
    df = df.copy()

    for metric in [PE, PB, PCFS]:
        if metric in df.columns:
            percentile_col = f'{metric}_percentile'
            df[percentile_col] = calculate_percentile(df[metric], window=window)

    return df


def normalize_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all normalized signals"""
    df = calculate_foreign_signals(df)
    df = calculate_valuation_percentiles(df)
    return df


# ============================================
# SECTION 2: Q1 - FOREIGN LEAD/LAG ANALYSIS
# ============================================

def prepare_lead_lag_data(df: pd.DataFrame, horizons: List[int] = None) -> pd.DataFrame:
    """Prepare data for lead-lag analysis"""
    if horizons is None:
        horizons = config.FORWARD_RETURN_HORIZONS

    df = df.copy()

    # Filter out invalid prices
    if CLOSE in df.columns:
        invalid_price_mask = (df[CLOSE].isna()) | (df[CLOSE] <= 0)
        num_invalid = invalid_price_mask.sum()
        if num_invalid > 0:
            print(f"Warning: Filtering out {num_invalid} rows with invalid close prices")
            df = df[~invalid_price_mask].copy()

    # Ensure excess returns
    if EXCESS_RETURN not in df.columns:
        if STOCK_RETURN in df.columns and MARKET_RETURN in df.columns:
            df[EXCESS_RETURN] = calculate_excess_return(df[STOCK_RETURN], df[MARKET_RETURN])

    # Calculate forward returns
    if CLOSE in df.columns:
        fwd_returns = calculate_forward_returns(df[CLOSE], horizons)

        # Remove duplicate columns before concat to avoid shape issues
        for col in fwd_returns.columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = pd.concat([df, fwd_returns], axis=1)

        # Calculate forward excess returns
        for h in horizons:
            fwd_ret_col = f'fwd_return_{h}d'
            fwd_excess_col = f'fwd_excess_return_{h}d'

            if fwd_ret_col in df.columns:
                if VNINDEX_CLOSE in df.columns:
                    fwd_market_ret = df[VNINDEX_CLOSE].pct_change(h).shift(-h)
                    # Use iloc to ensure we get a 1D Series
                    df[fwd_excess_col] = df[fwd_ret_col].iloc[:, 0] if isinstance(df[fwd_ret_col], pd.DataFrame) else df[fwd_ret_col] - fwd_market_ret
                else:
                    df[fwd_excess_col] = df[fwd_ret_col].iloc[:, 0] if isinstance(df[fwd_ret_col], pd.DataFrame) else df[fwd_ret_col]

                # Filter out inf values
                inf_mask = np.isinf(df[fwd_excess_col])
                if inf_mask.sum() > 0:
                    df.loc[inf_mask, fwd_excess_col] = np.nan

    return df


def create_quintiles_by_foreign_trading(df: pd.DataFrame, signal_col: str = FOREIGN_NET_BUY_VAL) -> pd.DataFrame:
    """Create quintiles based on foreign net buying (filters out zero values)"""
    df = df.copy()

    if signal_col in df.columns:
        # Filter out zero values
        df_nonzero = df[df[signal_col] != 0].copy()

        if len(df_nonzero) > 0:
            df_nonzero[QUINTILE] = create_quintiles(df_nonzero[signal_col], labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            df = df_nonzero

    return df


def analyze_foreign_lead_lag(df: pd.DataFrame, horizons: List[int] = None, signal_col: str = FOREIGN_NET_BUY_VAL) -> Dict:
    """Analyze if foreign investors predict returns"""
    if horizons is None:
        horizons = [5, 10]

    df = prepare_lead_lag_data(df, horizons)
    df = create_quintiles_by_foreign_trading(df, signal_col)

    results = {}

    for horizon in horizons:
        fwd_ret_col = f'fwd_return_{horizon}d'

        if fwd_ret_col not in df.columns or QUINTILE not in df.columns:
            continue

        # Calculate quintile statistics
        quintile_stats = df.groupby(QUINTILE)[fwd_ret_col].agg(['mean', 'std', 'count']).reset_index()

        # Test Q5 vs Q1
        q5_returns = df[df[QUINTILE] == 'Q5'][fwd_ret_col].dropna()
        q1_returns = df[df[QUINTILE] == 'Q1'][fwd_ret_col].dropna()

        ttest_result = ttest_two_groups(q5_returns, q1_returns)

        results[f'{horizon}d'] = {
            'quintile_stats': quintile_stats,
            'ttest': ttest_result,
            'q5_mean': q5_returns.mean() if len(q5_returns) > 0 else np.nan,
            'q1_mean': q1_returns.mean() if len(q1_returns) > 0 else np.nan,
            'spread': q5_returns.mean() - q1_returns.mean() if len(q5_returns) > 0 and len(q1_returns) > 0 else np.nan,
            # Add flattened ttest results for easier access
            'p_value': ttest_result.get(P_VALUE, np.nan),
            'significant': ttest_result.get('significant', False),
            't_stat': ttest_result.get(T_STAT, np.nan)
        }

    return results


# ============================================
# SECTION 3: Q4 - VALUATION ANALYSIS
# ============================================

def analyze_valuation_percentiles(df: pd.DataFrame, metric: str = PE, horizon: int = 10) -> Dict:
    """Analyze PE/PB percentiles vs returns"""
    df = df.copy()

    # Ensure percentiles are calculated
    df = calculate_valuation_percentiles(df)

    percentile_col = f'{metric}_percentile'
    fwd_ret_col = f'fwd_return_{horizon}d'

    if percentile_col not in df.columns:
        return {'error': f'Percentile column {percentile_col} not found'}

    # Calculate forward returns if needed
    if fwd_ret_col not in df.columns:
        df = prepare_lead_lag_data(df, [horizon])

    if fwd_ret_col not in df.columns:
        return {'error': f'Cannot calculate forward returns'}

    # Create deciles based on percentile
    df = create_deciles_by_valuation(df, metric)

    if DECILE not in df.columns:
        return {'error': 'Could not create deciles'}

    # Calculate decile statistics
    decile_stats = df.groupby(DECILE)[fwd_ret_col].agg(['mean', 'std', 'count']).reset_index()

    # Test D1 (cheap) vs D10 (expensive)
    d1_returns = df[df[DECILE] == 'D1'][fwd_ret_col].dropna()
    d10_returns = df[df[DECILE] == 'D10'][fwd_ret_col].dropna()

    ttest_result = ttest_two_groups(d1_returns, d10_returns)

    return {
        'decile_stats': decile_stats,
        'ttest': ttest_result,
        'd1_mean': d1_returns.mean() if len(d1_returns) > 0 else np.nan,
        'd10_mean': d10_returns.mean() if len(d10_returns) > 0 else np.nan,
        'spread': d1_returns.mean() - d10_returns.mean() if len(d1_returns) > 0 and len(d10_returns) > 0 else np.nan
    }


def create_deciles_by_valuation(df: pd.DataFrame, metric: str = PE) -> pd.DataFrame:
    """Create deciles based on valuation percentile"""
    df = df.copy()

    percentile_col = f'{metric}_percentile'

    if percentile_col in df.columns:
        df[DECILE] = create_deciles(df[percentile_col], labels=DECILE_LABELS)

    return df


# ============================================
# SECTION 4: Q5 - COMPOSITE SCORE
# ============================================

def build_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite score

    Formula: Composite = z(Foreign) - percentile(PE/PB)
    """
    df = df.copy()

    # Ensure all signals are calculated
    df = normalize_all_signals(df)

    # Start with foreign z-score
    if FOREIGN_ZSCORE in df.columns:
        composite = df[FOREIGN_ZSCORE].fillna(0)
    else:
        composite = pd.Series(0, index=df.index)

    # Subtract valuation percentile (lower percentile = cheaper = better)
    val_score = 0
    val_count = 0

    if PE_PERCENTILE in df.columns:
        val_score += df[PE_PERCENTILE].fillna(50) / 100  # 0-1 scale
        val_count += 1

    if PB_PERCENTILE in df.columns:
        val_score += df[PB_PERCENTILE].fillna(50) / 100
        val_count += 1

    if val_count > 0:
        composite -= (val_score / val_count)

    df[COMPOSITE_SCORE] = composite

    return df


def quintile_backtest(df: pd.DataFrame, horizon: int = 5) -> Dict:
    """Backtest quintile strategy"""
    df = df.copy()

    if COMPOSITE_SCORE not in df.columns:
        return {'error': 'No composite score'}

    # Create quintiles
    df[QUINTILE] = create_quintiles(df[COMPOSITE_SCORE])

    # Calculate forward returns
    if CLOSE in df.columns:
        df[f'fwd_return_{horizon}d'] = df[CLOSE].pct_change(horizon).shift(-horizon)

    fwd_ret_col = f'fwd_return_{horizon}d'

    if fwd_ret_col not in df.columns:
        return {'error': 'Cannot calculate returns'}

    # Calculate returns by quintile
    quintile_returns = df.groupby(QUINTILE)[fwd_ret_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()

    # Strategy returns (Long Q5, Short Q1)
    q5_returns = df[df[QUINTILE] == 'Q5'][fwd_ret_col].dropna()
    q1_returns = df[df[QUINTILE] == 'Q1'][fwd_ret_col].dropna()

    strategy_returns_mean = q5_returns.mean() - q1_returns.mean()
    q5_std = q5_returns.std()
    q1_std = q1_returns.std()
    strategy_returns_std = ((q5_std**2 + q1_std**2) ** 0.5) / (2 ** 0.5)

    # Sharpe ratio
    if strategy_returns_std > 0:
        periods_per_year = 252 // horizon
        sharpe = (strategy_returns_mean * periods_per_year) / (strategy_returns_std * (periods_per_year ** 0.5))
    else:
        sharpe = np.nan

    results = {
        'quintile_returns': quintile_returns,
        'strategy_returns_mean': strategy_returns_mean,
        'strategy_returns_std': strategy_returns_std,
        'sharpe_ratio': sharpe,
        'sample_size': len(df.dropna(subset=[COMPOSITE_SCORE, fwd_ret_col]))
    }

    return results


def capm_analysis(stock_returns: pd.Series, market_returns: pd.Series) -> Dict:
    """Simple CAPM analysis"""
    df = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()

    if len(df) < 10:
        return {'error': 'Insufficient data'}

    # Calculate beta
    covariance = df['stock'].cov(df['market'])
    market_var = df['market'].var()

    beta = covariance / market_var if market_var != 0 else 0

    # Calculate alpha
    mean_stock = df['stock'].mean()
    mean_market = df['market'].mean()

    alpha = mean_stock - beta * mean_market

    # Annualize
    periods_per_year = 252
    alpha_annual = alpha * periods_per_year
    mean_stock_annual = mean_stock * periods_per_year

    return {
        ALPHA: alpha_annual,
        BETA: beta,
        'mean_return': mean_stock_annual,
        'sample_size': len(df)
    }


# ============================================
# SECTION 5: RANKING (Cross-sectional)
# ============================================

def load_financial_metrics(financial_file: str, tickers: List[str]) -> pd.DataFrame:
    """Load financial metrics from Excel file"""
    all_data = []

    for ticker in tickers:
        try:
            df = pd.read_excel(financial_file, sheet_name=ticker)
            if 'ticker' not in df.columns:
                df['ticker'] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Create date column from year + quarter
    if 'year' in combined.columns and 'quarter' in combined.columns:
        combined[DATE] = pd.to_datetime(
            combined['year'].astype(str) + '-' +
            (combined['quarter'] * 3).astype(str) + '-01'
        )

    return combined


def calculate_cross_sectional_ranks(df: pd.DataFrame, metric: str, direction: str = 'higher_is_better',
                                    optimal_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    """Calculate cross-sectional ranks for a metric"""
    result = df.copy()

    # Group by date and calculate ranks
    for date, group in result.groupby(DATE):
        valid_data = group[group[metric].notna()].copy()

        if len(valid_data) == 0:
            continue

        if direction == 'higher_is_better':
            ranks = valid_data[metric].rank(ascending=False, method='min')
        elif direction == 'lower_is_better':
            ranks = valid_data[metric].rank(ascending=True, method='min')
        elif direction == 'optimal_range' and optimal_range:
            min_opt, max_opt = optimal_range
            distances = valid_data[metric].apply(
                lambda x: 0 if min_opt <= x <= max_opt else min(abs(x - min_opt), abs(x - max_opt))
            )
            ranks = distances.rank(ascending=True, method='min')
        else:
            ranks = pd.Series(np.nan, index=valid_data.index)

        result.loc[valid_data.index, f'{metric}_rank'] = ranks

    return result
