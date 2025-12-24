"""
Banking Flow Analysis - All Visualization Functions
Charts and plotting utilities for Streamlit pages
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import config

# ============================================
# SECTION 1: COMMON CHART UTILITIES
# ============================================

def create_bar_chart(data: pd.DataFrame,
                     x_col: str,
                     y_col: str,
                     title: str,
                     x_label: str = None,
                     y_label: str = None,
                     color_col: str = None,
                     colors: List[str] = None) -> go.Figure:
    """
    Create a bar chart

    Args:
        data: DataFrame
        x_col: X-axis column
        y_col: Y-axis column
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        color_col: Column for coloring
        colors: Custom colors

    Returns:
        Plotly figure
    """
    if colors is None and color_col is None:
        colors = [config.COLOR_PALETTE['primary']] * len(data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data[x_col],
        y=data[y_col],
        marker_color=colors if colors else None,
        marker=dict(color=data[color_col]) if color_col else None
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        height=config.CHART_HEIGHT,
        template='plotly_white'
    )

    return fig


def create_line_chart(data: pd.DataFrame,
                      x_col: str,
                      y_cols: List[str],
                      title: str,
                      x_label: str = None,
                      y_label: str = None,
                      colors: List[str] = None) -> go.Figure:
    """
    Create a line chart

    Args:
        data: DataFrame
        x_col: X-axis column
        y_cols: List of Y-axis columns
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        colors: Custom colors for each line

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if colors is None:
        colors = list(config.COLOR_PALETTE.values())[:len(y_cols)]

    for i, y_col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            name=y_col,
            line=dict(color=colors[i % len(colors)], width=2),
            mode='lines'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label or x_col,
        yaxis_title=y_label,
        height=config.CHART_HEIGHT,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_heatmap(data: pd.DataFrame,
                   title: str,
                   x_label: str = None,
                   y_label: str = None,
                   colorscale: str = 'RdYlGn') -> go.Figure:
    """
    Create a heatmap

    Args:
        data: DataFrame (will be used as matrix)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        colorscale: Color scale

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=colorscale,
        text=data.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 10}
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=config.CHART_HEIGHT
    )

    return fig


# ============================================
# SECTION 2: Q4 VALUATION CHARTS
# ============================================

def create_percentile_timeseries(df: pd.DataFrame,
                                 metric: str,
                                 ticker: str = "") -> go.Figure:
    """
    Create dual-axis chart: metric value and percentile over time

    Args:
        df: DataFrame with valuation data
        metric: Valuation metric (pe, pb, pcfs)
        ticker: Ticker name

    Returns:
        Plotly figure
    """
    percentile_col = f'{metric}_percentile'
    date_col = 'date'

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add metric value
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[metric],
            name=metric.upper(),
            line=dict(color=config.COLOR_PALETTE['primary'], width=2)
        ),
        secondary_y=False
    )

    # Add percentile with colored zones
    if percentile_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[percentile_col],
                name=f'{metric.upper()} Percentile',
                line=dict(color=config.COLOR_PALETTE['secondary'], width=2)
            ),
            secondary_y=True
        )

        # Add zone shading
        fig.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.1,
                     layer="below", line_width=0, secondary_y=True,
                     annotation_text="Cheap", annotation_position="right")
        fig.add_hrect(y0=40, y1=60, fillcolor="gray", opacity=0.1,
                     layer="below", line_width=0, secondary_y=True,
                     annotation_text="Fair", annotation_position="right")
        fig.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.1,
                     layer="below", line_width=0, secondary_y=True,
                     annotation_text="Expensive", annotation_position="right")

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=f"{metric.upper()} Value", secondary_y=False)
    fig.update_yaxes(title_text="Percentile (%)", secondary_y=True)

    fig.update_layout(
        title=f"{ticker} - {metric.upper()} Value and Historical Percentile",
        height=config.CHART_HEIGHT,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_decile_returns_chart(decile_stats: pd.DataFrame,
                                metric: str,
                                ticker: str = "") -> go.Figure:
    """
    Create bar chart of returns by valuation decile

    Args:
        decile_stats: DataFrame with decile statistics
        metric: Valuation metric
        ticker: Ticker name

    Returns:
        Plotly figure
    """
    decile_col = 'decile'

    fig = go.Figure()

    # Color bars by return (green for positive, red for negative)
    colors = ['green' if x > 0 else 'red' for x in decile_stats['mean']]

    fig.add_trace(go.Bar(
        x=decile_stats[decile_col],
        y=decile_stats['mean'],
        marker_color=colors,
        text=decile_stats['mean'].apply(lambda x: f"{x:.2%}"),
        textposition='outside',
        name='Mean Return'
    ))

    fig.update_layout(
        title=f"{ticker} - Forward Returns by {metric.upper()} Percentile Decile",
        xaxis_title=f"{metric.upper()} Percentile Decile (D1=0-10%, D10=90-100%)",
        yaxis_title="Mean Forward Return",
        yaxis_tickformat='.2%',
        height=config.CHART_HEIGHT,
        template='plotly_white'
    )

    return fig


# ============================================
# SECTION 3: Q1 FOREIGN FLOW CHARTS
# ============================================

def create_quintile_returns_chart(quintile_stats: pd.DataFrame,
                                  title: str = "Quintile Returns") -> go.Figure:
    """
    Create bar chart of returns by quintile

    Args:
        quintile_stats: DataFrame with quintile statistics
        title: Chart title

    Returns:
        Plotly figure
    """
    quintile_col = 'quintile'

    fig = go.Figure()

    # Use quintile colors
    colors = config.QUINTILE_COLORS

    fig.add_trace(go.Bar(
        x=quintile_stats[quintile_col],
        y=quintile_stats['mean'],
        marker_color=colors,
        text=quintile_stats['mean'].apply(lambda x: f"{x:.2%}"),
        textposition='outside',
        name='Mean Return',
        error_y=dict(
            type='data',
            array=quintile_stats['std'] if 'std' in quintile_stats.columns else None
        )
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Foreign Net Buy Quintile (Q1=Lowest, Q5=Highest)",
        yaxis_title="Mean Forward Return",
        yaxis_tickformat='.2%',
        height=config.CHART_HEIGHT,
        template='plotly_white'
    )

    return fig


def create_foreign_flow_timeseries(df: pd.DataFrame,
                                   ticker: str = "") -> go.Figure:
    """
    Create foreign flow timeseries chart

    Args:
        df: DataFrame with foreign trading data
        ticker: Ticker name

    Returns:
        Plotly figure
    """
    date_col = 'date'
    foreign_col = 'foreign_net_buy_val'
    close_col = 'close'

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add price
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[close_col],
            name="Price",
            line=dict(color=config.COLOR_PALETTE['primary'], width=1)
        ),
        secondary_y=False
    )

    # Add foreign flow as bars
    if foreign_col in df.columns:
        colors = ['green' if x > 0 else 'red' for x in df[foreign_col]]

        fig.add_trace(
            go.Bar(
                x=df[date_col],
                y=df[foreign_col],
                name="Foreign Net Buy",
                marker_color=colors,
                opacity=0.6
            ),
            secondary_y=True
        )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Foreign Net Buy Value", secondary_y=True)

    fig.update_layout(
        title=f"{ticker} - Price and Foreign Trading Flow",
        height=config.CHART_HEIGHT,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


# ============================================
# SECTION 4: Q5 COMPOSITE CHARTS
# ============================================

def create_composite_backtest_chart(quintile_returns: pd.DataFrame) -> go.Figure:
    """
    Create composite quintile backtest chart

    Args:
        quintile_returns: DataFrame with quintile returns

    Returns:
        Plotly figure
    """
    quintile_col = 'quintile'

    fig = go.Figure()

    # Use quintile colors
    colors = config.QUINTILE_COLORS

    fig.add_trace(go.Bar(
        x=quintile_returns[quintile_col],
        y=quintile_returns['mean'],
        marker_color=colors,
        text=quintile_returns['mean'].apply(lambda x: f"{x:.2%}"),
        textposition='outside',
        name='Mean Return',
        error_y=dict(
            type='data',
            array=quintile_returns['std'] if 'std' in quintile_returns.columns else None
        )
    ))

    fig.update_layout(
        title="Composite Score Quintile Backtest",
        xaxis_title="Composite Score Quintile (Q1=Lowest, Q5=Highest)",
        yaxis_title="Mean Forward Return",
        yaxis_tickformat='.2%',
        height=config.CHART_HEIGHT,
        template='plotly_white'
    )

    return fig


def create_valuation_gauge(percentile: float, metric: str) -> go.Figure:
    """
    Create a gauge chart showing percentile position
    
    Args:
        percentile: Percentile value (0-100)
        metric: Valuation metric name (PE, PB, PCFS)
    
    Returns:
        Plotly gauge figure
    """
    # Determine color based on percentile
    if percentile <= 20:
        color = '#2ca02c'  # Green - Cheap
        zone = 'Rẻ'
    elif percentile <= 40:
        color = '#1f77b4'  # Blue - Fair Low
        zone = 'Hợp lý thấp'
    elif percentile <= 60:
        color = '#7f7f7f'  # Gray - Neutral
        zone = 'Trung bình'
    elif percentile <= 80:
        color = '#ff7f0e'  # Orange - Fair High
        zone = 'Hợp lý cao'
    else:
        color = '#d62728'  # Red - Expensive
        zone = 'Đắt'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentile,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{metric.upper()} Percentile", 'font': {'size': 20}},
        number={'font': {'size': 50}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#d4edda'},  # Light green
                {'range': [20, 40], 'color': '#d1ecf1'},  # Light blue
                {'range': [40, 60], 'color': '#e2e3e5'},  # Light gray
                {'range': [60, 80], 'color': '#fff3cd'},  # Light yellow
                {'range': [80, 100], 'color': '#f8d7da'}  # Light red
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': percentile
            }
        }
    ))

    fig.update_layout(
        height=350,
        font={'color': "darkblue", 'family': "Arial"},
        annotations=[
            dict(
                text=f"<b>Vùng: {zone}</b>",
                x=0.5,
                y=-0.15,
                showarrow=False,
                font=dict(size=18, color=color),
                xanchor='center'
            )
        ]
    )
    
    return fig


def create_zone_comparison_chart(cheap_return: float, expensive_return: float, metric: str) -> go.Figure:
    """
    Create bar chart comparing cheap vs expensive zone returns
    
    Args:
        cheap_return: Return for cheap zone (0-20% percentile)
        expensive_return: Return for expensive zone (80-100% percentile)
        metric: Valuation metric name
    
    Returns:
        Plotly bar chart figure
    """
    fig = go.Figure(data=[
        go.Bar(
            x=['Rẻ (0-20%)', 'Đắt (80-100%)'],
            y=[cheap_return, expensive_return],
            marker_color=['#2ca02c', '#d62728'],
            text=[f'{cheap_return:.2%}', f'{expensive_return:.2%}'],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"So Sánh Lợi Nhuận: Vùng Rẻ vs Đắt ({metric.upper()})",
        xaxis_title="Vùng Phân Vị",
        yaxis_title="Lợi Nhuận Trung Bình",
        yaxis_tickformat='.2%',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig
