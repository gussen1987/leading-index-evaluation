"""Chart generation module using Plotly."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from risk_index.core.types import Regime
from risk_index.core.constants import (
    COL_COMPOSITE_FAST,
    COL_COMPOSITE_MEDIUM,
    COL_COMPOSITE_SLOW,
    COL_REGIME_FAST,
    COL_REGIME_MEDIUM,
    COL_REGIME_SLOW,
    COL_CHECKLIST_SCORE,
    ARTIFACTS_DIR,
)
from risk_index.core.utils_io import ensure_dir

# Color scheme
COLORS = {
    "risk_on": "#2ecc71",  # Green
    "neutral": "#f39c12",  # Orange
    "risk_off": "#e74c3c",  # Red
    "fast": "#3498db",  # Blue
    "medium": "#9b59b6",  # Purple
    "slow": "#1abc9c",  # Teal
}


def create_composite_timeseries(
    composites_df: pd.DataFrame,
    regimes_df: pd.DataFrame | None = None,
    title: str = "Composite Signals Over Time",
) -> go.Figure:
    """Create time series chart of composite signals with regime shading.


    Args:
        composites_df: DataFrame with composite scores
        regimes_df: Optional DataFrame with regime columns for shading
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Add composite lines
    for col, color, name in [
        (COL_COMPOSITE_FAST, COLORS["fast"], "Fast (4-8w)"),
        (COL_COMPOSITE_MEDIUM, COLORS["medium"], "Medium (13-26w)"),
        (COL_COMPOSITE_SLOW, COLORS["slow"], "Slow (26-52w)"),
    ]:
        if col in composites_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=composites_df.index,
                    y=composites_df[col],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                )
            )

    # Add threshold lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)

    # Add regime shading if provided
    if regimes_df is not None and COL_REGIME_MEDIUM in regimes_df.columns:
        add_regime_shading(fig, regimes_df[COL_REGIME_MEDIUM])

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Composite Z-Score",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
    )

    return fig


def add_regime_shading(fig: go.Figure, regime_series: pd.Series) -> None:
    """Add regime shading to a figure.

    Args:
        fig: Plotly figure
        regime_series: Series of regime labels
    """
    if regime_series.empty:
        return

    # Find regime spans
    spans = []
    current_regime = regime_series.iloc[0]
    start_date = regime_series.index[0]

    for i, (date, regime) in enumerate(regime_series.items()):
        if regime != current_regime or i == len(regime_series) - 1:
            spans.append((start_date, date, current_regime))
            current_regime = regime
            start_date = date

    # Add shapes for each span
    for start, end, regime in spans:
        if regime == Regime.RISK_ON.value:
            color = "rgba(46, 204, 113, 0.1)"
        elif regime == Regime.RISK_OFF.value:
            color = "rgba(231, 76, 60, 0.1)"
        else:
            continue  # Don't shade neutral

        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            line_width=0,
            layer="below",
        )


def create_block_bars(
    blocks_df: pd.DataFrame,
    date: pd.Timestamp | None = None,
    title: str = "Block Scores",
) -> go.Figure:
    """Create bar chart of current block scores.

    Args:
        blocks_df: DataFrame with block scores
        date: Date for snapshot (defaults to latest)
        title: Chart title

    Returns:
        Plotly figure
    """
    if date is None:
        date = blocks_df.index[-1]

    scores = blocks_df.loc[date].dropna().sort_values()

    colors = [COLORS["risk_on"] if v > 0 else COLORS["risk_off"] for v in scores.values]

    fig = go.Figure(
        go.Bar(
            x=scores.values,
            y=scores.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" for v in scores.values],
            textposition="outside",
        )
    )

    fig.add_vline(x=0, line_dash="solid", line_color="gray")

    fig.update_layout(
        title=f"{title} ({date.strftime('%Y-%m-%d')})",
        xaxis_title="Z-Score",
        yaxis_title="Block",
        height=max(400, len(scores) * 35),
        margin=dict(l=150),
    )

    return fig


def create_block_heatmap(
    blocks_df: pd.DataFrame,
    weeks: int = 52,
    title: str = "Block Scores Heatmap",
) -> go.Figure:
    """Create heatmap of block scores over time.

    Args:
        blocks_df: DataFrame with block scores
        weeks: Number of weeks to show
        title: Chart title

    Returns:
        Plotly figure
    """
    # Get last N weeks
    recent = blocks_df.tail(weeks)

    fig = go.Figure(
        go.Heatmap(
            z=recent.T.values,
            x=recent.index,
            y=recent.columns,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Z-Score"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Block",
        height=max(400, len(recent.columns) * 30),
    )

    return fig


def create_checklist_score_chart(
    checklist_df: pd.DataFrame,
    title: str = "Checklist Score Over Time",
) -> go.Figure:
    """Create checklist score time series.

    Args:
        checklist_df: DataFrame with checklist score
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Handle missing column
    if COL_CHECKLIST_SCORE not in checklist_df.columns:
        fig.add_annotation(
            text="No checklist score data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, height=400)
        return fig

    score = checklist_df[COL_CHECKLIST_SCORE].dropna()

    # Handle empty data
    if score.empty:
        fig.add_annotation(
            text="No data available for selected date range",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, height=400)
        return fig

    fig.add_trace(
        go.Scatter(
            x=score.index,
            y=score.values,  # Use .values explicitly
            mode="lines",
            name="Checklist Score",
            line=dict(color=COLORS["medium"], width=2),
            fill="tozeroy",
            fillcolor="rgba(155, 89, 182, 0.2)",
        )
    )

    # Add threshold lines
    fig.add_hline(y=75, line_dash="dash", line_color="green", annotation_text="Confirmed Risk-On")
    fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="On Watch")
    fig.add_hline(y=25, line_dash="dash", line_color="red", annotation_text="Risk-Off")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Score (0-100)",
        yaxis=dict(range=[0, 100]),
        height=400,
        hovermode="x unified",
    )

    return fig


def create_regime_distribution(
    regimes_df: pd.DataFrame,
    regime_col: str = COL_REGIME_MEDIUM,
    title: str = "Regime Distribution",
) -> go.Figure:
    """Create pie chart of regime time distribution.

    Args:
        regimes_df: DataFrame with regime columns
        regime_col: Which regime column to use
        title: Chart title

    Returns:
        Plotly figure
    """
    if regime_col not in regimes_df.columns:
        return go.Figure()

    counts = regimes_df[regime_col].value_counts()

    colors_map = {
        Regime.RISK_ON.value: COLORS["risk_on"],
        Regime.NEUTRAL.value: COLORS["neutral"],
        Regime.RISK_OFF.value: COLORS["risk_off"],
    }

    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            marker_colors=[colors_map.get(l, "gray") for l in counts.index],
            hole=0.4,
            textinfo="percent+label",
        )
    )

    fig.update_layout(title=title, height=400)

    return fig


def create_attribution_chart(
    attribution: dict,
    speed: str = "medium",
    title: str = "Composite Attribution",
) -> go.Figure:
    """Create waterfall chart showing block contributions.

    Args:
        attribution: Attribution dict from get_composite_attribution
        speed: Which composite speed
        title: Chart title

    Returns:
        Plotly figure
    """
    if speed not in attribution:
        return go.Figure()

    attr = attribution[speed]
    blocks = attr.get("blocks", {})

    if not blocks:
        return go.Figure()

    # Sort by contribution
    sorted_blocks = sorted(
        blocks.items(), key=lambda x: abs(x[1].get("contribution", 0)), reverse=True
    )

    names = []
    values = []
    colors = []

    for name, data in sorted_blocks:
        contrib = data.get("contribution", 0)
        names.append(name)
        values.append(contrib)
        colors.append(COLORS["risk_on"] if contrib > 0 else COLORS["risk_off"])

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    )

    fig.add_hline(y=0, line_dash="solid", line_color="gray")

    fig.update_layout(
        title=f"{title} ({speed.capitalize()})",
        xaxis_title="Block",
        yaxis_title="Contribution",
        height=400,
    )

    return fig


def save_chart_as_png(
    fig: go.Figure,
    filename: str,
    charts_dir: Path | None = None,
    width: int = 1200,
    height: int = 600,
    scale: int = 2,
) -> Path:
    """Save chart as PNG file.

    Args:
        fig: Plotly figure
        filename: Output filename (without extension)
        charts_dir: Output directory
        width: Image width
        height: Image height
        scale: Resolution scale (2 = 300 DPI approximately)

    Returns:
        Path to saved file
    """
    if charts_dir is None:
        charts_dir = ARTIFACTS_DIR / "charts"

    ensure_dir(charts_dir)
    path = charts_dir / f"{filename}.png"

    try:
        fig.write_image(str(path), width=width, height=height, scale=scale)
    except (ValueError, ImportError) as e:
        # Kaleido not installed or version incompatible
        print(f"Warning: Could not save {filename}.png - {e}")
        return None

    return path


def create_composition_pie(
    weights_df: pd.DataFrame,
    speed: str,
    title: str | None = None,
) -> go.Figure:
    """Create pie chart of composite block weights.

    Args:
        weights_df: DataFrame with 'block' and 'weight' columns
        speed: Composite speed (fast, medium, slow)
        title: Chart title (auto-generated if None)

    Returns:
        Plotly figure
    """
    if weights_df.empty:
        return go.Figure()

    if title is None:
        title = f"{speed.title()} Composite Weights"

    # Generate colors based on position
    colors = px.colors.qualitative.Set2[:len(weights_df)]

    fig = go.Figure(
        go.Pie(
            labels=weights_df["block"],
            values=weights_df["weight"],
            marker_colors=colors,
            hole=0.4,
            textinfo="percent+label",
            textposition="outside",
            pull=[0.02] * len(weights_df),  # Slight separation
        )
    )

    fig.update_layout(
        title=title,
        height=400,
        showlegend=False,
    )

    return fig


def create_equity_curves(
    backtest_results: dict[str, dict],
    title: str = "SPY Strategy Equity Curves",
) -> go.Figure:
    """Create equity curve chart comparing regime strategies.

    Args:
        backtest_results: Dict mapping speed name to backtest result dict
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Color mapping for each speed
    speed_colors = {
        "Fast": COLORS["fast"],
        "Medium": COLORS["medium"],
        "Slow": COLORS["slow"],
    }

    # Add strategy equity curves
    for speed, result in backtest_results.items():
        if result is None:
            continue

        equity = result.get("equity_curve")
        if equity is None:
            continue

        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity,
                mode="lines",
                name=f"{speed} Strategy",
                line=dict(color=speed_colors.get(speed, "gray"), width=2),
            )
        )

    # Add buy-and-hold line (from first valid result)
    first_result = next((r for r in backtest_results.values() if r is not None), None)
    if first_result and "buy_hold_equity" in first_result:
        bh_equity = first_result["buy_hold_equity"]
        fig.add_trace(
            go.Scatter(
                x=bh_equity.index,
                y=bh_equity,
                mode="lines",
                name="Buy & Hold",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value (Base 100)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
    )

    return fig


def create_regime_equity_with_shading(
    backtest_result: dict,
    regimes: pd.Series,
    title: str = "Strategy Equity with Regime Shading",
) -> go.Figure:
    """Create equity curve with regime period shading.

    Args:
        backtest_result: Single backtest result dict
        regimes: Regime series corresponding to the backtest
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    equity = backtest_result.get("equity_curve")
    bh_equity = backtest_result.get("buy_hold_equity")

    if equity is not None:
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity,
                mode="lines",
                name="Strategy",
                line=dict(color=COLORS["medium"], width=2),
            )
        )

    if bh_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=bh_equity.index,
                y=bh_equity,
                mode="lines",
                name="Buy & Hold",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )

    # Add regime shading
    if not regimes.empty:
        add_regime_shading(fig, regimes)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value (Base 100)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450,
    )

    return fig


def create_spy_regime_panel_chart(
    spy_prices: pd.Series,
    composite_scores: pd.DataFrame,
    regimes: pd.Series,
    title: str = "SPY with Risk Regime Indicator",
) -> go.Figure:
    """Create multi-panel chart like Daily Number Risk-On/Off indicator.

    Args:
        spy_prices: SPY price series
        composite_scores: DataFrame with composite scores
        regimes: Regime series for shading
        title: Chart title

    Returns:
        Plotly figure with 2 panels
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=["S&P 500 (SPY)", "Risk Regime Indicator"],
    )

    # Panel 1: SPY price
    if not spy_prices.empty:
        fig.add_trace(
            go.Scatter(
                x=spy_prices.index,
                y=spy_prices.values,
                name="SPY",
                line=dict(color="black", width=1.5),
                hovertemplate="SPY: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Panel 2: Composite score oscillator
    composite_col = COL_COMPOSITE_MEDIUM
    if composite_col in composite_scores.columns:
        score = composite_scores[composite_col].dropna()
        if not score.empty:
            # Color based on value
            colors = [
                COLORS["risk_on"] if v > 0.5 else
                COLORS["risk_off"] if v < -0.5 else
                COLORS["neutral"]
                for v in score.values
            ]
            fig.add_trace(
                go.Scatter(
                    x=score.index,
                    y=score.values,
                    name="Composite",
                    line=dict(color=COLORS["medium"], width=2),
                    fill="tozeroy",
                    fillcolor="rgba(155, 89, 182, 0.2)",
                    hovertemplate="Score: %{y:.2f}<extra></extra>",
                ),
                row=2, col=1,
            )

    # Add threshold lines to panel 2
    fig.add_hline(y=0.5, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # Add regime shading to both panels
    if not regimes.empty:
        _add_regime_shading_to_subplot(fig, regimes, row=1)
        _add_regime_shading_to_subplot(fig, regimes, row=2)

    fig.update_layout(
        title=title,
        height=600,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)

    return fig


def _add_regime_shading_to_subplot(
    fig: go.Figure,
    regime_series: pd.Series,
    row: int,
) -> None:
    """Add regime shading to a specific subplot row.

    Args:
        fig: Plotly figure with subplots
        regime_series: Series of regime labels
        row: Row number for the subplot
    """
    if regime_series.empty:
        return

    # Find regime spans
    spans = []
    current_regime = regime_series.iloc[0]
    start_date = regime_series.index[0]

    for i, (date, regime) in enumerate(regime_series.items()):
        if regime != current_regime or i == len(regime_series) - 1:
            spans.append((start_date, date, current_regime))
            current_regime = regime
            start_date = date

    # Add shapes for each span
    for start, end, regime in spans:
        if regime == Regime.RISK_ON.value:
            color = "rgba(46, 204, 113, 0.15)"
        elif regime == Regime.RISK_OFF.value:
            color = "rgba(231, 76, 60, 0.15)"
        else:
            continue  # Don't shade neutral

        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            line_width=0,
            layer="below",
            row=row, col=1,
        )


def save_all_charts(
    composites_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    checklist_df: pd.DataFrame,
    attribution: dict | None = None,
    charts_dir: Path | None = None,
) -> list[Path]:
    """Save all charts as PNG files.

    Args:
        composites_df: Composites DataFrame
        blocks_df: Blocks DataFrame
        regimes_df: Regimes DataFrame
        checklist_df: Checklist DataFrame
        attribution: Optional attribution dict
        charts_dir: Output directory

    Returns:
        List of saved file paths
    """
    saved = []

    def try_save(fig, name):
        path = save_chart_as_png(fig, name, charts_dir)
        if path:
            saved.append(path)

    # Composite time series
    fig = create_composite_timeseries(composites_df, regimes_df)
    try_save(fig, "composite_timeseries")

    # Block heatmap
    fig = create_block_heatmap(blocks_df)
    try_save(fig, "block_heatmap")

    # Current block bars
    fig = create_block_bars(blocks_df)
    try_save(fig, "block_bars_current")

    # Checklist score
    fig = create_checklist_score_chart(checklist_df)
    try_save(fig, "checklist_score")

    # Regime distribution
    fig = create_regime_distribution(regimes_df)
    try_save(fig, "regime_distribution")

    return saved
