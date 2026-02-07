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
    if COL_CHECKLIST_SCORE not in checklist_df.columns:
        return go.Figure()

    score = checklist_df[COL_CHECKLIST_SCORE]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=score.index,
            y=score,
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

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Score (0-100)",
        yaxis=dict(range=[0, 100]),
        height=400,
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
