"""Streamlit dashboard for interactive regime visualization."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from risk_index.core.constants import (
    PROCESSED_DIR,
    COL_COMPOSITE_FAST,
    COL_COMPOSITE_MEDIUM,
    COL_COMPOSITE_SLOW,
    COL_REGIME_FAST,
    COL_REGIME_MEDIUM,
    COL_REGIME_SLOW,
    COL_CONFIDENCE_FAST,
    COL_CONFIDENCE_MEDIUM,
    COL_CONFIDENCE_SLOW,
    COL_CHECKLIST_SCORE,
    COL_CHECKLIST_LABEL,
)
from risk_index.core.types import Regime
from risk_index.core.utils_io import read_parquet
from risk_index.reporting.charts import (
    create_composite_timeseries,
    create_block_bars,
    create_block_heatmap,
    create_checklist_score_chart,
    create_regime_distribution,
    COLORS,
)


# Page config
st.set_page_config(
    page_title="Risk Regime Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data():
    """Load processed data files."""
    data = {}

    try:
        data["features"] = read_parquet(PROCESSED_DIR / "features_latest.parquet")
    except FileNotFoundError:
        data["features"] = pd.DataFrame()

    try:
        data["blocks"] = read_parquet(PROCESSED_DIR / "blocks_latest.parquet")
    except FileNotFoundError:
        data["blocks"] = pd.DataFrame()

    try:
        data["composites"] = read_parquet(PROCESSED_DIR / "composites_latest.parquet")
    except FileNotFoundError:
        data["composites"] = pd.DataFrame()

    try:
        data["regimes"] = read_parquet(PROCESSED_DIR / "regimes_latest.parquet")
    except FileNotFoundError:
        data["regimes"] = pd.DataFrame()

    try:
        data["checklist"] = read_parquet(PROCESSED_DIR / "checklist_latest.parquet")
    except FileNotFoundError:
        data["checklist"] = pd.DataFrame()

    return data


def regime_color(regime: str) -> str:
    """Get color for regime."""
    if regime == Regime.RISK_ON.value:
        return COLORS["risk_on"]
    elif regime == Regime.RISK_OFF.value:
        return COLORS["risk_off"]
    return COLORS["neutral"]


def main():
    """Main dashboard function."""
    st.title("Risk Regime Dashboard")

    # Load data
    data = load_data()

    if data["composites"].empty:
        st.warning("No data found. Run the pipeline first to generate data.")
        st.code("python scripts/run_update.py")
        return

    # Sidebar
    st.sidebar.header("Settings")

    # Date range selector
    min_date = data["composites"].index.min()
    max_date = data["composites"].index.max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(max_date - pd.DateOffset(years=2), max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (data["composites"].index >= pd.Timestamp(start_date)) & (
            data["composites"].index <= pd.Timestamp(end_date)
        )
        filtered_composites = data["composites"][mask]
        filtered_blocks = data["blocks"][mask] if not data["blocks"].empty else data["blocks"]
        filtered_regimes = data["regimes"][mask] if not data["regimes"].empty else data["regimes"]
        filtered_checklist = data["checklist"][mask] if not data["checklist"].empty else data["checklist"]
    else:
        filtered_composites = data["composites"]
        filtered_blocks = data["blocks"]
        filtered_regimes = data["regimes"]
        filtered_checklist = data["checklist"]

    # Current regime display
    st.header("Current Regime")

    if not filtered_regimes.empty:
        latest = filtered_regimes.iloc[-1]
        latest_date = filtered_regimes.index[-1]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fast_regime = latest.get(COL_REGIME_FAST, "N/A")
            fast_conf = latest.get(COL_CONFIDENCE_FAST, 0) * 100
            st.metric(
                "Fast (4-8w)",
                fast_regime,
                f"Confidence: {fast_conf:.0f}%",
            )

        with col2:
            medium_regime = latest.get(COL_REGIME_MEDIUM, "N/A")
            medium_conf = latest.get(COL_CONFIDENCE_MEDIUM, 0) * 100
            st.metric(
                "Medium (13-26w)",
                medium_regime,
                f"Confidence: {medium_conf:.0f}%",
            )

        with col3:
            slow_regime = latest.get(COL_REGIME_SLOW, "N/A")
            slow_conf = latest.get(COL_CONFIDENCE_SLOW, 0) * 100
            st.metric(
                "Slow (26-52w)",
                slow_regime,
                f"Confidence: {slow_conf:.0f}%",
            )

        with col4:
            if not filtered_checklist.empty:
                checklist_latest = filtered_checklist.iloc[-1]
                score = checklist_latest.get(COL_CHECKLIST_SCORE, 50)
                label = checklist_latest.get(COL_CHECKLIST_LABEL, "N/A")
                st.metric("Checklist", f"{score:.0f}/100", label)

        st.caption(f"Data as of: {latest_date.strftime('%Y-%m-%d')}")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Composites", "Blocks", "Checklist", "Attribution"]
    )

    with tab1:
        st.subheader("Composite Signals Over Time")
        fig = create_composite_timeseries(filtered_composites, filtered_regimes)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Regime Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig = create_regime_distribution(filtered_regimes, COL_REGIME_MEDIUM, "Medium Regime")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_regime_distribution(filtered_regimes, COL_REGIME_SLOW, "Slow Regime")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Current Block Scores")

        if not filtered_blocks.empty:
            fig = create_block_bars(filtered_blocks)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Block Heatmap (Last 52 Weeks)")
            fig = create_block_heatmap(filtered_blocks)
            st.plotly_chart(fig, use_container_width=True)

            # Block drill-down
            st.subheader("Block Details")
            selected_block = st.selectbox(
                "Select block to explore",
                options=filtered_blocks.columns.tolist(),
            )

            if selected_block:
                block_series = filtered_blocks[selected_block]
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=block_series.index,
                        y=block_series,
                        mode="lines",
                        name=selected_block,
                        line=dict(color=COLORS["medium"], width=2),
                    )
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title=f"{selected_block} Over Time",
                    xaxis_title="Date",
                    yaxis_title="Z-Score",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Checklist Score Over Time")

        if not filtered_checklist.empty:
            fig = create_checklist_score_chart(filtered_checklist)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Current Checklist Items")

            # Get signal columns
            signal_cols = [c for c in filtered_checklist.columns if c.endswith("_signal")]

            if signal_cols:
                latest_signals = filtered_checklist[signal_cols].iloc[-1]

                # Create display table
                items_data = []
                for col in signal_cols:
                    item_id = col.replace("_signal", "")
                    signal = latest_signals[col]

                    if signal == "bull":
                        icon = "[OK]"
                    elif signal == "bear":
                        icon = "[X]"
                    else:
                        icon = "[!]"

                    items_data.append({
                        "Item": item_id.replace("_", " ").title(),
                        "Signal": f"{icon} {signal}",
                    })

                st.table(pd.DataFrame(items_data))

    with tab4:
        st.subheader("Regime Attribution")

        if not filtered_blocks.empty and not filtered_composites.empty:
            latest_date = filtered_blocks.index[-1]

            for speed, col in [
                ("Fast", COL_COMPOSITE_FAST),
                ("Medium", COL_COMPOSITE_MEDIUM),
                ("Slow", COL_COMPOSITE_SLOW),
            ]:
                if col in filtered_composites.columns:
                    st.markdown(f"**{speed} Composite**")

                    composite_val = filtered_composites.loc[latest_date, col]
                    st.write(f"Score: {composite_val:.2f}")

                    # Show block contributions
                    block_vals = filtered_blocks.loc[latest_date].dropna().sort_values()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Top Contributors (Positive)")
                        positive = block_vals[block_vals > 0].tail(5)
                        for name, val in positive.items():
                            st.write(f"  - {name}: +{val:.2f}")

                    with col2:
                        st.write("Top Contributors (Negative)")
                        negative = block_vals[block_vals < 0].head(5)
                        for name, val in negative.items():
                            st.write(f"  - {name}: {val:.2f}")

                    st.divider()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Data updates weekly after Friday close. "
        "Run `python scripts/run_update.py` to refresh."
    )


if __name__ == "__main__":
    main()
