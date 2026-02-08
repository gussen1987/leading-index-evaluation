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
    CONFIG_DIR,
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
from risk_index.core.utils_io import read_parquet, read_yaml
from risk_index.reporting.charts import (
    create_composite_timeseries,
    create_block_bars,
    create_block_heatmap,
    create_checklist_score_chart,
    create_regime_distribution,
    create_composition_pie,
    create_equity_curves,
    create_spy_regime_panel_chart,
    COLORS,
)
from risk_index.research.backtest_regimes import (
    run_all_regime_backtests,
    create_backtest_summary,
    load_spy_prices,
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


# Ticker/ratio descriptions for display
RATIO_DESCRIPTIONS = {
    # Equity Leadership
    "RSP_SPY": ("Equal Weight vs Cap Weight", "RSP / SPY", "Breadth"),
    "IWM_SPY": ("Small Cap vs Large Cap", "IWM / SPY", "Size"),
    "QQQ_SPY": ("Nasdaq vs S&P 500", "QQQ / SPY", "Growth"),
    "XLY_XLP": ("Discretionary vs Staples", "XLY / XLP", "Cyclical"),
    "XLF_XLU": ("Financials vs Utilities", "XLF / XLU", "Risk-On"),
    "XLK_XLU": ("Tech vs Utilities", "XLK / XLU", "Risk-On"),
    "XLI_XLU": ("Industrials vs Utilities", "XLI / XLU", "Risk-On"),
    "SMH_SPY": ("Semiconductors vs Market", "SMH / SPY", "Tech"),
    "IYT_SPY": ("Transports vs Market", "IYT / SPY", "Cyclical"),
    "XHB_SPY": ("Homebuilders vs Market", "XHB / SPY", "Housing"),
    "XRT_SPY": ("Retail vs Market", "XRT / SPY", "Consumer"),
    "IBB_SPY": ("Biotech vs Market", "IBB / SPY", "Spec"),
    # Factor Prefs
    "SPHB_SPLV": ("High Beta vs Low Vol", "SPHB / SPLV", "Risk"),
    "MTUM_USMV": ("Momentum vs Min Vol", "MTUM / USMV", "Factor"),
    "QUAL_USMV": ("Quality vs Min Vol", "QUAL / USMV", "Factor"),
    # Credit
    "HYG_IEF": ("High Yield vs Treasuries", "HYG / IEF", "Credit"),
    "LQD_IEF": ("IG Credit vs Treasuries", "LQD / IEF", "Credit"),
    "JNK_TLT": ("Junk Bonds vs Long Treasuries", "JNK / TLT", "Credit"),
    "EMB_AGG": ("EM Bonds vs US Agg", "EMB / AGG", "EM"),
    "EMB_HYG": ("EM vs US High Yield", "EMB / HYG", "EM"),
    "BAMLH0A0HYM2": ("HY Credit Spread (OAS)", "FRED", "Credit"),
    "BAMLC0A0CM": ("IG Credit Spread (OAS)", "FRED", "Credit"),
    "BAMLH0A1HYBB": ("BB Credit Spread", "FRED", "Credit"),
    "BAMLH0A2HYB": ("B Credit Spread", "FRED", "Credit"),
    # Rates
    "T10Y2Y": ("10Y-2Y Yield Spread", "FRED", "Curve"),
    "T10Y3M": ("10Y-3M Yield Spread", "FRED", "Curve"),
    "DFII10": ("10Y Real Yield (TIPS)", "FRED", "Rates"),
    "DFII5": ("5Y Real Yield (TIPS)", "FRED", "Rates"),
    "T10YIE": ("10Y Breakeven Inflation", "FRED", "Inflation"),
    "T5YIE": ("5Y Breakeven Inflation", "FRED", "Inflation"),
    "T5YIFR": ("5Y5Y Forward Inflation", "FRED", "Inflation"),
    "TIP_IEF": ("TIPS vs Treasuries", "TIP / IEF", "Inflation"),
    "TIP_TLT": ("TIPS vs Long Treasuries", "TIP / TLT", "Inflation"),
    # FX
    "AUDJPY": ("AUD/JPY Cross", "Yahoo", "Risk FX"),
    "CADJPY": ("CAD/JPY Cross", "Yahoo", "Risk FX"),
    "FXA_FXY": ("AUD vs JPY (ETF)", "FXA / FXY", "Risk FX"),
    "FXE_UUP": ("Euro vs Dollar", "FXE / UUP", "FX"),
    "UUP": ("US Dollar Index", "UUP", "Dollar"),
    "DTWEXBGS": ("Trade Weighted Dollar", "FRED", "Dollar"),
    # Commodities
    "COPX_GLD": ("Copper vs Gold", "COPX / GLD", "Growth"),
    "USO_GLD": ("Oil vs Gold", "USO / GLD", "Inflation"),
    "DBC_GLD": ("Commodities vs Gold", "DBC / GLD", "Risk"),
    "XME_GLD": ("Metals & Mining vs Gold", "XME / GLD", "Risk"),
    "SLV_GLD": ("Silver vs Gold", "SLV / GLD", "Industrial"),
    # Volatility
    "VIX": ("VIX Volatility", "CBOE", "Fear"),
    "VVIX": ("VIX of VIX", "CBOE", "Tail Risk"),
    "NFCI": ("Financial Conditions", "FRED", "Liquidity"),
    "ANFCI": ("Adjusted NFCI", "FRED", "Liquidity"),
    "STLFSI4": ("Financial Stress", "FRED", "Stress"),
    "TEDRATE": ("TED Spread", "FRED", "Liquidity"),
    "RATES_VOL_PROXY_21D": ("Bond Volatility", "Computed", "Vol"),
    # Global
    "EEM_SPY": ("EM vs US", "EEM / SPY", "Global"),
    "EFA_SPY": ("Developed ex-US vs US", "EFA / SPY", "Global"),
    "ACWX_SPY": ("World ex-US vs US", "ACWX / SPY", "Global"),
    "FXI_SPY": ("China vs US", "FXI / SPY", "Global"),
    # Defensive
    "XLU_SPY": ("Utilities vs Market", "XLU / SPY", "Defensive"),
    "TLT_SHY": ("Duration Preference", "TLT / SHY", "Rates"),
    "GLD_SPY": ("Gold vs Market", "GLD / SPY", "Safe Haven"),
    "GDXJ_GLD": ("Jr Miners vs Gold", "GDXJ / GLD", "Gold Beta"),
    # Housing
    "KRE_VNQ": ("Banks vs REITs", "KRE / VNQ", "Rates"),
    "PERMIT": ("Building Permits", "FRED", "Housing"),
    "HOUST": ("Housing Starts", "FRED", "Housing"),
    # Macro
    "ICSA": ("Initial Claims", "FRED", "Labor"),
    "CCSA": ("Continuing Claims", "FRED", "Labor"),
    "UMCSENT": ("Consumer Sentiment", "FRED", "Sentiment"),
    "INDPRO": ("Industrial Production", "FRED", "Activity"),
    "UNRATE": ("Unemployment Rate", "FRED", "Labor"),
}


def format_item_name(item_id: str) -> str:
    """Format checklist item ID for display."""
    if item_id in RATIO_DESCRIPTIONS:
        return RATIO_DESCRIPTIONS[item_id][0]
    return item_id.replace("_", " ").title()


def color_row_by_status(row: pd.Series) -> list[str]:
    """Apply row colors based on status column."""
    status = row.get("Status", "").upper()
    if status == "BULL":
        return ["background-color: rgba(46, 204, 113, 0.3)"] * len(row)
    elif status == "BEAR":
        return ["background-color: rgba(231, 76, 60, 0.3)"] * len(row)
    return ["background-color: rgba(243, 156, 18, 0.2)"] * len(row)


def get_block_details(block_name: str) -> pd.DataFrame:
    """Get detailed members for a block from universe.yml.

    Args:
        block_name: Name of the block

    Returns:
        DataFrame with member details
    """
    config_path = CONFIG_DIR / "universe.yml"
    if not config_path.exists():
        return pd.DataFrame()

    config = read_yaml(config_path)
    for block in config.get("blocks", []):
        if block["name"] == block_name:
            members = []
            for m in block.get("members", []):
                member_id = m["id"]
                desc, tickers, category = RATIO_DESCRIPTIONS.get(
                    member_id, (member_id, member_id, "")
                )
                inverted = m.get("invert", False)
                members.append({
                    "Ratio": member_id,
                    "Description": desc,
                    "Tickers": tickers,
                    "Category": category,
                    "Inverted": "Yes" if inverted else "",
                })
            return pd.DataFrame(members)
    return pd.DataFrame()


def get_composite_weights(speed: str) -> pd.DataFrame:
    """Load composite weights from config.

    Args:
        speed: Composite speed (fast, medium, slow)

    Returns:
        DataFrame with block and weight columns
    """
    config_path = CONFIG_DIR / "composites.yml"
    if not config_path.exists():
        return pd.DataFrame()

    config = read_yaml(config_path)
    composites = config.get("composites", [])

    for comp in composites:
        if comp.get("speed") == speed:
            blocks = comp.get("blocks", [])
            return pd.DataFrame([
                {"block": b["block"], "weight": b["weight"]}
                for b in blocks
            ])

    return pd.DataFrame()


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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Composites", "Blocks", "Checklist", "Attribution", "Composition", "Backtest", "Factor Leadership"]
    )

    with tab1:
        st.subheader("Composite Signals Over Time")
        fig = create_composite_timeseries(filtered_composites, filtered_regimes)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Regime Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig = create_regime_distribution(filtered_regimes, COL_REGIME_MEDIUM, "Medium Regime")
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = create_regime_distribution(filtered_regimes, COL_REGIME_SLOW, "Slow Regime")
            st.plotly_chart(fig, width="stretch")

    with tab2:
        st.subheader("Current Block Scores")

        if not filtered_blocks.empty:
            fig = create_block_bars(filtered_blocks)
            st.plotly_chart(fig, width="stretch")

            st.subheader("Block Heatmap (Last 52 Weeks)")
            fig = create_block_heatmap(filtered_blocks)
            st.plotly_chart(fig, width="stretch")

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
                st.plotly_chart(fig, width="stretch")

    with tab3:
        st.subheader("Checklist Score Over Time")

        if not filtered_checklist.empty:
            # Get signal columns
            signal_cols = [c for c in filtered_checklist.columns if c.endswith("_signal")]
            latest = filtered_checklist.iloc[-1]
            prev = filtered_checklist.iloc[-2] if len(filtered_checklist) > 1 else latest

            # Calculate bull count for summary
            bull_count = 0
            for col in signal_cols:
                if latest.get(col) == "bull":
                    bull_count += 1
            total = len(signal_cols) if signal_cols else 1

            # Regime label
            score_pct = bull_count / total * 100 if total > 0 else 50
            if score_pct >= 75:
                regime_label = "Confirmed Bull Market"
                regime_color = "green"
            elif score_pct >= 50:
                regime_label = "On Watch"
                regime_color = "orange"
            else:
                regime_label = "Risk-Off"
                regime_color = "red"

            # Display regime prominently
            st.markdown(
                f"### <span style='color:{regime_color}'>{regime_label}</span>",
                unsafe_allow_html=True
            )

            # Score metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Checklist Score",
                    f"{bull_count} of {total} Bullish",
                    f"{score_pct:.0f}%"
                )
            with col2:
                checklist_score = latest.get(COL_CHECKLIST_SCORE, 50)
                st.metric("Score (0-100)", f"{checklist_score:.0f}")
            with col3:
                checklist_label = latest.get(COL_CHECKLIST_LABEL, "N/A")
                st.metric("Label", checklist_label)

            # Chart
            fig = create_checklist_score_chart(filtered_checklist)
            st.plotly_chart(fig, width="stretch")

            st.subheader("Current Checklist Items")

            if signal_cols:
                # Create enhanced display table with direction
                items_data = []
                for col in signal_cols:
                    item_id = col.replace("_signal", "")
                    signal = latest.get(col, "neutral")
                    prev_signal = prev.get(col, signal)

                    # Direction of change
                    if signal == prev_signal:
                        direction = "Flat"
                        dir_icon = "-"
                    elif signal == "bull" and prev_signal != "bull":
                        direction = "Improving"
                        dir_icon = "+"
                    elif signal == "bear" and prev_signal != "bear":
                        direction = "Declining"
                        dir_icon = "-"
                    else:
                        direction = "Flat"
                        dir_icon = "-"

                    # Get description
                    desc = format_item_name(item_id)
                    tickers = RATIO_DESCRIPTIONS.get(item_id, ("", item_id, ""))[1]

                    items_data.append({
                        "Item": desc,
                        "Tickers": tickers,
                        "Status": signal.upper() if signal else "NEUTRAL",
                        "Direction": f"{dir_icon} {direction}",
                    })

                items_df = pd.DataFrame(items_data)

                # Apply styling
                styled_df = items_df.style.apply(color_row_by_status, axis=1)
                st.dataframe(styled_df, width="stretch", hide_index=True)

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

    with tab5:
        st.subheader("Composite Composition")
        st.markdown("Block weights are normalized to sum to 1.0 for each composite.")

        for speed in ["fast", "medium", "slow"]:
            st.markdown(f"### {speed.title()} Composite")

            weights_df = get_composite_weights(speed)

            if not weights_df.empty:
                col1, col2 = st.columns([1, 1])

                with col1:
                    # Add current block scores if available
                    if not filtered_blocks.empty:
                        latest_date = filtered_blocks.index[-1]
                        display_df = weights_df.copy()
                        display_df["Weight (%)"] = (display_df["weight"] * 100).round(1)

                        # Get current scores for each block
                        scores = []
                        contributions = []
                        for _, row in display_df.iterrows():
                            block_name = row["block"]
                            if block_name in filtered_blocks.columns:
                                score = filtered_blocks.loc[latest_date, block_name]
                                scores.append(round(score, 2))
                                contributions.append(round(score * row["weight"], 3))
                            else:
                                scores.append("-")
                                contributions.append("-")

                        display_df["Current Score"] = scores
                        display_df["Contribution"] = contributions
                        display_df = display_df[["block", "Weight (%)", "Current Score", "Contribution"]]
                        display_df.columns = ["Block", "Weight (%)", "Score", "Contribution"]
                    else:
                        display_df = weights_df.copy()
                        display_df["Weight (%)"] = (display_df["weight"] * 100).round(1)
                        display_df = display_df[["block", "Weight (%)"]]
                        display_df.columns = ["Block", "Weight (%)"]

                    st.dataframe(display_df, width="stretch", hide_index=True)

                with col2:
                    fig = create_composition_pie(weights_df, speed)
                    st.plotly_chart(fig, width="stretch")

                # Expandable block details
                st.markdown("**Block Details** (click to expand)")
                for _, row in weights_df.iterrows():
                    block_name = row["block"]
                    with st.expander(f"{block_name.replace('_', ' ').title()}"):
                        block_details = get_block_details(block_name)
                        if not block_details.empty:
                            st.dataframe(
                                block_details,
                                width="stretch",
                                hide_index=True,
                            )
                        else:
                            st.write("No details available.")

            st.divider()

    with tab6:
        st.subheader("SPY Regime Backtest")
        st.markdown(
            "Backtest SPY using regime signals: "
            "**Long during Risk-On (green), cash during Risk-Off (red)**"
        )

        if not filtered_regimes.empty:
            # Run backtests
            with st.spinner("Running backtests..."):
                try:
                    spy_prices = load_spy_prices(align_to_weekly=True)
                    backtest_results = run_all_regime_backtests(
                        filtered_regimes,
                        spy_prices,
                        strategy="long_risk_on",
                    )

                    # Summary metrics table
                    st.markdown("### Performance Summary")
                    summary_df = create_backtest_summary(backtest_results)

                    if not summary_df.empty:
                        # Format the dataframe for display
                        formatted_df = summary_df.copy()
                        for col in formatted_df.columns:
                            formatted_df[col] = formatted_df[col].apply(
                                lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x
                            )
                        st.dataframe(formatted_df, width="stretch")

                    # Equity curves chart
                    st.markdown("### Equity Curves")
                    fig = create_equity_curves(backtest_results)
                    st.plotly_chart(fig, width="stretch")

                    # Trade details in expanders
                    st.markdown("### Trade Details")

                    for speed, result in backtest_results.items():
                        if result is None:
                            continue

                        trades_df = result.get("trades")
                        if trades_df is None or trades_df.empty:
                            continue

                        with st.expander(f"{speed} Composite - {len(trades_df)} trades"):
                            # Format trades for display
                            display_trades = trades_df.copy()
                            display_trades["Entry Date"] = pd.to_datetime(display_trades["Entry Date"]).dt.strftime("%Y-%m-%d")
                            display_trades["Exit Date"] = pd.to_datetime(display_trades["Exit Date"]).dt.strftime("%Y-%m-%d")
                            display_trades["Entry Price"] = display_trades["Entry Price"].round(2)
                            display_trades["Exit Price"] = display_trades["Exit Price"].round(2)
                            display_trades["Return (%)"] = display_trades["Return (%)"].round(2)

                            st.dataframe(display_trades, width="stretch", hide_index=True)

                            # Quick stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Return", f"{result['total_return']:.1f}%")
                            with col2:
                                st.metric("Win Rate", f"{result['win_rate']:.1f}%")
                            with col3:
                                st.metric("Max Drawdown", f"{result['max_drawdown']:.1f}%")

                except Exception as e:
                    st.error(f"Error running backtest: {e}")
        else:
            st.warning("No regime data available for backtesting.")

    with tab7:
        st.subheader("Factor Leadership")
        st.markdown(
            "Current factor preferences based on ratio trends. "
            "**Green** = Risk-On factor leading, **Red** = Defensive factor leading."
        )

        if not filtered_blocks.empty and not data["features"].empty:
            latest_date = filtered_blocks.index[-1]

            # Factor definitions with their key ratios
            factors = {
                "Size": {
                    "ratio": "IWM_SPY",
                    "description": "Small Cap vs Large Cap",
                    "risk_on_label": "Small Cap",
                    "risk_off_label": "Large Cap",
                },
                "Style": {
                    "ratio": "QQQ_SPY",
                    "description": "Growth vs Value proxy",
                    "risk_on_label": "Growth",
                    "risk_off_label": "Value",
                },
                "Risk Appetite": {
                    "ratio": "SPHB_SPLV",
                    "description": "High Beta vs Low Vol",
                    "risk_on_label": "High Beta",
                    "risk_off_label": "Low Vol",
                },
                "Cyclical": {
                    "ratio": "XLY_XLP",
                    "description": "Discretionary vs Staples",
                    "risk_on_label": "Cyclical",
                    "risk_off_label": "Defensive",
                },
                "Credit": {
                    "ratio": "HYG_IEF",
                    "description": "High Yield vs Treasuries",
                    "risk_on_label": "Risk-On",
                    "risk_off_label": "Flight to Safety",
                },
                "Global": {
                    "ratio": "EEM_SPY",
                    "description": "Emerging Markets vs US",
                    "risk_on_label": "EM",
                    "risk_off_label": "US",
                },
            }

            # Get feature data for ratios
            features_df = data["features"]
            if not features_df.empty:
                latest_features = features_df.iloc[-1]

                cols = st.columns(3)
                for i, (factor_name, factor_info) in enumerate(factors.items()):
                    col_idx = i % 3
                    ratio_id = factor_info["ratio"]

                    # Look for z-score column
                    z_col = f"{ratio_id}_z_52w"
                    roc_col = f"{ratio_id}_roc_8w"

                    z_score = latest_features.get(z_col, 0) if z_col in latest_features else 0
                    roc = latest_features.get(roc_col, 0) if roc_col in latest_features else 0

                    # Determine leader
                    if z_score > 0.5:
                        leader = factor_info["risk_on_label"]
                        color = "green"
                    elif z_score < -0.5:
                        leader = factor_info["risk_off_label"]
                        color = "red"
                    else:
                        leader = "Neutral"
                        color = "orange"

                    # Trend direction
                    if roc > 0.02:
                        trend = "Strengthening"
                    elif roc < -0.02:
                        trend = "Weakening"
                    else:
                        trend = "Stable"

                    with cols[col_idx]:
                        st.markdown(f"**{factor_name}**")
                        st.markdown(f"<span style='color:{color};font-size:1.2em'>{leader}</span>", unsafe_allow_html=True)
                        st.caption(f"Z: {z_score:.2f} | {trend}")
                        st.caption(factor_info["description"])

                # Add multi-panel chart if SPY data available
                st.divider()
                st.markdown("### SPY with Regime Indicator")

                try:
                    spy_prices = load_spy_prices(align_to_weekly=True)
                    if not spy_prices.empty and not filtered_regimes.empty:
                        # Align SPY to the filtered date range
                        spy_filtered = spy_prices[
                            (spy_prices.index >= filtered_composites.index.min()) &
                            (spy_prices.index <= filtered_composites.index.max())
                        ]

                        regime_col = COL_REGIME_MEDIUM
                        if regime_col in filtered_regimes.columns:
                            fig = create_spy_regime_panel_chart(
                                spy_filtered,
                                filtered_composites,
                                filtered_regimes[regime_col],
                                title="SPY with Medium Regime Indicator"
                            )
                            st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.warning(f"Could not load SPY data: {e}")
        else:
            st.warning("No feature data available for factor analysis.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Data updates weekly after Friday close. "
        "Run `python scripts/run_update.py` to refresh."
    )


if __name__ == "__main__":
    main()
