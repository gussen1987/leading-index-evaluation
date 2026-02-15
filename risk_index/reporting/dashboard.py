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
    create_cumulative_ad_chart,
    create_pct_above_ma_timeseries,
    create_new_highs_lows_timeseries,
    create_regime_timeline,
    create_attribution_chart,
    create_tax_yoy_chart,
    create_ytd_comparison_chart,
    create_tax_vs_spy_chart,
    create_gig_economy_chart,
    COLORS,
)
from risk_index.research.backtest_regimes import (
    run_all_regime_backtests,
    create_backtest_summary,
    load_spy_prices,
)
from risk_index.pipeline.breadth_fetch import (
    fetch_all_breadth_data,
    fetch_finviz_breadth,
    prepare_heatmap_data,
    prepare_breadth_summary,
    fetch_breadth_timeseries,
)
from risk_index.pipeline.treasury_fetch import (
    prepare_treasury_indicators,
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
    except Exception:
        data["features"] = pd.DataFrame()

    try:
        data["blocks"] = read_parquet(PROCESSED_DIR / "blocks_latest.parquet")
    except Exception:
        data["blocks"] = pd.DataFrame()

    try:
        data["composites"] = read_parquet(PROCESSED_DIR / "composites_latest.parquet")
    except Exception:
        data["composites"] = pd.DataFrame()

    try:
        data["regimes"] = read_parquet(PROCESSED_DIR / "regimes_latest.parquet")
    except Exception:
        data["regimes"] = pd.DataFrame()

    try:
        data["checklist"] = read_parquet(PROCESSED_DIR / "checklist_latest.parquet")
    except Exception:
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


def load_dashboard_config() -> dict:
    """Load dashboard configuration from YAML.

    Returns:
        Dict with dashboard settings including sector thresholds
    """
    config_path = CONFIG_DIR / "dashboard.yml"
    if not config_path.exists():
        # Return defaults if config doesn't exist
        return {
            "sector_scorecard": {
                "trend_thresholds": [0.5, 0, -0.3, -0.7],
                "relative_strength_thresholds": [0.7, 0.2, -0.2, -0.5],
                "momentum_thresholds": [3, 1, -1, -3],
            }
        }

    return read_yaml(config_path)


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

    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh All Data", type="primary"):
        with st.spinner("Refreshing all data..."):
            fetch_all_breadth_data(
                include_exchanges=True,
                include_russell=True,
                use_cache=False,
                force_refresh=True,
            )
            prepare_treasury_indicators(years=5, use_cache=False)
        st.rerun()

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
        ["Composites", "Blocks", "Checklist", "Attribution", "Composition", "Backtest", "Factor Leadership", "Market Breadth", "Sector Scorecard", "Treasury Tax Flow"]
    )

    with tab1:
        st.subheader("Composite Signals Over Time")

        with st.expander("Understanding Composite Signals", expanded=False):
            st.markdown("""
**What are Composites?**
Composites combine multiple market indicators (blocks) into single scores that track
risk appetite across different time horizons.

**Three Timeframes:**
- **Fast (4-8 weeks):** Short-term momentum, reacts quickly to market shifts
- **Medium (13-26 weeks):** Primary trend indicator, best for tactical allocation
- **Slow (26-52 weeks):** Long-term trend, filters out noise

**Regime Interpretation:**
| Score | Regime | Meaning |
|-------|--------|---------|
| > 0.5 | Risk-On | Bullish conditions, favor equities |
| -0.5 to 0.5 | Neutral | Mixed signals, reduce risk or wait |
| < -0.5 | Risk-Off | Defensive, favor bonds/cash |

**Usage:** The Medium composite is typically used for portfolio allocation decisions.
Fast signals potential turns, Slow confirms sustained trends.
            """)

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

        # Regime Timeline Charts
        st.subheader("Regime Timeline")
        with st.expander("Understanding Regime Timelines", expanded=False):
            st.markdown("""
**Regime Timeline Charts**

These step-function charts show regime changes over time:
- **+1 (Risk-On):** Bullish conditions, green shading
- **0 (Neutral):** Mixed signals, no shading
- **-1 (Risk-Off):** Bearish conditions, red shading

**Interpretation:**
- Look for sustained periods in one regime (trend persistence)
- Frequent regime changes = choppy, unclear market
- Compare Fast vs Slow: Fast leads, Slow confirms
            """)

        # Display regime timelines for each speed
        if not filtered_regimes.empty:
            timeline_tabs = st.tabs(["Fast", "Medium", "Slow"])

            with timeline_tabs[0]:
                fig_fast = create_regime_timeline(filtered_regimes, COL_REGIME_FAST, "Fast Regime Timeline")
                st.plotly_chart(fig_fast, use_container_width=True)

            with timeline_tabs[1]:
                fig_medium = create_regime_timeline(filtered_regimes, COL_REGIME_MEDIUM, "Medium Regime Timeline")
                st.plotly_chart(fig_medium, use_container_width=True)

            with timeline_tabs[2]:
                fig_slow = create_regime_timeline(filtered_regimes, COL_REGIME_SLOW, "Slow Regime Timeline")
                st.plotly_chart(fig_slow, use_container_width=True)

    with tab2:
        st.subheader("Current Block Scores")

        with st.expander("Understanding Block Scores", expanded=False):
            st.markdown("""
**What are Blocks?**
Blocks are groups of related indicators measuring specific aspects of market risk:
- **Equity Leadership:** Which sectors/styles are leading (cyclical vs defensive)
- **Credit:** High yield spreads, investment grade conditions
- **FX/Global:** Currency risk appetite, international flows
- **Volatility:** VIX, stress indicators

**Score Interpretation (Z-Scores):**
| Score | Meaning |
|-------|---------|
| > +1.0 | Strong risk-on signal |
| +0.5 to +1.0 | Moderately bullish |
| -0.5 to +0.5 | Neutral |
| -1.0 to -0.5 | Moderately bearish |
| < -1.0 | Strong risk-off signal |

**Block Heatmap:** Shows how each block has evolved over time.
Green = risk-on, Red = risk-off. Look for persistent colors (trend) or divergences.
            """)

        if not filtered_blocks.empty:
            fig = create_block_bars(filtered_blocks)
            st.plotly_chart(fig, width="stretch")

            # Calculate weeks based on filtered data
            num_weeks = len(filtered_blocks)
            st.subheader(f"Block Heatmap (Last {num_weeks} Weeks)")
            fig = create_block_heatmap(filtered_blocks, weeks=num_weeks)
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

        with st.expander("Understanding the Bull Market Checklist", expanded=False):
            st.markdown("""
**What is the Checklist?**
A simple bull/bear scoring system based on key market ratios and indicators.
Each item is classified as BULL, BEAR, or NEUTRAL based on its trend.

**Checklist Criteria (similar to Daily Number):**
- **Breakout Level:** S&P 500 above recent highs
- **Trend Direction:** 200-day MA rising
- **Momentum Regime:** RSI in bullish zone
- **Breadth Thrust:** % of stocks at 20-day highs > 55%
- **Risk Regime:** Risk-On/Risk-Off indicator
- **Credit Conditions:** High yield spreads
- **Global Participation:** International markets participating

**Score Interpretation:**
| Score | Regime | Action |
|-------|--------|--------|
| 75-100% Bull | Confirmed Bull | Full equity allocation |
| 50-75% Bull | On Watch | Moderate allocation, watch for changes |
| <50% Bull | Risk-Off | Defensive, raise cash |

**Direction column:** Shows if each indicator is Improving, Declining, or Flat vs prior reading.
            """)

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

                    # Direction of change using signal ordering
                    SIGNAL_ORDER = {"bear": 0, "neutral": 1, "bull": 2}
                    current_val = SIGNAL_ORDER.get(signal, 1)
                    prev_val = SIGNAL_ORDER.get(prev_signal, 1)

                    if current_val > prev_val:
                        direction = "Improving"
                        dir_icon = "+"
                    elif current_val < prev_val:
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

        with st.expander("Understanding Attribution", expanded=False):
            st.markdown("""
**What is Attribution?**
Attribution analysis breaks down the composite score into contributions from each block.
This shows which factors are driving the current regime signal.

**Waterfall Chart:**
- Each bar shows a block's contribution to the total composite score
- **Green bars:** Positive contributions (risk-on signals)
- **Red bars:** Negative contributions (risk-off signals)
- Contributions = Block Score × Weight

**Usage:** Identify which market factors are supporting or opposing the current regime.
            """)

        if not filtered_blocks.empty and not filtered_composites.empty:
            latest_date = filtered_blocks.index[-1]

            # Create tabs for each speed
            attr_tabs = st.tabs(["Fast", "Medium", "Slow"])

            for tab_idx, (speed, col, speed_key) in enumerate([
                ("Fast", COL_COMPOSITE_FAST, "fast"),
                ("Medium", COL_COMPOSITE_MEDIUM, "medium"),
                ("Slow", COL_COMPOSITE_SLOW, "slow"),
            ]):
                with attr_tabs[tab_idx]:
                    if col in filtered_composites.columns:
                        composite_val = filtered_composites.loc[latest_date, col]

                        # Get weights for this composite
                        weights_df = get_composite_weights(speed_key)

                        if not weights_df.empty:
                            # Compute attribution data
                            attribution = {speed_key: {"blocks": {}}}
                            total_contribution = 0

                            for _, row in weights_df.iterrows():
                                block_name = row["block"]
                                weight = row["weight"]

                                if block_name in filtered_blocks.columns:
                                    block_score = filtered_blocks.loc[latest_date, block_name]
                                    if pd.notna(block_score):
                                        contribution = block_score * weight
                                        total_contribution += contribution
                                        attribution[speed_key]["blocks"][block_name] = {
                                            "score": block_score,
                                            "weight": weight,
                                            "contribution": contribution,
                                        }

                            attribution[speed_key]["total"] = total_contribution

                            # Display composite score
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric(f"{speed} Composite", f"{composite_val:.2f}")

                            # Create and display waterfall chart
                            fig = create_attribution_chart(attribution, speed_key, f"{speed} Composite Attribution")
                            st.plotly_chart(fig, use_container_width=True)

                            # Show contribution table
                            st.markdown("**Block Contributions**")
                            contrib_data = []
                            for block_name, data in attribution[speed_key]["blocks"].items():
                                contrib_data.append({
                                    "Block": block_name.replace("_", " ").title(),
                                    "Score": round(data["score"], 2),
                                    "Weight": f"{data['weight']*100:.1f}%",
                                    "Contribution": round(data["contribution"], 3),
                                })

                            # Sort by absolute contribution
                            contrib_df = pd.DataFrame(contrib_data)
                            if not contrib_df.empty:
                                contrib_df["abs_contrib"] = contrib_df["Contribution"].abs()
                                contrib_df = contrib_df.sort_values("abs_contrib", ascending=False).drop("abs_contrib", axis=1)

                                def color_contribution(val):
                                    if val > 0:
                                        return "background-color: rgba(46, 204, 113, 0.3)"
                                    elif val < 0:
                                        return "background-color: rgba(231, 76, 60, 0.3)"
                                    return ""

                                styled_contrib = contrib_df.style.map(
                                    color_contribution, subset=["Contribution"]
                                )
                                st.dataframe(styled_contrib, use_container_width=True, hide_index=True)
                        else:
                            st.warning(f"No weight configuration found for {speed} composite.")

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

        with st.expander("Understanding the Backtest", expanded=False):
            st.markdown("""
**What this shows:**
A historical simulation of trading SPY based on regime signals.

**Strategy:**
- **Risk-On (green):** Hold SPY (long equities)
- **Risk-Off (red):** Go to cash (0% return while waiting)

**Key Metrics:**
| Metric | What it means | Good value |
|--------|---------------|------------|
| **Total Return** | Cumulative gain/loss | Higher = better |
| **CAGR** | Annualized return | >8% beats bonds |
| **Max Drawdown** | Worst peak-to-trough loss | <20% is defensive |
| **Win Rate** | % of trades profitable | >50% for trend systems |
| **Sharpe Ratio** | Risk-adjusted return | >1.0 is good |

**Comparison:**
- **Buy & Hold:** Simply holding SPY the entire time
- **Strategy:** Following regime signals

**Note:** Past performance doesn't guarantee future results. This is for
educational purposes to understand how the regime model would have performed.
            """)

        st.markdown(
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

        with st.expander("Understanding Factor Leadership", expanded=False):
            st.markdown("""
**What is Factor Leadership?**
Factor leadership shows which market characteristics (factors) are currently being
rewarded by the market. This helps understand the "type" of market we're in.

**Key Factors:**
| Factor | Risk-On Leader | Risk-Off Leader | Measured By |
|--------|---------------|-----------------|-------------|
| **Size** | Small Cap | Large Cap | IWM/SPY ratio |
| **Style** | Growth | Value | QQQ/SPY ratio |
| **Risk Appetite** | High Beta | Low Volatility | SPHB/SPLV ratio |
| **Cyclical** | Discretionary | Staples | XLY/XLP ratio |
| **Credit** | High Yield | Treasuries | HYG/IEF ratio |
| **Global** | Emerging Markets | US | EEM/SPY ratio |

**How to Use:**
- Multiple factors showing Risk-On = confident bull market
- Mixed signals = rotation or transition
- Multiple Risk-Off = defensive posture warranted

**Z-Score thresholds:** >0.5 = Risk-On leading, <-0.5 = Risk-Off leading
            """)

        st.markdown(
            "**Green** = Risk-On factor leading, **Red** = Defensive factor leading."
        )

        if not filtered_blocks.empty and data.get("features") is not None and not data["features"].empty:
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
                    z_col = f"{ratio_id}__z_52w"
                    roc_col = f"{ratio_id}__roc_8w"

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

    with tab8:
        st.subheader("Market Breadth Heat Maps")

        # Explanation section
        with st.expander("What is Market Breadth? (Click to learn)", expanded=False):
            st.markdown("""
**Market breadth** measures how many stocks are participating in a market move.
It helps distinguish between healthy rallies (broad participation) and narrow rallies
(driven by a few stocks).

**Key Concepts:**
- **Advancers/Decliners**: Stocks that went up vs down on a given day
- **% Above Moving Averages**: Shows trend health across the market
- **New Highs/Lows**: Stocks making fresh price extremes
- **Overbought/Oversold**: RSI-based momentum readings

**Why it matters:** Strong markets show broad participation. When indices rise but
breadth narrows (fewer stocks participating), it can signal distribution or a
topping process. Conversely, improving breadth during a pullback suggests underlying strength.
            """)

        # Load breadth data with caching
        try:
            # Options for which indices to include
            col1, col2, col3 = st.columns(3)
            with col1:
                include_exchanges = st.checkbox("Include NYSE/NASDAQ", value=True,
                    help="Add NYSE (~2300) and NASDAQ (~4000) stocks")
            with col2:
                include_russell = st.checkbox("Include Russell 1000/2000", value=True,
                    help="Add Russell 1000 (~1000) and Russell 2000 (~2000) stocks")
            with col3:
                if st.button("Refresh All Data"):
                    with st.spinner("Fetching all breadth data (10-15 minutes)..."):
                        breadth_df = fetch_all_breadth_data(
                            include_exchanges=include_exchanges,
                            include_russell=include_russell,
                            lookback_days=10,
                            use_cache=False,
                            force_refresh=True,
                        )
                    st.rerun()

            with st.spinner("Loading breadth data..."):
                breadth_df = fetch_all_breadth_data(
                    include_exchanges=include_exchanges,
                    include_russell=include_russell,
                    lookback_days=10,
                    use_cache=True,
                )

            if breadth_df.empty:
                st.warning("No breadth data available. Click 'Refresh All Data' to fetch.")
            else:
                # Data info
                latest_date = breadth_df["date"].max()
                st.caption(f"Data as of: {pd.Timestamp(latest_date).strftime('%Y-%m-%d')}")

                # ------------------------------------------------------------------
                # 0. Breadth Summary (all indices at a glance)
                # ------------------------------------------------------------------
                st.markdown("### Breadth Summary")

                # Fetch Finviz current data for comparison
                finviz_data = fetch_finviz_breadth()

                summary_df = prepare_breadth_summary(breadth_df, finviz_data)
                if not summary_df.empty:
                    def color_summary_row(row):
                        styles = [""] * len(row)
                        # Color A/D Ratio
                        ad_ratio = row.get("A/D Ratio", 1)
                        if pd.notna(ad_ratio):
                            if ad_ratio > 1.5:
                                styles[3] = "background-color: rgba(46, 204, 113, 0.5)"
                            elif ad_ratio > 1.0:
                                styles[3] = "background-color: rgba(46, 204, 113, 0.2)"
                            elif ad_ratio < 0.67:
                                styles[3] = "background-color: rgba(231, 76, 60, 0.5)"
                            elif ad_ratio < 1.0:
                                styles[3] = "background-color: rgba(231, 76, 60, 0.2)"
                        # Color % > 200 DMA
                        pct_200 = row.get("% > 200 DMA")
                        if pd.notna(pct_200):
                            if pct_200 >= 70:
                                styles[5] = "background-color: rgba(46, 204, 113, 0.5)"
                            elif pct_200 >= 50:
                                styles[5] = "background-color: rgba(46, 204, 113, 0.2)"
                            elif pct_200 < 40:
                                styles[5] = "background-color: rgba(231, 76, 60, 0.4)"
                        return styles

                    styled_summary = summary_df.style.apply(color_summary_row, axis=1)
                    st.dataframe(styled_summary, width="stretch", hide_index=True)

                    # Add Finviz timestamp if available
                    if finviz_data and finviz_data.get("timestamp"):
                        st.caption(f"Finviz data: {finviz_data.get('timestamp', 'N/A')[:19]}")

                st.divider()

                # ------------------------------------------------------------------
                # 1. Advancers & Decliners Heat Map
                # ------------------------------------------------------------------
                st.markdown("### Advancers & Decliners")
                with st.expander("How to interpret", expanded=False):
                    st.markdown("""
**What it shows:** Count of stocks that closed higher (Advancers) or lower (Decliners) vs prior day.

**Interpretation:**
- **Strong day:** Advancers significantly outnumber Decliners (e.g., 400+ vs 100)
- **Weak day:** Decliners outnumber Advancers
- **Healthy market:** Consistently more advancers over time
- **Distribution:** Rising index with increasing decliners = warning sign
                    """)

                adv_dec_df = prepare_heatmap_data(breadth_df, "advancers")
                if not adv_dec_df.empty:
                    numeric_cols = [c for c in adv_dec_df.columns if c not in ["Index", "Metric"]]

                    def color_advancers_decliners(row):
                        styles = [""] * len(row)
                        metric = row.get("Metric", "")
                        for i, col in enumerate(row.index):
                            if col in numeric_cols:
                                val = row[col]
                                if pd.notna(val):
                                    if metric == "Advancers":
                                        intensity = min(val / 400, 1.0) * 0.6
                                        styles[i] = f"background-color: rgba(46, 204, 113, {intensity})"
                                    elif metric == "Decliners":
                                        intensity = min(val / 400, 1.0) * 0.6
                                        styles[i] = f"background-color: rgba(231, 76, 60, {intensity})"
                        return styles

                    styled_adv = adv_dec_df.style.apply(color_advancers_decliners, axis=1)
                    st.dataframe(styled_adv, width="stretch", hide_index=True)
                else:
                    st.info("No advancers/decliners data available.")

                st.divider()

                # ------------------------------------------------------------------
                # 2. Moving Average Heat Map
                # ------------------------------------------------------------------
                st.markdown("### % Above Moving Averages")
                with st.expander("How to interpret", expanded=False):
                    st.markdown("""
**What it shows:** Percentage of stocks trading above their 20, 50, 100, and 200-day moving averages.

**Thresholds:**
| Level | 20-Day | 50-Day | 200-Day | Interpretation |
|-------|--------|--------|---------|----------------|
| **Strong** | >70% | >70% | >70% | Healthy uptrend |
| **Neutral** | 40-70% | 40-70% | 40-70% | Mixed conditions |
| **Weak** | <40% | <40% | <40% | Bearish, oversold |

**Key insight:** Compare short-term (20-day) to long-term (200-day):
- 20-day much higher = near-term overbought, pullback likely
- 20-day much lower = oversold bounce opportunity
- All aligned high = strong bull market
                    """)

                ma_df = prepare_heatmap_data(breadth_df, "ma")
                if not ma_df.empty:
                    numeric_cols = [c for c in ma_df.columns if c not in ["Index", "Metric"]]

                    def color_ma_pct(row):
                        styles = [""] * len(row)
                        for i, col in enumerate(row.index):
                            if col in numeric_cols:
                                val = row[col]
                                if pd.notna(val):
                                    intensity = (val / 100) * 0.7
                                    if val >= 50:
                                        styles[i] = f"background-color: rgba(46, 204, 113, {intensity})"
                                    else:
                                        styles[i] = f"background-color: rgba(231, 76, 60, {(1 - val/100) * 0.7})"
                        return styles

                    styled_ma = ma_df.style.apply(color_ma_pct, axis=1)
                    st.dataframe(styled_ma, width="stretch", hide_index=True)
                else:
                    st.info("No moving average data available.")

                st.divider()

                # ------------------------------------------------------------------
                # 3. Golden Cross (50-Day > 200-Day MA)
                # ------------------------------------------------------------------
                st.markdown("### Golden Cross Breadth (50-Day > 200-Day MA)")
                with st.expander("How to interpret", expanded=False):
                    st.markdown("""
**What it shows:** Percentage of stocks with their 50-day MA above their 200-day MA (a "Golden Cross").

**Why it matters:** The Golden Cross is a classic bullish signal for individual stocks.
Measuring it across all index constituents shows overall trend strength.

**Thresholds:**
- **>70%:** Strong bull market, most stocks in uptrends
- **50-70%:** Healthy, but some sectors lagging
- **<50%:** Bearish undertone, fewer stocks in sustainable uptrends
- **<30%:** Bear market conditions

**Trend:** Rising % = improving breadth, Falling % = deteriorating conditions
                    """)

                gc_df = prepare_heatmap_data(breadth_df, "golden_cross")
                if not gc_df.empty:
                    numeric_cols = [c for c in gc_df.columns if c not in ["Index", "Metric"]]

                    def color_golden_cross(row):
                        styles = [""] * len(row)
                        for i, col in enumerate(row.index):
                            if col in numeric_cols:
                                val = row[col]
                                if pd.notna(val):
                                    if val >= 70:
                                        styles[i] = "background-color: rgba(46, 204, 113, 0.6)"
                                    elif val >= 50:
                                        styles[i] = "background-color: rgba(46, 204, 113, 0.3)"
                                    elif val >= 30:
                                        styles[i] = "background-color: rgba(243, 156, 18, 0.4)"
                                    else:
                                        styles[i] = "background-color: rgba(231, 76, 60, 0.5)"
                        return styles

                    styled_gc = gc_df.style.apply(color_golden_cross, axis=1)
                    st.dataframe(styled_gc, width="stretch", hide_index=True)
                else:
                    st.info("Golden Cross data not available. Refresh to compute.")

                st.divider()

                # ------------------------------------------------------------------
                # 4. Trend Count Model
                # ------------------------------------------------------------------
                st.markdown("### Trend Count Model (4 Criteria)")
                with st.expander("How to interpret", expanded=False):
                    st.markdown("""
**What it shows:** Counts of stocks meeting ALL 4 or NONE of 4 trend criteria:

**The 4 Criteria:**
1. 50-Day MA slope is rising (MA today > MA 5 days ago)
2. 200-Day MA slope is rising
3. Price > 50-Day MA
4. 50-Day MA > 200-Day MA

**Interpretation:**
- **4 of 4 (green):** Stocks in strong uptrends. High counts = healthy bull market
- **0 of 4 (red):** Stocks in strong downtrends. High counts = bear market conditions

**Typical ranges:**
- Bull market: 150-250 stocks at 4/4, <50 at 0/4
- Bear market: <50 stocks at 4/4, 150+ at 0/4
- Transition: Both counts moderate (50-100 each)
                    """)

                tc_df = prepare_heatmap_data(breadth_df, "trend_count")
                if not tc_df.empty:
                    numeric_cols = [c for c in tc_df.columns if c not in ["Index", "Metric"]]

                    def color_trend_count(row):
                        styles = [""] * len(row)
                        metric = row.get("Metric", "")
                        is_bullish = "4 of 4" in metric
                        for i, col in enumerate(row.index):
                            if col in numeric_cols:
                                val = row[col]
                                if pd.notna(val):
                                    if is_bullish:
                                        intensity = min(val / 200, 1.0) * 0.6
                                        styles[i] = f"background-color: rgba(46, 204, 113, {intensity})"
                                    else:
                                        intensity = min(val / 150, 1.0) * 0.6
                                        styles[i] = f"background-color: rgba(231, 76, 60, {intensity})"
                        return styles

                    styled_tc = tc_df.style.apply(color_trend_count, axis=1)
                    st.dataframe(styled_tc, width="stretch", hide_index=True)
                else:
                    st.info("Trend count data not available. Refresh to compute.")

                st.divider()

                # ------------------------------------------------------------------
                # 5. New Highs & Lows Heat Map
                # ------------------------------------------------------------------
                st.markdown("### New Highs & New Lows")
                with st.expander("How to interpret", expanded=False):
                    st.markdown("""
**What it shows:** Percentage of stocks at N-period highs or lows (1, 3, 6, 12 months).

**Interpretation:**
- **Highs > Lows:** Bullish, more stocks breaking out than breaking down
- **Lows > Highs:** Bearish, distribution underway
- **52-week highs >15%:** Strong momentum
- **52-week lows >15%:** Panic/capitulation (often a contrarian buy signal)

**Divergence signals:**
- Index at new high but fewer stocks at highs = negative divergence (warning)
- Index flat but new highs expanding = positive divergence (bullish)
                    """)

                hl_df = prepare_heatmap_data(breadth_df, "highs_lows")
                if not hl_df.empty:
                    numeric_cols = [c for c in hl_df.columns if c not in ["Index", "Metric"]]

                    def color_highs_lows(row):
                        styles = [""] * len(row)
                        metric = row.get("Metric", "")
                        is_high = "Highs" in metric
                        for i, col in enumerate(row.index):
                            if col in numeric_cols:
                                val = row[col]
                                if pd.notna(val):
                                    intensity = min(val / 40, 1.0) * 0.7
                                    if is_high:
                                        styles[i] = f"background-color: rgba(46, 204, 113, {intensity})"
                                    else:
                                        styles[i] = f"background-color: rgba(231, 76, 60, {intensity})"
                        return styles

                    styled_hl = hl_df.style.apply(color_highs_lows, axis=1)
                    st.dataframe(styled_hl, width="stretch", hide_index=True, height=500)
                else:
                    st.info("No highs/lows data available.")

                st.divider()

                # ------------------------------------------------------------------
                # 6. Overbought/Oversold Heat Map
                # ------------------------------------------------------------------
                st.markdown("### Overbought / Oversold (RSI-based)")
                with st.expander("How to interpret", expanded=False):
                    st.markdown("""
**What it shows:** Percentage of stocks with 14-day RSI above 70 (overbought) or below 30 (oversold).

**Interpretation:**
- **>25% Overbought:** Market running hot, pullback risk elevated
- **>20% Oversold:** Panic selling, bounce likely
- **Both low (<10%):** Consolidation, no extreme

**Contrarian signals:**
- Extreme oversold (>30% of stocks) often marks bottoms
- Extreme overbought after a rally can persist in strong trends
- In bear markets, oversold can stay oversold longer

**Note:** Overbought is NOT bearish by itself - strong markets can stay overbought.
It's more useful as a timing tool for entries after pullbacks.
                    """)

                ob_df = prepare_heatmap_data(breadth_df, "overbought")
                if not ob_df.empty:
                    numeric_cols = [c for c in ob_df.columns if c not in ["Index", "Metric"]]

                    def color_overbought_oversold(row):
                        styles = [""] * len(row)
                        metric = row.get("Metric", "")
                        for i, col in enumerate(row.index):
                            if col in numeric_cols:
                                val = row[col]
                                if pd.notna(val):
                                    intensity = min(val / 30, 1.0) * 0.7
                                    if "Overbought" in metric:
                                        styles[i] = f"background-color: rgba(46, 204, 113, {intensity})"
                                    else:
                                        styles[i] = f"background-color: rgba(231, 76, 60, {intensity})"
                        return styles

                    styled_ob = ob_df.style.apply(color_overbought_oversold, axis=1)
                    st.dataframe(styled_ob, width="stretch", hide_index=True)
                else:
                    st.info("No overbought/oversold data available.")

                st.divider()

                # ------------------------------------------------------------------
                # 7. 5-Year Breadth Trends (Time-Series Charts)
                # ------------------------------------------------------------------
                st.markdown("### 5-Year Breadth Trends")
                with st.expander("About these charts", expanded=False):
                    st.markdown("""
**5-Year Historical Breadth Charts**

These charts show breadth metrics over the past 5 years, providing longer-term context
for current readings. Look for:

- **A/D Line divergences:** Index making new highs while A/D line fails to confirm
- **% Above MA trends:** Long-term health of the market
- **New Highs vs Lows:** Expansion or contraction of leadership

**Note:** Initial calculation may take 2-3 minutes per index. Data is cached for 24 hours.
                    """)

                # Index selector for time-series
                ts_index = st.selectbox(
                    "Select Index for Time-Series Charts",
                    options=["SP500", "SP400", "SP600"],
                    index=0,
                    help="Choose which index to display 5-year breadth history"
                )

                ts_col1, ts_col2 = st.columns([1, 1])
                with ts_col1:
                    if st.button("Load 5-Year Data", key="load_ts_data"):
                        with st.spinner(f"Computing 5-year breadth for {ts_index} (2-3 min)..."):
                            ts_data = fetch_breadth_timeseries(
                                index_name=ts_index,
                                years=5,
                                use_cache=False,
                                force_refresh=True,
                            )
                            st.session_state[f"breadth_ts_{ts_index}"] = ts_data
                        st.rerun()

                # Try to load from session state or cache
                ts_key = f"breadth_ts_{ts_index}"
                if ts_key not in st.session_state:
                    # Try loading from cache (no computation)
                    ts_cache = fetch_breadth_timeseries(
                        index_name=ts_index,
                        years=5,
                        use_cache=True,
                        force_refresh=False,
                    )
                    if not ts_cache.empty:
                        st.session_state[ts_key] = ts_cache

                if ts_key in st.session_state and not st.session_state[ts_key].empty:
                    ts_data = st.session_state[ts_key]
                    st.caption(f"Showing {len(ts_data)} days of {ts_index} breadth history")

                    # Cumulative A/D Line chart
                    ad_chart = create_cumulative_ad_chart(ts_data, title=f"{ts_index} Cumulative A/D Line")
                    st.plotly_chart(ad_chart, use_container_width=True)

                    # % Above MA chart
                    ma_chart = create_pct_above_ma_timeseries(ts_data, title=f"{ts_index} % Above Moving Averages")
                    st.plotly_chart(ma_chart, use_container_width=True)

                    # New Highs vs Lows chart
                    hl_chart = create_new_highs_lows_timeseries(ts_data, title=f"{ts_index} New Highs vs Lows")
                    st.plotly_chart(hl_chart, use_container_width=True)
                else:
                    st.info("Click 'Load 5-Year Data' to compute and display historical breadth charts.")

                # Info
                st.divider()
                indices_loaded = breadth_df["index"].unique().tolist()
                st.caption(f"Indices loaded: {', '.join(indices_loaded)}. Uncheck boxes above to exclude indices.")

        except Exception as e:
            st.error(f"Error loading breadth data: {e}")
            import traceback
            st.code(traceback.format_exc())

    with tab9:
        st.subheader("Sector Scorecard")

        with st.expander("Understanding the Scorecard", expanded=False):
            st.markdown("""
**What is the Sector Scorecard?**
A letter-grade ranking of S&P sectors based on three models:

**The 3 Models:**
1. **Trend Model:** Is the sector in an uptrend? (Price vs MAs, MA slopes)
2. **Relative Strength:** Is the sector outperforming SPY? (Ratio trend)
3. **Momentum Model:** Is momentum positive? (Rate of change, RSI)

**Grade Scale:**
| Grade | Score | Meaning |
|-------|-------|---------|
| **A+** | All A's | Elite - strong across all measures |
| **A** | Mostly A's | Strong - favor for overweight |
| **B** | Mixed A/B | Neutral-positive - market weight |
| **C** | Mixed B/C | Neutral-negative - caution |
| **D** | Mostly D/F | Weak - underweight |
| **F** | All F's | Avoid - defensive sectors may do this in rallies |

**How to use:**
- **Overweight:** A+ and A sectors
- **Market weight:** B sectors
- **Underweight:** C, D, F sectors
- Watch for grade changes (improving or deteriorating)
            """)

        # Sector ETF definitions with their comparison ratio columns
        # Most sectors are compared to XLU (utilities) as risk-on vs defensive
        sector_etfs = {
            "Technology": {"etf": "XLK", "ratio": "XLK_XLU"},
            "Financials": {"etf": "XLF", "ratio": "XLF_XLU"},
            "Industrials": {"etf": "XLI", "ratio": "XLI_XLU"},
            "Consumer Discretionary": {"etf": "XLY", "ratio": "XLY_XLP"},
            "Consumer Staples": {"etf": "XLP", "ratio": "XLY_XLP", "invert": True},
            "Energy": {"etf": "XLE", "ratio": "USO_GLD"},
            "Utilities": {"etf": "XLU", "ratio": "XLU_SPY"},
            "Materials": {"etf": "XLB", "ratio": "XME_GLD"},
            "Health Care": {"etf": "XLV", "ratio": "IBB_SPY"},
            "Real Estate": {"etf": "XLRE", "ratio": "KRE_VNQ"},
            "Communication Services": {"etf": "XLC", "ratio": "QQQ_SPY"},
        }

        # Calculate sector scores from features data
        if data.get("features") is not None and not data["features"].empty:
            latest_features = data["features"].iloc[-1]
            scorecard_data = []

            # Load thresholds from config
            dashboard_config = load_dashboard_config()
            scorecard_config = dashboard_config.get("sector_scorecard", {})
            trend_thresholds = scorecard_config.get("trend_thresholds", [0.5, 0, -0.3, -0.7])
            rs_thresholds = scorecard_config.get("relative_strength_thresholds", [0.7, 0.2, -0.2, -0.5])
            mom_thresholds = scorecard_config.get("momentum_thresholds", [3, 1, -1, -3])

            # Convert to grades helper
            def score_to_grade(score, thresholds):
                if score > thresholds[0]:
                    return "A"
                elif score > thresholds[1]:
                    return "B"
                elif score > thresholds[2]:
                    return "C"
                elif score > thresholds[3]:
                    return "D"
                else:
                    return "F"

            for sector_name, info in sector_etfs.items():
                etf = info["etf"]
                ratio = info["ratio"]
                invert = info.get("invert", False)

                # Look for sector-related features (use double underscore)
                z_col = f"{ratio}__z_52w"
                roc_col = f"{ratio}__roc_8w"

                z_score = latest_features.get(z_col, 0) if z_col in latest_features.index else 0
                roc = latest_features.get(roc_col, 0) if roc_col in latest_features.index else 0

                # Invert if needed (e.g., Consumer Staples is inverse of XLY/XLP)
                if invert:
                    z_score = -z_score
                    roc = -roc

                # Trend model (based on z-score)
                trend_grade = score_to_grade(z_score, trend_thresholds)

                # Relative Strength (based on z-score direction)
                rs_grade = score_to_grade(z_score, rs_thresholds)

                # Momentum (based on rate of change)
                mom_grade = score_to_grade(roc * 100, mom_thresholds)

                # Overall grade
                grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
                avg_score = (grade_values[trend_grade] + grade_values[rs_grade] + grade_values[mom_grade]) / 3

                if avg_score >= 3.5:
                    overall = "A+"
                elif avg_score >= 3.0:
                    overall = "A"
                elif avg_score >= 2.5:
                    overall = "B"
                elif avg_score >= 1.5:
                    overall = "C"
                elif avg_score >= 0.5:
                    overall = "D"
                else:
                    overall = "F"

                scorecard_data.append({
                    "Sector": sector_name,
                    "ETF": etf,
                    "Trend": trend_grade,
                    "Rel Strength": rs_grade,
                    "Momentum": mom_grade,
                    "Overall": overall,
                    "Z-Score": round(z_score, 2),
                    "ROC": round(roc * 100, 1),
                })

            scorecard_df = pd.DataFrame(scorecard_data)

            # Sort by overall grade
            grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3, "D": 4, "F": 5}
            scorecard_df["sort_key"] = scorecard_df["Overall"].map(grade_order)
            scorecard_df = scorecard_df.sort_values("sort_key").drop("sort_key", axis=1)

            # Color styling function
            def color_grades(val):
                if val in ["A+", "A"]:
                    return "background-color: rgba(46, 204, 113, 0.5); color: black; font-weight: bold"
                elif val == "B":
                    return "background-color: rgba(46, 204, 113, 0.2); color: black"
                elif val == "C":
                    return "background-color: rgba(243, 156, 18, 0.3); color: black"
                elif val == "D":
                    return "background-color: rgba(231, 76, 60, 0.3); color: black"
                elif val == "F":
                    return "background-color: rgba(231, 76, 60, 0.5); color: black; font-weight: bold"
                return ""

            # Apply styling
            styled_scorecard = scorecard_df.style.map(
                color_grades,
                subset=["Trend", "Rel Strength", "Momentum", "Overall"]
            )

            st.dataframe(styled_scorecard, width="stretch", hide_index=True)

            # Legend
            st.markdown("""
**Grade Colors:**
<span style='background-color: rgba(46, 204, 113, 0.5); padding: 2px 8px;'>A/A+</span>
<span style='background-color: rgba(46, 204, 113, 0.2); padding: 2px 8px;'>B</span>
<span style='background-color: rgba(243, 156, 18, 0.3); padding: 2px 8px;'>C</span>
<span style='background-color: rgba(231, 76, 60, 0.3); padding: 2px 8px;'>D</span>
<span style='background-color: rgba(231, 76, 60, 0.5); padding: 2px 8px;'>F</span>
            """, unsafe_allow_html=True)

        else:
            st.warning("Feature data not available for sector scoring. Run the pipeline first.")

    with tab10:
        st.subheader("Treasury Tax Flow")

        with st.expander("Understanding Treasury Tax Flow (Deluard Methodology)", expanded=False):
            st.markdown("""
**What is Treasury Tax Flow Analysis?**

This tab implements Vincent Deluard's methodology for tracking Daily Treasury Statement (DTS)
tax collection data as real-time economic indicators.

**Why Tax Deposits Matter:**
- **Hard cash:** Cannot be revised like GDP or employment data
- **Real-time:** Updated daily by 4 PM ET
- **Comprehensive:** Covers all taxpayers (not a sample)
- **Leading:** Often leads official macro data by weeks

**Tax Categories:**
| Category | Economic Proxy |
|----------|----------------|
| **Withheld Income & Employment** | Labor market health, wage growth |
| **Corporate Income Taxes** | Corporate profit cycle (quarterly spikes) |
| **Non-Withheld/Self-Employment** | Gig economy, small business activity |
| **Total Federal Tax Deposits** | Aggregate economic signal |

**Interpretation:**
- **YoY Growth > 5%:** Strong economic conditions
- **YoY Growth 0-5%:** Moderate growth
- **YoY Growth < 0%:** Economic weakness, potential leading indicator of slowdown

**YTD Comparison:**
Compare cumulative deposits this year vs last year on the same calendar day.
Gap widening = accelerating growth; Gap narrowing = decelerating growth.

**Data Source:** U.S. Treasury Fiscal Data API (free, no key required)
            """)

        # Load treasury data (5 years for cumulative chart)
        try:
            with st.spinner("Loading treasury data..."):
                treasury_data = prepare_treasury_indicators(years=5, use_cache=True)

            if treasury_data["status"] == "failed":
                st.error("Failed to load treasury data. Check network connection or try again later.")
            else:
                # Data freshness indicator
                latest_date = treasury_data.get("latest_date")
                if latest_date:
                    days_old = (pd.Timestamp.now() - latest_date).days
                    if days_old <= 1:
                        freshness_color = "green"
                        freshness_label = "Current"
                    elif days_old <= 3:
                        freshness_color = "orange"
                        freshness_label = "Recent"
                    else:
                        freshness_color = "red"
                        freshness_label = "Stale"

                    st.markdown(
                        f"**Data as of:** {latest_date.strftime('%Y-%m-%d')} "
                        f"<span style='color:{freshness_color}'>({freshness_label})</span>",
                        unsafe_allow_html=True
                    )

                # Control panel
                st.markdown("### Controls")
                col1, col2 = st.columns(2)
                with col1:
                    rolling_window = st.selectbox(
                        "Smoothing Window",
                        options=[7, 28, 63],
                        index=1,
                        format_func=lambda x: {7: "1 Week", 28: "1 Month", 63: "1 Quarter"}[x],
                        help="Rolling window for smoothing daily noise"
                    )
                with col2:
                    selected_categories = st.multiselect(
                        "Tax Categories",
                        options=["withheld_yoy", "corporate_yoy", "non_withheld_yoy", "total_yoy"],
                        default=["withheld_yoy", "total_yoy"],
                        format_func=lambda x: {
                            "withheld_yoy": "Withheld Income",
                            "corporate_yoy": "Corporate",
                            "non_withheld_yoy": "Non-Withheld (Gig)",
                            "total_yoy": "Total"
                        }[x],
                    )

                st.divider()

                # Chart 1: YoY Growth by Category
                st.markdown("### Year-over-Year Growth by Tax Category")
                st.caption(
                    "**YoY growth** compares tax deposits to the same period last year. "
                    "Positive growth indicates economic expansion; negative growth suggests contraction. "
                    "Corporate taxes show quarterly spikes due to estimated payment deadlines."
                )
                yoy_df = treasury_data.get("yoy", pd.DataFrame())
                if not yoy_df.empty:
                    fig_yoy = create_tax_yoy_chart(
                        yoy_df,
                        categories=selected_categories if selected_categories else None,
                        title="Tax Deposit YoY Growth"
                    )
                    st.plotly_chart(fig_yoy, use_container_width=True)
                else:
                    st.info("YoY data not available. Refresh to compute.")

                st.divider()

                # Chart 2: YTD Cumulative Comparison
                st.markdown("### YTD Cumulative: This Year vs Last Year")
                st.caption(
                    "**YTD cumulative** tracks total tax deposits from January 1st to today, "
                    "compared to the same period last year. A widening gap above last year "
                    "indicates accelerating economic growth; a narrowing gap suggests deceleration."
                )
                ytd_comp_df = treasury_data.get("ytd_comparison", pd.DataFrame())

                ytd_category = st.selectbox(
                    "Select Category for YTD Comparison",
                    options=["withheld", "corporate", "non_withheld", "total"],
                    index=0,
                    format_func=lambda x: {
                        "withheld": "Withheld Income & Employment",
                        "corporate": "Corporate Income Taxes",
                        "non_withheld": "Non-Withheld (Gig Economy)",
                        "total": "Total Federal Deposits"
                    }[x],
                )

                if not ytd_comp_df.empty:
                    fig_ytd = create_ytd_comparison_chart(ytd_comp_df, category=ytd_category)
                    st.plotly_chart(fig_ytd, use_container_width=True)
                else:
                    st.info("YTD comparison data not available.")

                st.divider()

                # Chart 3: Gig Economy Indicator
                st.markdown("### Gig Economy Indicator")
                st.caption("Non-withheld taxes serve as a proxy for gig economy and self-employment activity.")
                raw_df = treasury_data.get("raw", pd.DataFrame())
                if not raw_df.empty:
                    fig_gig = create_gig_economy_chart(raw_df, rolling_window=rolling_window)
                    st.plotly_chart(fig_gig, use_container_width=True)
                else:
                    st.info("Gig economy data not available.")

                st.divider()

                # Chart 4: Tax Receipts vs SPY
                st.markdown("### Tax Receipts vs S&P 500")
                st.caption(
                    "**Leading indicator:** Changes in tax deposit growth often precede "
                    "equity market moves. Divergences between tax receipts and SPY can signal "
                    "upcoming market turning points."
                )

                try:
                    spy_prices = load_spy_prices(align_to_weekly=False)
                    if not spy_prices.empty and not raw_df.empty:
                        fig_spy = create_tax_vs_spy_chart(
                            raw_df,
                            spy_prices,
                            category="total",
                            rolling_window=rolling_window,
                        )
                        st.plotly_chart(fig_spy, use_container_width=True)
                    else:
                        st.info("SPY price data not available for overlay.")
                except Exception as e:
                    st.warning(f"Could not load SPY data: {e}")

                st.divider()

                # Chart 5: 5-Year Cumulative
                st.markdown("### 5-Year Cumulative Tax Deposits")
                st.caption(
                    "**Long-term trend:** Cumulative tax deposits since the start of the data series. "
                    "The slope indicates collection velocity - steeper slopes mean faster economic activity. "
                    "Flattening or declining slopes can signal economic slowdowns."
                )
                if not raw_df.empty:
                    from risk_index.reporting.charts import create_cumulative_tax_chart
                    fig_cumulative = create_cumulative_tax_chart(
                        raw_df,
                        categories=["withheld", "corporate", "non_withheld", "total"],
                        title="Cumulative Federal Tax Deposits"
                    )
                    st.plotly_chart(fig_cumulative, use_container_width=True)
                else:
                    st.info("Cumulative data not available.")

                # Refresh button
                st.divider()
                if st.button("Refresh Treasury Data"):
                    with st.spinner("Fetching fresh treasury data (may take 30-60 seconds)..."):
                        treasury_data = prepare_treasury_indicators(years=5, use_cache=False)
                    st.rerun()

        except Exception as e:
            st.error(f"Error loading treasury data: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Data updates weekly after Friday close. "
        "Run `python scripts/run_update.py` to refresh."
    )


if __name__ == "__main__":
    main()
