"""Excel export module for analysis workbooks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_index.core.config_schema import ChecklistConfig, UniverseConfig
from risk_index.core.constants import (
    EXPORTS_DIR,
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
from risk_index.core.utils_io import write_excel, ensure_dir
from risk_index.core.logger import get_logger
from risk_index.pipeline.regimes_build import get_regime_history

logger = get_logger(__name__)


def export_all_excel(
    weekly_df: pd.DataFrame,
    features_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    composites_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    checklist_df: pd.DataFrame,
    universe_cfg: UniverseConfig,
    checklist_cfg: ChecklistConfig,
    exports_dir: Path = EXPORTS_DIR,
) -> list[Path]:
    """Export all Excel workbooks.


    Args:
        weekly_df: Weekly aligned data
        features_df: Features DataFrame
        blocks_df: Blocks DataFrame
        composites_df: Composites DataFrame
        regimes_df: Regimes DataFrame
        checklist_df: Checklist DataFrame
        universe_cfg: Universe configuration
        checklist_cfg: Checklist configuration
        exports_dir: Export directory

    Returns:
        List of exported file paths
    """
    ensure_dir(exports_dir)
    exported = []

    # 1. Weekly data
    path = export_weekly_data(weekly_df, universe_cfg, exports_dir)
    exported.append(path)

    # 2. Block scores
    path = export_block_scores(blocks_df, exports_dir)
    exported.append(path)

    # 3. Composites and regimes
    path = export_composites(composites_df, regimes_df, exports_dir)
    exported.append(path)

    # 4. Checklist
    path = export_checklist(checklist_df, checklist_cfg, exports_dir)
    exported.append(path)

    # 5. Summary dashboard
    path = export_summary_dashboard(
        weekly_df, blocks_df, composites_df, regimes_df, checklist_df, exports_dir
    )
    exported.append(path)

    logger.info(f"Exported {len(exported)} Excel files to {exports_dir}")
    return exported


def export_weekly_data(
    weekly_df: pd.DataFrame,
    universe_cfg: UniverseConfig,
    exports_dir: Path,
) -> Path:
    """Export weekly data workbook.

    Sheets:
    - Prices: All weekly price series
    - Ratios: All computed ratios
    - Features: All transformed features

    Args:
        weekly_df: Weekly aligned DataFrame
        universe_cfg: Universe config
        exports_dir: Export directory

    Returns:
        Path to exported file
    """
    sheets = {}

    # Prices sheet - base series only
    base_series_ids = [s.id for s in universe_cfg.series]
    price_cols = [c for c in weekly_df.columns if c in base_series_ids]
    if price_cols:
        sheets["Prices"] = weekly_df[price_cols]

    # Ratios sheet
    ratio_ids = [r.id for r in universe_cfg.ratios]
    ratio_cols = [c for c in weekly_df.columns if c in ratio_ids]
    if ratio_cols:
        sheets["Ratios"] = weekly_df[ratio_cols]

    # Features sheet - transformed features
    feature_cols = [c for c in weekly_df.columns if "__" in c]
    if feature_cols:
        sheets["Features"] = weekly_df[feature_cols]

    path = exports_dir / "weekly_data_latest.xlsx"
    write_excel(sheets, path)

    return path


def export_block_scores(
    blocks_df: pd.DataFrame,
    exports_dir: Path,
) -> Path:
    """Export block scores workbook.

    Sheets:
    - Blocks: All block scores over time
    - Current: Latest block values with rankings

    Args:
        blocks_df: Blocks DataFrame
        exports_dir: Export directory

    Returns:
        Path to exported file
    """
    sheets = {}

    # Historical block scores
    sheets["Blocks"] = blocks_df

    # Current snapshot with rankings
    if not blocks_df.empty:
        latest = blocks_df.iloc[-1].to_frame(name="Score")
        latest["Rank"] = latest["Score"].rank(ascending=False)
        latest = latest.sort_values("Rank")
        sheets["Current"] = latest

    path = exports_dir / "block_scores_latest.xlsx"
    write_excel(sheets, path)

    return path


def export_composites(
    composites_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    exports_dir: Path,
) -> Path:
    """Export composites workbook.

    Sheets:
    - Composites: Fast/Med/Slow over time
    - Regimes: Regime labels + confidence
    - Transitions: Regime change log with dates

    Args:
        composites_df: Composites DataFrame
        regimes_df: Regimes DataFrame
        exports_dir: Export directory

    Returns:
        Path to exported file
    """
    sheets = {}

    # Composites
    composite_cols = [
        c for c in composites_df.columns if c.startswith("composite_")
    ]
    if composite_cols:
        sheets["Composites"] = composites_df[composite_cols]

    # Regimes with confidence
    regime_cols = []
    for speed in ["fast", "medium", "slow"]:
        regime_col = f"regime_{speed}"
        conf_col = f"confidence_{speed}"
        if regime_col in regimes_df.columns:
            regime_cols.append(regime_col)
        if conf_col in regimes_df.columns:
            regime_cols.append(conf_col)

    if regime_cols:
        sheets["Regimes"] = regimes_df[regime_cols]

    # Transitions
    transitions = get_regime_history(regimes_df)
    if not transitions.empty:
        sheets["Transitions"] = transitions

    path = exports_dir / "composites_latest.xlsx"
    write_excel(sheets, path)

    return path


def export_checklist(
    checklist_df: pd.DataFrame,
    checklist_cfg: ChecklistConfig,
    exports_dir: Path,
) -> Path:
    """Export checklist workbook.

    Sheets:
    - History: All checklist items over time
    - Current: Latest item statuses with weights
    - Score: Aggregate score history

    Args:
        checklist_df: Checklist DataFrame
        checklist_cfg: Checklist config
        exports_dir: Export directory

    Returns:
        Path to exported file
    """
    sheets = {}

    # Item history - signal columns
    signal_cols = [c for c in checklist_df.columns if c.endswith("_signal")]
    if signal_cols:
        sheets["History"] = checklist_df[signal_cols]

    # Current snapshot
    if not checklist_df.empty:
        latest_date = checklist_df.index[-1]
        current_data = []

        for item in checklist_cfg.items:
            signal_col = f"{item.id}_signal"
            score_col = f"{item.id}_score"

            row = {
                "Item": item.name,
                "Category": item.category,
                "Weight": item.weight,
            }

            if signal_col in checklist_df.columns:
                row["Signal"] = checklist_df.loc[latest_date, signal_col]

            if score_col in checklist_df.columns:
                row["Score"] = checklist_df.loc[latest_date, score_col]

            current_data.append(row)

        sheets["Current"] = pd.DataFrame(current_data)

    # Score history
    score_cols = [COL_CHECKLIST_SCORE, COL_CHECKLIST_LABEL]
    score_cols = [c for c in score_cols if c in checklist_df.columns]
    if score_cols:
        sheets["Score"] = checklist_df[score_cols]

    path = exports_dir / "checklist_latest.xlsx"
    write_excel(sheets, path)

    return path


def export_summary_dashboard(
    weekly_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    composites_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    checklist_df: pd.DataFrame,
    exports_dir: Path,
) -> Path:
    """Export summary dashboard workbook.

    Sheets:
    - Latest: Current snapshot of all key metrics
    - Statistics: Descriptive stats for all series

    Args:
        weekly_df: Weekly data
        blocks_df: Blocks DataFrame
        composites_df: Composites DataFrame
        regimes_df: Regimes DataFrame
        checklist_df: Checklist DataFrame
        exports_dir: Export directory

    Returns:
        Path to exported file
    """
    sheets = {}

    # Latest snapshot
    if not composites_df.empty:
        latest_date = composites_df.index[-1]
        latest = {"Date": latest_date.strftime("%Y-%m-%d")}

        # Composites
        for speed in ["fast", "medium", "slow"]:
            col = f"composite_{speed}"
            if col in composites_df.columns:
                latest[f"{speed.capitalize()} Composite"] = round(
                    composites_df.loc[latest_date, col], 3
                )

        # Regimes
        for speed in ["fast", "medium", "slow"]:
            regime_col = f"regime_{speed}"
            conf_col = f"confidence_{speed}"
            if regime_col in regimes_df.columns:
                latest[f"{speed.capitalize()} Regime"] = regimes_df.loc[latest_date, regime_col]
            if conf_col in regimes_df.columns:
                latest[f"{speed.capitalize()} Confidence"] = round(
                    regimes_df.loc[latest_date, conf_col], 2
                )

        # Checklist
        if COL_CHECKLIST_SCORE in checklist_df.columns:
            latest["Checklist Score"] = round(
                checklist_df.loc[latest_date, COL_CHECKLIST_SCORE], 1
            )
        if COL_CHECKLIST_LABEL in checklist_df.columns:
            latest["Checklist Label"] = checklist_df.loc[latest_date, COL_CHECKLIST_LABEL]

        sheets["Latest"] = pd.DataFrame([latest]).T
        sheets["Latest"].columns = ["Value"]

    # Statistics
    stats_df = weekly_df.describe().T
    stats_df["null_pct"] = weekly_df.isna().mean()
    sheets["Statistics"] = stats_df

    path = exports_dir / "summary_dashboard_latest.xlsx"
    write_excel(sheets, path)

    return path
