"""Data cleaning and weekly alignment module."""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from risk_index.core.config_schema import UniverseConfig, TransformsConfig
from risk_index.core.constants import WEEK_ANCHOR, MIN_COVERAGE_RATIO
from risk_index.core.utils_dates import resample_to_weekly
from risk_index.core.utils_math import (
    compute_ratio,
    rsi,
    realized_vol,
    above_ma,
    ma_slope,
    moving_average,
)
from risk_index.core.logger import get_logger

logger = get_logger(__name__)


def clean_and_align(
    raw_series: dict[str, pd.Series],
    universe_cfg: UniverseConfig,
    transforms_cfg: TransformsConfig,
) -> pd.DataFrame:
    """Clean raw data and align to weekly Friday frequency.


    Args:
        raw_series: Dict mapping series ID to raw daily data
        universe_cfg: Universe configuration
        transforms_cfg: Transforms configuration

    Returns:
        Weekly-aligned DataFrame with all series, ratios, and computed series
    """
    logger.info("Starting data cleaning and alignment")

    # Step 1: Combine all series into daily DataFrame
    daily_df = combine_to_daily(raw_series, universe_cfg, transforms_cfg)
    logger.info(f"Combined {len(daily_df.columns)} series into daily DataFrame")

    # Step 2: Resample to weekly (W-FRI)
    weekly_df = resample_to_weekly_df(daily_df)
    logger.info(f"Resampled to weekly: {len(weekly_df)} weeks")

    # Step 3: Forward-fill with series-specific limits
    weekly_df = apply_ffill(weekly_df, universe_cfg, transforms_cfg)

    # Step 4: Compute ratios
    weekly_df = compute_ratios(weekly_df, universe_cfg)
    logger.info(f"Computed {len(universe_cfg.ratios)} ratios")

    # Step 5: Compute derived series
    weekly_df = compute_derived_series(weekly_df, universe_cfg)
    logger.info(f"Computed {len(universe_cfg.computed)} derived series")

    # Step 6: Generate data quality report
    quality_report = generate_quality_report(weekly_df)
    logger.info(f"Data quality: {quality_report['coverage_pct']:.1f}% average coverage")

    return weekly_df


def combine_to_daily(
    raw_series: dict[str, pd.Series],
    universe_cfg: UniverseConfig,
    transforms_cfg: TransformsConfig,
) -> pd.DataFrame:
    """Combine all raw series into a single daily DataFrame.

    Args:
        raw_series: Dict of raw series
        universe_cfg: Universe config
        transforms_cfg: Transforms config

    Returns:
        Daily DataFrame
    """
    if not raw_series:
        return pd.DataFrame()

    # Find common date range
    all_indices = [s.index for s in raw_series.values() if len(s) > 0]
    if not all_indices:
        return pd.DataFrame()

    min_date = min(idx.min() for idx in all_indices)
    max_date = max(idx.max() for idx in all_indices)

    # Create daily date range (business days)
    daily_index = pd.date_range(start=min_date, end=max_date, freq="B")

    # Reindex all series to daily
    daily_data = {}
    for series_id, series in raw_series.items():
        if len(series) > 0:
            series.index = pd.to_datetime(series.index)
            daily_data[series_id] = series.reindex(daily_index)

    return pd.DataFrame(daily_data)


def resample_to_weekly_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily DataFrame to weekly Friday frequency.

    Args:
        daily_df: Daily DataFrame

    Returns:
        Weekly DataFrame
    """
    if daily_df.empty:
        return pd.DataFrame()

    # Use last value for each week ending Friday
    weekly_df = daily_df.resample(WEEK_ANCHOR).last()

    return weekly_df


def apply_ffill(
    df: pd.DataFrame,
    universe_cfg: UniverseConfig,
    transforms_cfg: TransformsConfig,
) -> pd.DataFrame:
    """Apply forward-fill with series-specific limits.

    Args:
        df: Weekly DataFrame
        universe_cfg: Universe config
        transforms_cfg: Transforms config

    Returns:
        Forward-filled DataFrame
    """
    result = df.copy()

    # Get series kind mapping
    series_kind = {}
    for series in universe_cfg.series:
        series_kind[series.id] = series.kind

    # Get ffill limits (convert from business days to weeks)
    ffill_limits = transforms_cfg.ffill_limits

    for col in result.columns:
        kind = series_kind.get(col, "price")

        # Convert business day limits to approximate weeks
        if kind == "price":
            limit = max(1, ffill_limits.price // 5)
        elif kind == "fx":
            limit = max(1, ffill_limits.fx // 5)
        elif kind == "vol":
            limit = max(1, ffill_limits.vol // 5)
        elif kind == "macro_daily":
            limit = max(1, ffill_limits.macro_daily // 5)
        elif kind == "macro_weekly":
            limit = max(1, ffill_limits.macro_weekly // 5)
        elif kind == "macro_monthly":
            limit = max(1, ffill_limits.macro_monthly // 5)
        else:
            limit = 4  # Default: 4 weeks

        result[col] = result[col].ffill(limit=limit)

    return result


def compute_ratios(df: pd.DataFrame, universe_cfg: UniverseConfig) -> pd.DataFrame:
    """Compute all ratios defined in universe config.

    Args:
        df: Weekly DataFrame with base series
        universe_cfg: Universe config

    Returns:
        DataFrame with ratio columns added
    """
    result = df.copy()

    for ratio in universe_cfg.ratios:
        if ratio.numerator in result.columns and ratio.denominator in result.columns:
            ratio_series = compute_ratio(
                result[ratio.numerator],
                result[ratio.denominator],
                invert=ratio.invert,
            )
            result[ratio.id] = ratio_series
        else:
            logger.warning(
                f"Cannot compute ratio {ratio.id}: missing "
                f"{'numerator ' + ratio.numerator if ratio.numerator not in result.columns else ''}"
                f"{'denominator ' + ratio.denominator if ratio.denominator not in result.columns else ''}"
            )

    return result


def compute_derived_series(df: pd.DataFrame, universe_cfg: UniverseConfig) -> pd.DataFrame:
    """Compute derived series (RSI, volatility, MA signals, etc.).

    Args:
        df: Weekly DataFrame
        universe_cfg: Universe config

    Returns:
        DataFrame with derived series added
    """
    result = df.copy()

    for computed in universe_cfg.computed:
        if computed.series not in result.columns:
            logger.warning(
                f"Cannot compute {computed.id}: base series {computed.series} not found"
            )
            continue

        base_series = result[computed.series]

        if computed.type == "realized_vol":
            # Convert window from days to weeks (approximate)
            window_weeks = max(1, computed.window_days // 5)
            result[computed.id] = realized_vol(base_series, window_weeks, annualize=True)

        elif computed.type == "rsi":
            window_weeks = max(2, computed.window_days // 5)
            result[computed.id] = rsi(base_series, window_weeks)

        elif computed.type == "above_ma":
            window_weeks = max(1, computed.window_days // 5)
            result[computed.id] = above_ma(base_series, window_weeks)

        elif computed.type == "ma_slope":
            window_weeks = max(1, computed.window_days // 5)
            result[computed.id] = ma_slope(base_series, window_weeks)

        elif computed.type == "moving_average":
            window_weeks = max(1, computed.window_days // 5)
            result[computed.id] = moving_average(base_series, window_weeks)

        else:
            logger.warning(f"Unknown computed type: {computed.type} for {computed.id}")

    return result


def generate_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Generate data quality report.

    Args:
        df: Weekly DataFrame

    Returns:
        Quality metrics dict
    """
    if df.empty:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "coverage_pct": 0,
            "columns_above_80pct": 0,
            "start_date": None,
            "end_date": None,
        }

    # Calculate coverage
    coverage_by_col = df.notna().mean()
    overall_coverage = coverage_by_col.mean()

    # Find columns with good coverage
    good_coverage = (coverage_by_col >= MIN_COVERAGE_RATIO).sum()

    # Calculate coverage over last 5 years
    five_years_ago = df.index.max() - pd.DateOffset(years=5)
    recent_df = df[df.index >= five_years_ago]
    recent_coverage = recent_df.notna().mean().mean() if not recent_df.empty else 0

    # Series with missing data
    missing_series = coverage_by_col[coverage_by_col < 0.5].index.tolist()

    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "coverage_pct": overall_coverage * 100,
        "recent_coverage_pct": recent_coverage * 100,
        "columns_above_80pct": int(good_coverage),
        "start_date": df.index.min().strftime("%Y-%m-%d"),
        "end_date": df.index.max().strftime("%Y-%m-%d"),
        "missing_series": missing_series[:10],  # Limit to first 10
        "coverage_by_column": coverage_by_col.to_dict(),
    }

    return report


def filter_by_coverage(
    df: pd.DataFrame,
    min_coverage: float = MIN_COVERAGE_RATIO,
) -> pd.DataFrame:
    """Filter DataFrame to dates with sufficient coverage.

    Args:
        df: Input DataFrame
        min_coverage: Minimum fraction of non-null columns required

    Returns:
        Filtered DataFrame
    """
    coverage = df.notna().mean(axis=1)
    return df[coverage >= min_coverage].copy()


def get_series_start_dates(df: pd.DataFrame) -> pd.Series:
    """Get first valid date for each series.

    Args:
        df: Weekly DataFrame

    Returns:
        Series mapping column name to first valid date
    """
    return df.apply(lambda col: col.first_valid_index())


def get_common_start_date(
    df: pd.DataFrame,
    required_series: list[str] | None = None,
    min_coverage: float = 0.80,
) -> pd.Timestamp:
    """Find earliest date where minimum coverage is met.

    Args:
        df: Weekly DataFrame
        required_series: If provided, these series must be present
        min_coverage: Minimum fraction of columns that must have data

    Returns:
        Earliest qualifying date
    """
    if required_series:
        # Filter to required series
        available = [s for s in required_series if s in df.columns]
        check_df = df[available]
    else:
        check_df = df

    # Calculate rolling coverage
    coverage = check_df.notna().mean(axis=1)
    qualifying = coverage[coverage >= min_coverage]

    if qualifying.empty:
        # Return first date if no qualifying dates
        return df.index.min()

    return qualifying.index.min()
