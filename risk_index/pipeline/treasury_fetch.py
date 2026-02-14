"""Treasury tax flow data fetching and indicator calculations.

Implements Vincent Deluard's methodology for using Daily Treasury Statement (DTS)
tax collection data as real-time economic indicators.

Tax deposits = "hard cash" that can't be revised, covers all taxpayers,
and leads official macro data.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from risk_index.core.constants import PROCESSED_DIR
from risk_index.core.logger import get_logger
from risk_index.sources.fiscal_data import FiscalDataProvider

logger = get_logger(__name__)

# Cache file for treasury data
TREASURY_CACHE_FILE = PROCESSED_DIR / "treasury_tax_latest.parquet"
TREASURY_TIMESERIES_CACHE = PROCESSED_DIR / "treasury_timeseries.parquet"

# Default rolling windows for smoothing daily noise
DEFAULT_ROLLING_WINDOWS = [7, 28, 63]  # 1 week, 1 month, 1 quarter

TaxCategory = Literal["withheld", "corporate", "non_withheld", "total"]


def fetch_tax_deposits(
    start: str | datetime | pd.Timestamp | None = None,
    end: str | datetime | pd.Timestamp | None = None,
    use_cache: bool = True,
    cache_hours: int = 12,
) -> pd.DataFrame:
    """Fetch raw tax deposit data from Treasury API.

    Args:
        start: Start date (defaults to 3 years ago)
        end: End date (defaults to today)
        use_cache: Whether to use cached data if available
        cache_hours: Maximum cache age in hours

    Returns:
        DataFrame with columns for each tax category, indexed by date
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Default date range: 3 years of history
    if start is None:
        start = pd.Timestamp.now() - pd.Timedelta(days=365 * 3)
    if end is None:
        end = pd.Timestamp.now()

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Check cache
    if use_cache and TREASURY_CACHE_FILE.exists():
        try:
            cache_mtime = datetime.fromtimestamp(TREASURY_CACHE_FILE.stat().st_mtime)
            cache_age_hours = (datetime.now() - cache_mtime).total_seconds() / 3600

            if cache_age_hours < cache_hours:
                logger.info(f"Loading cached treasury data (age: {cache_age_hours:.1f} hours)")
                cached_df = pd.read_parquet(TREASURY_CACHE_FILE)
                cached_df.index = pd.to_datetime(cached_df.index)

                # Check if cache covers our date range
                if cached_df.index.min() <= start and cached_df.index.max() >= end - pd.Timedelta(days=3):
                    return cached_df[(cached_df.index >= start) & (cached_df.index <= end)]
        except Exception as e:
            logger.warning(f"Could not read cache: {e}")

    # Fetch fresh data
    logger.info(f"Fetching treasury data from {start.date()} to {end.date()}...")
    start_time = time.time()

    try:
        provider = FiscalDataProvider()
        df = provider.fetch_all_categories(start, end)

        elapsed = time.time() - start_time
        logger.info(f"Treasury data fetch completed in {elapsed:.1f} seconds")

        # Cache result
        if not df.empty:
            try:
                df.to_parquet(TREASURY_CACHE_FILE)
                logger.info(f"Cached treasury data to {TREASURY_CACHE_FILE}")
            except Exception as e:
                logger.warning(f"Could not cache treasury data: {e}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch treasury data: {e}")

        # Try to return cached data even if stale
        if TREASURY_CACHE_FILE.exists():
            logger.info("Returning stale cached data")
            try:
                cached_df = pd.read_parquet(TREASURY_CACHE_FILE)
                cached_df.index = pd.to_datetime(cached_df.index)
                return cached_df[(cached_df.index >= start) & (cached_df.index <= end)]
            except Exception:
                pass

        return pd.DataFrame()


def calculate_rolling_sums(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Calculate rolling sums to reduce daily noise.

    Daily tax deposits are noisy due to payment timing. Rolling sums
    provide a cleaner signal.

    Args:
        df: DataFrame with tax category columns
        windows: Rolling window sizes in days (default: [7, 28, 63])

    Returns:
        DataFrame with rolling sum columns added
    """
    if df.empty:
        return df

    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS

    result = df.copy()

    for col in df.columns:
        for window in windows:
            col_name = f"{col}_sum_{window}d"
            result[col_name] = df[col].rolling(window=window, min_periods=max(1, window // 2)).sum()

    return result


def calculate_yoy_growth(
    df: pd.DataFrame,
    use_business_days: bool = True,
) -> pd.DataFrame:
    """Calculate year-over-year growth rates.

    Uses business day alignment to compare same-day-of-week values
    when use_business_days is True.

    Args:
        df: DataFrame with tax category columns
        use_business_days: Align by business days (252/year) vs calendar days (365)

    Returns:
        DataFrame with YoY growth rates (as percentages)
    """
    if df.empty:
        return df

    # Days to shift for YoY comparison
    shift_days = 252 if use_business_days else 365

    result = pd.DataFrame(index=df.index)

    for col in df.columns:
        if "_sum_" in col or "_yoy_" in col or "_ytd_" in col:
            continue  # Skip derived columns

        # For raw daily values, use rolling sum first for stability
        if col in ["withheld", "corporate", "non_withheld", "total"]:
            # Use 28-day rolling sum for YoY
            rolling_col = f"{col}_sum_28d"
            if rolling_col in df.columns:
                current = df[rolling_col]
            else:
                current = df[col].rolling(window=28, min_periods=14).sum()

            prior = current.shift(shift_days)

            # Calculate YoY growth as percentage
            yoy = ((current - prior) / prior.abs().replace(0, np.nan)) * 100
            result[f"{col}_yoy"] = yoy
        else:
            # For pre-computed rolling sums
            current = df[col]
            prior = current.shift(shift_days)
            yoy = ((current - prior) / prior.abs().replace(0, np.nan)) * 100
            result[f"{col}_yoy"] = yoy

    return result


def calculate_ytd_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate year-to-date cumulative totals.

    Resets at the start of each calendar year. Useful for comparing
    current year vs prior year on the same calendar date.

    Args:
        df: DataFrame with tax category columns

    Returns:
        DataFrame with YTD cumulative columns
    """
    if df.empty:
        return df

    result = pd.DataFrame(index=df.index)

    for col in df.columns:
        if "_sum_" in col or "_yoy_" in col or "_ytd_" in col:
            continue

        # Group by year and calculate cumsum
        series = df[col].copy()
        series.index = pd.to_datetime(series.index)

        ytd = series.groupby(series.index.year).cumsum()
        result[f"{col}_ytd"] = ytd

    return result


def calculate_ytd_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate this year vs last year YTD comparison.

    Creates aligned series for current year and prior year YTD values
    for direct comparison charting.

    Args:
        df: DataFrame with tax category columns

    Returns:
        DataFrame with current_ytd and prior_ytd columns for each category
    """
    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    current_year = df.index.max().year

    result_data = []

    for col in df.columns:
        if "_sum_" in col or "_yoy_" in col or "_ytd_" in col:
            continue

        series = df[col]

        # Current year data
        current_mask = series.index.year == current_year
        current_data = series[current_mask].copy()

        # Prior year data
        prior_mask = series.index.year == current_year - 1
        prior_data = series[prior_mask].copy()

        if current_data.empty or prior_data.empty:
            continue

        # Calculate YTD cumulative for each year
        current_ytd = current_data.cumsum()
        prior_ytd = prior_data.cumsum()

        # Align by day of year
        current_df = pd.DataFrame({
            "day_of_year": current_ytd.index.dayofyear,
            f"{col}_current_ytd": current_ytd.values,
        }).set_index("day_of_year")

        prior_df = pd.DataFrame({
            "day_of_year": prior_ytd.index.dayofyear,
            f"{col}_prior_ytd": prior_ytd.values,
        }).set_index("day_of_year")

        # Merge on day of year
        combined = current_df.join(prior_df, how="outer")
        result_data.append(combined)

    if not result_data:
        return pd.DataFrame()

    result = pd.concat(result_data, axis=1)
    result = result.sort_index()

    return result


def prepare_treasury_indicators(
    years: int = 3,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Prepare all treasury tax flow indicators for dashboard.

    Main entry point that fetches data and computes all derived indicators.

    Args:
        years: Years of history to fetch
        use_cache: Whether to use cached data

    Returns:
        Dict with keys:
        - "raw": Raw daily tax deposits
        - "rolling": Rolling sums (7/28/63 day)
        - "yoy": Year-over-year growth rates
        - "ytd": YTD cumulative totals
        - "ytd_comparison": This year vs last year comparison
        - "latest_date": Most recent data date
        - "status": "success", "partial", or "failed"
    """
    result = {
        "raw": pd.DataFrame(),
        "rolling": pd.DataFrame(),
        "yoy": pd.DataFrame(),
        "ytd": pd.DataFrame(),
        "ytd_comparison": pd.DataFrame(),
        "latest_date": None,
        "status": "failed",
    }

    # Fetch raw data
    start = pd.Timestamp.now() - pd.Timedelta(days=365 * years)
    end = pd.Timestamp.now()

    raw_df = fetch_tax_deposits(start, end, use_cache=use_cache)

    if raw_df.empty:
        logger.error("No treasury data available")
        return result

    result["raw"] = raw_df
    result["latest_date"] = raw_df.index.max()

    # Calculate derived indicators
    try:
        rolling_df = calculate_rolling_sums(raw_df)
        result["rolling"] = rolling_df

        yoy_df = calculate_yoy_growth(rolling_df)
        result["yoy"] = yoy_df

        ytd_df = calculate_ytd_cumulative(raw_df)
        result["ytd"] = ytd_df

        ytd_comp = calculate_ytd_comparison(raw_df)
        result["ytd_comparison"] = ytd_comp

        result["status"] = "success"
        logger.info(
            f"Treasury indicators prepared: {len(raw_df)} days, "
            f"latest: {result['latest_date'].date()}"
        )

    except Exception as e:
        logger.error(f"Error calculating treasury indicators: {e}")
        result["status"] = "partial"

    return result


def get_gig_economy_indicator(
    df: pd.DataFrame,
    rolling_window: int = 28,
) -> pd.Series:
    """Calculate gig economy indicator from non-withheld taxes.

    Non-withheld tax deposits (self-employment taxes) are a proxy for
    gig economy and small business activity.

    Args:
        df: DataFrame with "non_withheld" column
        rolling_window: Smoothing window in days

    Returns:
        Series with gig economy indicator (YoY growth of rolling sum)
    """
    if "non_withheld" not in df.columns:
        return pd.Series(dtype=float)

    # Calculate rolling sum
    rolling = df["non_withheld"].rolling(window=rolling_window, min_periods=rolling_window // 2).sum()

    # Calculate YoY growth
    prior = rolling.shift(252)  # Business days in a year
    yoy = ((rolling - prior) / prior.abs().replace(0, np.nan)) * 100

    yoy.name = "gig_economy_yoy"
    return yoy


def get_corporate_profits_indicator(
    df: pd.DataFrame,
    rolling_window: int = 63,
) -> pd.Series:
    """Calculate corporate profits indicator from corporate tax deposits.

    Corporate income tax deposits are a proxy for the profit cycle.
    Note: Quarterly spikes due to estimated tax payments.

    Args:
        df: DataFrame with "corporate" column
        rolling_window: Smoothing window (63 days = ~1 quarter)

    Returns:
        Series with corporate profits indicator (YoY growth)
    """
    if "corporate" not in df.columns:
        return pd.Series(dtype=float)

    # Use quarterly rolling sum to smooth out payment timing
    rolling = df["corporate"].rolling(window=rolling_window, min_periods=rolling_window // 2).sum()

    # Calculate YoY growth
    prior = rolling.shift(252)
    yoy = ((rolling - prior) / prior.abs().replace(0, np.nan)) * 100

    yoy.name = "corporate_profits_yoy"
    return yoy


def get_labor_market_indicator(
    df: pd.DataFrame,
    rolling_window: int = 28,
) -> pd.Series:
    """Calculate labor market indicator from withheld taxes.

    Withheld income and employment taxes are a proxy for wage growth
    and labor market health.

    Args:
        df: DataFrame with "withheld" column
        rolling_window: Smoothing window in days

    Returns:
        Series with labor market indicator (YoY growth)
    """
    if "withheld" not in df.columns:
        return pd.Series(dtype=float)

    rolling = df["withheld"].rolling(window=rolling_window, min_periods=rolling_window // 2).sum()

    prior = rolling.shift(252)
    yoy = ((rolling - prior) / prior.abs().replace(0, np.nan)) * 100

    yoy.name = "labor_market_yoy"
    return yoy


if __name__ == "__main__":
    # Test fetch
    print("Testing treasury data fetch...")
    indicators = prepare_treasury_indicators(years=2, use_cache=False)

    print(f"\nStatus: {indicators['status']}")
    print(f"Latest date: {indicators['latest_date']}")

    if not indicators["raw"].empty:
        print(f"\nRaw data shape: {indicators['raw'].shape}")
        print(indicators["raw"].tail(10))

    if not indicators["yoy"].empty:
        print(f"\nYoY growth (last 5 days):")
        print(indicators["yoy"].tail(5))
