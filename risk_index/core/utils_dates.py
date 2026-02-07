"""Date utilities for the risk index system."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from risk_index.core.constants import WEEK_ANCHOR


# US business day calendar
US_HOLIDAYS = USFederalHolidayCalendar()
US_BDAY = CustomBusinessDay(calendar=US_HOLIDAYS)


def get_friday_aligned_dates(
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Generate Friday-aligned date range.


    Args:
        start: Start date
        end: End date (defaults to today)

    Returns:
        DatetimeIndex with weekly Friday frequency
    """
    if end is None:
        end = pd.Timestamp.now()

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    return pd.date_range(start=start, end=end, freq=WEEK_ANCHOR)


def align_to_friday(date: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """Align a date to the nearest previous Friday.

    Args:
        date: Input date

    Returns:
        Aligned Friday date
    """
    date = pd.Timestamp(date)
    days_since_friday = (date.weekday() - 4) % 7
    return date - timedelta(days=days_since_friday)


def get_last_friday() -> pd.Timestamp:
    """Get the most recent Friday (including today if Friday)."""
    today = pd.Timestamp.now().normalize()
    return align_to_friday(today)


def is_business_day(date: str | datetime | pd.Timestamp) -> bool:
    """Check if date is a US business day."""
    date = pd.Timestamp(date)
    return bool(pd.bdate_range(date, date, freq=US_BDAY).size > 0)


def business_days_between(
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
) -> int:
    """Count business days between two dates."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return len(pd.bdate_range(start, end, freq=US_BDAY))


def weeks_between(
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
) -> int:
    """Count weeks between two dates."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return (end - start).days // 7


def get_date_range_for_window(
    end_date: str | datetime | pd.Timestamp,
    window_weeks: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get date range for a lookback window.

    Args:
        end_date: End of the window
        window_weeks: Number of weeks to look back

    Returns:
        Tuple of (start_date, end_date)
    """
    end = pd.Timestamp(end_date)
    start = end - timedelta(weeks=window_weeks)
    return start, end


def resample_to_weekly(
    series: pd.Series,
    agg: str = "last",
) -> pd.Series:
    """Resample a series to weekly Friday frequency.

    Args:
        series: Input series with datetime index
        agg: Aggregation method ('last', 'first', 'mean', 'sum')

    Returns:
        Weekly resampled series
    """
    resampler = series.resample(WEEK_ANCHOR)

    if agg == "last":
        return resampler.last()
    elif agg == "first":
        return resampler.first()
    elif agg == "mean":
        return resampler.mean()
    elif agg == "sum":
        return resampler.sum()
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")


def get_horizon_date(
    date: str | datetime | pd.Timestamp,
    horizon_weeks: int,
) -> pd.Timestamp:
    """Get the date N weeks in the future.

    Args:
        date: Starting date
        horizon_weeks: Number of weeks ahead

    Returns:
        Future date
    """
    return pd.Timestamp(date) + timedelta(weeks=horizon_weeks)


def filter_dates_by_coverage(
    df: pd.DataFrame,
    min_coverage: float = 0.80,
) -> pd.DataFrame:
    """Filter DataFrame to dates with sufficient coverage.

    Args:
        df: Input DataFrame
        min_coverage: Minimum fraction of non-null columns required

    Returns:
        Filtered DataFrame
    """
    coverage = df.notna().mean(axis=1)
    return df[coverage >= min_coverage]
