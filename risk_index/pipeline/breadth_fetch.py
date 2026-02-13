"""Market breadth data fetching and calculation module.

Fetches S&P 500, 400, 600 constituents and calculates breadth metrics:
- Advancers/Decliners
- % Above 50/200 day moving averages
- % at N-month highs/lows
- % Overbought/Oversold (RSI-based)
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from risk_index.core.constants import DATA_DIR, PROCESSED_DIR
from risk_index.core.logger import get_logger

logger = get_logger(__name__)

# Directory for cached constituent lists
CONSTITUENTS_DIR = DATA_DIR / "constituents"
BREADTH_CACHE_FILE = PROCESSED_DIR / "breadth_latest.parquet"

# Wikipedia URLs for constituent lists
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"

IndexType = Literal["SP500", "SP400", "SP600"]

# User-agent header to avoid Wikipedia blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_wikipedia_table(url: str) -> list[pd.DataFrame]:
    """Fetch HTML tables from Wikipedia with proper headers.

    Args:
        url: Wikipedia URL

    Returns:
        List of DataFrames from HTML tables
    """
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def ensure_dirs() -> None:
    """Ensure required directories exist."""
    CONSTITUENTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_sp500_constituents(use_cache: bool = True) -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia or cache.

    Args:
        use_cache: Whether to use cached CSV if available

    Returns:
        List of ticker symbols
    """
    ensure_dirs()
    cache_file = CONSTITUENTS_DIR / "sp500_constituents.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file)
        return df["Symbol"].tolist()

    try:
        tables = fetch_wikipedia_table(SP500_URL)
        df = tables[0]
        # Clean ticker symbols (replace . with - for Yahoo Finance)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

        # Cache the result
        pd.DataFrame({"Symbol": tickers}).to_csv(cache_file, index=False)
        logger.info(f"Fetched {len(tickers)} S&P 500 constituents from Wikipedia")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 constituents: {e}")
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            return df["Symbol"].tolist()
        return []


def get_sp400_constituents(use_cache: bool = True) -> list[str]:
    """Fetch S&P 400 tickers from Wikipedia or cache.

    Args:
        use_cache: Whether to use cached CSV if available

    Returns:
        List of ticker symbols
    """
    ensure_dirs()
    cache_file = CONSTITUENTS_DIR / "sp400_constituents.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file)
        return df["Symbol"].tolist()

    try:
        tables = fetch_wikipedia_table(SP400_URL)
        df = tables[0]
        # Clean ticker symbols
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

        # Cache the result
        pd.DataFrame({"Symbol": tickers}).to_csv(cache_file, index=False)
        logger.info(f"Fetched {len(tickers)} S&P 400 constituents from Wikipedia")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 400 constituents: {e}")
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            return df["Symbol"].tolist()
        return []


def get_sp600_constituents(use_cache: bool = True) -> list[str]:
    """Fetch S&P 600 tickers from Wikipedia or cache.

    Args:
        use_cache: Whether to use cached CSV if available

    Returns:
        List of ticker symbols
    """
    ensure_dirs()
    cache_file = CONSTITUENTS_DIR / "sp600_constituents.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file)
        return df["Symbol"].tolist()

    try:
        tables = fetch_wikipedia_table(SP600_URL)
        df = tables[0]
        # Clean ticker symbols
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

        # Cache the result
        pd.DataFrame({"Symbol": tickers}).to_csv(cache_file, index=False)
        logger.info(f"Fetched {len(tickers)} S&P 600 constituents from Wikipedia")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 600 constituents: {e}")
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            return df["Symbol"].tolist()
        return []


def get_all_constituents(use_cache: bool = True) -> dict[IndexType, list[str]]:
    """Get all S&P index constituents.

    Returns:
        Dict mapping index name to list of tickers
    """
    return {
        "SP500": get_sp500_constituents(use_cache),
        "SP400": get_sp400_constituents(use_cache),
        "SP600": get_sp600_constituents(use_cache),
    }


def fetch_price_data(
    tickers: list[str],
    period: str = "1y",
    progress: bool = False,
) -> pd.DataFrame:
    """Fetch OHLC data for multiple tickers.

    Args:
        tickers: List of ticker symbols
        period: Data period (e.g., "1y", "2y")
        progress: Show download progress

    Returns:
        DataFrame with MultiIndex columns (ticker, OHLC)
    """
    try:
        df = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=progress,
            threads=True,
        )

        if df.empty:
            logger.warning("Price download returned empty DataFrame")
            return pd.DataFrame()

        # Ensure timezone-naive index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df
    except Exception as e:
        logger.error(f"Failed to fetch price data: {e}")
        return pd.DataFrame()


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index.

    Args:
        prices: Price series
        period: RSI period (default 14)

    Returns:
        RSI series (0-100)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def count_advancers_decliners(close_df: pd.DataFrame, date: pd.Timestamp) -> tuple[int, int]:
    """Count advancers and decliners for a given date.

    Args:
        close_df: DataFrame of close prices (columns = tickers)
        date: Date to check

    Returns:
        Tuple of (advancers, decliners)
    """
    if date not in close_df.index:
        return 0, 0

    idx = close_df.index.get_loc(date)
    if idx == 0:
        return 0, 0

    today = close_df.iloc[idx]
    yesterday = close_df.iloc[idx - 1]

    changes = today - yesterday
    advancers = (changes > 0).sum()
    decliners = (changes < 0).sum()

    return int(advancers), int(decliners)


def pct_above_ma(close_df: pd.DataFrame, date: pd.Timestamp, ma_period: int) -> float:
    """Calculate percentage of stocks above their moving average.

    Args:
        close_df: DataFrame of close prices
        date: Date to check
        ma_period: Moving average period in days

    Returns:
        Percentage (0-100) of stocks above MA
    """
    if date not in close_df.index:
        return np.nan

    idx = close_df.index.get_loc(date)
    if idx < ma_period:
        return np.nan

    # Calculate MA for each stock
    ma_values = close_df.iloc[idx - ma_period + 1 : idx + 1].mean()
    current_prices = close_df.iloc[idx]

    valid_mask = ~(current_prices.isna() | ma_values.isna())
    if valid_mask.sum() == 0:
        return np.nan

    above_ma = (current_prices > ma_values) & valid_mask
    return float(above_ma.sum() / valid_mask.sum() * 100)


def pct_at_high(close_df: pd.DataFrame, date: pd.Timestamp, lookback_days: int) -> float:
    """Calculate percentage of stocks at N-day highs.

    Args:
        close_df: DataFrame of close prices
        date: Date to check
        lookback_days: Number of trading days to look back

    Returns:
        Percentage (0-100) at highs
    """
    if date not in close_df.index:
        return np.nan

    idx = close_df.index.get_loc(date)
    if idx < lookback_days:
        return np.nan

    window_data = close_df.iloc[idx - lookback_days : idx + 1]
    current_prices = close_df.iloc[idx]
    period_highs = window_data.max()

    valid_mask = ~(current_prices.isna() | period_highs.isna())
    if valid_mask.sum() == 0:
        return np.nan

    at_high = (current_prices >= period_highs) & valid_mask
    return float(at_high.sum() / valid_mask.sum() * 100)


def pct_at_low(close_df: pd.DataFrame, date: pd.Timestamp, lookback_days: int) -> float:
    """Calculate percentage of stocks at N-day lows.

    Args:
        close_df: DataFrame of close prices
        date: Date to check
        lookback_days: Number of trading days to look back

    Returns:
        Percentage (0-100) at lows
    """
    if date not in close_df.index:
        return np.nan

    idx = close_df.index.get_loc(date)
    if idx < lookback_days:
        return np.nan

    window_data = close_df.iloc[idx - lookback_days : idx + 1]
    current_prices = close_df.iloc[idx]
    period_lows = window_data.min()

    valid_mask = ~(current_prices.isna() | period_lows.isna())
    if valid_mask.sum() == 0:
        return np.nan

    at_low = (current_prices <= period_lows) & valid_mask
    return float(at_low.sum() / valid_mask.sum() * 100)


def pct_above_20ma(close_df: pd.DataFrame, date: pd.Timestamp) -> float:
    """Calculate percentage of stocks above 20-day moving average."""
    return pct_above_ma(close_df, date, 20)


def pct_above_100ma(close_df: pd.DataFrame, date: pd.Timestamp) -> float:
    """Calculate percentage of stocks above 100-day moving average."""
    return pct_above_ma(close_df, date, 100)


def pct_golden_cross(close_df: pd.DataFrame, date: pd.Timestamp) -> float:
    """Calculate percentage of stocks with 50-Day MA > 200-Day MA (Golden Cross).

    Args:
        close_df: DataFrame of close prices
        date: Date to check

    Returns:
        Percentage (0-100) of stocks in Golden Cross
    """
    if date not in close_df.index:
        return np.nan

    idx = close_df.index.get_loc(date)
    if idx < 200:
        return np.nan

    # Calculate 50 and 200 day MAs
    ma50 = close_df.iloc[idx - 49 : idx + 1].mean()
    ma200 = close_df.iloc[idx - 199 : idx + 1].mean()

    valid_mask = ~(ma50.isna() | ma200.isna())
    if valid_mask.sum() == 0:
        return np.nan

    golden_cross = (ma50 > ma200) & valid_mask
    return float(golden_cross.sum() / valid_mask.sum() * 100)


def compute_trend_count(close_df: pd.DataFrame, date: pd.Timestamp) -> tuple[int, int, int]:
    """Count stocks meeting trend criteria.

    Criteria (4 total):
    1. 50-Day MA slope is rising (MA today > MA 5 days ago)
    2. 200-Day MA slope is rising
    3. Close > 50-Day MA
    4. 50-Day MA > 200-Day MA

    Args:
        close_df: DataFrame of close prices
        date: Date to check

    Returns:
        Tuple of (count_4_of_4, count_0_of_4, valid_count)
    """
    if date not in close_df.index:
        return 0, 0, 0

    idx = close_df.index.get_loc(date)
    if idx < 205:  # Need enough history
        return 0, 0, 0

    count_4_of_4 = 0
    count_0_of_4 = 0
    valid_count = 0

    for ticker in close_df.columns:
        prices = close_df[ticker].iloc[: idx + 1].dropna()
        if len(prices) < 205:
            continue

        valid_count += 1

        # Calculate MAs
        ma50_today = prices.iloc[-50:].mean()
        ma50_5d_ago = prices.iloc[-55:-5].mean()
        ma200_today = prices.iloc[-200:].mean()
        ma200_5d_ago = prices.iloc[-205:-5].mean()
        close_today = prices.iloc[-1]

        # Check 4 criteria
        criteria_met = 0
        if ma50_today > ma50_5d_ago:  # 50MA rising
            criteria_met += 1
        if ma200_today > ma200_5d_ago:  # 200MA rising
            criteria_met += 1
        if close_today > ma50_today:  # Close > 50MA
            criteria_met += 1
        if ma50_today > ma200_today:  # 50MA > 200MA (Golden Cross)
            criteria_met += 1

        if criteria_met == 4:
            count_4_of_4 += 1
        elif criteria_met == 0:
            count_0_of_4 += 1

    return count_4_of_4, count_0_of_4, valid_count


def pct_rsi_overbought(close_df: pd.DataFrame, date: pd.Timestamp, threshold: float = 70) -> float:
    """Calculate percentage of stocks with RSI above threshold (overbought).

    Args:
        close_df: DataFrame of close prices
        date: Date to check
        threshold: RSI threshold for overbought (default 70)

    Returns:
        Percentage (0-100) overbought
    """
    if date not in close_df.index:
        return np.nan

    idx = close_df.index.get_loc(date)
    if idx < 14:  # Need at least 14 days for RSI
        return np.nan

    overbought_count = 0
    valid_count = 0

    for ticker in close_df.columns:
        prices = close_df[ticker].iloc[: idx + 1].dropna()
        if len(prices) >= 15:
            rsi = compute_rsi(prices)
            if not rsi.empty and not np.isnan(rsi.iloc[-1]):
                valid_count += 1
                if rsi.iloc[-1] > threshold:
                    overbought_count += 1

    if valid_count == 0:
        return np.nan

    return float(overbought_count / valid_count * 100)


def pct_rsi_oversold(close_df: pd.DataFrame, date: pd.Timestamp, threshold: float = 30) -> float:
    """Calculate percentage of stocks with RSI below threshold (oversold).

    Args:
        close_df: DataFrame of close prices
        date: Date to check
        threshold: RSI threshold for oversold (default 30)

    Returns:
        Percentage (0-100) oversold
    """
    if date not in close_df.index:
        return np.nan

    idx = close_df.index.get_loc(date)
    if idx < 14:
        return np.nan

    oversold_count = 0
    valid_count = 0

    for ticker in close_df.columns:
        prices = close_df[ticker].iloc[: idx + 1].dropna()
        if len(prices) >= 15:
            rsi = compute_rsi(prices)
            if not rsi.empty and not np.isnan(rsi.iloc[-1]):
                valid_count += 1
                if rsi.iloc[-1] < threshold:
                    oversold_count += 1

    if valid_count == 0:
        return np.nan

    return float(oversold_count / valid_count * 100)


def compute_breadth_metrics_for_index(
    tickers: list[str],
    index_name: str,
    lookback_days: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """Compute all breadth metrics for a single index.

    Args:
        tickers: List of ticker symbols
        index_name: Name of the index (e.g., "SP500")
        lookback_days: Number of trading days to compute
        progress_callback: Optional callback for progress updates

    Returns:
        DataFrame with breadth metrics for the index
    """
    if not tickers:
        return pd.DataFrame()

    logger.info(f"Fetching price data for {len(tickers)} {index_name} constituents...")

    # Fetch 1 year of data (need ~252 days for 200 DMA + 12 month highs)
    df = fetch_price_data(tickers, period="1y", progress=False)

    if df.empty:
        logger.warning(f"No price data for {index_name}")
        return pd.DataFrame()

    # Extract Close prices
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close_df = df["Close"]
        else:
            logger.warning(f"No Close prices in data for {index_name}")
            return pd.DataFrame()
    else:
        close_df = df[["Close"]] if "Close" in df.columns else df

    # Get the last N+1 trading days
    trading_days = close_df.index[-lookback_days - 1 :]

    metrics_list = []

    for i, date in enumerate(trading_days):
        if progress_callback:
            progress_callback(i, len(trading_days), index_name)

        advancers, decliners = count_advancers_decliners(close_df, date)
        trend_4of4, trend_0of4, trend_valid = compute_trend_count(close_df, date)

        metrics = {
            "date": date,
            "index": index_name,
            "advancers": advancers,
            "decliners": decliners,
            "pct_above_20ma": pct_above_ma(close_df, date, 20),
            "pct_above_50ma": pct_above_ma(close_df, date, 50),
            "pct_above_100ma": pct_above_ma(close_df, date, 100),
            "pct_above_200ma": pct_above_ma(close_df, date, 200),
            "pct_golden_cross": pct_golden_cross(close_df, date),
            "pct_1mo_highs": pct_at_high(close_df, date, 21),
            "pct_3mo_highs": pct_at_high(close_df, date, 63),
            "pct_6mo_highs": pct_at_high(close_df, date, 126),
            "pct_12mo_highs": pct_at_high(close_df, date, 252),
            "pct_1mo_lows": pct_at_low(close_df, date, 21),
            "pct_3mo_lows": pct_at_low(close_df, date, 63),
            "pct_6mo_lows": pct_at_low(close_df, date, 126),
            "pct_12mo_lows": pct_at_low(close_df, date, 252),
            "pct_overbought": pct_rsi_overbought(close_df, date),
            "pct_oversold": pct_rsi_oversold(close_df, date),
            "trend_count_4of4": trend_4of4,
            "trend_count_0of4": trend_0of4,
            "trend_valid_stocks": trend_valid,
        }
        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)


def compute_all_breadth_metrics(
    lookback_days: int = 10,
    use_cached_constituents: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """Compute breadth metrics for all S&P indices.

    Args:
        lookback_days: Number of trading days to compute
        use_cached_constituents: Use cached constituent lists
        progress_callback: Optional callback for progress updates

    Returns:
        DataFrame with all breadth metrics
    """
    all_constituents = get_all_constituents(use_cache=use_cached_constituents)

    all_metrics = []

    for index_name, tickers in all_constituents.items():
        if not tickers:
            logger.warning(f"No constituents for {index_name}, skipping")
            continue

        logger.info(f"Computing breadth metrics for {index_name} ({len(tickers)} stocks)...")
        metrics_df = compute_breadth_metrics_for_index(
            tickers,
            index_name,
            lookback_days,
            progress_callback,
        )

        if not metrics_df.empty:
            all_metrics.append(metrics_df)

    if not all_metrics:
        return pd.DataFrame()

    combined = pd.concat(all_metrics, ignore_index=True)
    return combined


def fetch_breadth_data(
    lookback_days: int = 10,
    use_cache: bool = True,
    force_refresh: bool = False,
    progress_callback=None,
) -> pd.DataFrame:
    """Main entry point to fetch breadth data with caching.

    Args:
        lookback_days: Number of trading days of history
        use_cache: Use cached breadth data if available
        force_refresh: Force refresh even if cache exists
        progress_callback: Optional callback for progress updates

    Returns:
        DataFrame with breadth metrics
    """
    ensure_dirs()

    # Check cache
    if use_cache and not force_refresh and BREADTH_CACHE_FILE.exists():
        try:
            cache_mtime = datetime.fromtimestamp(BREADTH_CACHE_FILE.stat().st_mtime)
            cache_age_hours = (datetime.now() - cache_mtime).total_seconds() / 3600

            # Use cache if less than 12 hours old
            if cache_age_hours < 12:
                logger.info(f"Loading cached breadth data (age: {cache_age_hours:.1f} hours)")
                return pd.read_parquet(BREADTH_CACHE_FILE)
        except Exception as e:
            logger.warning(f"Could not read cache: {e}")

    # Compute fresh data
    logger.info("Computing fresh breadth data (this may take 2-3 minutes)...")
    start_time = time.time()

    df = compute_all_breadth_metrics(
        lookback_days=lookback_days,
        use_cached_constituents=True,
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time
    logger.info(f"Breadth computation completed in {elapsed:.1f} seconds")

    # Cache result
    if not df.empty:
        try:
            df.to_parquet(BREADTH_CACHE_FILE, index=False)
            logger.info(f"Cached breadth data to {BREADTH_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Could not cache breadth data: {e}")

    return df


def prepare_heatmap_data(
    breadth_df: pd.DataFrame,
    metric_type: Literal["advancers", "ma", "highs_lows", "overbought", "golden_cross", "trend_count"],
) -> pd.DataFrame:
    """Prepare breadth data for heat map display.

    Args:
        breadth_df: Raw breadth metrics DataFrame
        metric_type: Type of heat map to prepare

    Returns:
        DataFrame formatted for heat map display with dates as columns
    """
    if breadth_df.empty:
        return pd.DataFrame()

    # Get unique dates sorted descending (current first)
    dates = sorted(breadth_df["date"].unique(), reverse=True)

    # Map index names to display names
    index_display = {
        "SP500": "S&P 500 - Large Cap",
        "SP400": "S&P 400 - Mid Cap",
        "SP600": "S&P 600 - Small Cap",
    }

    rows = []

    if metric_type == "advancers":
        # Advancers & Decliners
        for idx in ["SP500", "SP400", "SP600"]:
            idx_data = breadth_df[breadth_df["index"] == idx].set_index("date")

            # Advancers row
            adv_row = {"Index": index_display.get(idx, idx), "Metric": "Advancers"}
            for i, date in enumerate(dates):
                col_name = "Current" if i == 0 else f"{i}D Ago"
                adv_row[col_name] = idx_data.loc[date, "advancers"] if date in idx_data.index else np.nan
            rows.append(adv_row)

            # Decliners row
            dec_row = {"Index": index_display.get(idx, idx), "Metric": "Decliners"}
            for i, date in enumerate(dates):
                col_name = "Current" if i == 0 else f"{i}D Ago"
                dec_row[col_name] = idx_data.loc[date, "decliners"] if date in idx_data.index else np.nan
            rows.append(dec_row)

    elif metric_type == "ma":
        # Moving Average % - now includes 20, 50, 100, 200 day MAs
        ma_metrics = [
            ("pct_above_20ma", "% Above 20 Day MA"),
            ("pct_above_50ma", "% Above 50 Day MA"),
            ("pct_above_100ma", "% Above 100 Day MA"),
            ("pct_above_200ma", "% Above 200 Day MA"),
        ]
        for idx in ["SP500", "SP400", "SP600"]:
            idx_data = breadth_df[breadth_df["index"] == idx].set_index("date")

            for metric_col, metric_label in ma_metrics:
                row = {"Index": index_display.get(idx, idx), "Metric": metric_label}
                for i, date in enumerate(dates):
                    col_name = "Current" if i == 0 else f"{i}D Ago"
                    if metric_col in idx_data.columns and date in idx_data.index:
                        val = idx_data.loc[date, metric_col]
                        row[col_name] = round(val, 1) if pd.notna(val) else np.nan
                    else:
                        row[col_name] = np.nan
                rows.append(row)

    elif metric_type == "golden_cross":
        # Golden Cross (50MA > 200MA) percentage
        for idx in ["SP500", "SP400", "SP600"]:
            idx_data = breadth_df[breadth_df["index"] == idx].set_index("date")

            row = {"Index": index_display.get(idx, idx), "Metric": "% 50-Day > 200-Day MA"}
            for i, date in enumerate(dates):
                col_name = "Current" if i == 0 else f"{i}D Ago"
                if "pct_golden_cross" in idx_data.columns and date in idx_data.index:
                    val = idx_data.loc[date, "pct_golden_cross"]
                    row[col_name] = round(val, 1) if pd.notna(val) else np.nan
                else:
                    row[col_name] = np.nan
            rows.append(row)

    elif metric_type == "trend_count":
        # Trend Count (4 of 4 and 0 of 4 criteria)
        for idx in ["SP500", "SP400", "SP600"]:
            idx_data = breadth_df[breadth_df["index"] == idx].set_index("date")

            # 4 of 4 row
            row_4of4 = {"Index": index_display.get(idx, idx), "Metric": "Stocks: 4 of 4 Trend Criteria"}
            for i, date in enumerate(dates):
                col_name = "Current" if i == 0 else f"{i}D Ago"
                if "trend_count_4of4" in idx_data.columns and date in idx_data.index:
                    row_4of4[col_name] = int(idx_data.loc[date, "trend_count_4of4"])
                else:
                    row_4of4[col_name] = np.nan
            rows.append(row_4of4)

            # 0 of 4 row
            row_0of4 = {"Index": index_display.get(idx, idx), "Metric": "Stocks: 0 of 4 Trend Criteria"}
            for i, date in enumerate(dates):
                col_name = "Current" if i == 0 else f"{i}D Ago"
                if "trend_count_0of4" in idx_data.columns and date in idx_data.index:
                    row_0of4[col_name] = int(idx_data.loc[date, "trend_count_0of4"])
                else:
                    row_0of4[col_name] = np.nan
            rows.append(row_0of4)

    elif metric_type == "highs_lows":
        # New Highs & Lows
        metrics_map = {
            "pct_1mo_highs": "% at 1 Month Highs",
            "pct_3mo_highs": "% at 3 Month Highs",
            "pct_6mo_highs": "% at 6 Month Highs",
            "pct_12mo_highs": "% at 12 Month Highs",
            "pct_1mo_lows": "% at 1 Month Lows",
            "pct_3mo_lows": "% at 3 Month Lows",
            "pct_6mo_lows": "% at 6 Month Lows",
            "pct_12mo_lows": "% at 12 Month Lows",
        }

        for idx in ["SP500", "SP400", "SP600"]:
            idx_data = breadth_df[breadth_df["index"] == idx].set_index("date")

            for metric_col, metric_label in metrics_map.items():
                row = {"Index": index_display.get(idx, idx), "Metric": metric_label}
                for i, date in enumerate(dates):
                    col_name = "Current" if i == 0 else f"{i}D Ago"
                    val = idx_data.loc[date, metric_col] if date in idx_data.index else np.nan
                    row[col_name] = round(val, 1) if not np.isnan(val) else np.nan
                rows.append(row)

    elif metric_type == "overbought":
        # Overbought/Oversold
        for idx in ["SP500", "SP400", "SP600"]:
            idx_data = breadth_df[breadth_df["index"] == idx].set_index("date")

            # Overbought row
            ob_row = {"Index": index_display.get(idx, idx), "Metric": "% Overbought"}
            for i, date in enumerate(dates):
                col_name = "Current" if i == 0 else f"{i}D Ago"
                val = idx_data.loc[date, "pct_overbought"] if date in idx_data.index else np.nan
                ob_row[col_name] = round(val, 1) if not np.isnan(val) else np.nan
            rows.append(ob_row)

            # Oversold row
            os_row = {"Index": index_display.get(idx, idx), "Metric": "% Oversold"}
            for i, date in enumerate(dates):
                col_name = "Current" if i == 0 else f"{i}D Ago"
                val = idx_data.loc[date, "pct_oversold"] if date in idx_data.index else np.nan
                os_row[col_name] = round(val, 1) if not np.isnan(val) else np.nan
            rows.append(os_row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test fetch
    print("Testing breadth data fetch...")
    df = fetch_breadth_data(lookback_days=5, use_cache=False, force_refresh=True)
    print(f"\nFetched {len(df)} rows of breadth data")
    print(df.head(20))
