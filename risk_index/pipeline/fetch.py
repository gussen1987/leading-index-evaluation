"""Data fetching module for the risk index pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from risk_index.core.config_schema import SourcesConfig, UniverseConfig
from risk_index.core.constants import CACHE_DIR, CACHE_MAX_AGE_DAYS
from risk_index.core.utils_io import (
    ensure_dir,
    get_cache_path,
    is_cache_stale,
    read_parquet,
    write_parquet,
)
from risk_index.core.logger import get_logger
from risk_index.sources.yahoo import YahooProvider
from risk_index.sources.fred import FredProvider
from risk_index.core.exceptions import DataFetchError

logger = get_logger(__name__)


def fetch_all(
    sources_cfg: SourcesConfig,
    universe_cfg: UniverseConfig,
    cache_dir: Path = CACHE_DIR,
    force_refresh: bool = False,
    start_date: str = "1990-01-01",
) -> dict[str, pd.Series]:
    """Fetch all series from Yahoo and FRED.


    Args:
        sources_cfg: Sources configuration
        universe_cfg: Universe configuration
        cache_dir: Directory for caching data
        force_refresh: If True, ignore cache and re-fetch
        start_date: Start date for data fetch

    Returns:
        Dict mapping series ID to price/value Series
    """
    ensure_dir(cache_dir)

    results = {}
    failed = []

    # Initialize providers
    yahoo = None
    fred = None

    if sources_cfg.yahoo.enabled:
        yahoo = YahooProvider(
            rate_limit=sources_cfg.yahoo.rate_limit_per_minute,
            retry_attempts=sources_cfg.yahoo.retry_attempts,
            retry_delay=sources_cfg.yahoo.retry_delay_seconds,
        )

    if sources_cfg.fred.enabled:
        try:
            fred = FredProvider(
                rate_limit=sources_cfg.fred.rate_limit_per_minute,
                retry_attempts=sources_cfg.fred.retry_attempts,
                retry_delay=sources_cfg.fred.retry_delay_seconds,
            )
        except Exception as e:
            logger.warning(f"Could not initialize FRED provider: {e}")

    # Group series by source for batch fetching
    yahoo_tickers = []
    fred_tickers = []

    for series in universe_cfg.series:
        cache_path = get_cache_path(series.source, series.id)

        # Check cache
        if not force_refresh and cache_path.exists() and not is_cache_stale(
            cache_path, CACHE_MAX_AGE_DAYS
        ):
            try:
                cached = read_parquet(cache_path)
                if isinstance(cached, pd.DataFrame):
                    # Extract series from single-column DataFrame
                    results[series.id] = cached.iloc[:, 0]
                else:
                    results[series.id] = cached
                logger.info(f"Using cached data for {series.id}")
                continue
            except Exception as e:
                logger.warning(f"Could not read cache for {series.id}: {e}")

        if series.source == "yahoo":
            yahoo_tickers.append(series)
        elif series.source == "fred":
            fred_tickers.append(series)

    # Batch fetch Yahoo tickers
    if yahoo and yahoo_tickers:
        logger.info(f"Fetching {len(yahoo_tickers)} Yahoo tickers")
        ticker_list = [s.ticker for s in yahoo_tickers]
        id_map = {s.ticker: s.id for s in yahoo_tickers}

        try:
            batch_results = yahoo.fetch_batch(ticker_list, start_date)

            for ticker, series in batch_results.items():
                series_id = id_map.get(ticker)
                if series_id:
                    results[series_id] = series
                    # Cache the result
                    cache_path = get_cache_path("yahoo", series_id)
                    write_parquet(series.to_frame(name=series_id), cache_path)

        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")

        # Fetch any that failed in batch individually
        for series_cfg in yahoo_tickers:
            if series_cfg.id not in results:
                try:
                    data = yahoo.fetch(series_cfg.ticker, start_date)
                    results[series_cfg.id] = data
                    cache_path = get_cache_path("yahoo", series_cfg.id)
                    write_parquet(data.to_frame(name=series_cfg.id), cache_path)
                except DataFetchError:
                    # Try fallbacks
                    fetched = False
                    for fallback in series_cfg.fallbacks:
                        try:
                            if fallback.source == "yahoo" and yahoo:
                                data = yahoo.fetch(fallback.ticker, start_date)
                                results[series_cfg.id] = data
                                cache_path = get_cache_path("yahoo", series_cfg.id)
                                write_parquet(data.to_frame(name=series_cfg.id), cache_path)
                                fetched = True
                                logger.info(f"Used fallback {fallback.ticker} for {series_cfg.id}")
                                break
                        except DataFetchError:
                            continue

                    if not fetched:
                        failed.append(series_cfg.id)
                        logger.warning(f"Could not fetch {series_cfg.id} (no fallbacks worked)")

    # Fetch FRED tickers individually
    if fred and fred_tickers:
        logger.info(f"Fetching {len(fred_tickers)} FRED tickers")

        for series_cfg in fred_tickers:
            try:
                data = fred.fetch(series_cfg.ticker, start_date)
                results[series_cfg.id] = data
                # Cache the result
                cache_path = get_cache_path("fred", series_cfg.id)
                write_parquet(data.to_frame(name=series_cfg.id), cache_path)

            except DataFetchError as e:
                failed.append(series_cfg.id)
                logger.warning(f"Could not fetch {series_cfg.id}: {e}")

    # Log summary
    logger.info(
        f"Fetch complete: {len(results)} succeeded, {len(failed)} failed",
        extra={"succeeded": len(results), "failed": len(failed), "failed_ids": failed},
    )

    return results


def fetch_single(
    series_id: str,
    ticker: str,
    source: str,
    sources_cfg: SourcesConfig,
    cache_dir: Path = CACHE_DIR,
    force_refresh: bool = False,
    start_date: str = "1990-01-01",
) -> pd.Series | None:
    """Fetch a single series.

    Args:
        series_id: Series identifier
        ticker: Ticker symbol
        source: Data source ('yahoo' or 'fred')
        sources_cfg: Sources configuration
        cache_dir: Cache directory
        force_refresh: If True, ignore cache
        start_date: Start date for fetch

    Returns:
        Data series or None if failed
    """
    cache_path = get_cache_path(source, series_id)

    # Check cache
    if not force_refresh and cache_path.exists() and not is_cache_stale(
        cache_path, CACHE_MAX_AGE_DAYS
    ):
        try:
            cached = read_parquet(cache_path)
            if isinstance(cached, pd.DataFrame):
                return cached.iloc[:, 0]
            return cached
        except Exception:
            pass

    # Fetch fresh data
    try:
        if source == "yahoo" and sources_cfg.yahoo.enabled:
            provider = YahooProvider(
                rate_limit=sources_cfg.yahoo.rate_limit_per_minute,
                retry_attempts=sources_cfg.yahoo.retry_attempts,
            )
            data = provider.fetch(ticker, start_date)

        elif source == "fred" and sources_cfg.fred.enabled:
            provider = FredProvider(
                rate_limit=sources_cfg.fred.rate_limit_per_minute,
                retry_attempts=sources_cfg.fred.retry_attempts,
            )
            data = provider.fetch(ticker, start_date)
        else:
            return None

        # Cache result
        ensure_dir(cache_dir)
        write_parquet(data.to_frame(name=series_id), cache_path)

        return data

    except DataFetchError:
        return None


def clear_cache(cache_dir: Path = CACHE_DIR) -> int:
    """Clear all cached data.

    Args:
        cache_dir: Cache directory

    Returns:
        Number of files deleted
    """
    if not cache_dir.exists():
        return 0

    count = 0
    for f in cache_dir.glob("*.parquet"):
        f.unlink()
        count += 1

    logger.info(f"Cleared {count} cached files")
    return count
