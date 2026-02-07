"""FRED (Federal Reserve Economic Data) provider."""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from risk_index.sources.base import DataProvider
from risk_index.core.exceptions import DataFetchError, ConfigError
from risk_index.core.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


# Expected publication lags for FRED series (days)
EXPECTED_LAGS = {
    "NFCI": 7,  # Weekly, ~1 week lag
    "ICSA": 7,  # Weekly claims, ~1 week
    "PERMIT": 45,  # Monthly, ~6 weeks lag
    "UMCSENT": 30,  # Monthly
    "DGS10": 1,  # Daily rates, ~1 day
    "DGS2": 1,
    "T10Y2Y": 1,
    "DFII10": 1,
    "T10YIE": 1,
    "BAMLH0A0HYM2": 2,  # Daily spreads
    "BAMLC0A0CM": 2,
    "DTWEXBGS": 1,
}


class FredProvider(DataProvider):
    """FRED data provider using fredapi."""


    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: int = 120,
        retry_attempts: int = 3,
        retry_delay: float = 0.5,
    ):
        """Initialize FRED provider.

        Args:
            api_key: FRED API key (defaults to FRED_API_KEY env var)
            rate_limit: Requests per minute
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Raises:
            ConfigError: If API key is not provided
        """
        super().__init__("fred", rate_limit, retry_attempts)
        self.retry_delay = retry_delay
        self._last_request_time = 0.0

        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ConfigError(
                "FRED API key required. Set FRED_API_KEY environment variable or pass api_key parameter.",
                field="FRED_API_KEY",
            )

        # Lazy import fredapi to allow running without it if not using FRED
        try:
            from fredapi import Fred

            self._fred = Fred(api_key=self.api_key)
        except ImportError:
            raise ConfigError(
                "fredapi package required. Install with: pip install fredapi",
                field="fredapi",
            )

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limit."""
        min_interval = 60.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def check_series_status(self, series_id: str, series: pd.Series) -> str:
        """Check if FRED series is current, delayed, or discontinued.

        Args:
            series_id: FRED series ID
            series: Fetched data series

        Returns:
            Status: 'current', 'delayed', or 'discontinued'
        """
        if series.empty:
            return "discontinued"

        last_date = series.index[-1]
        days_stale = (pd.Timestamp.now() - last_date).days

        expected_lag = EXPECTED_LAGS.get(series_id, 30)
        max_acceptable_lag = expected_lag + 14  # buffer

        if days_stale <= max_acceptable_lag:
            return "current"
        elif days_stale <= 90:
            return "delayed"
        else:
            return "discontinued"

    def fetch(
        self,
        ticker: str,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.Series:
        """Fetch data from FRED.

        Args:
            ticker: FRED series ID
            start: Start date
            end: End date (defaults to today)

        Returns:
            Value series

        Raises:
            DataFetchError: If fetch fails
        """
        if end is None:
            end = pd.Timestamp.now()

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                self._rate_limit_wait()

                series = self._fred.get_series(
                    ticker,
                    observation_start=start.strftime("%Y-%m-%d"),
                    observation_end=end.strftime("%Y-%m-%d"),
                )

                if series is None or (isinstance(series, pd.Series) and series.empty):
                    raise DataFetchError(
                        f"No data returned for {ticker}",
                        source="fred",
                        ticker=ticker,
                    )

                series.name = ticker
                series.index = pd.to_datetime(series.index)

                # Check status and log accordingly
                status = self.check_series_status(ticker, series)
                if status == "discontinued":
                    logger.error(
                        f"FRED series {ticker} appears discontinued",
                        extra={"ticker": ticker, "source": "fred", "status": status},
                    )
                elif status == "delayed":
                    logger.warning(
                        f"FRED series {ticker} is delayed (last: {series.index[-1]})",
                        extra={"ticker": ticker, "source": "fred", "status": status},
                    )

                logger.info(
                    f"Fetched {len(series)} rows for {ticker}",
                    extra={
                        "ticker": ticker,
                        "source": "fred",
                        "rows": len(series),
                        "status": status,
                    },
                )
                return series

            except DataFetchError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.retry_attempts} failed for {ticker}: {e}",
                    extra={"ticker": ticker, "source": "fred", "attempt": attempt + 1},
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise DataFetchError(
            f"Failed after {self.retry_attempts} attempts: {last_error}",
            source="fred",
            ticker=ticker,
        )

    def validate_ticker(self, ticker: str) -> bool:
        """Validate that FRED series exists.

        Args:
            ticker: FRED series ID

        Returns:
            True if series exists
        """
        try:
            self._rate_limit_wait()
            info = self._fred.get_series_info(ticker)
            return info is not None
        except Exception:
            return False

    def get_series_info(self, ticker: str) -> dict | None:
        """Get metadata for a FRED series.

        Args:
            ticker: FRED series ID

        Returns:
            Series info dict or None
        """
        try:
            self._rate_limit_wait()
            info = self._fred.get_series_info(ticker)
            if info is not None:
                return info.to_dict()
            return None
        except Exception:
            return None
