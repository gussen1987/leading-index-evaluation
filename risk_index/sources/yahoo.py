"""Yahoo Finance data provider."""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import yfinance as yf

from risk_index.sources.base import DataProvider
from risk_index.core.exceptions import DataFetchError
from risk_index.core.logger import get_logger

logger = get_logger(__name__)


class YahooProvider(DataProvider):
    """Yahoo Finance data provider using yfinance."""


    def __init__(self, rate_limit: int = 60, retry_attempts: int = 3, retry_delay: float = 1.0):
        """Initialize Yahoo Finance provider.

        Args:
            rate_limit: Requests per minute
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        super().__init__("yahoo", rate_limit, retry_attempts)
        self.retry_delay = retry_delay
        self._last_request_time = 0.0

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limit."""
        min_interval = 60.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def fetch(
        self,
        ticker: str,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.Series:
        """Fetch daily adjusted close prices from Yahoo Finance.

        Args:
            ticker: Yahoo Finance ticker symbol
            start: Start date
            end: End date (defaults to today)

        Returns:
            Adjusted close price series

        Raises:
            DataFetchError: If fetch fails after all retries
        """
        if end is None:
            end = pd.Timestamp.now()

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                self._rate_limit_wait()

                yf_ticker = yf.Ticker(ticker)
                df = yf_ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                )

                if df.empty:
                    raise DataFetchError(
                        f"No data returned for {ticker}",
                        source="yahoo",
                        ticker=ticker,
                    )

                # Use Close price (already adjusted when auto_adjust=True)
                series = df["Close"].copy()
                series.name = ticker
                series.index = pd.to_datetime(series.index).tz_localize(None)

                logger.info(
                    f"Fetched {len(series)} rows for {ticker}",
                    extra={"ticker": ticker, "source": "yahoo", "rows": len(series)},
                )
                return series

            except DataFetchError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.retry_attempts} failed for {ticker}: {e}",
                    extra={"ticker": ticker, "source": "yahoo", "attempt": attempt + 1},
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise DataFetchError(
            f"Failed after {self.retry_attempts} attempts: {last_error}",
            source="yahoo",
            ticker=ticker,
        )

    def fetch_batch(
        self,
        tickers: list[str],
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> dict[str, pd.Series]:
        """Fetch data for multiple tickers efficiently.

        Args:
            tickers: List of ticker symbols
            start: Start date
            end: End date

        Returns:
            Dict mapping ticker to price series
        """
        if end is None:
            end = pd.Timestamp.now()

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        results = {}

        try:
            self._rate_limit_wait()

            # yfinance can download multiple tickers at once
            df = yf.download(
                tickers,
                start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if df.empty:
                logger.warning("Batch download returned empty DataFrame")
                return results

            # Handle single vs multiple ticker response format
            if len(tickers) == 1:
                ticker = tickers[0]
                if "Close" in df.columns:
                    series = df["Close"].copy()
                    series.name = ticker
                    series.index = pd.to_datetime(series.index).tz_localize(None)
                    results[ticker] = series
            else:
                # Multiple tickers: df has MultiIndex columns
                if "Close" in df.columns.get_level_values(0):
                    close_df = df["Close"]
                    for ticker in tickers:
                        if ticker in close_df.columns:
                            series = close_df[ticker].dropna()
                            if not series.empty:
                                series.name = ticker
                                series.index = pd.to_datetime(series.index).tz_localize(None)
                                results[ticker] = series

            logger.info(
                f"Batch fetched {len(results)}/{len(tickers)} tickers",
                extra={"source": "yahoo", "requested": len(tickers), "fetched": len(results)},
            )

        except Exception as e:
            logger.error(f"Batch download failed: {e}", extra={"source": "yahoo"})
            # Fall back to individual fetches
            for ticker in tickers:
                try:
                    results[ticker] = self.fetch(ticker, start, end)
                except DataFetchError as fetch_err:
                    logger.warning(f"Could not fetch {ticker}: {fetch_err}")

        return results

    def validate_ticker(self, ticker: str) -> bool:
        """Validate that ticker exists.

        Args:
            ticker: Ticker symbol

        Returns:
            True if ticker has data
        """
        try:
            self._rate_limit_wait()
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="5d")
            return not hist.empty
        except Exception:
            return False
