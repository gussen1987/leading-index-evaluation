"""Abstract base class for data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd


class DataProvider(ABC):
    """Abstract base class for data providers."""


    def __init__(self, name: str, rate_limit: int = 60, retry_attempts: int = 3):
        """Initialize data provider.

        Args:
            name: Provider name
            rate_limit: Requests per minute
            retry_attempts: Number of retry attempts on failure
        """
        self.name = name
        self.rate_limit = rate_limit
        self.retry_attempts = retry_attempts

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.Series:
        """Fetch data for a ticker.

        Args:
            ticker: Ticker symbol
            start: Start date
            end: End date (defaults to today)

        Returns:
            Price/value series with datetime index
        """
        pass

    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """Validate that ticker exists and has data.

        Args:
            ticker: Ticker symbol

        Returns:
            True if ticker is valid
        """
        pass

    def get_status(self) -> dict[str, Any]:
        """Get provider status information.

        Returns:
            Status dict with name, rate_limit, etc.
        """
        return {
            "name": self.name,
            "rate_limit": self.rate_limit,
            "retry_attempts": self.retry_attempts,
        }
