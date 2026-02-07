"""Custom exceptions for the risk index system."""

from __future__ import annotations


class RiskIndexError(Exception):
    """Base exception for risk index errors."""


    pass


class ConfigError(RiskIndexError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, field: str | None = None):
        self.field = field
        super().__init__(f"Configuration error{f' in {field}' if field else ''}: {message}")


class DataFetchError(RiskIndexError):
    """Raised when data fetching fails."""

    def __init__(self, message: str, source: str | None = None, ticker: str | None = None):
        self.source = source
        self.ticker = ticker
        context = []
        if source:
            context.append(f"source={source}")
        if ticker:
            context.append(f"ticker={ticker}")
        context_str = f" ({', '.join(context)})" if context else ""
        super().__init__(f"Data fetch error{context_str}: {message}")


class AlignmentError(RiskIndexError):
    """Raised when data alignment fails."""

    def __init__(self, message: str, series: str | None = None):
        self.series = series
        super().__init__(f"Alignment error{f' for {series}' if series else ''}: {message}")


class ValidationError(RiskIndexError):
    """Raised when data validation fails."""

    def __init__(self, message: str, check: str | None = None):
        self.check = check
        super().__init__(f"Validation error{f' ({check})' if check else ''}: {message}")
