"""U.S. Treasury Fiscal Data API provider.

Implements Vincent Deluard's methodology for tracking Daily Treasury Statement (DTS)
tax collection data as real-time economic indicators.

API Documentation: https://fiscaldata.treasury.gov/api-documentation/
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Literal

import pandas as pd
import requests

from risk_index.sources.base import DataProvider
from risk_index.core.exceptions import DataFetchError
from risk_index.core.logger import get_logger

logger = get_logger(__name__)

# Treasury Fiscal Data API endpoint
API_BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
DTS_ENDPOINT = "/v1/accounting/dts/deposits_withdrawals_operating_cash"

# Tax deposit categories - based on actual API field values
# These are the transaction_catg values for tax-related deposits
TAX_CATEGORY_PATTERNS = {
    "withheld": ["taxes - withheld individual/fica"],
    "corporate": ["taxes - corporate income"],
    "non_withheld": [
        "taxes - non withheld ind/seca electronic",
        "taxes - non withheld ind/seca other",
    ],
    # "total" is computed as sum of all tax categories, not a separate field
}


TaxCategory = Literal["withheld", "corporate", "non_withheld", "total"]


class FiscalDataProvider(DataProvider):
    """U.S. Treasury Fiscal Data API provider.

    Provides access to Daily Treasury Statement data including tax deposits.
    Free API, no key required. Updates daily by 4 PM ET.
    """

    def __init__(
        self,
        rate_limit: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize Fiscal Data provider.

        Args:
            rate_limit: Requests per minute (API has no hard limit, be conservative)
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        super().__init__("fiscal_data", rate_limit, retry_attempts)
        self.retry_delay = retry_delay
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "RiskIndexDashboard/1.0",
            "Accept": "application/json",
        })

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limit."""
        min_interval = 60.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, params: dict) -> dict:
        """Make API request with retry logic.

        Args:
            params: Query parameters

        Returns:
            API response as dict

        Raises:
            DataFetchError: If request fails after retries
        """
        url = f"{API_BASE_URL}{DTS_ENDPOINT}"

        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                self._rate_limit_wait()
                response = self._session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(
                    f"Fiscal Data API attempt {attempt + 1}/{self.retry_attempts} failed: {e}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise DataFetchError(
            f"Failed after {self.retry_attempts} attempts: {last_error}",
            source="fiscal_data",
            ticker="dts",
        )

    def fetch_raw_deposits(
        self,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch raw deposit data from DTS.

        Args:
            start: Start date
            end: End date (defaults to today)

        Returns:
            DataFrame with all deposit categories
        """
        start = pd.Timestamp(start)
        end = pd.Timestamp(end) if end else pd.Timestamp.now()

        # API uses ISO date format
        params = {
            "filter": f"record_date:gte:{start.strftime('%Y-%m-%d')},"
                      f"record_date:lte:{end.strftime('%Y-%m-%d')}",
            "sort": "record_date",
            "page[size]": 10000,  # Max page size
        }

        all_data = []
        page = 1

        while True:
            params["page[number]"] = page
            response = self._make_request(params)

            data = response.get("data", [])
            if not data:
                break

            all_data.extend(data)

            # Check for more pages
            meta = response.get("meta", {})
            total_pages = meta.get("total-pages", 1)
            if page >= total_pages:
                break
            page += 1

        if not all_data:
            logger.warning(f"No DTS data found for {start} to {end}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Convert record_date to datetime
        df["record_date"] = pd.to_datetime(df["record_date"])

        logger.info(f"Fetched {len(df)} DTS records from {start.date()} to {end.date()}")
        return df

    def _normalize_category_name(self, name: str) -> str | None:
        """Match transaction_catg to our tax category buckets.

        Args:
            name: Raw category name from API (transaction_catg field)

        Returns:
            Normalized category key or None if not a tracked category
        """
        name_lower = name.lower().strip()

        # Check if this matches any of our tax category patterns
        for category, patterns in TAX_CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if name_lower == pattern:
                    return category

        return None

    def fetch(
        self,
        ticker: str,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.Series:
        """Fetch specific tax category time series.

        Args:
            ticker: Tax category ("withheld", "corporate", "non_withheld", "total")
            start: Start date
            end: End date (defaults to today)

        Returns:
            Daily values series for the category

        Raises:
            DataFetchError: If category not found or fetch fails
        """
        if not self.validate_ticker(ticker):
            raise DataFetchError(
                f"Invalid tax category: {ticker}. "
                f"Valid categories: withheld, corporate, non_withheld, total",
                source="fiscal_data",
                ticker=ticker,
            )

        raw_df = self.fetch_raw_deposits(start, end)

        if raw_df.empty:
            raise DataFetchError(
                f"No data returned for {ticker}",
                source="fiscal_data",
                ticker=ticker,
            )

        # Filter to deposits only
        deposit_mask = raw_df["transaction_type"].str.lower().str.startswith("deposit")
        deposit_df = raw_df[deposit_mask].copy()

        # If requesting "total", sum all tax categories
        if ticker == "total":
            # Filter to all tax-related deposits
            all_values = []
            for _, row in deposit_df.iterrows():
                record_date = row["record_date"]
                transaction_catg = row.get("transaction_catg", "")
                category = self._normalize_category_name(transaction_catg)

                if category is not None:  # Any recognized tax category
                    try:
                        amt_str = row.get("transaction_today_amt") or row.get("today_amt", "0")
                        amount = float(str(amt_str).replace(",", ""))
                        all_values.append({"date": record_date, "value": amount})
                    except (ValueError, TypeError):
                        pass

            if not all_values:
                raise DataFetchError(
                    f"No matching records for category: {ticker}",
                    source="fiscal_data",
                    ticker=ticker,
                )

            result_df = pd.DataFrame(all_values)
            result = result_df.groupby("date")["value"].sum()
            result.name = ticker
            result = result.sort_index()
            logger.info(f"Fetched {len(result)} daily values for {ticker}")
            return result

        # Filter to the requested category and aggregate by date
        values = []

        for _, row in deposit_df.iterrows():
            record_date = row["record_date"]
            transaction_catg = row.get("transaction_catg", "")

            category = self._normalize_category_name(transaction_catg)

            if category == ticker:
                try:
                    amt_str = row.get("transaction_today_amt") or row.get("today_amt", "0")
                    amount = float(str(amt_str).replace(",", ""))
                    values.append({"date": record_date, "value": amount})
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse amount for {ticker} on {record_date}: {e}")

        if not values:
            raise DataFetchError(
                f"No matching records for category: {ticker}",
                source="fiscal_data",
                ticker=ticker,
            )

        result_df = pd.DataFrame(values)

        # Aggregate by date (sum if multiple entries per day)
        result = result_df.groupby("date")["value"].sum()
        result.name = ticker
        result = result.sort_index()

        logger.info(f"Fetched {len(result)} daily values for {ticker}")
        return result

    def fetch_all_categories(
        self,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch all tax categories as a DataFrame.

        Args:
            start: Start date
            end: End date (defaults to today)

        Returns:
            DataFrame with columns for each tax category, indexed by date
        """
        raw_df = self.fetch_raw_deposits(start, end)

        if raw_df.empty:
            return pd.DataFrame()

        # Filter to deposits only
        deposit_mask = raw_df["transaction_type"].str.lower().str.startswith("deposit")
        deposit_df = raw_df[deposit_mask].copy()

        # Process into category columns (excluding "total" - we'll compute it)
        category_data = {cat: {} for cat in ["withheld", "corporate", "non_withheld"]}

        for _, row in deposit_df.iterrows():
            record_date = row["record_date"]
            transaction_catg = row.get("transaction_catg", "")

            category = self._normalize_category_name(transaction_catg)

            if category and category in category_data:
                try:
                    amt_str = row.get("transaction_today_amt") or row.get("today_amt", "0")
                    amount = float(str(amt_str).replace(",", ""))

                    # Aggregate by date
                    if record_date in category_data[category]:
                        category_data[category][record_date] += amount
                    else:
                        category_data[category][record_date] = amount
                except (ValueError, TypeError):
                    pass

        # Convert to DataFrame
        result = pd.DataFrame(category_data)
        result.index.name = "date"
        result = result.sort_index()

        # Compute "total" as sum of all categories
        if not result.empty:
            result["total"] = result.sum(axis=1)

        # Drop rows where all values are NaN
        result = result.dropna(how="all")

        logger.info(f"Fetched {len(result)} days of tax deposit data across {len(result.columns)} categories")
        return result

    def validate_ticker(self, ticker: str) -> bool:
        """Validate that ticker is a valid tax category.

        Args:
            ticker: Tax category name

        Returns:
            True if valid category
        """
        return ticker in ["withheld", "corporate", "non_withheld", "total"]

    def get_latest_date(self) -> pd.Timestamp | None:
        """Get the most recent date with data available.

        Returns:
            Latest available date or None
        """
        try:
            # Fetch last few days to find latest
            end = pd.Timestamp.now()
            start = end - pd.Timedelta(days=7)

            raw_df = self.fetch_raw_deposits(start, end)
            if raw_df.empty:
                return None

            return raw_df["record_date"].max()
        except Exception as e:
            logger.warning(f"Could not determine latest date: {e}")
            return None
