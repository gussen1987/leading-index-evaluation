"""42 Macro Risk Matrix regime detection.

Implements Darius Dale's Risk Matrix methodology:
- ~40 market prices scored daily via VAMS (Volatility-Adjusted Momentum Signal)
- Each asset's bullish/bearish state maps to 4 regime confirmations
- Dominant regime = highest share of confirming markets
- Daily frequency, cached with 12-hour TTL
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from risk_index.core.constants import PROCESSED_DIR, CONFIG_DIR, CACHE_DIR
from risk_index.core.logger import get_logger
from risk_index.core.utils_io import read_yaml

logger = get_logger(__name__)

MATRIX_CACHE_FILE = PROCESSED_DIR / "risk_matrix_latest.parquet"


def _load_config() -> dict:
    """Load risk matrix config from macro_regimes.yml."""
    config = read_yaml(CONFIG_DIR / "macro_regimes.yml")
    return config.get("risk_matrix", {})


def _fetch_yahoo_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch daily close prices from Yahoo Finance for multiple tickers."""
    try:
        import yfinance as yf
        # Batch download
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return pd.DataFrame()

        # Handle single vs multi-ticker output
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            # Single ticker case
            prices = data[["Close"]]
            prices.columns = tickers[:1]

        prices.index = pd.to_datetime(prices.index)
        return prices
    except Exception as e:
        logger.warning(f"Yahoo batch fetch failed: {e}")
        return pd.DataFrame()


def _fetch_fred_daily(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch a FRED series for daily data (spreads, NFCI)."""
    try:
        from risk_index.sources.fred import FredProvider
        provider = FredProvider()
        data = provider.fetch(series_id, start, end)
        data.index = pd.to_datetime(data.index)
        return data.dropna()
    except Exception as e:
        logger.warning(f"Failed to fetch FRED {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)


def compute_vams(prices: pd.Series, lookback: int = 63) -> pd.Series:
    """Compute Volatility-Adjusted Momentum Signal.

    VAMS = (Price - SMA) / (StdDev * sqrt(lookback))

    Positive = bullish momentum, negative = bearish.
    """
    if prices.empty or len(prices) < lookback:
        return pd.Series(dtype=float)
    min_periods = max(1, int(lookback * 0.67))
    sma = prices.rolling(lookback, min_periods=min_periods).mean()
    vol = prices.rolling(lookback, min_periods=min_periods).std()
    # Avoid div by zero
    vol = vol.replace(0, np.nan)
    vams = (prices - sma) / (vol * np.sqrt(lookback))
    return vams


def _vams_state(vams_value: float, threshold: float = 0.0) -> int:
    """Convert VAMS value to state: +1 (bullish), -1 (bearish), 0 (neutral)."""
    if pd.isna(vams_value):
        return 0
    if vams_value > threshold:
        return 1
    elif vams_value < -threshold:
        return -1
    return 0


def _get_regime_confirmations(state: int, polarity: float) -> list[str]:
    """Map an asset's VAMS state + polarity to confirmed regimes.

    Logic:
        Risk asset (polarity > 0):
            bullish → Goldilocks, Reflation
            bearish → Inflation, Deflation
        Defensive/vol asset (polarity < 0):
            bullish (vol up, USD up) → Inflation, Deflation
            bearish (vol down, USD down) → Goldilocks, Reflation
        Neutral (polarity == 0): confirms nothing
    """
    if state == 0 or polarity == 0:
        return []

    effective = state * np.sign(polarity)  # +1 if aligned, -1 if contrary

    if effective > 0:
        return ["goldilocks", "reflation"]
    else:
        return ["inflation", "deflation"]


def prepare_risk_matrix(
    lookback: int = 63,
    years: int = 5,
    use_cache: bool = True,
    cache_hours: int = 12,
) -> dict:
    """Prepare Risk Matrix data for dashboard.

    Returns dict with:
        - regime_counts: DataFrame (daily, 4 cols) of confirming asset counts
        - regime_shares: DataFrame (daily, 4 cols) of % shares
        - dominant_regime: Series of dominant regime labels
        - vams_all: DataFrame of VAMS values for all assets
        - category_scores: DataFrame with per-category average VAMS
        - current_snapshot: DataFrame with current asset-level detail
        - cacri: Series (Cyclical Asset/Counter-cyclical Risk Indicator)
        - latest_date: most recent date
        - status: "success", "partial", or "failed"
    """
    result = {
        "regime_counts": pd.DataFrame(),
        "regime_shares": pd.DataFrame(),
        "dominant_regime": pd.Series(dtype=str),
        "vams_all": pd.DataFrame(),
        "category_scores": pd.DataFrame(),
        "current_snapshot": pd.DataFrame(),
        "cacri": pd.Series(dtype=float),
        "latest_date": None,
        "status": "failed",
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache
    if use_cache and MATRIX_CACHE_FILE.exists():
        try:
            cache_mtime = datetime.fromtimestamp(MATRIX_CACHE_FILE.stat().st_mtime)
            cache_age_hours = (datetime.now() - cache_mtime).total_seconds() / 3600
            if cache_age_hours < cache_hours:
                logger.info(f"Loading cached risk matrix data (age: {cache_age_hours:.1f}h)")
                cached = pd.read_parquet(MATRIX_CACHE_FILE)
                cached.index = pd.to_datetime(cached.index)
                return _unpack_matrix_cache(cached)
        except Exception as e:
            logger.warning(f"Could not read matrix cache: {e}")

    config = _load_config()
    if not config:
        logger.error("No risk_matrix config found in macro_regimes.yml")
        return result

    default_lookback = config.get("default_lookback", 63)
    if lookback != 63:
        default_lookback = lookback
    vams_threshold = config.get("vams_threshold", 0.0)

    start = pd.Timestamp.now() - pd.Timedelta(days=365 * years)
    end = pd.Timestamp.now()

    logger.info(f"Fetching risk matrix data ({years} years, lookback={default_lookback})...")
    start_time = time.time()

    # Collect all tickers by source
    yahoo_tickers = []
    fred_tickers = []
    asset_meta = {}  # ticker -> {name, category, polarity, source}

    for category, assets in config.get("assets", {}).items():
        for asset in assets:
            ticker = asset["ticker"]
            source = asset.get("source", "yahoo")
            asset_meta[ticker] = {
                "name": asset.get("name", ticker),
                "category": category,
                "polarity": asset.get("polarity", 0),
                "source": source,
            }
            if source == "yahoo":
                yahoo_tickers.append(ticker)
            elif source == "fred":
                fred_tickers.append(ticker)

    # Fetch Yahoo prices in batch
    all_prices = pd.DataFrame()
    if yahoo_tickers:
        yahoo_prices = _fetch_yahoo_prices(yahoo_tickers, start, end)
        if not yahoo_prices.empty:
            all_prices = yahoo_prices

    # Fetch FRED series individually
    for tid in fred_tickers:
        fred_data = _fetch_fred_daily(tid, start, end)
        if not fred_data.empty:
            all_prices[tid] = fred_data

    if all_prices.empty:
        logger.error("No price data fetched")
        return result

    # Forward-fill gaps (weekends, holidays) then drop leading NaN
    all_prices = all_prices.ffill().dropna(how="all")

    # Compute ratios from config
    for ratio_def in config.get("ratios", []):
        num_ticker = ratio_def["numerator"]
        den_ticker = ratio_def["denominator"]
        ratio_name = f"{num_ticker}/{den_ticker}"

        if num_ticker in all_prices.columns and den_ticker in all_prices.columns:
            ratio_series = all_prices[num_ticker] / all_prices[den_ticker].replace(0, np.nan)
            all_prices[ratio_name] = ratio_series
            asset_meta[ratio_name] = {
                "name": ratio_def.get("name", ratio_name),
                "category": "ratios",
                "polarity": ratio_def.get("polarity", 1),
                "source": "computed",
            }

    # Compute VAMS for all assets
    vams_all = pd.DataFrame(index=all_prices.index)
    for ticker in all_prices.columns:
        if ticker in asset_meta:
            vams_all[ticker] = compute_vams(all_prices[ticker], default_lookback)

    vams_all = vams_all.dropna(how="all")

    if vams_all.empty:
        logger.error("No VAMS data computed")
        return result

    # Compute regime confirmations for each day
    regimes = ["goldilocks", "reflation", "inflation", "deflation"]
    regime_counts = pd.DataFrame(0, index=vams_all.index, columns=regimes, dtype=float)

    for ticker in vams_all.columns:
        if ticker not in asset_meta:
            continue
        polarity = asset_meta[ticker]["polarity"]
        for idx in vams_all.index:
            val = vams_all.loc[idx, ticker]
            state = _vams_state(val, vams_threshold)
            confirmed = _get_regime_confirmations(state, polarity)
            for r in confirmed:
                regime_counts.loc[idx, r] += abs(polarity)  # Weight by |polarity|

    # Convert to shares (%)
    total_weight = regime_counts.sum(axis=1).replace(0, np.nan)
    regime_shares = regime_counts.div(total_weight, axis=0) * 100

    # Dominant regime
    dominant_regime = regime_shares.idxmax(axis=1)

    # CACRI: Inflation + Deflation share (counter-cyclical asset risk indicator)
    cacri = regime_shares.get("inflation", 0) + regime_shares.get("deflation", 0)

    # Category scores (average VAMS per category)
    category_data = {}
    for ticker in vams_all.columns:
        if ticker in asset_meta:
            cat = asset_meta[ticker]["category"]
            if cat not in category_data:
                category_data[cat] = []
            category_data[cat].append(vams_all[ticker])

    category_scores = pd.DataFrame({
        cat: pd.concat(series_list, axis=1).mean(axis=1)
        for cat, series_list in category_data.items()
    })

    # Current snapshot for asset table
    latest_idx = vams_all.index[-1]
    snapshot_rows = []
    for ticker in vams_all.columns:
        if ticker not in asset_meta:
            continue
        meta = asset_meta[ticker]
        vams_val = vams_all.loc[latest_idx, ticker]
        state = _vams_state(vams_val, vams_threshold)
        snapshot_rows.append({
            "Ticker": ticker,
            "Name": meta["name"],
            "Category": meta["category"],
            "Polarity": meta["polarity"],
            "VAMS": round(vams_val, 3) if pd.notna(vams_val) else None,
            "Signal": "Bullish" if state > 0 else ("Bearish" if state < 0 else "Neutral"),
        })

    current_snapshot = pd.DataFrame(snapshot_rows)

    elapsed = time.time() - start_time
    logger.info(f"Risk matrix computed in {elapsed:.1f}s ({len(vams_all)} days, {len(vams_all.columns)} assets)")

    result["regime_counts"] = regime_counts
    result["regime_shares"] = regime_shares
    result["dominant_regime"] = dominant_regime
    result["vams_all"] = vams_all
    result["category_scores"] = category_scores
    result["current_snapshot"] = current_snapshot
    result["cacri"] = cacri
    result["latest_date"] = latest_idx
    result["status"] = "success" if len(vams_all.columns) >= 15 else "partial"

    # Cache core time-series data
    try:
        cache_df = regime_shares.copy()
        cache_df["dominant_regime"] = dominant_regime
        cache_df["cacri"] = cacri
        # Add a few VAMS columns for the heatmap
        for col in vams_all.columns:
            cache_df[f"vams_{col}"] = vams_all[col]

        cache_df.to_parquet(MATRIX_CACHE_FILE)
        logger.info(f"Cached risk matrix data to {MATRIX_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Could not cache risk matrix: {e}")

    return result


def _unpack_matrix_cache(cached: pd.DataFrame) -> dict:
    """Unpack cached parquet back into result dict."""
    regimes = ["goldilocks", "reflation", "inflation", "deflation"]
    regime_cols = [c for c in regimes if c in cached.columns]
    vams_cols = [c for c in cached.columns if c.startswith("vams_")]

    regime_shares = cached[regime_cols] if regime_cols else pd.DataFrame()
    dominant = cached.get("dominant_regime", pd.Series(dtype=str))
    cacri_series = cached.get("cacri", pd.Series(dtype=float))

    # Reconstruct VAMS DataFrame
    vams_all = pd.DataFrame()
    if vams_cols:
        vams_all = cached[vams_cols].copy()
        vams_all.columns = [c[5:] for c in vams_cols]  # strip "vams_" prefix

    # Rebuild current snapshot from latest VAMS
    snapshot_rows = []
    if not vams_all.empty:
        config = _load_config()
        asset_meta = {}
        for category, assets in config.get("assets", {}).items():
            for asset in assets:
                asset_meta[asset["ticker"]] = {
                    "name": asset.get("name", asset["ticker"]),
                    "category": category,
                    "polarity": asset.get("polarity", 0),
                }
        for ratio_def in config.get("ratios", []):
            rname = f"{ratio_def['numerator']}/{ratio_def['denominator']}"
            asset_meta[rname] = {
                "name": ratio_def.get("name", rname),
                "category": "ratios",
                "polarity": ratio_def.get("polarity", 1),
            }

        latest_idx = vams_all.index[-1]
        for ticker in vams_all.columns:
            meta = asset_meta.get(ticker, {"name": ticker, "category": "unknown", "polarity": 0})
            vams_val = vams_all.loc[latest_idx, ticker]
            state = _vams_state(vams_val, 0.0)
            snapshot_rows.append({
                "Ticker": ticker,
                "Name": meta["name"],
                "Category": meta["category"],
                "Polarity": meta["polarity"],
                "VAMS": round(vams_val, 3) if pd.notna(vams_val) else None,
                "Signal": "Bullish" if state > 0 else ("Bearish" if state < 0 else "Neutral"),
            })

    return {
        "regime_counts": pd.DataFrame(),  # Not cached (can be recomputed)
        "regime_shares": regime_shares,
        "dominant_regime": dominant,
        "vams_all": vams_all,
        "category_scores": pd.DataFrame(),
        "current_snapshot": pd.DataFrame(snapshot_rows),
        "cacri": cacri_series,
        "latest_date": cached.index.max() if not cached.empty else None,
        "status": "success",
    }


if __name__ == "__main__":
    print("Testing risk matrix fetch...")
    d = prepare_risk_matrix(lookback=63, years=3, use_cache=False)
    print(f"Status: {d['status']}")
    print(f"Latest: {d['latest_date']}")
    if not d["dominant_regime"].empty:
        print(f"Dominant regime: {d['dominant_regime'].iloc[-1]}")
        print(f"\nRegime shares (latest):")
        print(d["regime_shares"].iloc[-1])
