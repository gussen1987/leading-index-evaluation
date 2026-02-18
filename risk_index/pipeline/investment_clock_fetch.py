"""Investment Clock macro regime detection.

Implements Trevor Greetham's Investment Clock methodology:
- 8 growth indicators + 7 inflation indicators from FRED
- Dual moving-average scoring (6m vs 12m)
- 4 quadrants: Recovery, Overheat, Stagflation, Reflation
- Monthly frequency, cached with 12-hour TTL
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from risk_index.core.constants import PROCESSED_DIR, CONFIG_DIR
from risk_index.core.logger import get_logger
from risk_index.core.utils_io import read_yaml

logger = get_logger(__name__)

CLOCK_CACHE_FILE = PROCESSED_DIR / "investment_clock_latest.parquet"


def _load_config() -> dict:
    """Load investment clock config from macro_regimes.yml."""
    config = read_yaml(CONFIG_DIR / "macro_regimes.yml")
    return config.get("investment_clock", {})


def _fetch_fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch a single FRED series, returning empty Series on failure."""
    try:
        from risk_index.sources.fred import FredProvider
        provider = FredProvider()
        data = provider.fetch(series_id, start, end)
        data.index = pd.to_datetime(data.index)
        return data.dropna()
    except Exception as e:
        logger.warning(f"Failed to fetch {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)


def _to_monthly(series: pd.Series) -> pd.Series:
    """Resample to monthly frequency (end of month)."""
    if series.empty:
        return series
    return series.resample("ME").last().dropna()


def _compute_yoy(series: pd.Series) -> pd.Series:
    """Compute year-over-year percent change for monthly data."""
    if series.empty or len(series) < 13:
        return pd.Series(dtype=float)
    return series.pct_change(12) * 100


def _dual_ma_score(series: pd.Series, short: int = 6, long: int = 12) -> pd.Series:
    """Score via dual moving average rule.

    +1 if above both MAs, -1 if below both, 0 otherwise.
    """
    if series.empty or len(series) < long + 1:
        return pd.Series(dtype=float)
    ma_s = series.rolling(short, min_periods=short).mean()
    ma_l = series.rolling(long, min_periods=long).mean()
    score = pd.Series(
        np.where(
            (series > ma_s) & (series > ma_l), 1,
            np.where((series < ma_s) & (series < ma_l), -1, 0)
        ),
        index=series.index,
    )
    return score


def _score_indicator(
    raw: pd.Series,
    transform: str,
    invert: bool,
    ma_short: int,
    ma_long: int,
) -> pd.Series:
    """Score a single indicator: transform → monthly → dual-MA."""
    monthly = _to_monthly(raw)
    if monthly.empty:
        return pd.Series(dtype=float)

    if transform == "yoy":
        scored_series = _compute_yoy(monthly)
    else:
        scored_series = monthly

    if scored_series.empty:
        return pd.Series(dtype=float)

    score = _dual_ma_score(scored_series, ma_short, ma_long)

    if invert:
        score = -score

    return score


def _classify_quadrant(growth_sum: float, inflation_sum: float) -> str:
    """Map growth/inflation sums to quadrant name."""
    if growth_sum > 0 and inflation_sum <= 0:
        return "recovery"
    elif growth_sum > 0 and inflation_sum > 0:
        return "overheat"
    elif growth_sum <= 0 and inflation_sum > 0:
        return "stagflation"
    else:
        return "reflation"


def prepare_investment_clock(
    years: int = 15,
    use_cache: bool = True,
    cache_hours: int = 12,
) -> dict:
    """Prepare Investment Clock data for dashboard.

    Returns dict with:
        - growth_score: Series of summed growth scores
        - inflation_score: Series of summed inflation scores
        - quadrant: Series of quadrant labels
        - growth_components: DataFrame of individual growth indicator scores
        - inflation_components: DataFrame of individual inflation indicator scores
        - growth_detail: DataFrame with raw values + MAs for display
        - inflation_detail: DataFrame with raw values + MAs for display
        - latest_date: most recent date
        - status: "success", "partial", or "failed"
    """
    result = {
        "growth_score": pd.Series(dtype=float),
        "inflation_score": pd.Series(dtype=float),
        "quadrant": pd.Series(dtype=str),
        "growth_components": pd.DataFrame(),
        "inflation_components": pd.DataFrame(),
        "growth_detail": pd.DataFrame(),
        "inflation_detail": pd.DataFrame(),
        "latest_date": None,
        "status": "failed",
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache
    if use_cache and CLOCK_CACHE_FILE.exists():
        try:
            cache_mtime = datetime.fromtimestamp(CLOCK_CACHE_FILE.stat().st_mtime)
            cache_age_hours = (datetime.now() - cache_mtime).total_seconds() / 3600
            if cache_age_hours < cache_hours:
                logger.info(f"Loading cached investment clock data (age: {cache_age_hours:.1f}h)")
                cached = pd.read_parquet(CLOCK_CACHE_FILE)
                cached.index = pd.to_datetime(cached.index)
                return _unpack_cache(cached)
        except Exception as e:
            logger.warning(f"Could not read clock cache: {e}")

    # Load config
    config = _load_config()
    if not config:
        logger.error("No investment_clock config found in macro_regimes.yml")
        return result

    scoring = config.get("scoring", {})
    ma_short = scoring.get("ma_short", 6)
    ma_long = scoring.get("ma_long", 12)

    start = pd.Timestamp.now() - pd.Timedelta(days=365 * years)
    end = pd.Timestamp.now()

    logger.info(f"Fetching investment clock data ({years} years)...")
    start_time = time.time()

    # Score growth indicators
    growth_scores = {}
    growth_details = {}
    for ind in config.get("growth_indicators", []):
        sid = ind["series_id"]
        raw = _fetch_fred_series(sid, start, end)
        if raw.empty:
            continue
        score = _score_indicator(raw, ind.get("transform", "level"), ind.get("invert", False), ma_short, ma_long)
        if not score.empty:
            growth_scores[sid] = score
            # Store detail for display
            monthly = _to_monthly(raw)
            growth_details[sid] = monthly

    # Score inflation indicators
    inflation_scores = {}
    inflation_details = {}
    for ind in config.get("inflation_indicators", []):
        sid = ind["series_id"]
        raw = _fetch_fred_series(sid, start, end)
        if raw.empty:
            continue
        score = _score_indicator(raw, ind.get("transform", "level"), ind.get("invert", False), ma_short, ma_long)
        if not score.empty:
            inflation_scores[sid] = score
            monthly = _to_monthly(raw)
            inflation_details[sid] = monthly

    if not growth_scores and not inflation_scores:
        logger.error("No indicators could be fetched")
        return result

    # Combine into DataFrames
    growth_df = pd.DataFrame(growth_scores)
    inflation_df = pd.DataFrame(inflation_scores)

    # Align indices
    common_idx = growth_df.index.intersection(inflation_df.index)
    if common_idx.empty:
        # Use whichever is available
        common_idx = growth_df.index if not growth_df.empty else inflation_df.index

    growth_df = growth_df.reindex(common_idx).fillna(0)
    inflation_df = inflation_df.reindex(common_idx).fillna(0)

    # Sum scores
    growth_sum = growth_df.sum(axis=1)
    inflation_sum = inflation_df.sum(axis=1)

    # Classify quadrants
    quadrant = pd.Series(
        [_classify_quadrant(g, i) for g, i in zip(growth_sum, inflation_sum)],
        index=common_idx,
    )

    elapsed = time.time() - start_time
    logger.info(f"Investment clock computed in {elapsed:.1f}s ({len(common_idx)} months)")

    result["growth_score"] = growth_sum
    result["inflation_score"] = inflation_sum
    result["quadrant"] = quadrant
    result["growth_components"] = growth_df
    result["inflation_components"] = inflation_df
    result["growth_detail"] = pd.DataFrame(growth_details).reindex(common_idx)
    result["inflation_detail"] = pd.DataFrame(inflation_details).reindex(common_idx)
    result["latest_date"] = common_idx.max() if len(common_idx) > 0 else None
    result["status"] = "success" if len(growth_scores) >= 4 and len(inflation_scores) >= 4 else "partial"

    # Cache
    try:
        cache_df = pd.DataFrame({
            "growth_score": growth_sum,
            "inflation_score": inflation_sum,
            "quadrant": quadrant,
        })
        # Add component columns
        for col in growth_df.columns:
            cache_df[f"g_{col}"] = growth_df[col]
        for col in inflation_df.columns:
            cache_df[f"i_{col}"] = inflation_df[col]

        cache_df.to_parquet(CLOCK_CACHE_FILE)
        logger.info(f"Cached investment clock data to {CLOCK_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Could not cache investment clock: {e}")

    return result


def _unpack_cache(cached: pd.DataFrame) -> dict:
    """Unpack cached parquet back into result dict."""
    result = {
        "growth_score": cached.get("growth_score", pd.Series(dtype=float)),
        "inflation_score": cached.get("inflation_score", pd.Series(dtype=float)),
        "quadrant": cached.get("quadrant", pd.Series(dtype=str)),
        "growth_components": pd.DataFrame(),
        "inflation_components": pd.DataFrame(),
        "growth_detail": pd.DataFrame(),
        "inflation_detail": pd.DataFrame(),
        "latest_date": cached.index.max() if not cached.empty else None,
        "status": "success",
    }

    # Extract component columns
    g_cols = [c for c in cached.columns if c.startswith("g_")]
    i_cols = [c for c in cached.columns if c.startswith("i_")]

    if g_cols:
        g_df = cached[g_cols].copy()
        g_df.columns = [c[2:] for c in g_cols]
        result["growth_components"] = g_df

    if i_cols:
        i_df = cached[i_cols].copy()
        i_df.columns = [c[2:] for c in i_cols]
        result["inflation_components"] = i_df

    return result


if __name__ == "__main__":
    print("Testing investment clock fetch...")
    d = prepare_investment_clock(years=10, use_cache=False)
    print(f"Status: {d['status']}")
    print(f"Latest: {d['latest_date']}")
    if not d["quadrant"].empty:
        print(f"Current quadrant: {d['quadrant'].iloc[-1]}")
        print(f"Growth score: {d['growth_score'].iloc[-1]}")
        print(f"Inflation score: {d['inflation_score'].iloc[-1]}")
