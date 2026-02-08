"""Lead-lag analysis module for computing Information Coefficients."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal

from risk_index.core.logger import get_logger
from risk_index.core.utils_math import forward_return, information_coefficient

logger = get_logger(__name__)


def compute_lead_lag_matrix(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    horizons: list[int] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.DataFrame:
    """Compute Information Coefficient (rank correlation) for each feature
    against forward returns at multiple horizons.

    Args:
        features_df: DataFrame with features (columns) indexed by date
        target_series: Price series to compute forward returns from
        horizons: List of forward horizons in weeks (default: [4, 8, 13, 26, 52])
        method: Correlation method ('spearman' or 'pearson')

    Returns:
        DataFrame with features as rows, horizons as columns, values = IC
    """
    if horizons is None:
        horizons = [4, 8, 13, 26, 52]

    logger.info(f"Computing lead-lag matrix for {len(features_df.columns)} features x {len(horizons)} horizons")

    # Compute forward returns for each horizon
    forward_returns = {}
    for h in horizons:
        forward_returns[h] = forward_return(target_series, horizon=h, log=True)
        logger.debug(f"Computed forward return for horizon {h}w")

    # Compute IC for each feature x horizon
    results = {}
    for feature in features_df.columns:
        feature_ics = {}
        for h in horizons:
            ic = information_coefficient(
                features_df[feature],
                forward_returns[h],
                method=method,
            )
            feature_ics[h] = ic
        results[feature] = feature_ics

    # Create DataFrame
    ic_matrix = pd.DataFrame(results).T
    ic_matrix.columns = [f"{h}w" for h in horizons]
    ic_matrix.index.name = "feature"

    logger.info(f"Lead-lag matrix shape: {ic_matrix.shape}")
    return ic_matrix


def compute_ic_with_stats(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    horizons: list[int] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
    rolling_window: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute IC with additional statistics.

    Args:
        features_df: DataFrame with features
        target_series: Price series
        horizons: Forward horizons in weeks
        method: Correlation method
        rolling_window: If provided, compute rolling IC stats over this many periods

    Returns:
        Dict with:
            - 'ic_matrix': Base IC matrix (features x horizons)
            - 'ic_tstat': T-statistics for IC significance
            - 'ic_pvalue': P-values for IC
            - 'sign_consistency': Fraction of periods with consistent sign
    """
    if horizons is None:
        horizons = [4, 8, 13, 26, 52]

    logger.info(f"Computing IC with stats for {len(features_df.columns)} features")

    # Compute forward returns
    forward_returns = {h: forward_return(target_series, horizon=h, log=True) for h in horizons}

    # Results containers
    ic_values = {}
    ic_tstat = {}
    ic_pvalue = {}
    sign_consistency = {}

    for feature in features_df.columns:
        ic_values[feature] = {}
        ic_tstat[feature] = {}
        ic_pvalue[feature] = {}
        sign_consistency[feature] = {}

        for h in horizons:
            # Align data
            aligned = pd.concat([features_df[feature], forward_returns[h]], axis=1).dropna()
            if len(aligned) < 30:
                ic_values[feature][h] = np.nan
                ic_tstat[feature][h] = np.nan
                ic_pvalue[feature][h] = np.nan
                sign_consistency[feature][h] = np.nan
                continue

            # Compute full-sample IC
            ic = information_coefficient(aligned.iloc[:, 0], aligned.iloc[:, 1], method=method)
            ic_values[feature][h] = ic

            # Compute t-statistic for IC (Fisher z-transform)
            n = len(aligned)
            if abs(ic) < 0.9999:
                z = np.arctanh(ic)
                se = 1.0 / np.sqrt(n - 3)
                t = z / se
                # Two-tailed p-value
                from scipy import stats
                pval = 2 * (1 - stats.norm.cdf(abs(t)))
            else:
                t = np.nan
                pval = np.nan

            ic_tstat[feature][h] = t
            ic_pvalue[feature][h] = pval

            # Compute sign consistency using rolling windows
            if rolling_window is not None and len(aligned) >= rolling_window * 2:
                rolling_ics = []
                for i in range(rolling_window, len(aligned)):
                    window_data = aligned.iloc[i - rolling_window : i]
                    rolling_ic = information_coefficient(
                        window_data.iloc[:, 0], window_data.iloc[:, 1], method=method
                    )
                    if not np.isnan(rolling_ic):
                        rolling_ics.append(rolling_ic)

                if rolling_ics:
                    expected_sign = np.sign(ic)
                    consistent = sum(1 for r in rolling_ics if np.sign(r) == expected_sign)
                    sign_consistency[feature][h] = consistent / len(rolling_ics)
                else:
                    sign_consistency[feature][h] = np.nan
            else:
                sign_consistency[feature][h] = np.nan

    # Convert to DataFrames
    def to_df(d, horizons):
        df = pd.DataFrame(d).T
        df.columns = [f"{h}w" for h in horizons]
        df.index.name = "feature"
        return df

    return {
        "ic_matrix": to_df(ic_values, horizons),
        "ic_tstat": to_df(ic_tstat, horizons),
        "ic_pvalue": to_df(ic_pvalue, horizons),
        "sign_consistency": to_df(sign_consistency, horizons),
    }


def compute_rolling_ic(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    horizon: int,
    window: int = 52,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.DataFrame:
    """Compute rolling IC for all features over time.

    Args:
        features_df: DataFrame with features
        target_series: Price series
        horizon: Forward horizon in weeks
        window: Rolling window size in weeks
        method: Correlation method

    Returns:
        DataFrame with dates as index, features as columns, values = rolling IC
    """
    logger.info(f"Computing rolling IC with window={window}w for horizon={horizon}w")

    # Compute forward return
    fwd_ret = forward_return(target_series, horizon=horizon, log=True)

    # Align all data
    aligned = features_df.copy()
    aligned["_fwd_ret"] = fwd_ret
    aligned = aligned.dropna(subset=["_fwd_ret"])

    if len(aligned) < window + 10:
        logger.warning(f"Insufficient data for rolling IC: {len(aligned)} rows, need {window + 10}")
        return pd.DataFrame()

    # Compute rolling IC for each feature
    rolling_ics = {}
    for feature in features_df.columns:
        feature_ics = []
        dates = []
        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i - window : i]
            if window_data[feature].notna().sum() >= window // 2:
                ic = information_coefficient(
                    window_data[feature], window_data["_fwd_ret"], method=method
                )
                feature_ics.append(ic)
            else:
                feature_ics.append(np.nan)
            dates.append(aligned.index[i])

        rolling_ics[feature] = pd.Series(feature_ics, index=dates)

    result = pd.DataFrame(rolling_ics)
    logger.info(f"Rolling IC shape: {result.shape}")
    return result


def rank_features_by_ic(
    ic_matrix: pd.DataFrame,
    horizons_to_average: list[str] | None = None,
    min_abs_ic: float = 0.0,
) -> pd.DataFrame:
    """Rank features by average absolute IC across horizons.

    Args:
        ic_matrix: IC matrix (features x horizons)
        horizons_to_average: List of horizon columns to include (e.g., ['4w', '8w'])
        min_abs_ic: Minimum absolute IC to include

    Returns:
        DataFrame with features ranked by IC, including:
            - avg_abs_ic: Average absolute IC
            - avg_ic: Average IC (preserving sign)
            - best_horizon: Horizon with highest absolute IC
            - best_ic: IC at best horizon
    """
    if horizons_to_average is None:
        horizons_to_average = ic_matrix.columns.tolist()

    subset = ic_matrix[horizons_to_average]

    # Drop features with all NaN values
    valid_mask = subset.notna().any(axis=1)
    subset_valid = subset[valid_mask]

    def get_best_ic(row):
        """Get IC at horizon with highest absolute IC, handling NaN."""
        valid_row = row.dropna()
        if len(valid_row) == 0:
            return np.nan
        best_idx = valid_row.abs().idxmax()
        return row[best_idx]

    ranking = pd.DataFrame({
        "avg_abs_ic": subset_valid.abs().mean(axis=1),
        "avg_ic": subset_valid.mean(axis=1),
        "std_ic": subset_valid.std(axis=1),
        "best_horizon": subset_valid.abs().idxmax(axis=1, skipna=True),
        "best_ic": subset_valid.apply(get_best_ic, axis=1),
        "n_significant": (subset_valid.abs() >= min_abs_ic).sum(axis=1),
    })

    # Filter by min_abs_ic
    if min_abs_ic > 0:
        ranking = ranking[ranking["avg_abs_ic"] >= min_abs_ic]

    # Sort by average absolute IC
    ranking = ranking.sort_values("avg_abs_ic", ascending=False)

    return ranking


def identify_horizon_specialists(
    ic_matrix: pd.DataFrame,
    min_abs_ic: float = 0.05,
    min_ratio: float = 1.5,
) -> dict[str, list[str]]:
    """Identify features that specialize in specific horizons.

    Args:
        ic_matrix: IC matrix (features x horizons)
        min_abs_ic: Minimum absolute IC to be considered
        min_ratio: Minimum ratio of best horizon IC to average IC

    Returns:
        Dict mapping horizon to list of specialist features
    """
    specialists = {col: [] for col in ic_matrix.columns}

    for feature in ic_matrix.index:
        row = ic_matrix.loc[feature]
        abs_row = row.abs()

        # Find best horizon
        best_horizon = abs_row.idxmax()
        best_ic = abs_row[best_horizon]
        avg_ic = abs_row.mean()

        # Check if specialist
        if best_ic >= min_abs_ic and (avg_ic == 0 or best_ic / avg_ic >= min_ratio):
            specialists[best_horizon].append(feature)

    return specialists
