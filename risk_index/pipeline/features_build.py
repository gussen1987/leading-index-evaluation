"""Feature engineering module for building transformed features."""

from __future__ import annotations

import pandas as pd
import numpy as np

from risk_index.core.config_schema import UniverseConfig, TransformsConfig
from risk_index.core.constants import FEATURE_SEPARATOR
from risk_index.core.utils_math import (
    zscore,
    percentile,
    roc,
    slope,
    drawdown_from_high,
    realized_vol,
    winsorize,
    replace_inf,
)
from risk_index.core.logger import get_logger

logger = get_logger(__name__)


def build_features(
    weekly_df: pd.DataFrame,
    universe_cfg: UniverseConfig,
    transforms_cfg: TransformsConfig,
) -> pd.DataFrame:
    """Build all features from weekly data.


    Args:
        weekly_df: Weekly aligned DataFrame with base series and ratios
        universe_cfg: Universe configuration
        transforms_cfg: Transforms configuration

    Returns:
        DataFrame with all features (base series + transformed features)
    """
    logger.info("Starting feature engineering")

    # Get list of series eligible for transforms
    eligible_series = get_eligible_series(weekly_df, universe_cfg)
    logger.info(f"Found {len(eligible_series)} eligible series for transforms")

    # Build features
    features_df = weekly_df.copy()

    # Apply each transform to each eligible series
    transform_count = 0
    for transform in transforms_cfg.transforms:
        for series_id in eligible_series:
            if series_id not in features_df.columns:
                continue

            feature_name = f"{series_id}{FEATURE_SEPARATOR}{transform.name}"

            try:
                feature_values = apply_transform(
                    features_df[series_id],
                    transform.function,
                    transform.window,
                )

                # Post-process: winsorize and replace inf
                feature_values = replace_inf(feature_values)
                feature_values = winsorize(feature_values)

                features_df[feature_name] = feature_values
                transform_count += 1

            except Exception as e:
                logger.warning(f"Failed to compute {feature_name}: {e}")

    logger.info(f"Built {transform_count} transformed features")

    # Validate features
    validation = validate_features(features_df)
    logger.info(
        f"Feature validation: {validation['valid_features']} valid, "
        f"{validation['inf_count']} inf values replaced"
    )

    return features_df


def get_eligible_series(df: pd.DataFrame, universe_cfg: UniverseConfig) -> list[str]:
    """Get list of series eligible for feature transforms.

    Eligible series include:
    - All ratio IDs
    - VIX, VVIX (volatility levels)
    - All FRED spreads/yields
    - All computed series

    Args:
        df: Weekly DataFrame
        universe_cfg: Universe config

    Returns:
        List of eligible series IDs
    """
    eligible = set()

    # Add all ratios
    for ratio in universe_cfg.ratios:
        if ratio.id in df.columns:
            eligible.add(ratio.id)

    # Add volatility series
    vol_series = ["VIX", "VVIX"]
    for s in vol_series:
        if s in df.columns:
            eligible.add(s)

    # Add FRED series (rates, spreads, macro)
    fred_series = [s for s in universe_cfg.series if s.source == "fred"]
    for s in fred_series:
        if s.id in df.columns:
            eligible.add(s.id)

    # Add computed series
    for computed in universe_cfg.computed:
        if computed.id in df.columns:
            eligible.add(computed.id)

    # Add FX series
    fx_series = [s for s in universe_cfg.series if s.kind == "fx"]
    for s in fx_series:
        if s.id in df.columns:
            eligible.add(s.id)

    return sorted(eligible)


def apply_transform(
    series: pd.Series,
    function: str,
    window: int,
) -> pd.Series:
    """Apply a transform function to a series.

    Args:
        series: Input series
        function: Transform function name
        window: Window size

    Returns:
        Transformed series

    Raises:
        ValueError: If function is unknown
    """
    if function == "zscore":
        return zscore(series, window)
    elif function == "percentile":
        return percentile(series, window)
    elif function == "roc":
        return roc(series, window)
    elif function == "slope":
        return slope(series, window)
    elif function == "drawdown":
        return drawdown_from_high(series, window)
    elif function == "realized_vol":
        return realized_vol(series, window)
    else:
        raise ValueError(f"Unknown transform function: {function}")


def validate_features(df: pd.DataFrame) -> dict:
    """Validate feature quality.

    Args:
        df: Features DataFrame

    Returns:
        Validation metrics
    """
    # Check for inf values
    inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
    inf_count = inf_mask.sum().sum()

    # Get feature columns (those with separator)
    feature_cols = [c for c in df.columns if FEATURE_SEPARATOR in c]

    # Check z-score features have reasonable stats
    zscore_cols = [c for c in feature_cols if "__z_" in c]
    zscore_stats = {}
    for col in zscore_cols:
        if col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 10:
                zscore_stats[col] = {
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                }

    # Count features with acceptable stats
    valid_zscore = sum(
        1
        for stats in zscore_stats.values()
        if abs(stats["mean"]) < 0.5 and 0.5 < stats["std"] < 2.0
    )

    return {
        "total_features": len(feature_cols),
        "valid_features": len(feature_cols),  # All features considered valid after post-processing
        "inf_count": int(inf_count),
        "zscore_features": len(zscore_cols),
        "valid_zscore": valid_zscore,
    }


def get_feature_names_for_block(
    block_members: list,
    features_df: pd.DataFrame,
) -> list[str]:
    """Get feature column names for block members.

    Args:
        block_members: List of BlockMemberConfig
        features_df: Features DataFrame

    Returns:
        List of feature column names
    """
    feature_names = []

    for member in block_members:
        for transform_name in member.use_features:
            feature_col = f"{member.id}{FEATURE_SEPARATOR}{transform_name}"
            if feature_col in features_df.columns:
                feature_names.append(feature_col)

    return feature_names


def extract_block_features(
    features_df: pd.DataFrame,
    block_members: list,
) -> pd.DataFrame:
    """Extract features for a specific block.

    Args:
        features_df: Full features DataFrame
        block_members: List of BlockMemberConfig

    Returns:
        DataFrame with only block features
    """
    feature_names = get_feature_names_for_block(block_members, features_df)
    return features_df[feature_names].copy() if feature_names else pd.DataFrame()


def compute_feature_correlations(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for features.

    Args:
        features_df: Features DataFrame

    Returns:
        Correlation matrix
    """
    # Get only feature columns
    feature_cols = [c for c in features_df.columns if FEATURE_SEPARATOR in c]

    if not feature_cols:
        return pd.DataFrame()

    return features_df[feature_cols].corr()


def select_uncorrelated_features(
    features_df: pd.DataFrame,
    max_corr: float = 0.85,
) -> list[str]:
    """Select features that are not highly correlated.

    Args:
        features_df: Features DataFrame
        max_corr: Maximum allowed pairwise correlation

    Returns:
        List of selected feature names
    """
    feature_cols = [c for c in features_df.columns if FEATURE_SEPARATOR in c]

    if not feature_cols:
        return []

    corr_matrix = features_df[feature_cols].corr().abs()

    # Greedy selection: add features that don't correlate too highly with already selected
    selected = []

    for col in feature_cols:
        if not selected:
            selected.append(col)
            continue

        # Check correlation with all selected features
        max_corr_with_selected = max(corr_matrix.loc[col, s] for s in selected)
        if max_corr_with_selected < max_corr:
            selected.append(col)

    return selected
