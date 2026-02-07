"""Block scoring module for aggregating features into block-level signals."""

from __future__ import annotations

import pandas as pd
import numpy as np

from risk_index.core.config_schema import UniverseConfig, CompositesConfig
from risk_index.core.constants import FEATURE_SEPARATOR, MIN_COVERAGE_RATIO
from risk_index.core.utils_math import cross_sectional_mean, correlation_matrix
from risk_index.core.logger import get_logger

logger = get_logger(__name__)


def build_blocks(
    features_df: pd.DataFrame,
    universe_cfg: UniverseConfig,
    composites_cfg: CompositesConfig,
) -> pd.DataFrame:
    """Build block scores from features.


    Args:
        features_df: DataFrame with all features
        universe_cfg: Universe configuration
        composites_cfg: Composites configuration

    Returns:
        DataFrame with block scores
    """
    logger.info("Building block scores")

    block_scores = {}

    for block in universe_cfg.blocks:
        block_score = compute_block_score(features_df, block)
        if block_score is not None:
            block_scores[block.name] = block_score
            logger.info(f"Computed block score for {block.name}")
        else:
            logger.warning(f"Could not compute block score for {block.name}")

    blocks_df = pd.DataFrame(block_scores, index=features_df.index)

    # Check block correlations and apply policy
    blocks_df = apply_correlation_policy(blocks_df, composites_cfg.correlation_policy)

    # Validate blocks
    validation = validate_blocks(blocks_df)
    logger.info(
        f"Block validation: {validation['valid_blocks']}/{validation['total_blocks']} valid, "
        f"coverage: {validation['avg_coverage']:.1%}"
    )

    return blocks_df


def compute_block_score(
    features_df: pd.DataFrame,
    block,
) -> pd.Series | None:
    """Compute a single block score.

    Args:
        features_df: DataFrame with all features
        block: BlockConfig object

    Returns:
        Block score series or None if insufficient data
    """
    member_features = []

    for member in block.members:
        member_series_list = []

        for transform_name in member.use_features:
            feature_col = f"{member.id}{FEATURE_SEPARATOR}{transform_name}"

            if feature_col in features_df.columns:
                feature_data = features_df[feature_col].copy()

                # Apply inversion if needed
                if member.invert:
                    feature_data = -feature_data

                member_series_list.append(feature_data)

        if member_series_list:
            # Average member's features
            member_avg = pd.concat(member_series_list, axis=1).mean(axis=1)
            member_features.append(member_avg)

    if len(member_features) < 2:
        logger.warning(f"Block {block.name} has fewer than 2 valid members")
        return None

    # Combine all member features
    members_df = pd.concat(member_features, axis=1)

    # Block score = cross-sectional mean of member features
    block_score = members_df.mean(axis=1)

    # Set to NaN where too few features are valid
    valid_count = members_df.notna().sum(axis=1)
    block_score[valid_count < 2] = np.nan

    block_score.name = block.name

    return block_score


def apply_correlation_policy(
    blocks_df: pd.DataFrame,
    correlation_policy,
) -> pd.DataFrame:
    """Apply correlation policy to block scores.

    Args:
        blocks_df: DataFrame with block scores
        correlation_policy: CorrelationPolicyConfig

    Returns:
        Adjusted blocks DataFrame
    """
    if blocks_df.empty or len(blocks_df.columns) < 2:
        return blocks_df

    corr = blocks_df.corr()

    # Find highly correlated block pairs
    high_corr_pairs = []
    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1 :]:
            if abs(corr.loc[col1, col2]) > correlation_policy.max_block_corr:
                high_corr_pairs.append((col1, col2, corr.loc[col1, col2]))
                logger.warning(
                    f"High correlation between {col1} and {col2}: {corr.loc[col1, col2]:.2f}"
                )

    # For now, log warnings but don't modify (weights will be adjusted in composite)
    if high_corr_pairs:
        logger.info(
            f"Found {len(high_corr_pairs)} highly correlated block pairs. "
            f"Will apply {correlation_policy.action_if_exceeded} policy."
        )

    return blocks_df


def get_correlated_block_pairs(
    blocks_df: pd.DataFrame,
    max_corr: float = 0.90,
) -> list[tuple[str, str, float]]:
    """Get list of highly correlated block pairs.

    Args:
        blocks_df: DataFrame with block scores
        max_corr: Maximum allowed correlation

    Returns:
        List of (block1, block2, correlation) tuples
    """
    if blocks_df.empty or len(blocks_df.columns) < 2:
        return []

    corr = blocks_df.corr().abs()
    pairs = []

    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1 :]:
            if corr.loc[col1, col2] > max_corr:
                pairs.append((col1, col2, corr.loc[col1, col2]))

    return pairs


def validate_blocks(blocks_df: pd.DataFrame) -> dict:
    """Validate block scores.

    Args:
        blocks_df: DataFrame with block scores

    Returns:
        Validation metrics
    """
    if blocks_df.empty:
        return {
            "total_blocks": 0,
            "valid_blocks": 0,
            "avg_coverage": 0.0,
            "coverage_by_block": {},
        }

    # Calculate coverage for each block
    coverage = blocks_df.notna().mean()

    # Consider block valid if coverage > 80%
    valid_blocks = (coverage >= MIN_COVERAGE_RATIO).sum()

    # Calculate coverage over last 5 years
    five_years_ago = blocks_df.index.max() - pd.DateOffset(years=5)
    recent = blocks_df[blocks_df.index >= five_years_ago]
    recent_coverage = recent.notna().mean() if not recent.empty else pd.Series()

    return {
        "total_blocks": len(blocks_df.columns),
        "valid_blocks": int(valid_blocks),
        "avg_coverage": float(coverage.mean()),
        "recent_avg_coverage": float(recent_coverage.mean()) if len(recent_coverage) > 0 else 0.0,
        "coverage_by_block": coverage.to_dict(),
    }


def get_block_details(
    features_df: pd.DataFrame,
    block,
) -> pd.DataFrame:
    """Get detailed feature contributions for a block.

    Args:
        features_df: DataFrame with all features
        block: BlockConfig object

    Returns:
        DataFrame with member-level feature values
    """
    details = {}

    for member in block.members:
        for transform_name in member.use_features:
            feature_col = f"{member.id}{FEATURE_SEPARATOR}{transform_name}"

            if feature_col in features_df.columns:
                label = f"{member.id} ({transform_name})"
                value = features_df[feature_col].copy()

                if member.invert:
                    label += " [inv]"
                    value = -value

                details[label] = value

    return pd.DataFrame(details, index=features_df.index)


def get_block_attribution(
    features_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    universe_cfg: UniverseConfig,
    date: pd.Timestamp | None = None,
) -> dict[str, dict]:
    """Get attribution breakdown for each block at a specific date.

    Args:
        features_df: DataFrame with all features
        blocks_df: DataFrame with block scores
        universe_cfg: Universe configuration
        date: Date for attribution (defaults to latest)

    Returns:
        Dict mapping block name to attribution dict
    """
    if date is None:
        date = blocks_df.index[-1]

    attribution = {}

    for block in universe_cfg.blocks:
        if block.name not in blocks_df.columns:
            continue

        block_attr = {
            "score": blocks_df.loc[date, block.name] if date in blocks_df.index else np.nan,
            "members": {},
        }

        for member in block.members:
            member_features = {}
            for transform_name in member.use_features:
                feature_col = f"{member.id}{FEATURE_SEPARATOR}{transform_name}"
                if feature_col in features_df.columns and date in features_df.index:
                    value = features_df.loc[date, feature_col]
                    if member.invert:
                        value = -value
                    member_features[transform_name] = value

            if member_features:
                block_attr["members"][member.id] = {
                    "features": member_features,
                    "invert": member.invert,
                    "avg": np.nanmean(list(member_features.values())),
                }

        attribution[block.name] = block_attr

    return attribution
