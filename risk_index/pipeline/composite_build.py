"""Composite signal construction module."""

from __future__ import annotations

import pandas as pd
import numpy as np

from risk_index.core.config_schema import CompositesConfig
from risk_index.core.constants import (
    COL_COMPOSITE_FAST,
    COL_COMPOSITE_MEDIUM,
    COL_COMPOSITE_SLOW,
)
from risk_index.core.utils_math import zscore
from risk_index.core.logger import get_logger

logger = get_logger(__name__)


def build_composites(
    blocks_df: pd.DataFrame,
    composites_cfg: CompositesConfig,
) -> pd.DataFrame:
    """Build composite signals from block scores.


    Args:
        blocks_df: DataFrame with block scores
        composites_cfg: Composites configuration

    Returns:
        DataFrame with composite scores
    """
    logger.info("Building composite signals")

    composites = {}

    for composite_def in composites_cfg.composites:
        composite_score = compute_composite(
            blocks_df,
            composite_def,
            composites_cfg.correlation_policy,
        )

        if composite_score is not None:
            # Map speed to column name
            col_name = {
                "fast": COL_COMPOSITE_FAST,
                "medium": COL_COMPOSITE_MEDIUM,
                "slow": COL_COMPOSITE_SLOW,
            }.get(composite_def.speed, f"composite_{composite_def.speed}")

            composites[col_name] = composite_score
            logger.info(f"Built {col_name} composite")
        else:
            logger.warning(f"Could not build {composite_def.speed} composite")

    composites_df = pd.DataFrame(composites, index=blocks_df.index)

    # Validate composites
    validation = validate_composites(composites_df)
    logger.info(f"Composite validation: pairwise correlations = {validation['pairwise_corr']}")

    return composites_df


def compute_composite(
    blocks_df: pd.DataFrame,
    composite_def,
    correlation_policy,
) -> pd.Series | None:
    """Compute a single composite signal.

    Args:
        blocks_df: DataFrame with block scores
        composite_def: CompositeDefinition
        correlation_policy: CorrelationPolicyConfig

    Returns:
        Composite score series or None
    """
    # Get weights for each block
    weights = {}
    for block_weight in composite_def.blocks:
        if block_weight.block in blocks_df.columns:
            weights[block_weight.block] = block_weight.weight

    if not weights:
        logger.warning(f"No blocks found for composite {composite_def.name}")
        return None

    # Check for correlated blocks and adjust weights if needed
    adjusted_weights = adjust_weights_for_correlation(
        blocks_df,
        weights,
        correlation_policy,
    )

    # Compute weighted sum
    weighted_sum = pd.Series(0.0, index=blocks_df.index)
    total_weight = 0.0

    for block_name, weight in adjusted_weights.items():
        if block_name in blocks_df.columns:
            block_data = blocks_df[block_name].fillna(0)
            weighted_sum += block_data * weight
            total_weight += weight

    if total_weight == 0:
        return None

    # Normalize by total weight
    raw_composite = weighted_sum / total_weight

    # Apply rolling z-score standardization
    composite_z = zscore(raw_composite, composite_def.z_window)

    composite_z.name = composite_def.name

    return composite_z


def adjust_weights_for_correlation(
    blocks_df: pd.DataFrame,
    weights: dict[str, float],
    correlation_policy,
) -> dict[str, float]:
    """Adjust block weights based on correlation policy.

    Args:
        blocks_df: DataFrame with block scores
        weights: Original weights
        correlation_policy: CorrelationPolicyConfig

    Returns:
        Adjusted weights
    """
    block_names = list(weights.keys())

    if len(block_names) < 2:
        return weights

    # Get correlations for blocks in this composite
    available = [b for b in block_names if b in blocks_df.columns]
    if len(available) < 2:
        return weights

    corr = blocks_df[available].corr()

    # Find highly correlated pairs
    adjusted_weights = weights.copy()

    for i, b1 in enumerate(available):
        for b2 in available[i + 1 :]:
            if abs(corr.loc[b1, b2]) > correlation_policy.max_block_corr:
                if correlation_policy.action_if_exceeded == "cap_combined_weight":
                    # Cap combined weight
                    combined = adjusted_weights.get(b1, 0) + adjusted_weights.get(b2, 0)
                    if combined > correlation_policy.max_combined_weight:
                        scale = correlation_policy.max_combined_weight / combined
                        if b1 in adjusted_weights:
                            adjusted_weights[b1] *= scale
                        if b2 in adjusted_weights:
                            adjusted_weights[b2] *= scale

                        logger.info(
                            f"Scaled weights for {b1}, {b2} due to high correlation "
                            f"({corr.loc[b1, b2]:.2f})"
                        )

    return adjusted_weights


def validate_composites(composites_df: pd.DataFrame) -> dict:
    """Validate composite signals.

    Args:
        composites_df: DataFrame with composite scores

    Returns:
        Validation metrics
    """
    if composites_df.empty or len(composites_df.columns) < 2:
        return {"pairwise_corr": {}, "max_corr": 0.0, "valid": True}

    corr = composites_df.corr()

    pairwise = {}
    max_corr = 0.0

    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            pair_corr = abs(corr.loc[c1, c2])
            pairwise[f"{c1} vs {c2}"] = round(pair_corr, 3)
            max_corr = max(max_corr, pair_corr)

    # Valid if pairwise correlations < 0.92
    valid = max_corr < 0.92

    if not valid:
        logger.warning(f"Composites too correlated: max |r| = {max_corr:.2f}")

    return {
        "pairwise_corr": pairwise,
        "max_corr": round(max_corr, 3),
        "valid": valid,
    }


def get_composite_weights(composites_cfg: CompositesConfig) -> dict[str, dict[str, float]]:
    """Get block weights for each composite.

    Args:
        composites_cfg: Composites configuration

    Returns:
        Dict mapping composite speed to block weights
    """
    result = {}
    for composite in composites_cfg.composites:
        weights = {bw.block: bw.weight for bw in composite.blocks}
        result[composite.speed] = weights
    return result


def get_composite_attribution(
    blocks_df: pd.DataFrame,
    composites_df: pd.DataFrame,
    composites_cfg: CompositesConfig,
    date: pd.Timestamp | None = None,
) -> dict[str, dict]:
    """Get attribution breakdown for composites.

    Args:
        blocks_df: DataFrame with block scores
        composites_df: DataFrame with composite scores
        composites_cfg: Composites configuration
        date: Date for attribution (defaults to latest)

    Returns:
        Dict mapping composite name to attribution
    """
    if date is None:
        date = composites_df.index[-1]

    attribution = {}

    for composite_def in composites_cfg.composites:
        col_name = {
            "fast": COL_COMPOSITE_FAST,
            "medium": COL_COMPOSITE_MEDIUM,
            "slow": COL_COMPOSITE_SLOW,
        }.get(composite_def.speed)

        if col_name not in composites_df.columns:
            continue

        attr = {
            "score": composites_df.loc[date, col_name] if date in composites_df.index else np.nan,
            "blocks": {},
        }

        # Get block contributions
        total_weight = sum(bw.weight for bw in composite_def.blocks)

        for block_weight in composite_def.blocks:
            if block_weight.block in blocks_df.columns and date in blocks_df.index:
                block_score = blocks_df.loc[date, block_weight.block]
                contribution = (block_score * block_weight.weight) / total_weight

                attr["blocks"][block_weight.block] = {
                    "score": block_score,
                    "weight": block_weight.weight,
                    "contribution": contribution,
                }

        attribution[composite_def.speed] = attr

    return attribution
