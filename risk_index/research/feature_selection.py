"""Feature selection module with statistical gates and redundancy removal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np

from risk_index.core.logger import get_logger
from risk_index.core.constants import FAST_HORIZONS, MEDIUM_HORIZONS, SLOW_HORIZONS

logger = get_logger(__name__)


@dataclass
class SelectionRules:
    """Feature selection rules from backtest config."""
    min_abs_ic: float = 0.03
    min_horizons: int = 2
    min_windows_pass: float = 0.60
    min_sign_consistency: float = 0.70
    max_pair_corr: float = 0.85


def select_features(
    wf_aggregated: pd.DataFrame,
    features_df: pd.DataFrame,
    selection_rules: SelectionRules | dict | None = None,
    fast_horizons: list[int] | None = None,
    medium_horizons: list[int] | None = None,
    slow_horizons: list[int] | None = None,
) -> dict[str, list[str]]:
    """Apply selection gates from config/backtest.yml.

    Selection gates:
    - min_abs_ic: 0.03 (must predict with IC >= 0.03)
    - min_horizons: 2 (must work at 2+ horizons)
    - min_windows_pass: 0.60 (pass in 60%+ of walk-forward windows)
    - min_sign_consistency: 0.70 (sign stable 70%+ of time)
    - max_pair_corr: 0.85 (drop redundant features)

    Args:
        wf_aggregated: Aggregated walk-forward results (feature × horizon)
        features_df: Original features DataFrame for correlation check
        selection_rules: SelectionRules or dict with selection parameters
        fast_horizons: Horizons for fast composite (default: [4, 8])
        medium_horizons: Horizons for medium composite (default: [13, 26])
        slow_horizons: Horizons for slow composite (default: [26, 52])

    Returns:
        Dict with selected features per speed:
        {
            'fast': [selected_features],
            'medium': [selected_features],
            'slow': [selected_features],
        }
    """
    # Use defaults if not provided
    if selection_rules is None:
        selection_rules = SelectionRules()
    elif isinstance(selection_rules, dict):
        selection_rules = SelectionRules(**selection_rules)

    if fast_horizons is None:
        fast_horizons = FAST_HORIZONS
    if medium_horizons is None:
        medium_horizons = MEDIUM_HORIZONS
    if slow_horizons is None:
        slow_horizons = SLOW_HORIZONS

    logger.info(f"Selecting features with rules: {selection_rules}")

    results = {
        "fast": _select_for_speed(
            wf_aggregated, features_df, fast_horizons, selection_rules, "fast"
        ),
        "medium": _select_for_speed(
            wf_aggregated, features_df, medium_horizons, selection_rules, "medium"
        ),
        "slow": _select_for_speed(
            wf_aggregated, features_df, slow_horizons, selection_rules, "slow"
        ),
    }

    for speed, selected in results.items():
        logger.info(f"{speed}: {len(selected)} features selected")

    return results


def _select_for_speed(
    wf_aggregated: pd.DataFrame,
    features_df: pd.DataFrame,
    horizons: list[int],
    rules: SelectionRules,
    speed_name: str,
) -> list[str]:
    """Select features for a specific speed category.

    Args:
        wf_aggregated: Aggregated walk-forward results
        features_df: Original features for correlation
        horizons: Target horizons for this speed
        rules: Selection rules
        speed_name: Name for logging

    Returns:
        List of selected feature names
    """
    if wf_aggregated.empty:
        logger.warning(f"Empty walk-forward results for {speed_name}")
        return []

    # Reset index if needed
    if isinstance(wf_aggregated.index, pd.MultiIndex):
        wf_df = wf_aggregated.reset_index()
    else:
        wf_df = wf_aggregated.copy()

    # Filter to relevant horizons
    horizon_mask = wf_df["horizon"].isin(horizons)
    subset = wf_df[horizon_mask].copy()

    if subset.empty:
        logger.warning(f"No data for horizons {horizons}")
        return []

    # Apply selection gates
    candidates = _apply_selection_gates(subset, rules, horizons)

    if not candidates:
        logger.warning(f"No features passed selection gates for {speed_name}")
        return []

    # Remove redundant features
    selected = _remove_redundant_features(
        candidates, features_df, rules.max_pair_corr
    )

    return selected


def _apply_selection_gates(
    wf_subset: pd.DataFrame,
    rules: SelectionRules,
    horizons: list[int],
) -> list[str]:
    """Apply IC and consistency gates.

    Args:
        wf_subset: Walk-forward results for target horizons
        rules: Selection rules
        horizons: Target horizons

    Returns:
        List of features passing all gates
    """
    passing_features = []

    # Group by feature
    for feature, group in wf_subset.groupby("feature"):
        # Gate 1: Minimum absolute IC
        mean_abs_ic = group["mean_test_ic"].abs().mean()
        if mean_abs_ic < rules.min_abs_ic:
            continue

        # Gate 2: Must be predictive at min_horizons
        significant_horizons = (group["mean_test_ic"].abs() >= rules.min_abs_ic).sum()
        if significant_horizons < rules.min_horizons:
            continue

        # Gate 3: Pass rate across walk-forward windows
        avg_pass_rate = group["pass_rate"].mean()
        if pd.isna(avg_pass_rate) or avg_pass_rate < rules.min_windows_pass:
            continue

        # Gate 4: Sign consistency
        avg_sign_consistency = group["sign_consistency"].mean()
        if pd.isna(avg_sign_consistency) or avg_sign_consistency < rules.min_sign_consistency:
            continue

        passing_features.append(feature)

    logger.debug(f"Features passing gates: {len(passing_features)}")
    return passing_features


def _remove_redundant_features(
    candidates: list[str],
    features_df: pd.DataFrame,
    max_corr: float,
) -> list[str]:
    """Remove highly correlated features, keeping the first in list (higher IC).

    Args:
        candidates: List of candidate features (assumed sorted by IC)
        features_df: Features DataFrame for correlation
        max_corr: Maximum allowed pairwise correlation

    Returns:
        List of uncorrelated features
    """
    if len(candidates) <= 1:
        return candidates

    # Filter to available features
    available = [f for f in candidates if f in features_df.columns]

    if len(available) <= 1:
        return available

    # Compute correlation matrix
    subset_df = features_df[available].dropna(how="all")
    if len(subset_df) < 10:
        return available

    corr_matrix = subset_df.corr().abs()

    # Greedy selection
    selected = []
    for feature in available:
        if not selected:
            selected.append(feature)
            continue

        # Check correlation with all selected
        max_corr_with_selected = max(
            corr_matrix.loc[feature, s] if feature in corr_matrix.index and s in corr_matrix.columns else 0
            for s in selected
        )

        if max_corr_with_selected < max_corr:
            selected.append(feature)

    logger.debug(f"After redundancy removal: {len(selected)}/{len(available)} features")
    return selected


def compute_feature_scores(
    wf_aggregated: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute composite scores for feature ranking.

    Args:
        wf_aggregated: Aggregated walk-forward results
        horizons: Optional horizons to filter

    Returns:
        DataFrame with features and composite scores
    """
    if wf_aggregated.empty:
        return pd.DataFrame()

    # Reset index if needed
    if isinstance(wf_aggregated.index, pd.MultiIndex):
        wf_df = wf_aggregated.reset_index()
    else:
        wf_df = wf_aggregated.copy()

    # Filter horizons
    if horizons is not None:
        wf_df = wf_df[wf_df["horizon"].isin(horizons)]

    # Compute scores per feature
    scores = []
    for feature, group in wf_df.groupby("feature"):
        # Average metrics across horizons
        avg_test_ic = group["mean_test_ic"].mean()
        avg_abs_ic = group["mean_test_ic"].abs().mean()
        avg_sign_consistency = group["sign_consistency"].mean()
        avg_pass_rate = group["pass_rate"].mean()
        std_test_ic = group["std_test_ic"].mean()

        # Information ratio (IC / volatility)
        ir = avg_test_ic / std_test_ic if std_test_ic > 0 else 0

        # Composite score: weighted combination
        # Higher IC, higher consistency, higher pass rate = better
        composite = (
            0.40 * avg_abs_ic +
            0.25 * avg_sign_consistency +
            0.20 * avg_pass_rate +
            0.15 * (ir / 2)  # Scaled IR contribution
        )

        scores.append({
            "feature": feature,
            "avg_test_ic": avg_test_ic,
            "avg_abs_ic": avg_abs_ic,
            "sign_consistency": avg_sign_consistency,
            "pass_rate": avg_pass_rate,
            "information_ratio": ir,
            "composite_score": composite,
        })

    result = pd.DataFrame(scores)
    if not result.empty:
        result = result.sort_values("composite_score", ascending=False)
        result = result.reset_index(drop=True)

    return result


def get_selection_report(
    wf_aggregated: pd.DataFrame,
    selected: dict[str, list[str]],
    selection_rules: SelectionRules | dict,
) -> dict:
    """Generate detailed selection report.

    Args:
        wf_aggregated: Walk-forward results
        selected: Dict of selected features per speed
        selection_rules: Selection rules used

    Returns:
        Dict with selection statistics and reasoning
    """
    if isinstance(selection_rules, dict):
        selection_rules = SelectionRules(**selection_rules)

    # Get all unique features
    all_features = set()
    if isinstance(wf_aggregated.index, pd.MultiIndex):
        all_features = set(wf_aggregated.index.get_level_values("feature"))
    else:
        all_features = set(wf_aggregated["feature"].unique())

    # Track rejection reasons
    rejections = {
        "low_ic": [],
        "few_horizons": [],
        "low_pass_rate": [],
        "low_sign_consistency": [],
        "redundant": [],
    }

    # Analyze each feature (simplified - actual rejection tracking would need more state)
    report = {
        "rules_applied": {
            "min_abs_ic": selection_rules.min_abs_ic,
            "min_horizons": selection_rules.min_horizons,
            "min_windows_pass": selection_rules.min_windows_pass,
            "min_sign_consistency": selection_rules.min_sign_consistency,
            "max_pair_corr": selection_rules.max_pair_corr,
        },
        "total_features_evaluated": len(all_features),
        "selected_counts": {speed: len(features) for speed, features in selected.items()},
        "selected_features": selected,
    }

    return report


def assign_features_to_blocks(
    selected_features: list[str],
    universe_blocks: dict,
) -> dict[str, list[str]]:
    """Map selected features back to their signal blocks.

    Args:
        selected_features: List of selected feature names (e.g., "HYG_IEF__z_52w")
        universe_blocks: Block definitions from universe config

    Returns:
        Dict mapping block name to list of selected features for that block
    """
    # Parse feature name to get base series ID
    # Feature format: {series_id}__{transform_name}
    feature_to_base = {}
    for feature in selected_features:
        if "__" in feature:
            base = feature.split("__")[0]
            feature_to_base[feature] = base
        else:
            feature_to_base[feature] = feature

    # Map base series to blocks
    base_to_block = {}
    for block_name, block_config in universe_blocks.items():
        members = block_config.get("members", [])
        for member in members:
            member_id = member.get("id") if isinstance(member, dict) else member
            base_to_block[member_id] = block_name

    # Assign features to blocks
    block_features = {}
    unassigned = []

    for feature, base in feature_to_base.items():
        if base in base_to_block:
            block = base_to_block[base]
            if block not in block_features:
                block_features[block] = []
            block_features[block].append(feature)
        else:
            unassigned.append(feature)

    if unassigned:
        logger.warning(f"{len(unassigned)} features not assigned to blocks")
        block_features["_unassigned"] = unassigned

    return block_features
