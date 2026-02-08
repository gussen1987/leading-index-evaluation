"""Weight optimization module for block and feature weights."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import numpy as np
from scipy import optimize

from risk_index.core.logger import get_logger
from risk_index.core.utils_math import forward_return, information_coefficient

logger = get_logger(__name__)


def optimize_block_weights(
    blocks_df: pd.DataFrame,
    target_series: pd.Series,
    horizon: int,
    method: Literal["equal", "inverse_variance", "max_ic", "ic_weighted"] = "equal",
    min_weight: float = 0.05,
    max_weight: float = 0.50,
) -> dict[str, float]:
    """Optimize block weights for target horizon.

    Methods:
    - equal: 1/N weighting (baseline)
    - inverse_variance: Weight by 1/variance
    - ic_weighted: Weight proportional to IC
    - max_ic: Maximize information coefficient (constrained optimization)

    Args:
        blocks_df: DataFrame with block scores as columns
        target_series: Price series for forward returns
        horizon: Forward horizon in weeks
        method: Optimization method
        min_weight: Minimum weight per block
        max_weight: Maximum weight per block

    Returns:
        Dict mapping block name to weight
    """
    block_names = blocks_df.columns.tolist()
    n_blocks = len(block_names)

    if n_blocks == 0:
        return {}

    logger.info(f"Optimizing weights for {n_blocks} blocks using method: {method}")

    # Compute forward returns
    fwd_ret = forward_return(target_series, horizon=horizon, log=True)

    # Align data
    aligned = blocks_df.copy()
    aligned["_fwd_ret"] = fwd_ret
    aligned = aligned.dropna()

    if len(aligned) < 52:
        logger.warning(f"Insufficient data for optimization: {len(aligned)} rows")
        return {name: 1.0 / n_blocks for name in block_names}

    if method == "equal":
        weights = {name: 1.0 / n_blocks for name in block_names}

    elif method == "inverse_variance":
        weights = _inverse_variance_weights(aligned[block_names], min_weight, max_weight)

    elif method == "ic_weighted":
        weights = _ic_weighted(aligned[block_names], aligned["_fwd_ret"], min_weight, max_weight)

    elif method == "max_ic":
        weights = _maximize_ic(aligned[block_names], aligned["_fwd_ret"], min_weight, max_weight)

    else:
        raise ValueError(f"Unknown optimization method: {method}")

    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    logger.info(f"Optimized weights: {weights}")
    return weights


def _inverse_variance_weights(
    blocks_df: pd.DataFrame,
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    """Compute inverse variance weights."""
    variances = blocks_df.var()
    inv_var = 1.0 / variances.replace(0, np.nan)
    inv_var = inv_var.fillna(inv_var.median())

    # Normalize
    weights = inv_var / inv_var.sum()

    # Apply bounds
    weights = weights.clip(lower=min_weight, upper=max_weight)

    return weights.to_dict()


def _ic_weighted(
    blocks_df: pd.DataFrame,
    fwd_ret: pd.Series,
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    """Weight proportional to individual IC."""
    ics = {}
    for col in blocks_df.columns:
        ic = information_coefficient(blocks_df[col], fwd_ret, method="spearman")
        ics[col] = abs(ic) if not np.isnan(ic) else 0

    # Normalize
    total_ic = sum(ics.values())
    if total_ic == 0:
        return {k: 1.0 / len(ics) for k in ics}

    weights = {k: v / total_ic for k, v in ics.items()}

    # Apply bounds
    weights = {k: max(min_weight, min(max_weight, v)) for k, v in weights.items()}

    return weights


def _maximize_ic(
    blocks_df: pd.DataFrame,
    fwd_ret: pd.Series,
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    """Maximize IC through constrained optimization."""
    block_names = blocks_df.columns.tolist()
    n_blocks = len(block_names)

    blocks_array = blocks_df.values
    returns_array = fwd_ret.values

    def negative_ic(weights):
        """Negative IC (to minimize)."""
        weights = np.array(weights)
        composite = blocks_array @ weights
        # Spearman correlation
        from scipy.stats import spearmanr
        mask = ~(np.isnan(composite) | np.isnan(returns_array))
        if mask.sum() < 10:
            return 1.0  # Penalty
        corr, _ = spearmanr(composite[mask], returns_array[mask])
        return -corr if not np.isnan(corr) else 1.0

    # Initial guess: equal weights
    x0 = np.ones(n_blocks) / n_blocks

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Bounds: each weight between min and max
    bounds = [(min_weight, max_weight) for _ in range(n_blocks)]

    # Optimize
    try:
        result = optimize.minimize(
            negative_ic,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        if result.success:
            weights = result.x
        else:
            logger.warning(f"Optimization did not converge: {result.message}")
            weights = x0

    except Exception as e:
        logger.warning(f"Optimization failed: {e}")
        weights = x0

    return dict(zip(block_names, weights))


def optimize_composite_weights(
    composites_df: pd.DataFrame,
    target_series: pd.Series,
    horizons: list[int],
    method: Literal["equal", "inverse_variance", "max_ic"] = "equal",
) -> dict[str, float]:
    """Optimize weights across multiple composites.

    Args:
        composites_df: DataFrame with composite scores
        target_series: Price series
        horizons: List of horizons to average over
        method: Optimization method

    Returns:
        Dict mapping composite name to weight
    """
    if composites_df.empty:
        return {}

    composite_names = composites_df.columns.tolist()

    # Compute average IC across horizons for each composite
    avg_ics = {}
    for composite in composite_names:
        horizon_ics = []
        for h in horizons:
            fwd_ret = forward_return(target_series, horizon=h, log=True)
            ic = information_coefficient(composites_df[composite], fwd_ret)
            if not np.isnan(ic):
                horizon_ics.append(ic)
        avg_ics[composite] = np.mean(horizon_ics) if horizon_ics else 0

    if method == "equal":
        weights = {k: 1.0 / len(composite_names) for k in composite_names}

    elif method == "inverse_variance":
        variances = composites_df.var()
        inv_var = 1.0 / variances.replace(0, variances.median())
        weights = (inv_var / inv_var.sum()).to_dict()

    elif method == "max_ic":
        # Weight by IC magnitude
        total = sum(abs(v) for v in avg_ics.values())
        if total > 0:
            weights = {k: abs(v) / total for k, v in avg_ics.items()}
        else:
            weights = {k: 1.0 / len(composite_names) for k in composite_names}

    return weights


def generate_optimized_composites_config(
    block_weights: dict[str, dict[str, float]],
    base_config: dict | None = None,
) -> dict:
    """Generate optimized composites.yml config.

    Args:
        block_weights: Dict mapping speed -> {block: weight}
        base_config: Optional base config to modify

    Returns:
        New composites config dict
    """
    if base_config is None:
        base_config = {
            "correlation_policy": {
                "max_block_corr": 0.90,
                "action_if_exceeded": "cap_combined_weight",
                "max_combined_weight": 0.25,
            },
            "composites": [],
        }

    composites = []

    # Speed configurations
    speed_configs = {
        "fast": {"z_window": 104, "target_horizons": [4, 8]},
        "medium": {"z_window": 156, "target_horizons": [13, 26]},
        "slow": {"z_window": 260, "target_horizons": [26, 52]},
    }

    for speed, config in speed_configs.items():
        if speed not in block_weights:
            continue

        weights = block_weights[speed]
        if not weights:
            continue

        # Convert weights to block list
        blocks = [
            {"block": block, "weight": round(weight, 3)}
            for block, weight in sorted(weights.items(), key=lambda x: -x[1])
            if weight > 0.01  # Filter tiny weights
        ]

        composite = {
            "name": f"{speed}_composite",
            "speed": speed,
            "z_window": config["z_window"],
            "target_horizons": config["target_horizons"],
            "objective": f"Optimized for {speed} horizon IC maximization",
            "blocks": blocks,
        }
        composites.append(composite)

    return {
        "correlation_policy": base_config.get("correlation_policy", {}),
        "composites": composites,
    }


def compare_weights(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    blocks_df: pd.DataFrame,
    target_series: pd.Series,
    horizon: int,
) -> dict:
    """Compare old vs new weights in terms of IC.

    Args:
        old_weights: Previous weight configuration
        new_weights: New optimized weights
        blocks_df: Block scores DataFrame
        target_series: Target price series
        horizon: Forward horizon

    Returns:
        Comparison dict with ICs and improvement
    """
    fwd_ret = forward_return(target_series, horizon=horizon, log=True)

    def compute_composite_ic(weights):
        """Compute IC for given weights."""
        aligned = blocks_df.copy()
        aligned["_fwd_ret"] = fwd_ret
        aligned = aligned.dropna()

        if len(aligned) < 10:
            return np.nan

        composite = sum(
            aligned[block] * weight
            for block, weight in weights.items()
            if block in aligned.columns
        ) / sum(weights.values())

        return information_coefficient(composite, aligned["_fwd_ret"])

    old_ic = compute_composite_ic(old_weights)
    new_ic = compute_composite_ic(new_weights)

    improvement = (new_ic - old_ic) / abs(old_ic) if old_ic != 0 and not np.isnan(old_ic) else np.nan

    return {
        "old_ic": old_ic,
        "new_ic": new_ic,
        "ic_improvement": new_ic - old_ic if not (np.isnan(old_ic) or np.isnan(new_ic)) else np.nan,
        "pct_improvement": improvement * 100 if not np.isnan(improvement) else np.nan,
        "old_weights": old_weights,
        "new_weights": new_weights,
    }


def validate_weights(
    weights: dict[str, float],
    blocks_df: pd.DataFrame,
    target_series: pd.Series,
    horizons: list[int],
    min_ic: float = 0.05,
) -> dict:
    """Validate optimized weights on multiple horizons.

    Args:
        weights: Optimized weights
        blocks_df: Block scores
        target_series: Target prices
        horizons: Horizons to validate on
        min_ic: Minimum acceptable IC

    Returns:
        Validation results dict
    """
    results = {
        "weights": weights,
        "horizon_ics": {},
        "avg_ic": None,
        "passes_threshold": False,
    }

    for h in horizons:
        fwd_ret = forward_return(target_series, horizon=h, log=True)
        aligned = blocks_df.copy()
        aligned["_fwd_ret"] = fwd_ret
        aligned = aligned.dropna()

        if len(aligned) < 10:
            results["horizon_ics"][h] = np.nan
            continue

        composite = sum(
            aligned[block] * weight
            for block, weight in weights.items()
            if block in aligned.columns
        ) / sum(weights.values())

        ic = information_coefficient(composite, aligned["_fwd_ret"])
        results["horizon_ics"][h] = ic

    valid_ics = [v for v in results["horizon_ics"].values() if not np.isnan(v)]
    results["avg_ic"] = np.mean(valid_ics) if valid_ics else np.nan
    results["passes_threshold"] = results["avg_ic"] >= min_ic if not np.isnan(results["avg_ic"]) else False

    return results
