"""Attribution module for explaining regime drivers."""

from __future__ import annotations

import pandas as pd
import numpy as np

from risk_index.core.config_schema import UniverseConfig, CompositesConfig
from risk_index.core.constants import FEATURE_SEPARATOR
from risk_index.core.logger import get_logger

logger = get_logger(__name__)


def compute_full_attribution(
    features_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    composites_df: pd.DataFrame,
    universe_cfg: UniverseConfig,
    composites_cfg: CompositesConfig,
    date: pd.Timestamp | None = None,
) -> dict:
    """Compute full attribution breakdown.


    Args:
        features_df: Features DataFrame
        blocks_df: Blocks DataFrame
        composites_df: Composites DataFrame
        universe_cfg: Universe configuration
        composites_cfg: Composites configuration
        date: Date for attribution (defaults to latest)

    Returns:
        Attribution dict with block and feature contributions
    """
    if date is None:
        date = composites_df.index[-1]

    attribution = {
        "date": date.strftime("%Y-%m-%d"),
        "composites": {},
    }

    for composite_def in composites_cfg.composites:
        speed = composite_def.speed
        col_name = f"composite_{speed}"

        if col_name not in composites_df.columns:
            continue

        composite_score = (
            composites_df.loc[date, col_name] if date in composites_df.index else np.nan
        )

        # Get block contributions
        block_contributions = []
        total_weight = sum(bw.weight for bw in composite_def.blocks)

        for block_weight in composite_def.blocks:
            block_name = block_weight.block

            if block_name not in blocks_df.columns:
                continue

            block_score = blocks_df.loc[date, block_name] if date in blocks_df.index else np.nan
            weighted_contrib = (block_score * block_weight.weight) / total_weight if total_weight > 0 else 0

            block_contributions.append({
                "block": block_name,
                "score": round(block_score, 3) if not pd.isna(block_score) else None,
                "weight": block_weight.weight,
                "contribution": round(weighted_contrib, 3) if not pd.isna(weighted_contrib) else None,
            })

        # Sort by absolute contribution
        block_contributions.sort(key=lambda x: abs(x["contribution"] or 0), reverse=True)

        attribution["composites"][speed] = {
            "score": round(composite_score, 3) if not pd.isna(composite_score) else None,
            "blocks": block_contributions,
        }

    # Get top feature contributors
    attribution["top_positive_features"] = get_top_features(features_df, universe_cfg, date, top_n=5, positive=True)
    attribution["top_negative_features"] = get_top_features(features_df, universe_cfg, date, top_n=5, positive=False)

    return attribution


def get_top_features(
    features_df: pd.DataFrame,
    universe_cfg: UniverseConfig,
    date: pd.Timestamp,
    top_n: int = 5,
    positive: bool = True,
) -> list[dict]:
    """Get top contributing features.

    Args:
        features_df: Features DataFrame
        universe_cfg: Universe configuration
        date: Date for analysis
        top_n: Number of features to return
        positive: If True, return highest; if False, return lowest

    Returns:
        List of feature dicts with values
    """
    if date not in features_df.index:
        return []

    row = features_df.loc[date]

    # Get only transformed features
    feature_cols = [c for c in row.index if FEATURE_SEPARATOR in c]

    if not feature_cols:
        return []

    feature_values = row[feature_cols].dropna()

    if feature_values.empty:
        return []

    # Sort
    if positive:
        sorted_features = feature_values.sort_values(ascending=False)
    else:
        sorted_features = feature_values.sort_values(ascending=True)

    result = []
    for feat_name, value in sorted_features.head(top_n).items():
        # Parse feature name
        parts = feat_name.split(FEATURE_SEPARATOR)
        series_id = parts[0] if parts else feat_name
        transform = parts[1] if len(parts) > 1 else ""

        result.append({
            "feature": feat_name,
            "series": series_id,
            "transform": transform,
            "value": round(value, 3),
        })

    return result


def compute_week_over_week_changes(
    features_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    composites_df: pd.DataFrame,
    date: pd.Timestamp | None = None,
) -> dict:
    """Compute week-over-week changes in key metrics.

    Args:
        features_df: Features DataFrame
        blocks_df: Blocks DataFrame
        composites_df: Composites DataFrame
        date: Current date (defaults to latest)

    Returns:
        Dict with WoW changes
    """
    if date is None:
        date = composites_df.index[-1]

    # Find previous week
    date_idx = composites_df.index.get_loc(date)
    if date_idx == 0:
        return {"date": date.strftime("%Y-%m-%d"), "changes": {}}

    prev_date = composites_df.index[date_idx - 1]

    changes = {
        "date": date.strftime("%Y-%m-%d"),
        "prev_date": prev_date.strftime("%Y-%m-%d"),
        "composites": {},
        "blocks": {},
    }

    # Composite changes
    for col in composites_df.columns:
        if "composite" in col:
            curr = composites_df.loc[date, col]
            prev = composites_df.loc[prev_date, col]
            change = curr - prev if not (pd.isna(curr) or pd.isna(prev)) else None
            changes["composites"][col] = {
                "current": round(curr, 3) if not pd.isna(curr) else None,
                "previous": round(prev, 3) if not pd.isna(prev) else None,
                "change": round(change, 3) if change is not None else None,
            }

    # Block changes
    for col in blocks_df.columns:
        curr = blocks_df.loc[date, col] if date in blocks_df.index else np.nan
        prev = blocks_df.loc[prev_date, col] if prev_date in blocks_df.index else np.nan
        change = curr - prev if not (pd.isna(curr) or pd.isna(prev)) else None
        changes["blocks"][col] = {
            "current": round(curr, 3) if not pd.isna(curr) else None,
            "previous": round(prev, 3) if not pd.isna(prev) else None,
            "change": round(change, 3) if change is not None else None,
        }

    # Sort blocks by absolute change
    changes["blocks"] = dict(
        sorted(
            changes["blocks"].items(),
            key=lambda x: abs(x[1]["change"] or 0),
            reverse=True,
        )
    )

    return changes


def get_key_drivers(
    attribution: dict,
    changes: dict,
) -> dict:
    """Extract key drivers from attribution and changes.

    Args:
        attribution: Attribution dict
        changes: WoW changes dict

    Returns:
        Key drivers summary
    """
    drivers = {
        "bullish": [],
        "bearish": [],
        "improving": [],
        "deteriorating": [],
    }

    # From attribution - top positive/negative features
    for feat in attribution.get("top_positive_features", [])[:3]:
        drivers["bullish"].append(f"{feat['series']} ({feat['transform']}): {feat['value']:.2f}")

    for feat in attribution.get("top_negative_features", [])[:3]:
        drivers["bearish"].append(f"{feat['series']} ({feat['transform']}): {feat['value']:.2f}")

    # From changes - improving/deteriorating blocks
    block_changes = changes.get("blocks", {})
    for block_name, data in list(block_changes.items())[:5]:
        change = data.get("change")
        if change is not None:
            if change > 0.1:
                drivers["improving"].append(f"{block_name}: +{change:.2f}")
            elif change < -0.1:
                drivers["deteriorating"].append(f"{block_name}: {change:.2f}")

    return drivers


def format_attribution_text(attribution: dict, changes: dict) -> str:
    """Format attribution as human-readable text.

    Args:
        attribution: Attribution dict
        changes: WoW changes dict

    Returns:
        Formatted text string
    """
    lines = [f"Risk Regime Attribution ({attribution['date']})", "=" * 50, ""]

    # Composite scores
    lines.append("COMPOSITE SCORES:")
    for speed, data in attribution.get("composites", {}).items():
        score = data.get("score", "N/A")
        lines.append(f"  {speed.capitalize()}: {score}")
    lines.append("")

    # Key drivers
    drivers = get_key_drivers(attribution, changes)

    if drivers["bullish"]:
        lines.append("BULLISH SIGNALS:")
        for d in drivers["bullish"]:
            lines.append(f"  + {d}")
        lines.append("")

    if drivers["bearish"]:
        lines.append("BEARISH SIGNALS:")
        for d in drivers["bearish"]:
            lines.append(f"  - {d}")
        lines.append("")

    # Week-over-week changes
    lines.append("WEEK-OVER-WEEK CHANGES:")
    if drivers["improving"]:
        lines.append("  Improving:")
        for d in drivers["improving"]:
            lines.append(f"    {d}")

    if drivers["deteriorating"]:
        lines.append("  Deteriorating:")
        for d in drivers["deteriorating"]:
            lines.append(f"    {d}")

    return "\n".join(lines)
