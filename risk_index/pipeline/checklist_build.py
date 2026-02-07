"""Bull Market Checklist engine."""

from __future__ import annotations

import pandas as pd
import numpy as np

from risk_index.core.config_schema import ChecklistConfig
from risk_index.core.types import Signal, ChecklistLabel
from risk_index.core.constants import (
    FEATURE_SEPARATOR,
    CHECKLIST_SCORE_BULL,
    CHECKLIST_SCORE_WATCH,
    CHECKLIST_SCORE_BEAR,
    COL_CHECKLIST_SCORE,
    COL_CHECKLIST_LABEL,
)
from risk_index.core.logger import get_logger

logger = get_logger(__name__)


def build_checklist(
    features_df: pd.DataFrame,
    composites_df: pd.DataFrame,
    checklist_cfg: ChecklistConfig,
) -> pd.DataFrame:
    """Build checklist scores and labels.


    Args:
        features_df: DataFrame with all features
        composites_df: DataFrame with composite scores
        checklist_cfg: Checklist configuration

    Returns:
        DataFrame with item scores and aggregate score
    """
    logger.info("Building checklist")

    # Combine features and composites for rule evaluation
    combined_df = features_df.join(composites_df, how="outer")

    result = pd.DataFrame(index=combined_df.index)

    # Evaluate each checklist item
    item_scores = {}
    item_signals = {}

    for item in checklist_cfg.items:
        signal = evaluate_item(combined_df, item)
        score = signal_to_score(signal)

        item_signals[f"{item.id}_signal"] = signal
        item_scores[f"{item.id}_score"] = score

    # Add item columns to result
    for col, values in item_signals.items():
        result[col] = values
    for col, values in item_scores.items():
        result[col] = values

    # Calculate aggregate score
    result[COL_CHECKLIST_SCORE] = compute_aggregate_score(
        {item.id: item_scores[f"{item.id}_score"] for item in checklist_cfg.items},
        {item.id: item.weight for item in checklist_cfg.items},
    )

    # Assign labels
    result[COL_CHECKLIST_LABEL] = result[COL_CHECKLIST_SCORE].apply(
        lambda x: score_to_label(x, checklist_cfg.score_thresholds)
    )

    # Validate checklist
    validation = validate_checklist(result, checklist_cfg)
    logger.info(f"Checklist validation: {validation['items_evaluated']}/{len(checklist_cfg.items)} items")

    return result


def evaluate_item(
    df: pd.DataFrame,
    item,
) -> pd.Series:
    """Evaluate a single checklist item.

    Args:
        df: DataFrame with features and composites
        item: ChecklistItemConfig

    Returns:
        Series of Signal values
    """
    return evaluate_rule(df, item.rule)


def evaluate_rule(
    df: pd.DataFrame,
    rule,
) -> pd.Series:
    """Evaluate a rule and return signal series.

    Args:
        df: DataFrame with data
        rule: Rule configuration

    Returns:
        Series of Signal values
    """
    if rule.type == "threshold_rule":
        return evaluate_threshold_rule(df, rule)
    elif rule.type == "trend_rule":
        return evaluate_trend_rule(df, rule)
    elif rule.type == "compound_rule":
        return evaluate_compound_rule(df, rule)
    else:
        logger.warning(f"Unknown rule type: {rule.type}")
        return pd.Series(Signal.WATCH.value, index=df.index)


def evaluate_threshold_rule(
    df: pd.DataFrame,
    rule,
) -> pd.Series:
    """Evaluate a threshold-based rule.

    Args:
        df: DataFrame with data
        rule: ThresholdRuleConfig

    Returns:
        Series of Signal values
    """
    # Get the series to evaluate
    if rule.feature:
        col = f"{rule.series}{FEATURE_SEPARATOR}{rule.feature}"
    else:
        col = rule.series

    if col not in df.columns:
        logger.warning(f"Column {col} not found for threshold rule")
        return pd.Series(Signal.WATCH.value, index=df.index)

    values = df[col]
    signal = pd.Series(Signal.WATCH.value, index=df.index)

    if rule.operator in ("gt", "gte"):
        # Bull if above bull threshold, Bear if below bear threshold
        if rule.operator == "gt":
            bull_mask = values > rule.bull_threshold
            bear_mask = values <= rule.bear_threshold
        else:
            bull_mask = values >= rule.bull_threshold
            bear_mask = values < rule.bear_threshold

        signal[bull_mask] = Signal.BULL.value
        signal[bear_mask] = Signal.BEAR.value

    elif rule.operator in ("lt", "lte"):
        # Bull if below bull threshold, Bear if above bear threshold
        if rule.operator == "lt":
            bull_mask = values < rule.bull_threshold
            bear_mask = values >= rule.bear_threshold
        else:
            bull_mask = values <= rule.bull_threshold
            bear_mask = values > rule.bear_threshold

        signal[bull_mask] = Signal.BULL.value
        signal[bear_mask] = Signal.BEAR.value

    return signal


def evaluate_trend_rule(
    df: pd.DataFrame,
    rule,
) -> pd.Series:
    """Evaluate a trend-based rule.

    Args:
        df: DataFrame with data
        rule: TrendRuleConfig

    Returns:
        Series of Signal values
    """
    col = f"{rule.series}{FEATURE_SEPARATOR}{rule.feature}"

    if col not in df.columns:
        logger.warning(f"Column {col} not found for trend rule")
        return pd.Series(Signal.WATCH.value, index=df.index)

    values = df[col]
    signal = pd.Series(Signal.WATCH.value, index=df.index)

    # Bull if trend > bull threshold, Bear if trend < bear threshold
    signal[values > rule.bull_threshold] = Signal.BULL.value
    signal[values < rule.bear_threshold] = Signal.BEAR.value

    return signal


def evaluate_compound_rule(
    df: pd.DataFrame,
    rule,
) -> pd.Series:
    """Evaluate a compound rule with multiple conditions.

    Args:
        df: DataFrame with data
        rule: CompoundRuleConfig

    Returns:
        Series of Signal values
    """
    condition_signals = []
    for condition in rule.conditions:
        cond_signal = evaluate_rule(df, condition)
        condition_signals.append(cond_signal)

    if not condition_signals:
        return pd.Series(Signal.WATCH.value, index=df.index)

    # Combine signals based on logic
    signals_df = pd.DataFrame(condition_signals).T

    if rule.logic == "all":
        # All must be BULL for BULL, any BEAR for BEAR
        all_bull = (signals_df == Signal.BULL.value).all(axis=1)
        any_bear = (signals_df == Signal.BEAR.value).any(axis=1)

        signal = pd.Series(Signal.WATCH.value, index=df.index)
        signal[all_bull] = Signal.BULL.value
        signal[any_bear & ~all_bull] = Signal.BEAR.value

    else:  # "any"
        # Any BULL for BULL, all BEAR for BEAR
        any_bull = (signals_df == Signal.BULL.value).any(axis=1)
        all_bear = (signals_df == Signal.BEAR.value).all(axis=1)

        signal = pd.Series(Signal.WATCH.value, index=df.index)
        signal[any_bull] = Signal.BULL.value
        signal[all_bear & ~any_bull] = Signal.BEAR.value

    return signal


def signal_to_score(signal: pd.Series) -> pd.Series:
    """Convert signal series to numeric score.

    Args:
        signal: Series of Signal values

    Returns:
        Series of scores (1.0, 0.5, 0.0)
    """
    score_map = {
        Signal.BULL.value: CHECKLIST_SCORE_BULL,
        Signal.WATCH.value: CHECKLIST_SCORE_WATCH,
        Signal.BEAR.value: CHECKLIST_SCORE_BEAR,
    }
    return signal.map(score_map)


def compute_aggregate_score(
    item_scores: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """Compute weighted aggregate checklist score.

    Args:
        item_scores: Dict mapping item ID to score series
        weights: Dict mapping item ID to weight

    Returns:
        Aggregate score series [0, 100]
    """
    if not item_scores:
        return pd.Series(50.0)

    # Get common index
    first_series = list(item_scores.values())[0]
    index = first_series.index

    weighted_sum = pd.Series(0.0, index=index)
    total_weight = 0.0

    for item_id, score_series in item_scores.items():
        weight = weights.get(item_id, 1.0)
        weighted_sum += score_series.fillna(CHECKLIST_SCORE_WATCH) * weight
        total_weight += weight

    if total_weight == 0:
        return pd.Series(50.0, index=index)

    # Normalize to 0-100 scale
    raw_score = weighted_sum / total_weight
    scaled_score = raw_score * 100

    return scaled_score


def score_to_label(score: float, thresholds: dict) -> str:
    """Convert score to label.

    Args:
        score: Aggregate score [0, 100]
        thresholds: Dict with 'confirmed_risk_on' and 'on_watch' thresholds

    Returns:
        Label string
    """
    if pd.isna(score):
        return ChecklistLabel.ON_WATCH.value

    if score >= thresholds.get("confirmed_risk_on", 75):
        return ChecklistLabel.CONFIRMED_RISK_ON.value
    elif score >= thresholds.get("on_watch", 50):
        return ChecklistLabel.ON_WATCH.value
    else:
        return ChecklistLabel.RISK_OFF.value


def validate_checklist(
    result_df: pd.DataFrame,
    checklist_cfg: ChecklistConfig,
) -> dict:
    """Validate checklist results.

    Args:
        result_df: Checklist results DataFrame
        checklist_cfg: Checklist configuration

    Returns:
        Validation metrics
    """
    items_evaluated = sum(
        1
        for item in checklist_cfg.items
        if f"{item.id}_score" in result_df.columns
    )

    # Get score statistics
    score = result_df.get(COL_CHECKLIST_SCORE, pd.Series())

    if score.empty:
        return {"items_evaluated": 0, "total_items": len(checklist_cfg.items)}

    # Count label distribution
    labels = result_df.get(COL_CHECKLIST_LABEL, pd.Series())
    label_dist = labels.value_counts(normalize=True).to_dict() if not labels.empty else {}

    return {
        "items_evaluated": items_evaluated,
        "total_items": len(checklist_cfg.items),
        "score_mean": round(score.mean(), 1),
        "score_min": round(score.min(), 1),
        "score_max": round(score.max(), 1),
        "label_distribution": label_dist,
    }


def get_checklist_detail(
    result_df: pd.DataFrame,
    checklist_cfg: ChecklistConfig,
    date: pd.Timestamp | None = None,
) -> list[dict]:
    """Get detailed checklist breakdown for a specific date.

    Args:
        result_df: Checklist results DataFrame
        checklist_cfg: Checklist configuration
        date: Date for detail (defaults to latest)

    Returns:
        List of item detail dicts
    """
    if date is None:
        date = result_df.index[-1]

    details = []

    for item in checklist_cfg.items:
        signal_col = f"{item.id}_signal"
        score_col = f"{item.id}_score"

        detail = {
            "id": item.id,
            "name": item.name,
            "category": item.category,
            "weight": item.weight,
            "description": item.description,
        }

        if signal_col in result_df.columns and date in result_df.index:
            detail["signal"] = result_df.loc[date, signal_col]

        if score_col in result_df.columns and date in result_df.index:
            detail["score"] = result_df.loc[date, score_col]

        details.append(detail)

    return details


def get_checklist_summary(
    result_df: pd.DataFrame,
    date: pd.Timestamp | None = None,
) -> dict:
    """Get checklist summary for a specific date.

    Args:
        result_df: Checklist results DataFrame
        date: Date for summary (defaults to latest)

    Returns:
        Summary dict
    """
    if date is None:
        date = result_df.index[-1]

    summary = {"date": date.strftime("%Y-%m-%d")}

    if COL_CHECKLIST_SCORE in result_df.columns and date in result_df.index:
        summary["score"] = round(result_df.loc[date, COL_CHECKLIST_SCORE], 1)

    if COL_CHECKLIST_LABEL in result_df.columns and date in result_df.index:
        summary["label"] = result_df.loc[date, COL_CHECKLIST_LABEL]

    # Count signals
    signal_cols = [c for c in result_df.columns if c.endswith("_signal")]
    if signal_cols and date in result_df.index:
        row = result_df.loc[date, signal_cols]
        summary["bull_count"] = (row == Signal.BULL.value).sum()
        summary["watch_count"] = (row == Signal.WATCH.value).sum()
        summary["bear_count"] = (row == Signal.BEAR.value).sum()

    return summary
